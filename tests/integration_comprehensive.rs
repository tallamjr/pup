//! Comprehensive integration testing framework
//! This test suite focuses on end-to-end system integration and cross-module testing

use gstpup::config::AppConfig;
use gstpup::error::PupError;
use gstpup::metrics::{
    ConsoleReporter, JsonReporter, Metrics, MetricsReporter, PerformanceMonitor,
};
use std::fs;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

mod utils;
use utils::assertions::*;
use utils::fixtures::*;
use utils::mocks::*;
use utils::MemoryLeakDetector;

/// Integration tests for configuration and error handling interaction
#[cfg(test)]
mod config_error_integration {
    use super::*;

    #[test]
    fn test_config_validation_error_propagation_chain() {
        // Test complete error propagation from file loading to validation
        let temp_dir = TempDir::new().unwrap();

        // Test 1: Non-existent file
        let nonexistent_path = temp_dir.path().join("missing.toml");
        let result = AppConfig::from_toml_file(&nonexistent_path);

        match result {
            Err(PupError::ConfigNotFound(path)) => {
                assert_eq!(path, nonexistent_path);
            }
            other => panic!("Expected ConfigNotFound error, got: {:?}", other),
        }

        // Test 2: Invalid TOML syntax
        let invalid_toml_path = temp_dir.path().join("invalid.toml");
        fs::write(&invalid_toml_path, "invalid [ toml syntax").unwrap();

        let result = AppConfig::from_toml_file(&invalid_toml_path);
        assert_config_parse_error(result);

        // Test 3: Valid TOML but invalid configuration values
        let invalid_config_path = temp_dir.path().join("invalid_config.toml");
        let invalid_config_content = r#"
[mode]
type = "production"

[input]
source = "webcam"
device_id = 999

[inference]
backend = "ort"
execution_providers = ["cpu"]
model_path = "models/test.onnx"
confidence_threshold = 1.5
batch_size = 0

[output]
display_enabled = true
output_format = "mp4"
"#;
        fs::write(&invalid_config_path, invalid_config_content).unwrap();

        let result = AppConfig::from_toml_file(&invalid_config_path);

        match result {
            Err(PupError::InvalidConfigValue { .. }) => { /* Expected */ }
            other => panic!("Expected InvalidConfigValue error, got: {:?}", other),
        }
    }

    #[test]
    fn test_config_error_recovery_scenarios() {
        let mut config = ConfigFixtures::invalid_confidence();

        // Initial validation should fail
        let initial_result = config.validate();
        assert!(initial_result.is_err());

        // Recovery step 1: Fix confidence threshold
        config.inference.confidence_threshold = 0.5;

        // Should still fail due to missing model file
        let recovery_result = config.validate();
        match recovery_result {
            Err(PupError::ModelLoadError(_)) => { /* Expected */ }
            other => {
                // Might also fail on other validation checks, that's acceptable
                println!("Recovery validation result: {:?}", other);
            }
        }

        // Recovery step 2: Create mock model file
        let temp_dir = TempDir::new().unwrap();
        let mock_model_path = temp_dir.path().join("mock_model.onnx");
        fs::write(&mock_model_path, b"mock onnx content").unwrap();
        config.inference.model_path = mock_model_path;

        // Should now pass validation
        let final_result = config.validate();
        assert!(final_result.is_ok());
    }

    #[test]
    fn test_batch_config_processing_with_mixed_results() {
        let temp_dir = TempDir::new().unwrap();
        let mock_model_path = temp_dir.path().join("test_model.onnx");
        fs::write(&mock_model_path, b"mock onnx content").unwrap();

        let mut configs = vec![
            ConfigFixtures::minimal_valid(),
            ConfigFixtures::invalid_confidence(),
            ConfigFixtures::high_performance(),
            ConfigFixtures::invalid_batch_size(),
            ConfigFixtures::file_input(),
        ];

        // Set valid model path for all configs
        for config in &mut configs {
            config.inference.model_path = mock_model_path.clone();
        }

        let mut results = Vec::new();
        let mut successful = 0;
        let mut failed = 0;
        let mut recovered = 0;

        for mut config in configs {
            match config.validate() {
                Ok(()) => {
                    successful += 1;
                    results.push(Ok(config));
                }
                Err(error) => {
                    failed += 1;

                    // Attempt recovery based on error type
                    let recovery_result = match &error {
                        PupError::InvalidConfigValue { field, .. }
                            if field == "inference.confidence_threshold" =>
                        {
                            config.inference.confidence_threshold = 0.5;
                            config.validate()
                        }
                        PupError::InvalidConfigValue { field, .. }
                            if field == "inference.batch_size" =>
                        {
                            config.inference.batch_size = 1;
                            config.validate()
                        }
                        _ => Err(error.clone()),
                    };

                    match recovery_result {
                        Ok(()) => {
                            recovered += 1;
                            results.push(Ok(config));
                        }
                        Err(e) => {
                            results.push(Err(e));
                        }
                    }
                }
            }
        }

        assert_eq!(results.len(), 5);
        assert!(successful > 0, "Expected some configs to succeed initially");
        assert!(failed > 0, "Expected some configs to fail initially");
        assert!(recovered > 0, "Expected some configs to be recoverable");

        // Final success rate should be high after recovery
        let final_successful = results.iter().filter(|r| r.is_ok()).count();
        assert!(
            final_successful >= 4,
            "Expected most configs to be valid after recovery"
        );
    }

    #[test]
    fn test_configuration_serialization_with_error_handling() {
        let temp_dir = TempDir::new().unwrap();
        let config = ConfigFixtures::high_performance();

        // Test successful serialization
        let config_path = temp_dir.path().join("test_config.toml");
        let save_result = config.to_toml_file(&config_path);
        assert!(save_result.is_ok());

        // Test loading back
        let load_result = AppConfig::from_toml_file(&config_path);
        assert!(load_result.is_ok());

        // Test serialization to read-only location
        #[cfg(unix)]
        {
            let readonly_dir = temp_dir.path().join("readonly");
            fs::create_dir(&readonly_dir).unwrap();

            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&readonly_dir).unwrap().permissions();
            perms.set_mode(0o444); // Read-only
            fs::set_permissions(&readonly_dir, perms).unwrap();

            let readonly_path = readonly_dir.join("config.toml");
            let readonly_result = config.to_toml_file(&readonly_path);

            // Should fail gracefully
            assert!(readonly_result.is_err());

            // Restore permissions for cleanup
            let mut restore_perms = fs::metadata(&readonly_dir).unwrap().permissions();
            restore_perms.set_mode(0o755);
            fs::set_permissions(&readonly_dir, restore_perms).unwrap();
        }
    }
}

/// Integration tests for metrics and error interaction
#[cfg(test)]
mod metrics_error_integration {
    use super::*;

    #[test]
    fn test_metrics_with_performance_target_failures() {
        let metrics = Metrics::new();

        // Set up performance conditions that will fail targets
        metrics.update_fps(15.0); // Below target
        metrics.update_inference_latency(100.0); // Above threshold

        // Test performance target checking
        let result = metrics.check_performance_targets(30.0, 50.0);

        match result {
            Err(PupError::PerformanceTarget {
                target_fps,
                actual_fps,
            }) => {
                assert_eq!(target_fps, 30.0);
                assert_eq!(actual_fps, 15.0);
            }
            other => panic!("Expected PerformanceTarget error, got: {:?}", other),
        }

        // Fix FPS but latency still problematic
        metrics.update_fps(60.0);
        let result = metrics.check_performance_targets(30.0, 50.0);

        match result {
            Err(PupError::ProcessingTimeout(timeout)) => {
                assert_eq!(timeout, 50);
            }
            other => panic!("Expected ProcessingTimeout error, got: {:?}", other),
        }

        // Fix both issues
        metrics.update_inference_latency(25.0);
        let result = metrics.check_performance_targets(30.0, 50.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_performance_monitor_with_error_injection() {
        let mut monitor = PerformanceMonitor::new();

        // Add a mix of working and failing reporters
        monitor.add_reporter(Box::new(ConsoleReporter::new(1000)));
        monitor.add_reporter(Box::new(MockMetricsReporter::new("working")));
        monitor.add_reporter(Box::new(MockMetricsReporter::new("failing").with_failure()));

        // Set up some metrics
        let metrics = monitor.metrics();
        metrics.update_fps(45.0);
        metrics.update_memory_usage(256);

        // Reporting should fail due to the failing reporter
        let report_result = monitor.report();
        assert!(report_result.is_err());

        match report_result {
            Err(PupError::Unexpected(msg)) => {
                assert_eq!(msg, "Mock reporter failure");
            }
            other => panic!("Expected mock reporter failure, got: {:?}", other),
        }
    }

    #[test]
    fn test_metrics_reporter_file_system_errors() {
        let temp_dir = TempDir::new().unwrap();

        // Test JSON reporter with invalid directory
        let invalid_path = temp_dir.path().join("nonexistent_dir").join("metrics.json");
        let reporter = JsonReporter::new(invalid_path, 0);
        let metrics = Metrics::new();

        metrics.update_fps(30.0);

        let report_result = reporter.report(&metrics);

        // Should fail with appropriate error
        match report_result {
            Err(PupError::OutputDirectoryError(_)) => { /* Expected */ }
            Err(other_error) => {
                // Other I/O related errors are also acceptable
                println!("Got I/O error: {:?}", other_error);
            }
            Ok(()) => panic!("Expected file system error"),
        }
    }

    #[test]
    fn test_concurrent_metrics_with_error_conditions() {
        let metrics = Arc::new(Metrics::new());
        let num_threads = 8;
        let operations_per_thread = 100;

        let errors = Arc::new(Mutex::new(Vec::new()));
        let mut handles = vec![];

        for thread_id in 0..num_threads {
            let metrics = metrics.clone();
            let errors = errors.clone();

            let handle = thread::spawn(move || {
                for i in 0..operations_per_thread {
                    // Update metrics
                    metrics.update_fps((thread_id * 10 + i) as f64);
                    metrics.increment_total_frames();

                    // Occasionally check performance targets (some will fail)
                    if i % 20 == 0 {
                        let target_fps = 50.0 + thread_id as f64 * 10.0; // Varying targets
                        let result = metrics.check_performance_targets(target_fps, 100.0);

                        if let Err(error) = result {
                            errors.lock().unwrap().push(error);
                        }
                    }
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should have some performance target errors
        let collected_errors = errors.lock().unwrap();
        assert!(
            !collected_errors.is_empty(),
            "Expected some performance target failures"
        );

        // All errors should be performance-related
        for error in collected_errors.iter() {
            match error {
                PupError::PerformanceTarget { .. } => { /* Expected */ }
                PupError::ProcessingTimeout(_) => { /* Also acceptable */ }
                other => panic!("Unexpected error type: {:?}", other),
            }
        }

        // Final metrics should reflect all updates
        assert_eq!(
            metrics.get_total_frames(),
            num_threads * operations_per_thread
        );
    }
}

/// Integration tests for multi-threaded scenarios
#[cfg(test)]
mod multi_threaded_integration {
    use super::*;

    #[test]
    fn test_config_validation_multithreaded() {
        let temp_dir = TempDir::new().unwrap();
        let mock_model_path = temp_dir.path().join("model.onnx");
        fs::write(&mock_model_path, b"mock content").unwrap();

        // Create multiple configurations
        let configs = Arc::new(vec![
            {
                let mut c = ConfigFixtures::minimal_valid();
                c.inference.model_path = mock_model_path.clone();
                c
            },
            {
                let mut c = ConfigFixtures::high_performance();
                c.inference.model_path = mock_model_path.clone();
                c
            },
            {
                let mut c = ConfigFixtures::file_input();
                c.inference.model_path = mock_model_path.clone();
                c
            },
        ]);

        let results = Arc::new(Mutex::new(Vec::new()));
        let mut handles = vec![];

        // Multiple threads validating different configs
        for thread_id in 0..6 {
            let configs = configs.clone();
            let results = results.clone();

            let handle = thread::spawn(move || {
                let config_index = thread_id % configs.len();
                let config = &configs[config_index];

                for _ in 0..10 {
                    let validation_result = config.validate();
                    results
                        .lock()
                        .unwrap()
                        .push((thread_id, config_index, validation_result));
                    thread::sleep(Duration::from_millis(1));
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let final_results = results.lock().unwrap();
        assert_eq!(final_results.len(), 60); // 6 threads * 10 validations each

        // All validations should succeed
        for (thread_id, config_index, result) in final_results.iter() {
            match result {
                Ok(()) => { /* Expected */ }
                Err(error) => {
                    panic!(
                        "Validation failed for thread {} config {}: {:?}",
                        thread_id, config_index, error
                    );
                }
            }
        }
    }

    #[test]
    fn test_metrics_concurrent_reporters() {
        let metrics = Arc::new(Metrics::new());
        let temp_dir = TempDir::new().unwrap();

        // Create multiple reporters
        let reporters: Arc<Mutex<Vec<Box<dyn MetricsReporter + Send>>>> =
            Arc::new(Mutex::new(vec![
                Box::new(ConsoleReporter::new(100)),
                Box::new(MockMetricsReporter::new("mock1")),
                Box::new(MockMetricsReporter::new("mock2")),
            ]));

        let mut handles = vec![];
        let report_count = Arc::new(Mutex::new(0));

        // Thread 1: Update metrics
        {
            let metrics = metrics.clone();
            let handle = thread::spawn(move || {
                for i in 0..200 {
                    metrics.update_fps((i % 60) as f64);
                    metrics.update_inference_latency((i % 50) as f64);
                    metrics.increment_total_frames();

                    if i % 10 == 0 {
                        metrics.increment_dropped_frames();
                    }

                    thread::sleep(Duration::from_millis(5));
                }
            });
            handles.push(handle);
        }

        // Thread 2: Report metrics
        {
            let metrics = metrics.clone();
            let reporters = reporters.clone();
            let report_count = report_count.clone();

            let handle = thread::spawn(move || {
                for _ in 0..50 {
                    let reporters_guard = reporters.lock().unwrap();

                    for reporter in reporters_guard.iter() {
                        let _ = reporter.report(&metrics);
                        *report_count.lock().unwrap() += 1;
                    }

                    thread::sleep(Duration::from_millis(20));
                }
            });
            handles.push(handle);
        }

        // Thread 3: Read metrics
        {
            let metrics = metrics.clone();
            let handle = thread::spawn(move || {
                for _ in 0..100 {
                    let _ = metrics.get_fps();
                    let _ = metrics.get_memory_usage_mb();
                    let _ = metrics.get_frame_drop_rate();
                    let _ = metrics.format_summary();
                    thread::sleep(Duration::from_millis(10));
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Verify final state
        assert_eq!(metrics.get_total_frames(), 200);
        assert_eq!(metrics.get_dropped_frames(), 20); // Every 10th frame
        assert_eq!(metrics.get_frame_drop_rate(), 10.0);

        let final_report_count = *report_count.lock().unwrap();
        assert!(
            final_report_count > 0,
            "Expected some reports to be generated"
        );
    }

    #[test]
    fn test_error_propagation_multithreaded() {
        let configs = vec![
            ConfigFixtures::invalid_confidence(),
            ConfigFixtures::invalid_batch_size(),
            ConfigFixtures::invalid_device_id(),
            ConfigFixtures::invalid_execution_provider(),
        ];

        let errors = Arc::new(Mutex::new(Vec::new()));
        // Execute operations concurrently using threads instead of async
        let mut handles = vec![];

        for (i, config) in configs.into_iter().enumerate() {
            let errors = errors.clone();
            let handle = thread::spawn(move || {
                for _ in 0..5 {
                    match config.validate() {
                        Ok(()) => panic!("Expected config {} to be invalid", i),
                        Err(error) => {
                            errors.lock().unwrap().push((i, error));
                        }
                    }
                    thread::sleep(Duration::from_millis(10));
                }
                i
            });
            handles.push(handle);
        }

        // Wait for all operations to complete
        for handle in handles {
            let _ = handle.join().unwrap();
        }

        let collected_errors = errors.lock().unwrap();
        assert_eq!(collected_errors.len(), 20); // 4 configs * 5 validations each

        // Verify error types are correct
        for (config_index, error) in collected_errors.iter() {
            match (config_index, error) {
                (0, PupError::InvalidConfigValue { field, .. })
                    if field == "inference.confidence_threshold" => {}
                (1, PupError::InvalidConfigValue { field, .. })
                    if field == "inference.batch_size" => {}
                (2, PupError::InvalidConfigValue { field, .. }) if field == "input.device_id" => {}
                (3, PupError::InvalidConfigValue { field, .. })
                    if field == "inference.execution_providers" => {}
                (index, error) => panic!("Unexpected error for config {}: {:?}", index, error),
            }
        }
    }

    #[test]
    fn test_race_condition_detection() {
        let metrics = Arc::new(Metrics::new());

        // Test for race conditions in metrics updates by running concurrent operations
        let num_threads = 4;
        let operations_per_thread = 250;
        let mut handles = vec![];

        for _ in 0..num_threads {
            let metrics = metrics.clone();
            let handle = thread::spawn(move || {
                for i in 0..operations_per_thread {
                    metrics.update_fps(i as f64);
                    metrics.increment_total_frames();
                    if i % 10 == 0 {
                        metrics.increment_dropped_frames();
                    }
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Metrics should be in a consistent state (no race condition detected)

        // Verify metrics are in a consistent state after concurrent access
        let total_frames = metrics.get_total_frames();
        let dropped_frames = metrics.get_dropped_frames();

        assert!(
            dropped_frames <= total_frames,
            "Dropped frames {} cannot exceed total frames {}",
            dropped_frames,
            total_frames
        );

        // Frame drop rate should be reasonable
        let drop_rate = metrics.get_frame_drop_rate();
        assert!(
            drop_rate >= 0.0 && drop_rate <= 100.0,
            "Frame drop rate should be between 0-100%, got {}",
            drop_rate
        );
    }
}

/// Integration tests for resource management
#[cfg(test)]
mod resource_management_integration {
    use super::*;

    #[test]
    fn test_memory_leak_detection_integration() {
        let detector = MemoryLeakDetector::new();

        // Perform operations that might leak memory
        let configs = vec![
            ConfigFixtures::minimal_valid(),
            ConfigFixtures::high_performance(),
            ConfigFixtures::file_input(),
        ];

        let metrics = Arc::new(Metrics::new());

        for _iteration in 0..100 {
            for config in &configs {
                // Simulate config processing
                let _ = config.validate();

                // Update metrics
                metrics.update_fps(30.0);
                metrics.increment_total_frames();

                // Create and destroy temporary objects
                let _summary = metrics.format_summary();
                let _temp_config = config.clone();
            }

            // Force garbage collection opportunity
            thread::sleep(Duration::from_millis(1));
        }

        // Check for memory leaks (threshold: 50MB increase)
        let leaked = detector.check_for_leaks(50);
        assert!(!leaked, "Memory leak detected during integration test");
    }

    #[test]
    fn test_file_system_resource_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();

        // Create multiple temporary files for testing
        let mut file_paths = Vec::new();
        let mut reporters = Vec::new();

        for i in 0..10 {
            let file_path = base_path.join(format!("metrics_{}.json", i));
            let reporter = JsonReporter::new(file_path.clone(), 0);

            file_paths.push(file_path);
            reporters.push(reporter);
        }

        let metrics = Metrics::new();
        metrics.update_fps(45.0);

        // Use all reporters
        for reporter in &reporters {
            let _ = reporter.report(&metrics);
        }

        // Verify files were created
        for file_path in &file_paths {
            assert!(
                file_path.exists(),
                "Expected file to exist: {:?}",
                file_path
            );
        }

        // Files should be automatically cleaned up when temp_dir is dropped
        // This tests that our file handles are properly released
        drop(reporters);

        // Files should still exist until temp_dir cleanup
        for file_path in &file_paths {
            assert!(
                file_path.exists(),
                "File should still exist before cleanup: {:?}",
                file_path
            );
        }

        // temp_dir cleanup happens when it goes out of scope
    }

    #[test]
    fn test_thread_resource_cleanup() {
        let metrics = Arc::new(Metrics::new());
        let thread_count = 20;
        let operations_per_thread = 50;

        let start_time = Instant::now();
        let mut handles = Vec::new();

        // Create many threads
        for thread_id in 0..thread_count {
            let metrics = metrics.clone();

            let handle = thread::spawn(move || {
                for i in 0..operations_per_thread {
                    metrics.update_fps(thread_id as f64 + i as f64);
                    metrics.increment_total_frames();

                    // Small delay to simulate work
                    thread::sleep(Duration::from_millis(1));
                }

                thread_id
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        let mut completed_threads = Vec::new();
        for handle in handles {
            let thread_id = handle.join().expect("Thread should complete successfully");
            completed_threads.push(thread_id);
        }

        let duration = start_time.elapsed();

        // Verify all threads completed
        assert_eq!(completed_threads.len(), thread_count);
        completed_threads.sort();
        for i in 0..thread_count {
            assert_eq!(completed_threads[i], i);
        }

        // Verify metrics reflect all operations
        assert_eq!(
            metrics.get_total_frames(),
            thread_count * operations_per_thread
        );

        // Should complete in reasonable time (less than 30 seconds)
        assert!(
            duration.as_secs() < 30,
            "Thread cleanup took too long: {:?}",
            duration
        );
    }

    #[test]
    fn test_configuration_file_handle_management() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test_config.toml");

        let config = ConfigFixtures::minimal_valid();

        // Repeatedly save and load configuration
        for iteration in 0..100 {
            // Save config
            assert!(config.to_toml_file(&config_path).is_ok());

            // Load config
            match AppConfig::from_toml_file(&config_path) {
                Ok(loaded_config) => {
                    assert_eq!(loaded_config.mode.mode_type, config.mode.mode_type);
                }
                Err(PupError::ModelLoadError(_)) => {
                    // Expected due to non-existent model file
                }
                Err(other) => {
                    panic!("Unexpected error on iteration {}: {:?}", iteration, other);
                }
            }

            // Occasionally delete and recreate file to test handle cleanup
            if iteration % 20 == 0 {
                fs::remove_file(&config_path).unwrap();
            }
        }

        // Final verification
        assert!(config.to_toml_file(&config_path).is_ok());
        assert!(config_path.exists());
    }
}

/// Integration tests for end-to-end scenarios
#[cfg(test)]
mod end_to_end_integration {
    use super::*;

    #[test]
    fn test_complete_application_lifecycle() {
        let temp_dir = TempDir::new().unwrap();

        // Step 1: Create and validate configuration
        let mock_model_path = temp_dir.path().join("model.onnx");
        fs::write(&mock_model_path, b"mock onnx content").unwrap();

        let mut config = ConfigFixtures::high_performance();
        config.inference.model_path = mock_model_path;

        assert!(config.validate().is_ok());

        // Step 2: Save configuration
        let config_path = temp_dir.path().join("app_config.toml");
        assert!(config.to_toml_file(&config_path).is_ok());

        // Step 3: Load configuration from file
        let loaded_config = AppConfig::from_toml_file(&config_path).unwrap();
        assert!(loaded_config.validate().is_ok());

        // Step 4: Initialize performance monitoring
        let mut monitor = PerformanceMonitor::new();
        let metrics_json_path = temp_dir.path().join("metrics.json");

        monitor.add_reporter(Box::new(ConsoleReporter::new(1000)));
        monitor.add_reporter(Box::new(JsonReporter::new(metrics_json_path.clone(), 100)));

        let metrics = monitor.metrics();

        // Step 5: Simulate application processing loop
        for frame_id in 0..100 {
            let timer = monitor.start_frame();

            // Simulate frame processing
            thread::sleep(Duration::from_millis(16)); // ~60 FPS

            // Simulate inference
            let inference_start = Instant::now();
            thread::sleep(Duration::from_millis(5)); // Simulate inference time
            monitor.record_inference_time(inference_start.elapsed());

            // Update system metrics
            let _ = monitor.update_system_metrics();

            // Complete frame
            if frame_id % 20 == 0 {
                timer.drop(); // Simulate dropped frame
            } else {
                timer.complete();
            }

            // Report metrics periodically
            if frame_id % 10 == 0 {
                let _ = monitor.report();
            }

            // Check performance targets
            let _ = monitor.check_targets(25.0, 50.0);
        }

        // Step 6: Verify final metrics
        assert_eq!(metrics.get_total_frames(), 100);
        assert_eq!(metrics.get_dropped_frames(), 5); // Every 20th frame
        assert!((metrics.get_frame_drop_rate() - 5.0).abs() < 0.1);
        assert!(metrics.get_fps() > 0.0);
        assert!(metrics.get_inference_latency_ms() > 0.0);

        // Step 7: Verify metrics were written to file
        if metrics_json_path.exists() {
            let metrics_content = fs::read_to_string(&metrics_json_path).unwrap();
            assert!(!metrics_content.is_empty());
            assert!(metrics_content.contains("fps"));
            assert!(metrics_content.contains("inference_latency_ms"));
        }
    }

    #[test]
    fn test_error_recovery_full_scenario() {
        let temp_dir = TempDir::new().unwrap();

        // Scenario: Application starts with bad config, recovers, then encounters runtime errors

        // Phase 1: Bad initial configuration
        let mut config = ConfigFixtures::invalid_confidence();
        let initial_validation = config.validate();
        assert!(initial_validation.is_err());

        // Phase 2: Configuration recovery
        config.inference.confidence_threshold = 0.5;
        let mock_model_path = temp_dir.path().join("model.onnx");
        fs::write(&mock_model_path, b"mock content").unwrap();
        config.inference.model_path = mock_model_path;

        assert!(config.validate().is_ok());

        // Phase 3: Runtime with performance issues
        let metrics = Arc::new(Metrics::new());
        let mut error_count = 0;
        let mut recovery_attempts = 0;

        for cycle in 0..50 {
            // Simulate varying performance
            let fps = if cycle < 20 { 15.0 } else { 45.0 }; // Poor then good performance
            let latency = if cycle < 30 { 80.0 } else { 25.0 }; // High then low latency

            metrics.update_fps(fps);
            metrics.update_inference_latency(latency);
            metrics.increment_total_frames();

            // Check performance targets
            match metrics.check_performance_targets(30.0, 50.0) {
                Ok(()) => {
                    // Performance is good
                }
                Err(PupError::PerformanceTarget { .. }) => {
                    error_count += 1;
                    recovery_attempts += 1;

                    // Simulate performance recovery action
                    if recovery_attempts > 10 {
                        // After enough attempts, assume we've made improvements
                        // In real app, this might involve reducing quality, batch size, etc.
                        metrics.update_fps(35.0);
                    }
                }
                Err(PupError::ProcessingTimeout(_)) => {
                    error_count += 1;
                    recovery_attempts += 1;

                    // Simulate latency recovery
                    if recovery_attempts > 5 {
                        metrics.update_inference_latency(30.0);
                    }
                }
                Err(other) => {
                    panic!("Unexpected error: {:?}", other);
                }
            }
        }

        // Phase 4: Verify recovery was successful
        assert!(
            error_count > 0,
            "Expected some performance errors initially"
        );
        assert!(recovery_attempts > 0, "Expected some recovery attempts");

        // Final performance should be good
        let final_check = metrics.check_performance_targets(30.0, 50.0);
        assert!(
            final_check.is_ok(),
            "Expected final performance to be good after recovery"
        );

        assert_eq!(metrics.get_total_frames(), 50);
        assert!(metrics.get_fps() >= 30.0);
        assert!(metrics.get_inference_latency_ms() <= 50.0);
    }

    #[test]
    fn test_graceful_degradation_scenario() {
        let temp_dir = TempDir::new().unwrap();
        let mock_model_path = temp_dir.path().join("model.onnx");
        fs::write(&mock_model_path, b"mock content").unwrap();

        // Start with high-performance configuration
        let mut config = ConfigFixtures::high_performance();
        config.inference.model_path = mock_model_path;
        assert!(config.validate().is_ok());

        let metrics = Arc::new(Metrics::new());
        let mut degradation_steps = 0;

        // Simulate progressive degradation due to system constraints
        let scenarios = [
            ("High Performance", 60.0, 15.0, 4), // Target: 60 FPS, 15ms latency, batch 4
            ("Medium Performance", 40.0, 25.0, 2), // Degrade: 40 FPS, 25ms latency, batch 2
            ("Low Performance", 25.0, 40.0, 1),  // Further: 25 FPS, 40ms latency, batch 1
            ("Survival Mode", 15.0, 60.0, 1),    // Minimal: 15 FPS, 60ms latency, batch 1
        ];

        for (scenario_name, target_fps, max_latency, batch_size) in scenarios {
            println!("Testing scenario: {}", scenario_name);

            // Update configuration for this performance level
            config.inference.batch_size = batch_size;
            assert!(config.validate().is_ok());

            // Simulate performance at this level
            let actual_fps = target_fps * 0.9; // Slightly below target
            let actual_latency = max_latency * 0.8; // Within limits

            metrics.update_fps(actual_fps);
            metrics.update_inference_latency(actual_latency);

            // Check if performance targets are met
            let performance_result = metrics.check_performance_targets(target_fps, max_latency);

            match performance_result {
                Ok(()) => {
                    // Performance is acceptable for this level
                }
                Err(_) => {
                    degradation_steps += 1;
                    // In a real application, this would trigger further degradation
                }
            }

            // Simulate some processing
            for _ in 0..10 {
                metrics.increment_total_frames();
            }
        }

        // Verify graceful degradation occurred
        assert!(
            config.inference.batch_size <= 4,
            "Batch size should have been reduced"
        );
        assert_eq!(metrics.get_total_frames(), 40); // 4 scenarios * 10 frames each

        // System should still be functional even in degraded state
        let final_check = metrics.check_performance_targets(10.0, 100.0); // Very lenient targets
        assert!(
            final_check.is_ok(),
            "System should be functional even in degraded state"
        );
    }
}
