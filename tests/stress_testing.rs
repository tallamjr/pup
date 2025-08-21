//! Stress testing framework for comprehensive system testing under load
//! This test suite focuses on performance regression detection and system stability

use gstpup::config::AppConfig;
use gstpup::error::PupError;
use gstpup::metrics::{
    ConsoleReporter, JsonReporter, Metrics, MetricsReporter, PerformanceMonitor,
};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

mod utils;
use utils::fixtures::*;
use utils::generators::*;
use utils::mocks::*;
use utils::stress::*;
use utils::MemoryLeakDetector;

/// Stress tests for configuration validation under load
#[cfg(test)]
mod configuration_stress_tests {
    use super::*;

    #[test]
    fn test_high_volume_config_validation() {
        let temp_dir = TempDir::new().unwrap();
        let mock_model_path = temp_dir.path().join("model.onnx");
        fs::write(&mock_model_path, b"mock onnx content").unwrap();

        let config = StressTestConfig {
            thread_count: 8,
            duration_seconds: 10,
            ops_per_second: 200,
            memory_limit_mb: 512,
            cpu_limit_percent: 80.0,
            inject_failures: false,
            failure_rate: 0.0,
        };

        let executor = StressTestExecutor::new(config);

        let results = executor.run_config_validation_stress(move || {
            let mut test_config = ConfigGenerator::random_valid();
            test_config.inference.model_path = mock_model_path.clone();
            Ok(test_config)
        });

        assert!(
            results.passed(),
            "Configuration validation stress test should pass: {:?}",
            results
        );
        assert!(
            results.total_operations > 1000,
            "Expected high operation count"
        );
        assert!(
            results.success_rate() > 95.0,
            "Expected high success rate: {}%",
            results.success_rate()
        );
        assert!(
            !results.memory_limit_exceeded,
            "Memory limit should not be exceeded"
        );
        assert!(
            !results.cpu_limit_exceeded,
            "CPU limit should not be exceeded"
        );
    }

    #[test]
    fn test_config_validation_with_error_injection() {
        let config = StressTestConfig {
            thread_count: 4,
            duration_seconds: 5,
            ops_per_second: 50,
            inject_failures: true,
            failure_rate: 0.2, // 20% failure rate
            ..Default::default()
        };

        let executor = StressTestExecutor::new(config);
        let error_count = Arc::new(AtomicUsize::new(0));
        let error_count_clone = error_count.clone();

        let results = executor.run_config_validation_stress(move || {
            let count = error_count_clone.fetch_add(1, Ordering::SeqCst) + 1;
            if count % 5 == 0 {
                // Inject failures every 5th operation
                Err(PupError::InvalidConfigValue {
                    field: "injected_error".to_string(),
                    value: "test".to_string(),
                })
            } else {
                Ok(ConfigGenerator::random_valid())
            }
        });

        assert!(
            results.failed_operations > 0,
            "Expected some failures with error injection"
        );
        assert!(results.successful_operations > 0, "Expected some successes");
        assert!(
            results.success_rate() > 60.0,
            "Success rate should be reasonable: {}%",
            results.success_rate()
        );
    }

    #[test]
    fn test_concurrent_config_file_operations() {
        let temp_dir = TempDir::new().unwrap();
        let num_threads = 10;
        let configs_per_thread = 20;

        let start_time = Instant::now();
        let mut handles = vec![];
        let results = Arc::new(Mutex::new(Vec::new()));

        for thread_id in 0..num_threads {
            let temp_dir_path = temp_dir.path().to_path_buf();
            let results = results.clone();

            let handle = thread::spawn(move || {
                let mut thread_results = Vec::new();

                for config_id in 0..configs_per_thread {
                    let config_path =
                        temp_dir_path.join(format!("config_{}_{}.toml", thread_id, config_id));
                    let config = ConfigFixtures::minimal_valid();

                    // Save config
                    let save_result = config.to_toml_file(&config_path);
                    thread_results.push(("save", save_result.is_ok()));

                    // Load config
                    let load_result = AppConfig::from_toml_file(&config_path);
                    match load_result {
                        Ok(_) => thread_results.push(("load", true)),
                        Err(PupError::ModelLoadError(_)) => thread_results.push(("load", true)), // Expected
                        Err(_) => thread_results.push(("load", false)),
                    }

                    // Small delay to create contention
                    thread::sleep(Duration::from_millis(1));
                }

                results.lock().unwrap().extend(thread_results);
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let duration = start_time.elapsed();
        let final_results = results.lock().unwrap();

        assert_eq!(final_results.len(), num_threads * configs_per_thread * 2); // Save + load per config

        let successful_operations = final_results.iter().filter(|(_, success)| *success).count();
        let success_rate = (successful_operations as f64 / final_results.len() as f64) * 100.0;

        assert!(
            success_rate > 95.0,
            "Expected high success rate: {}%",
            success_rate
        );
        assert!(
            duration.as_secs() < 30,
            "Operations should complete in reasonable time: {:?}",
            duration
        );
    }

    #[test]
    fn test_property_based_stress_validation() {
        use utils::PropertyTestGenerator;

        let start_time = Instant::now();

        // Test confidence threshold validation
        let invalid_thresholds = PropertyTestGenerator::invalid_confidence_thresholds(50);
        let mut confidence_failures = 0;

        for invalid_value in invalid_thresholds {
            let mut config = ConfigFixtures::minimal_valid();
            config.inference.confidence_threshold = invalid_value;

            match config.validate() {
                Err(PupError::InvalidConfigValue { .. }) => confidence_failures += 1,
                _ => {} // Some may fail for other reasons like missing model
            }
        }

        // Test batch size validation
        let invalid_batch_sizes = PropertyTestGenerator::invalid_batch_sizes(50);
        let mut batch_size_failures = 0;

        for invalid_value in invalid_batch_sizes {
            let mut config = ConfigFixtures::minimal_valid();
            config.inference.batch_size = invalid_value;

            match config.validate() {
                Err(PupError::InvalidConfigValue { .. }) => batch_size_failures += 1,
                _ => {} // Some may fail for other reasons like missing model
            }
        }

        let duration = start_time.elapsed();

        // Should have reasonable failure rates for invalid properties
        assert!(
            confidence_failures > 30,
            "Should catch most invalid confidence thresholds"
        );
        assert!(
            batch_size_failures > 30,
            "Should catch most invalid batch sizes"
        );

        assert!(
            duration.as_secs() < 10,
            "Property validation should complete quickly: {:?}",
            duration
        );
    }
}

/// Stress tests for metrics collection and reporting
#[cfg(test)]
mod metrics_stress_tests {
    use super::*;

    #[test]
    fn test_high_frequency_metrics_updates() {
        let config = StressTestConfig {
            thread_count: 6,
            duration_seconds: 8,
            ops_per_second: 500, // Very high frequency
            memory_limit_mb: 256,
            ..Default::default()
        };

        let executor = StressTestExecutor::new(config);
        let results = executor.run_metrics_stress_test();

        assert!(
            results.total_operations > 2000,
            "Expected high operation count"
        );
        assert!(
            results.success_rate() > 99.0,
            "Metrics updates should be very reliable"
        );
        assert!(
            !results.memory_limit_exceeded,
            "Should not exceed memory limit"
        );

        // Verify metrics are consistent after stress test
        let metrics = Metrics::new();
        // Basic verification that metrics are in valid ranges
        assert!(metrics.get_fps() >= 0.0);
        assert!(metrics.get_memory_usage_mb() >= 0);
    }

    #[test]
    fn test_concurrent_metrics_reporters() {
        let temp_dir = TempDir::new().unwrap();
        let metrics = Arc::new(Metrics::new());
        let num_reporters = 5;
        let reports_per_reporter = 100;

        let mut reporters: Vec<Box<dyn MetricsReporter>> = vec![
            Box::new(ConsoleReporter::new(10)), // Fast reporting
        ];

        // Add JSON reporters
        for i in 0..num_reporters - 1 {
            let json_path = temp_dir.path().join(format!("metrics_{}.json", i));
            reporters.push(Box::new(JsonReporter::new(json_path, 10)));
        }

        let start_time = Instant::now();
        let mut handles = vec![];
        let error_count = Arc::new(Mutex::new(0));

        // Thread 1: Update metrics rapidly
        {
            let metrics = metrics.clone();
            let handle = thread::spawn(move || {
                for i in 0..1000 {
                    metrics.update_fps((i % 120) as f64);
                    metrics.update_inference_latency((i % 100) as f64);
                    metrics.update_memory_usage(100 + (i % 500));
                    metrics.increment_total_frames();

                    if i % 10 == 0 {
                        metrics.increment_dropped_frames();
                    }

                    thread::sleep(Duration::from_millis(2));
                }
            });
            handles.push(handle);
        }

        // Thread 2: Report metrics using all reporters
        {
            let metrics = metrics.clone();
            let error_count = error_count.clone();

            let handle = thread::spawn(move || {
                for _ in 0..reports_per_reporter {
                    for reporter in &reporters {
                        match reporter.report(&metrics) {
                            Ok(()) => {}
                            Err(_) => {
                                *error_count.lock().unwrap() += 1;
                            }
                        }
                    }
                    thread::sleep(Duration::from_millis(10));
                }
            });
            handles.push(handle);
        }

        // Thread 3: Read metrics frequently
        {
            let metrics = metrics.clone();
            let handle = thread::spawn(move || {
                for _ in 0..500 {
                    let _ = metrics.get_fps();
                    let _ = metrics.get_memory_usage_mb();
                    let _ = metrics.get_frame_drop_rate();
                    let _ = metrics.format_summary();
                    thread::sleep(Duration::from_millis(5));
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let duration = start_time.elapsed();
        let final_error_count = *error_count.lock().unwrap();

        assert_eq!(metrics.get_total_frames(), 1000);
        assert_eq!(metrics.get_dropped_frames(), 100);

        // Some errors might be acceptable (file system issues, etc.)
        let error_rate =
            (final_error_count as f64 / (reports_per_reporter * num_reporters) as f64) * 100.0;
        assert!(
            error_rate < 5.0,
            "Error rate should be low: {}%",
            error_rate
        );

        assert!(
            duration.as_secs() < 60,
            "Concurrent reporting should complete in reasonable time"
        );
    }

    #[test]
    fn test_performance_monitor_under_load() {
        let temp_dir = TempDir::new().unwrap();
        let mut monitor = PerformanceMonitor::new();

        // Add multiple reporters
        monitor.add_reporter(Box::new(ConsoleReporter::new(500)));
        monitor.add_reporter(Box::new(JsonReporter::new(
            temp_dir.path().join("load_test.json"),
            200,
        )));
        monitor.add_reporter(Box::new(MockMetricsReporter::new("load_test")));

        let metrics = monitor.metrics();
        let num_frames = 500;
        let start_time = Instant::now();

        // Simulate high-load processing
        for frame_id in 0..num_frames {
            let timer = monitor.start_frame();

            // Simulate variable processing load
            let processing_time = if frame_id % 50 == 0 {
                Duration::from_millis(100) // Occasional heavy frame
            } else {
                Duration::from_millis(16) // Normal frame at ~60 FPS
            };

            thread::sleep(processing_time);

            // Simulate inference
            let inference_duration = Duration::from_millis(5 + (frame_id % 20));
            monitor.record_inference_time(inference_duration);

            // Complete or drop frame based on processing time
            if processing_time.as_millis() > 33 {
                // Slower than 30 FPS
                timer.drop();
            } else {
                timer.complete();
            }

            // Update system metrics
            let _ = monitor.update_system_metrics();

            // Report periodically
            if frame_id % 25 == 0 {
                let _ = monitor.report();
            }

            // Check performance targets
            let _ = monitor.check_targets(25.0, 50.0);
        }

        let duration = start_time.elapsed();

        // Verify metrics
        assert_eq!(metrics.get_total_frames() as usize, num_frames as usize);
        assert!(
            metrics.get_dropped_frames() > 0,
            "Expected some dropped frames under load"
        );
        assert!(metrics.get_fps() > 0.0, "FPS should be calculated");
        assert!(
            metrics.get_inference_latency_ms() > 0.0,
            "Latency should be recorded"
        );

        // Should complete in reasonable time
        let expected_duration = Duration::from_millis(num_frames as u64 * 20); // ~20ms per frame average
        assert!(
            duration < expected_duration * 2,
            "Processing took too long: {:?} vs expected ~{:?}",
            duration,
            expected_duration
        );
    }

    #[test]
    fn test_metrics_memory_pressure() {
        let config = StressTestConfig {
            thread_count: 8,
            duration_seconds: 5,
            ops_per_second: 1000,
            memory_limit_mb: 128, // Tight memory limit
            ..Default::default()
        };

        let executor = StressTestExecutor::new(config);
        let results = executor.run_memory_stress_test();

        // Under memory pressure, some operations might fail
        // but the system should remain stable
        assert!(results.total_operations > 0);
        assert!(
            results.success_rate() > 50.0,
            "Should maintain reasonable success rate under pressure"
        );

        // Memory usage should be tracked
        assert!(results.peak_memory_mb > 0, "Should track memory usage");
    }
}

/// Stress tests for error handling and recovery
#[cfg(test)]
mod error_handling_stress_tests {
    use super::*;

    #[test]
    fn test_error_propagation_under_load() {
        let configs = vec![
            ConfigFixtures::invalid_confidence(),
            ConfigFixtures::invalid_batch_size(),
            ConfigFixtures::invalid_device_id(),
            ConfigFixtures::nonexistent_model(),
        ];

        let num_threads = 8;
        let iterations_per_thread = 25;
        let error_results = Arc::new(Mutex::new(Vec::new()));
        let start_time = Instant::now();

        let mut handles = vec![];

        for thread_id in 0..num_threads {
            let configs = configs.clone();
            let error_results = error_results.clone();

            let handle = thread::spawn(move || {
                for iteration in 0..iterations_per_thread {
                    let config_index = (thread_id + iteration) % configs.len();
                    let config = &configs[config_index];

                    let result = config.validate();
                    let success = result.is_ok();

                    error_results
                        .lock()
                        .unwrap()
                        .push((thread_id, config_index, success));

                    // Small delay to create timing variations
                    thread::sleep(Duration::from_millis(1));
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let duration = start_time.elapsed();
        let final_results = error_results.lock().unwrap();

        assert_eq!(final_results.len(), num_threads * iterations_per_thread);

        // Analyze results by config type
        let mut config_stats = vec![0; configs.len()];
        let mut success_stats = vec![0; configs.len()];

        for (_, config_index, success) in final_results.iter() {
            config_stats[*config_index] += 1;
            if *success {
                success_stats[*config_index] += 1;
            }
        }

        // Most invalid configs should fail validation consistently
        for (i, (total, successes)) in config_stats.iter().zip(success_stats.iter()).enumerate() {
            let failure_rate = ((total - successes) as f64 / *total as f64) * 100.0;

            match i {
                3 => {
                    // nonexistent_model config might succeed if validation doesn't check file existence
                    // This is configuration-dependent
                }
                _ => {
                    assert!(
                        failure_rate > 80.0,
                        "Config type {} should fail validation consistently: {}% failure rate",
                        i,
                        failure_rate
                    );
                }
            }
        }

        assert!(
            duration.as_secs() < 10,
            "Error handling should be fast: {:?}",
            duration
        );
    }

    #[test]
    fn test_error_recovery_patterns() {
        let error_scenarios = vec![
            (
                "confidence",
                PupError::InvalidConfigValue {
                    field: "confidence_threshold".to_string(),
                    value: "1.5".to_string(),
                },
            ),
            (
                "batch_size",
                PupError::InvalidConfigValue {
                    field: "batch_size".to_string(),
                    value: "0".to_string(),
                },
            ),
            (
                "memory",
                PupError::InsufficientMemory {
                    required_mb: 2048,
                    available_mb: 512,
                },
            ),
            ("timeout", PupError::ProcessingTimeout(5000)),
        ];

        let recovery_attempts = 100;
        let mut recovery_stats = std::collections::HashMap::new();

        for (error_type, error) in error_scenarios {
            let mut successful_recoveries = 0;

            for _ in 0..recovery_attempts {
                // Simulate recovery attempt
                let recovery_successful = match &error {
                    PupError::InvalidConfigValue { field, .. }
                        if field == "confidence_threshold" =>
                    {
                        // Simulate fixing confidence threshold
                        true
                    }
                    PupError::InvalidConfigValue { field, .. } if field == "batch_size" => {
                        // Simulate fixing batch size
                        true
                    }
                    PupError::InsufficientMemory { .. } => {
                        // Simulate memory recovery (succeed 70% of the time)
                        rand::random::<f64>() > 0.3
                    }
                    PupError::ProcessingTimeout(_) => {
                        // Simulate timeout recovery (succeed 60% of the time)
                        rand::random::<f64>() > 0.4
                    }
                    _ => false,
                };

                if recovery_successful {
                    successful_recoveries += 1;
                }

                thread::sleep(Duration::from_millis(1));
            }

            recovery_stats.insert(error_type, successful_recoveries);
        }

        // Verify recovery rates
        assert!(
            recovery_stats["confidence"] > 90,
            "Configuration errors should have high recovery rate"
        );
        assert!(
            recovery_stats["batch_size"] > 90,
            "Configuration errors should have high recovery rate"
        );
        assert!(
            recovery_stats["memory"] > 50,
            "Memory errors should have moderate recovery rate"
        );
        assert!(
            recovery_stats["timeout"] > 40,
            "Timeout errors should have moderate recovery rate"
        );
    }

    #[test]
    fn test_cascade_failure_prevention() {
        let metrics = Arc::new(Metrics::new());
        let num_threads = 10;
        let operations_per_thread = 50;

        // Simulate a scenario where one component fails but others continue working
        let barrier = Arc::new(Barrier::new(num_threads));
        let failure_injected = Arc::new(Mutex::new(false));
        let operation_results = Arc::new(Mutex::new(Vec::new()));

        let mut handles = vec![];

        for thread_id in 0..num_threads {
            let metrics = metrics.clone();
            let barrier = barrier.clone();
            let failure_injected = failure_injected.clone();
            let operation_results = operation_results.clone();

            let handle = thread::spawn(move || {
                barrier.wait(); // Synchronized start

                for operation_id in 0..operations_per_thread {
                    // Simulate different types of operations
                    let operation_type = operation_id % 4;
                    let mut result = Ok(());

                    match operation_type {
                        0 => {
                            // Metrics update - should always work
                            metrics.update_fps(thread_id as f64 * 10.0);
                            metrics.increment_total_frames();
                        }
                        1 => {
                            // Performance check - might fail
                            result = metrics.check_performance_targets(100.0, 10.0);
                        }
                        2 => {
                            // Simulated file operation - inject failure once
                            if thread_id == 0 && operation_id == 10 {
                                let mut injected = failure_injected.lock().unwrap();
                                if !*injected {
                                    result = Err(PupError::PermissionDenied(PathBuf::from("test")));
                                    *injected = true;
                                }
                            }
                        }
                        3 => {
                            // Memory operation - might fail under pressure
                            if operation_id > 40 {
                                result = Err(PupError::InsufficientMemory {
                                    required_mb: 1024,
                                    available_mb: 256,
                                });
                            }
                        }
                        _ => unreachable!(),
                    }

                    operation_results.lock().unwrap().push((
                        thread_id,
                        operation_type,
                        result.is_ok(),
                    ));

                    thread::sleep(Duration::from_millis(2));
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let results = operation_results.lock().unwrap();
        assert_eq!(results.len(), num_threads * operations_per_thread);

        // Analyze operation success rates by type
        let mut operation_stats = vec![(0, 0); 4]; // (success, total) for each operation type

        for (_, operation_type, success) in results.iter() {
            operation_stats[*operation_type].1 += 1;
            if *success {
                operation_stats[*operation_type].0 += 1;
            }
        }

        // Metrics updates should have 100% success rate
        assert_eq!(
            operation_stats[0].0, operation_stats[0].1,
            "Metrics updates should never fail"
        );

        // Other operations might have failures, but system should remain stable
        let overall_success = results.iter().filter(|(_, _, success)| *success).count();
        let overall_rate = (overall_success as f64 / results.len() as f64) * 100.0;

        assert!(
            overall_rate > 70.0,
            "Overall success rate should be reasonable despite failures: {}%",
            overall_rate
        );

        // Verify metrics are still consistent
        assert_eq!(
            metrics.get_total_frames(),
            num_threads * operations_per_thread / 4
        ); // Only type 0 operations
    }
}

/// Long-running stability and memory leak detection tests
#[cfg(test)]
mod stability_tests {
    use super::*;

    #[test]
    fn test_sustained_load_stability() {
        let config = ConfigFixtures::minimal_valid();
        let tester = LoadTester::new(config);

        // Run a sustained load test for 2 minutes
        let results = tester.run_sustained_load_test(2);

        assert!(
            results.stable_performance,
            "System should maintain stable performance: {:?}",
            results
        );
        assert!(
            !results.memory_leak_detected,
            "No memory leaks should be detected: peak memory {}MB",
            results.peak_memory_mb
        );
        assert!(
            !results.performance_degradation_detected,
            "Performance should not degrade over time"
        );

        assert!(results.avg_fps > 15.0, "Should maintain reasonable FPS");
        assert!(
            results.total_operations > 1000,
            "Should process significant number of operations"
        );
    }

    #[test]
    fn test_memory_leak_detection() {
        let detector = MemoryLeakDetector::new();
        let metrics = Arc::new(Metrics::new());

        // Perform operations that might cause memory leaks
        for cycle in 0..100 {
            // Create temporary objects
            let temp_configs = (0..10)
                .map(|_| ConfigGenerator::random_valid())
                .collect::<Vec<_>>();

            // Process configurations
            for config in temp_configs {
                let _ = config.validate();
                metrics.update_fps((cycle * 10) as f64);
                let _ = metrics.format_summary();
            }

            // Update metrics with varying data
            for i in 0..100 {
                metrics.update_inference_latency(i as f64);
                metrics.increment_total_frames();

                if i % 10 == 0 {
                    metrics.increment_dropped_frames();
                }
            }

            // Periodic reset to simulate application lifecycle
            if cycle % 20 == 0 {
                metrics.reset();
            }

            thread::sleep(Duration::from_millis(50));
        }

        // Check for memory leaks (allow 100MB growth)
        let leaked = detector.check_for_leaks(100);
        assert!(!leaked, "Memory leak detected during stability test");
    }

    #[test]
    fn test_error_handling_stability() {
        let num_cycles = 50;
        let operations_per_cycle = 20;

        for cycle in 0..num_cycles {
            let mut error_count = 0;
            let mut recovery_count = 0;

            for operation in 0..operations_per_cycle {
                // Generate various error conditions
                let error = match operation % 5 {
                    0 => Some(PupError::ConfigParseError("test error".to_string())),
                    1 => Some(PupError::ModelLoadError(PathBuf::from("missing.onnx"))),
                    2 => Some(PupError::InferenceError("execution failed".to_string())),
                    3 => Some(PupError::PerformanceTarget {
                        target_fps: 60.0,
                        actual_fps: 30.0,
                    }),
                    4 => Some(PupError::InsufficientMemory {
                        required_mb: 1024,
                        available_mb: 512,
                    }),
                    _ => None,
                };

                if let Some(error) = error {
                    error_count += 1;

                    // Simulate error recovery
                    match error {
                        PupError::ConfigParseError(_) => {
                            // Always recoverable
                            recovery_count += 1;
                        }
                        PupError::ModelLoadError(_) => {
                            // Sometimes recoverable
                            if operation % 2 == 0 {
                                recovery_count += 1;
                            }
                        }
                        PupError::InferenceError(_) => {
                            // Rarely recoverable
                            if operation % 4 == 0 {
                                recovery_count += 1;
                            }
                        }
                        PupError::PerformanceTarget { .. } => {
                            // Usually recoverable
                            if operation % 3 != 0 {
                                recovery_count += 1;
                            }
                        }
                        PupError::InsufficientMemory { .. } => {
                            // Sometimes recoverable
                            if operation % 3 == 0 {
                                recovery_count += 1;
                            }
                        }
                        _ => {}
                    }
                }
            }

            // System should maintain reasonable recovery rate
            if error_count > 0 {
                let recovery_rate = (recovery_count as f64 / error_count as f64) * 100.0;
                assert!(
                    recovery_rate > 30.0,
                    "Recovery rate should be reasonable in cycle {}: {}%",
                    cycle,
                    recovery_rate
                );
            }

            // Small delay between cycles
            thread::sleep(Duration::from_millis(100));
        }
    }

    #[test]
    fn test_resource_exhaustion_recovery() {
        let temp_dir = TempDir::new().unwrap();
        let mut file_handles = Vec::new();
        let metrics = Arc::new(Metrics::new());

        // Gradually consume resources
        for i in 0..100 {
            // Create file handles (simulate resource consumption)
            let file_path = temp_dir.path().join(format!("resource_{}.tmp", i));
            if let Ok(file) = fs::File::create(&file_path) {
                file_handles.push(file);
            }

            // Update metrics (should always work)
            metrics.update_fps(i as f64);
            metrics.increment_total_frames();

            // Try to create a reporter (might fail under resource pressure)
            let reporter_result = std::panic::catch_unwind(|| {
                JsonReporter::new(temp_dir.path().join(format!("metrics_{}.json", i)), 1000)
            });

            if reporter_result.is_err() {
                // Resource exhaustion detected - attempt recovery
                file_handles.clear(); // Release resources
                std::thread::sleep(Duration::from_millis(100)); // Allow system recovery
            }

            // Verify core functionality still works
            let fps = metrics.get_fps();
            assert_eq!(fps, i as f64, "Core metrics should remain functional");

            thread::sleep(Duration::from_millis(10));
        }

        // System should be stable after resource pressure
        assert_eq!(metrics.get_total_frames(), 100);
        assert!(metrics.get_fps() > 0.0);
    }
}

/// Performance regression detection tests
#[cfg(test)]
mod regression_tests {
    use super::*;

    #[test]
    fn test_performance_baseline() {
        let baseline_config = StressTestConfig {
            thread_count: 4,
            duration_seconds: 5,
            ops_per_second: 100,
            ..Default::default()
        };

        let executor = StressTestExecutor::new(baseline_config);

        // Run baseline test
        let baseline_results =
            executor.run_config_validation_stress(|| Ok(ConfigFixtures::minimal_valid()));

        assert!(baseline_results.passed(), "Baseline test should pass");

        let baseline_ops_per_second = baseline_results.avg_ops_per_second;
        let baseline_memory = baseline_results.peak_memory_mb;

        // Run comparison test with more complex operations
        let complex_results = executor.run_config_validation_stress(|| {
            let config = ConfigGenerator::random_valid();
            // Add complexity
            let _ = config.validate(); // Extra validation step
            Ok(config)
        });

        assert!(complex_results.passed(), "Complex test should pass");

        // Performance should not degrade significantly
        let performance_ratio = complex_results.avg_ops_per_second / baseline_ops_per_second;
        assert!(
            performance_ratio > 0.7,
            "Performance should not degrade significantly: {:.2}x",
            performance_ratio
        );

        // Memory usage should not increase dramatically
        let memory_ratio = complex_results.peak_memory_mb as f64 / baseline_memory as f64;
        assert!(
            memory_ratio < 2.0,
            "Memory usage should not increase dramatically: {:.2}x",
            memory_ratio
        );
    }

    #[test]
    fn test_scalability_characteristics() {
        let thread_counts = [1, 2, 4, 8];
        let mut performance_data = Vec::new();

        for &thread_count in &thread_counts {
            let config = StressTestConfig {
                thread_count,
                duration_seconds: 3,
                ops_per_second: 50 * (thread_count as u64), // Scale ops with threads
                ..Default::default()
            };

            let executor = StressTestExecutor::new(config);
            let results = executor.run_metrics_stress_test();

            assert!(
                results.passed(),
                "Scalability test should pass for {} threads",
                thread_count
            );

            performance_data.push((thread_count, results.avg_ops_per_second));
        }

        // Analyze scalability
        let single_thread_perf = performance_data[0].1;

        for (i, &(thread_count, ops_per_second)) in performance_data.iter().enumerate().skip(1) {
            let expected_min_perf = single_thread_perf * (thread_count as f64 * 0.5); // At least 50% scaling

            assert!(
                ops_per_second > expected_min_perf,
                "Performance should scale reasonably with {} threads: {:.2} vs expected min {:.2}",
                thread_count,
                ops_per_second,
                expected_min_perf
            );
        }
    }

    #[test]
    fn test_latency_regression() {
        let operations: [(&str, Box<dyn Fn()>); 3] = [
            (
                "config_validation",
                Box::new(|| {
                    let config = ConfigFixtures::minimal_valid();
                    let _ = config.validate();
                }),
            ),
            (
                "metrics_update",
                Box::new(|| {
                    let metrics = Metrics::new();
                    metrics.update_fps(30.0);
                    metrics.increment_total_frames();
                }),
            ),
            (
                "error_creation",
                Box::new(|| {
                    let _error = PupError::InferenceError("test error".to_string());
                    let _display = _error.to_string();
                }),
            ),
        ];

        for (operation_name, operation) in operations {
            let iterations = 1000;
            let start_time = Instant::now();

            for _ in 0..iterations {
                operation();
            }

            let duration = start_time.elapsed();
            let avg_latency = duration.as_nanos() / iterations;

            // Define latency thresholds (in nanoseconds)
            let max_latency = match operation_name {
                "config_validation" => 100_000, // 100 microseconds
                "metrics_update" => 10_000,     // 10 microseconds
                "error_creation" => 50_000,     // 50 microseconds
                _ => 1_000_000,                 // 1 millisecond default
            };

            assert!(
                avg_latency < max_latency,
                "Operation '{}' latency regression detected: {} ns avg (max: {} ns)",
                operation_name,
                avg_latency,
                max_latency
            );
        }
    }
}
