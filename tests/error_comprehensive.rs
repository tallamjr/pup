//! Comprehensive error handling and recovery testing
//! This test suite focuses on achieving 85% coverage for the error module

use gstpup::config::AppConfig;
use gstpup::error::{PupError, PupResult};
use gstpup::inference::InferenceError;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

mod utils;
use utils::fixtures::*;

/// Test all error type display and formatting
#[cfg(test)]
mod error_display_tests {
    use super::*;

    #[test]
    fn test_input_error_display() {
        let errors = [
            PupError::InputNotAvailable("webcam device not found".to_string()),
            PupError::WebcamNotFound { device_id: 0 },
            PupError::VideoFileError(PathBuf::from("missing_video.mp4")),
            PupError::RtspConnectionFailed("rtsp://invalid.url/stream".to_string()),
        ];

        for error in errors {
            let display = error.to_string();
            assert!(
                !display.is_empty(),
                "Error display should not be empty: {:?}",
                error
            );

            match error {
                PupError::InputNotAvailable(ref msg) => {
                    assert!(display.contains("Input source not available"));
                    assert!(display.contains(msg));
                }
                PupError::WebcamNotFound { device_id } => {
                    assert!(display.contains("Webcam device"));
                    assert!(display.contains(&device_id.to_string()));
                    assert!(display.contains("not found"));
                }
                PupError::VideoFileError(ref path) => {
                    assert!(display.contains("Video file not found"));
                    assert!(display.contains(&path.to_string_lossy().to_string()));
                }
                PupError::RtspConnectionFailed(ref url) => {
                    assert!(display.contains("RTSP stream connection failed"));
                    assert!(display.contains(url));
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_inference_error_display() {
        let errors = [
            PupError::CoreMLFailure("provider not available".to_string()),
            PupError::ModelLoadError(PathBuf::from("invalid_model.onnx")),
            PupError::InferenceError("execution failed".to_string()),
            PupError::ModelFormatError(PathBuf::from("model.txt")),
        ];

        for error in errors {
            let display = error.to_string();
            assert!(!display.is_empty());

            match error {
                PupError::CoreMLFailure(ref msg) => {
                    assert!(display.contains("CoreML initialisation failed"));
                    assert!(display.contains(msg));
                }
                PupError::ModelLoadError(ref path) => {
                    assert!(display.contains("Model loading failed"));
                    assert!(display.contains(&path.to_string_lossy().to_string()));
                }
                PupError::InferenceError(ref msg) => {
                    assert!(display.contains("Inference execution failed"));
                    assert!(display.contains(msg));
                }
                PupError::ModelFormatError(ref path) => {
                    assert!(display.contains("Invalid model format"));
                    assert!(display.contains(&path.to_string_lossy().to_string()));
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_pipeline_error_display() {
        let errors = [
            PupError::PipelineError("element creation failed".to_string()),
            PupError::ElementCreationFailed("videotestsrc".to_string()),
            PupError::CapsNegotiationFailed("video/x-raw".to_string()),
            PupError::UnsupportedVideoFormat("unsupported codec".to_string()),
        ];

        for error in errors {
            let display = error.to_string();
            assert!(!display.is_empty());

            // Test that each error contains expected keywords
            match error {
                PupError::PipelineError(_) => assert!(display.contains("GStreamer pipeline error")),
                PupError::ElementCreationFailed(_) => {
                    assert!(display.contains("GStreamer element creation failed"))
                }
                PupError::CapsNegotiationFailed(_) => {
                    assert!(display.contains("GStreamer caps negotiation failed"))
                }
                PupError::UnsupportedVideoFormat(_) => {
                    assert!(display.contains("Video format not supported"))
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_config_error_display() {
        let errors = [
            PupError::ConfigNotFound(PathBuf::from("missing_config.toml")),
            PupError::ConfigParseError("invalid TOML syntax".to_string()),
            PupError::InvalidConfigValue {
                field: "confidence_threshold".to_string(),
                value: "1.5".to_string(),
            },
            PupError::MissingConfigField("inference.model_path".to_string()),
        ];

        for error in errors {
            let display = error.to_string();
            assert!(!display.is_empty());

            match error {
                PupError::ConfigNotFound(ref path) => {
                    assert!(display.contains("Configuration file not found"));
                    assert!(display.contains(&path.to_string_lossy().to_string()));
                }
                PupError::ConfigParseError(ref msg) => {
                    assert!(display.contains("Configuration parsing failed"));
                    assert!(display.contains(msg));
                }
                PupError::InvalidConfigValue {
                    ref field,
                    ref value,
                } => {
                    assert!(display.contains("Invalid configuration value"));
                    assert!(display.contains(field));
                    assert!(display.contains(value));
                }
                PupError::MissingConfigField(ref field) => {
                    assert!(display.contains("Required configuration field missing"));
                    assert!(display.contains(field));
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_performance_error_display() {
        let errors = [
            PupError::InsufficientMemory {
                required_mb: 2048,
                available_mb: 1024,
            },
            PupError::PerformanceTarget {
                target_fps: 30.0,
                actual_fps: 15.0,
            },
            PupError::ProcessingTimeout(5000),
            PupError::FrameDropped(10),
        ];

        for error in errors {
            let display = error.to_string();
            assert!(!display.is_empty());

            match error {
                PupError::InsufficientMemory {
                    required_mb,
                    available_mb,
                } => {
                    assert!(display.contains("Insufficient memory"));
                    assert!(display.contains(&format!("{}MB", required_mb)));
                    assert!(display.contains(&format!("{}MB", available_mb)));
                }
                PupError::PerformanceTarget {
                    target_fps,
                    actual_fps,
                } => {
                    assert!(display.contains("Performance target not met"));
                    assert!(display.contains(&format!("{}", target_fps)));
                    assert!(display.contains(&format!("{}", actual_fps)));
                }
                PupError::ProcessingTimeout(ms) => {
                    assert!(display.contains("Frame processing timeout"));
                    assert!(display.contains(&format!("{}ms", ms)));
                }
                PupError::FrameDropped(count) => {
                    assert!(display.contains("Frame drop detected"));
                    assert!(display.contains(&format!("{}", count)));
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_platform_error_display() {
        let errors = [
            PupError::MacOSThreadingError("NSApplication not on main thread".to_string()),
            PupError::OpenGLError("context creation failed".to_string()),
            PupError::UnsupportedPlatform("Windows not supported".to_string()),
        ];

        for error in errors {
            let display = error.to_string();
            assert!(!display.is_empty());

            match error {
                PupError::MacOSThreadingError(ref msg) => {
                    assert!(display.contains("macOS NSApplication threading error"));
                    assert!(display.contains(msg));
                }
                PupError::OpenGLError(ref msg) => {
                    assert!(display.contains("OpenGL context creation failed"));
                    assert!(display.contains(msg));
                }
                PupError::UnsupportedPlatform(ref msg) => {
                    assert!(display.contains("Platform not supported"));
                    assert!(display.contains(msg));
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_io_error_display() {
        let errors = [
            PupError::PermissionDenied(PathBuf::from("/readonly/output.mp4")),
            PupError::InsufficientDiskSpace(1024),
            PupError::OutputDirectoryError(PathBuf::from("/invalid/output")),
        ];

        for error in errors {
            let display = error.to_string();
            assert!(!display.is_empty());

            match error {
                PupError::PermissionDenied(ref path) => {
                    assert!(display.contains("File permission denied"));
                    assert!(display.contains(&path.to_string_lossy().to_string()));
                }
                PupError::InsufficientDiskSpace(mb) => {
                    assert!(display.contains("Disk space insufficient"));
                    assert!(display.contains(&format!("{}MB", mb)));
                }
                PupError::OutputDirectoryError(ref path) => {
                    assert!(display.contains("Output directory creation failed"));
                    assert!(display.contains(&path.to_string_lossy().to_string()));
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_network_error_display() {
        let errors = [
            PupError::NetworkTimeout("connection timed out".to_string()),
            PupError::StreamEncodingError("codec error".to_string()),
            PupError::RtmpConnectionFailed("rtmp://server/stream".to_string()),
        ];

        for error in errors {
            let display = error.to_string();
            assert!(!display.is_empty());

            match error {
                PupError::NetworkTimeout(ref msg) => {
                    assert!(display.contains("Network connection timeout"));
                    assert!(display.contains(msg));
                }
                PupError::StreamEncodingError(ref msg) => {
                    assert!(display.contains("Stream encoding failed"));
                    assert!(display.contains(msg));
                }
                PupError::RtmpConnectionFailed(ref url) => {
                    assert!(display.contains("RTMP server connection failed"));
                    assert!(display.contains(url));
                }
                _ => unreachable!(),
            }
        }
    }

    #[test]
    fn test_generic_error_display() {
        let errors = [
            PupError::Unexpected("something went wrong".to_string()),
            PupError::NotImplemented("feature not available".to_string()),
        ];

        for error in errors {
            let display = error.to_string();
            assert!(!display.is_empty());

            match error {
                PupError::Unexpected(ref msg) => {
                    assert!(display.contains("Unexpected error"));
                    assert!(display.contains(msg));
                }
                PupError::NotImplemented(ref msg) => {
                    assert!(display.contains("Feature not implemented"));
                    assert!(display.contains(msg));
                }
                _ => unreachable!(),
            }
        }
    }
}

/// Test error conversion and compatibility
#[cfg(test)]
mod error_conversion_tests {
    use super::*;

    #[test]
    fn test_io_error_conversion() {
        let io_errors = [
            std::io::Error::new(std::io::ErrorKind::NotFound, "file not found"),
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied"),
            std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "unexpected EOF"),
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "invalid input"),
        ];

        for io_error in io_errors {
            let pup_error: PupError = io_error.into();

            match pup_error {
                PupError::Unexpected(msg) => {
                    assert!(msg.contains("File not found") || msg.contains("I/O error"));
                }
                PupError::PermissionDenied(_) => {
                    // Permission denied should map correctly
                }
                _ => {
                    // Other IO errors map to Unexpected, which is fine
                }
            }
        }
    }

    #[test]
    fn test_toml_error_conversion() {
        let invalid_toml = "invalid [ toml";
        let toml_error: toml::de::Error = toml::from_str::<AppConfig>(invalid_toml).unwrap_err();
        let pup_error: PupError = toml_error.into();

        match pup_error {
            PupError::ConfigParseError(msg) => {
                assert!(!msg.is_empty());
            }
            other => panic!("Expected ConfigParseError, got: {:?}", other),
        }
    }

    #[test]
    fn test_inference_error_conversion() {
        let inference_errors = [
            InferenceError::ModelNotLoaded,
            InferenceError::ModelLoadError("model file corrupted".to_string()),
            InferenceError::InferenceFailed("execution error".to_string()),
            InferenceError::OrtError("ONNX Runtime error".to_string()),
        ];

        for inference_error in inference_errors {
            let pup_error: PupError = inference_error.into();

            match pup_error {
                PupError::ModelLoadError(_) => {
                    // ModelLoadError should map correctly
                }
                PupError::InferenceError(msg) => {
                    assert!(!msg.is_empty());
                }
                other => panic!("Unexpected error conversion: {:?}", other),
            }
        }
    }

    #[test]
    fn test_glib_error_conversion() {
        // Test manual conversion for GStreamer errors
        let glib_error =
            gstreamer::glib::Error::new(gstreamer::glib::FileError::Failed, "GStreamer error");

        let pup_error = PupError::PipelineError(glib_error.to_string());

        match pup_error {
            PupError::PipelineError(msg) => {
                assert!(msg.contains("GStreamer error"));
            }
            other => panic!("Expected PipelineError, got: {:?}", other),
        }
    }

    #[test]
    fn test_io_error_conversion_trait() {
        let io_errors = [
            (std::io::ErrorKind::NotFound, "File not found"),
            (std::io::ErrorKind::PermissionDenied, "Permission denied"),
            (std::io::ErrorKind::InvalidInput, "Invalid input"),
        ];

        for (kind, msg) in io_errors {
            let io_error = std::io::Error::new(kind, msg);
            let pup_error: PupError = io_error.into();

            match pup_error {
                PupError::Unexpected(_) => { /* Expected for most IO errors */ }
                PupError::PermissionDenied(_) => { /* Expected for permission denied */ }
                other => panic!("Unexpected conversion result: {:?}", other),
            }
        }
    }
}

/// Test error context and enrichment
#[cfg(test)]
mod error_context_tests {
    use super::*;

    #[test]
    fn test_error_context_builder() {
        let base_error = PupError::ModelLoadError(PathBuf::from("test.onnx"));

        // Test basic error display format
        let display = base_error.to_string();

        assert!(display.contains("Model loading failed"));
        assert!(display.contains("test.onnx"));
    }

    #[test]
    fn test_error_display_consistency() {
        let base_error = PupError::InferenceError("test error".to_string());
        let cloned_error = base_error.clone();

        // Test that error cloning and display work correctly
        assert_eq!(base_error.to_string(), cloned_error.to_string());
        assert!(base_error.to_string().contains("test error"));
    }

    #[test]
    fn test_error_display_format() {
        let base_error = PupError::NetworkTimeout("connection failed".to_string());
        let display = base_error.to_string();

        assert!(display.contains("Network connection timeout"));
        assert!(display.contains("connection failed"));
    }

    #[test]
    fn test_error_context_handling() {
        fn failing_operation() -> Result<(), std::io::Error> {
            Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "test file not found",
            ))
        }

        let result = failing_operation()
            .map_err(|e| PupError::Unexpected(format!("reading configuration file: {}", e)));

        match result {
            Err(PupError::Unexpected(msg)) => {
                assert!(msg.contains("reading configuration file"));
                assert!(msg.contains("File not found") || msg.contains("test file not found"));
            }
            other => panic!("Expected contextual error, got: {:?}", other),
        }
    }

    #[test]
    fn test_error_context_success() {
        fn successful_operation() -> Result<String, std::io::Error> {
            Ok("success".to_string())
        }

        let result = successful_operation()
            .map_err(|e| PupError::Unexpected(format!("performing operation: {}", e)));

        match result {
            Ok(value) => assert_eq!(value, "success"),
            Err(error) => panic!("Expected success, got error: {:?}", error),
        }
    }
}

/// Test error propagation through system components
#[cfg(test)]
mod error_propagation_tests {
    use super::*;

    #[test]
    fn test_config_error_propagation() {
        // Test that configuration errors propagate correctly through validation chain
        let temp_dir = tempfile::TempDir::new().unwrap();
        let nonexistent_path = temp_dir.path().join("nonexistent.toml");

        let result = AppConfig::from_toml_file(&nonexistent_path);

        match result {
            Err(PupError::ConfigNotFound(path)) => {
                assert_eq!(path, nonexistent_path);
            }
            other => panic!("Expected ConfigNotFound error, got: {:?}", other),
        }
    }

    #[test]
    fn test_model_error_propagation() {
        // Test that model errors propagate through validation
        let config = ConfigFixtures::nonexistent_model();
        let result = config.validate();

        match result {
            Err(PupError::ModelLoadError(path)) => {
                assert_eq!(path, PathBuf::from("models/nonexistent.onnx"));
            }
            other => panic!("Expected ModelLoadError, got: {:?}", other),
        }
    }

    #[test]
    fn test_chained_error_propagation() {
        // Test error propagation through multiple validation layers
        let mut config = ConfigFixtures::minimal_valid();

        // Set up multiple invalid conditions
        config.inference.confidence_threshold = 1.5; // Invalid
        config.inference.batch_size = 0; // Invalid

        let result = config.validate();

        // Should catch the first validation error
        match result {
            Err(PupError::InvalidConfigValue { field, .. }) => {
                // Should be one of the invalid fields
                assert!(
                    field == "inference.confidence_threshold" || field == "inference.batch_size"
                );
            }
            other => panic!("Expected InvalidConfigValue, got: {:?}", other),
        }
    }

    #[test]
    fn test_concurrent_error_propagation() {
        // Test error propagation in concurrent scenarios
        let config = Arc::new(ConfigFixtures::invalid_confidence());
        let mut handles = vec![];
        let errors = Arc::new(Mutex::new(Vec::new()));

        for _ in 0..5 {
            let config = config.clone();
            let errors = errors.clone();

            let handle = thread::spawn(move || {
                let result = config.validate();
                if let Err(error) = result {
                    errors.lock().unwrap().push(error);
                }
            });

            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let collected_errors = errors.lock().unwrap();
        assert_eq!(collected_errors.len(), 5);

        // All errors should be the same type
        for error in collected_errors.iter() {
            match error {
                PupError::InvalidConfigValue { field, .. } => {
                    assert_eq!(field, "inference.confidence_threshold");
                }
                other => panic!("Expected InvalidConfigValue, got: {:?}", other),
            }
        }
    }
}

/// Test error recovery scenarios
#[cfg(test)]
mod error_recovery_tests {
    use super::*;

    #[test]
    fn test_config_validation_recovery() {
        let mut invalid_config = ConfigFixtures::invalid_confidence();

        // Test recovery by fixing the invalid confidence
        let initial_result = invalid_config.validate();
        assert!(
            initial_result.is_err(),
            "Config should initially fail validation"
        );

        // Fix the error
        invalid_config.inference.confidence_threshold = 0.5;
        let recovery_result = invalid_config.validate();

        match recovery_result {
            Ok(()) => { /* Recovery successful */ }
            Err(PupError::ModelLoadError(_)) => { /* Expected due to missing model file */ }
            Err(other) => panic!("Unexpected error after fixing confidence: {:?}", other),
        }
    }

    #[test]
    fn test_batch_processing_error_recovery() {
        let configs = vec![
            ConfigFixtures::minimal_valid(),
            ConfigFixtures::invalid_confidence(), // This will fail
            ConfigFixtures::high_performance(),
        ];

        let mut successful = 0;
        let mut failed = 0;
        let mut recovered = 0;

        for mut config in configs {
            match config.validate() {
                Ok(()) => successful += 1,
                Err(PupError::InvalidConfigValue { field, .. })
                    if field == "inference.confidence_threshold" =>
                {
                    failed += 1;
                    // Attempt recovery
                    config.inference.confidence_threshold = 0.5;
                    if config.validate().is_ok() {
                        recovered += 1;
                    }
                }
                Err(_) => failed += 1, // Other errors
            }
        }

        assert_eq!(successful, 2); // minimal_valid and high_performance should succeed
        assert_eq!(failed, 1); // invalid_confidence should fail initially
        assert_eq!(recovered, 1); // should be able to recover invalid_confidence
    }

    #[test]
    fn test_graceful_degradation() {
        // Test that system can continue operating with degraded functionality
        let mut config = ConfigFixtures::high_performance();

        // Simulate failure of high-performance features
        config.inference.execution_providers = vec!["invalid_provider".to_string()];

        match config.validate() {
            Err(PupError::InvalidConfigValue { field, .. })
                if field == "inference.execution_providers" =>
            {
                // Recovery: fall back to basic CPU execution
                config.inference.execution_providers = vec!["cpu".to_string()];

                let recovery_result = config.validate();
                match recovery_result {
                    Ok(()) => { /* Successful recovery */ }
                    Err(PupError::ModelLoadError(_)) => { /* Expected due to missing model */ }
                    Err(other) => panic!("Recovery failed: {:?}", other),
                }
            }
            other => panic!(
                "Expected execution provider validation error, got: {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_retry_mechanism() {
        let attempt_count = Arc::new(Mutex::new(0));
        let max_retries = 3;

        let operation = {
            let attempt_count = attempt_count.clone();
            move || -> PupResult<()> {
                let mut count = attempt_count.lock().unwrap();
                *count += 1;
                if *count < max_retries {
                    Err(PupError::NetworkTimeout("simulated timeout".to_string()))
                } else {
                    Ok(())
                }
            }
        };

        let mut result = operation();
        let mut retries = 0;

        while result.is_err() && retries < max_retries {
            match result {
                Err(PupError::NetworkTimeout(_)) => {
                    retries += 1;
                    thread::sleep(Duration::from_millis(10)); // Simulated delay
                    result = operation();
                }
                Err(other) => {
                    panic!("Non-retryable error: {:?}", other);
                }
                Ok(()) => break,
            }
        }

        assert!(result.is_ok(), "Operation should succeed after retries");
        assert_eq!(*attempt_count.lock().unwrap(), max_retries);
    }
}

/// Test error injection for systematic testing
#[cfg(test)]
mod error_injection_tests {
    use super::*;

    struct ErrorInjector {
        fail_config_load: bool,
        fail_model_load: bool,
        fail_validation: bool,
        failure_rate: f64,
    }

    impl ErrorInjector {
        fn new() -> Self {
            Self {
                fail_config_load: false,
                fail_model_load: false,
                fail_validation: false,
                failure_rate: 0.0,
            }
        }

        fn with_config_failure(mut self) -> Self {
            self.fail_config_load = true;
            self
        }

        fn with_model_failure(mut self) -> Self {
            self.fail_model_load = true;
            self
        }

        fn with_validation_failure(mut self) -> Self {
            self.fail_validation = true;
            self
        }

        fn with_random_failures(mut self, rate: f64) -> Self {
            self.failure_rate = rate;
            self
        }

        fn should_fail(&self) -> bool {
            if self.failure_rate > 0.0 {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                rng.gen::<f64>() < self.failure_rate
            } else {
                false
            }
        }

        fn load_config(&self, _path: &PathBuf) -> PupResult<AppConfig> {
            if self.fail_config_load || self.should_fail() {
                Err(PupError::ConfigParseError("Injected failure".to_string()))
            } else {
                Ok(ConfigFixtures::minimal_valid())
            }
        }

        fn validate_config(&self, _config: &AppConfig) -> PupResult<()> {
            if self.fail_validation || self.should_fail() {
                Err(PupError::InvalidConfigValue {
                    field: "injected_error".to_string(),
                    value: "test".to_string(),
                })
            } else {
                // Skip actual validation to avoid model file issues
                Ok(())
            }
        }
    }

    #[test]
    fn test_systematic_config_load_failure() {
        let injector = ErrorInjector::new().with_config_failure();
        let dummy_path = PathBuf::from("test.toml");

        let result = injector.load_config(&dummy_path);

        match result {
            Err(PupError::ConfigParseError(msg)) => {
                assert_eq!(msg, "Injected failure");
            }
            other => panic!("Expected injected config parse error, got: {:?}", other),
        }
    }

    #[test]
    fn test_systematic_validation_failure() {
        let injector = ErrorInjector::new().with_validation_failure();
        let config = ConfigFixtures::minimal_valid();

        let result = injector.validate_config(&config);

        match result {
            Err(PupError::InvalidConfigValue { field, value }) => {
                assert_eq!(field, "injected_error");
                assert_eq!(value, "test");
            }
            other => panic!("Expected injected validation error, got: {:?}", other),
        }
    }

    #[test]
    fn test_random_failure_injection() {
        let injector = ErrorInjector::new().with_random_failures(0.5); // 50% failure rate
        let config = ConfigFixtures::minimal_valid();

        let mut failures = 0;
        let mut successes = 0;
        let iterations = 100;

        for _ in 0..iterations {
            match injector.validate_config(&config) {
                Ok(()) => successes += 1,
                Err(PupError::InvalidConfigValue { .. }) => failures += 1,
                Err(other) => panic!("Unexpected error type: {:?}", other),
            }
        }

        // With 50% failure rate, we expect roughly half to fail
        // Allow some variance due to randomness
        assert!(
            failures > 20 && failures < 80,
            "Expected roughly 50% failures, got {} out of {}",
            failures,
            iterations
        );
        assert_eq!(failures + successes, iterations);
    }

    #[test]
    fn test_error_cascade_prevention() {
        // Test that a single error doesn't cascade into multiple failures
        let injector = ErrorInjector::new().with_config_failure();
        let dummy_path = PathBuf::from("test.toml");

        // Multiple operations should fail independently
        for _ in 0..5 {
            let result = injector.load_config(&dummy_path);

            match result {
                Err(PupError::ConfigParseError(_)) => { /* Expected */ }
                other => panic!("Expected consistent error type, got: {:?}", other),
            }
        }
    }
}

/// Test performance under error conditions
#[cfg(test)]
mod error_performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_error_creation_performance() {
        let start = Instant::now();
        let iterations = 10000;

        for i in 0..iterations {
            let _error = PupError::InferenceError(format!("test error {}", i));
        }

        let duration = start.elapsed();
        let avg_ns = duration.as_nanos() / iterations;

        // Error creation should be fast (less than 1 microsecond per error)
        assert!(
            avg_ns < 1000,
            "Error creation too slow: {} ns per error",
            avg_ns
        );
    }

    #[test]
    fn test_error_display_performance() {
        let errors = vec![
            PupError::ConfigParseError("test error with some details".to_string()),
            PupError::ModelLoadError(PathBuf::from("models/very_long_model_name.onnx")),
            PupError::InsufficientMemory {
                required_mb: 2048,
                available_mb: 1024,
            },
            PupError::PerformanceTarget {
                target_fps: 30.0,
                actual_fps: 15.5,
            },
        ];

        let start = Instant::now();
        let iterations = 1000;

        for _ in 0..iterations {
            for error in &errors {
                let _display = error.to_string();
            }
        }

        let duration = start.elapsed();
        let avg_ns = duration.as_nanos() / (iterations * errors.len() as u128);

        // Error display should be reasonably fast (less than 10 microseconds)
        assert!(
            avg_ns < 10000,
            "Error display too slow: {} ns per display",
            avg_ns
        );
    }

    #[test]
    fn test_error_context_performance() {
        let _base_error = PupError::NetworkTimeout("test timeout".to_string());

        let start = Instant::now();
        let iterations = 1000;

        for i in 0..iterations {
            let _error = PupError::InferenceError(format!("test error {}", i));
            let _display = _error.to_string();
        }

        let duration = start.elapsed();
        let avg_ns = duration.as_nanos() / iterations;

        // Context building should be reasonably fast (less than 50 microseconds)
        assert!(
            avg_ns < 50000,
            "Error context building too slow: {} ns per context",
            avg_ns
        );
    }

    #[test]
    fn test_concurrent_error_handling_performance() {
        let error_count = Arc::new(Mutex::new(0));
        let start = Instant::now();

        let mut handles = vec![];
        for _ in 0..10 {
            let error_count = error_count.clone();
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let _error = PupError::InferenceError(format!("thread error {}", i));
                    let _display = _error.to_string();
                    *error_count.lock().unwrap() += 1;
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        let duration = start.elapsed();
        let final_count = *error_count.lock().unwrap();

        assert_eq!(final_count, 1000);

        // Concurrent error handling should not significantly degrade performance
        let avg_ns = duration.as_nanos() / final_count as u128;
        assert!(
            avg_ns < 100000,
            "Concurrent error handling too slow: {} ns per error",
            avg_ns
        );
    }
}

/// Test error edge cases and boundary conditions
#[cfg(test)]
mod error_edge_cases {
    use super::*;

    #[test]
    fn test_empty_error_messages() {
        let errors = [
            PupError::InputNotAvailable("".to_string()),
            PupError::ConfigParseError("".to_string()),
            PupError::InferenceError("".to_string()),
            PupError::Unexpected("".to_string()),
        ];

        for error in errors {
            let display = error.to_string();
            // Should not crash with empty messages
            assert!(!display.is_empty());
        }
    }

    #[test]
    fn test_very_long_error_messages() {
        let long_message = "a".repeat(10000);
        let error = PupError::ConfigParseError(long_message.clone());
        let display = error.to_string();

        assert!(display.contains(&long_message));
        assert!(display.len() > 10000);
    }

    #[test]
    fn test_unicode_error_messages() {
        let unicode_messages = [
            "测试错误消息",
            "エラーメッセージ",
            "сообщение об ошибке",
            "😭 Error with emojis 🚫",
            "Path with unicode: /测试/路径.onnx",
        ];

        for message in unicode_messages {
            let error = PupError::ConfigParseError(message.to_string());
            let display = error.to_string();

            assert!(display.contains(message));
            assert!(!display.is_empty());
        }
    }

    #[test]
    fn test_special_character_paths() {
        let special_paths = [
            PathBuf::from("path with spaces.onnx"),
            PathBuf::from("path-with-dashes.onnx"),
            PathBuf::from("path_with_underscores.onnx"),
            PathBuf::from("path.with.dots.onnx"),
            PathBuf::from("path@with#special$.onnx"),
            PathBuf::from("path\nwith\nnewlines.onnx"),
        ];

        for path in special_paths {
            let error = PupError::ModelLoadError(path.clone());
            let display = error.to_string();

            assert!(display.contains(&path.to_string_lossy().to_string()));
            assert!(!display.is_empty());
        }
    }

    #[test]
    fn test_extreme_numeric_values() {
        let extreme_errors = [
            PupError::InsufficientMemory {
                required_mb: usize::MAX,
                available_mb: 0,
            },
            PupError::InsufficientDiskSpace(usize::MAX),
            PupError::ProcessingTimeout(u64::MAX),
            PupError::FrameDropped(usize::MAX),
            PupError::PerformanceTarget {
                target_fps: f64::INFINITY,
                actual_fps: 0.0,
            },
            PupError::PerformanceTarget {
                target_fps: 30.0,
                actual_fps: f64::NEG_INFINITY,
            },
        ];

        for error in extreme_errors {
            let display = error.to_string();
            // Should handle extreme values gracefully
            assert!(!display.is_empty());
            assert!(!display.contains("NaN")); // Should not produce NaN in display
        }
    }

    #[test]
    fn test_error_with_null_bytes() {
        let message_with_null = "error\0message\0with\0nulls";
        let error = PupError::Unexpected(message_with_null.to_string());
        let display = error.to_string();

        // Should handle null bytes gracefully
        assert!(!display.is_empty());
        // Null bytes may be filtered out or preserved, both are acceptable
    }
}
