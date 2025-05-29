//! Integration tests for production-ready features
//!
//! These tests validate the core functionality needed for production deployment,
//! including error handling, configuration validation, and performance monitoring.

use pup::{
    PupError, PupResult, 
    Metrics, PerformanceMonitor, ConsoleReporter, JsonReporter,
    AppConfig, InferenceConfig, InputConfig, ModeConfig, OutputConfig, PreprocessingConfig
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tempfile::NamedTempFile;

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_pup_error_display_formats() {
        let error = PupError::InputNotAvailable("webcam".to_string());
        assert_eq!(error.to_string(), "Input source not available: webcam");

        let error = PupError::CoreMLFailure("provider not available".to_string());
        assert_eq!(
            error.to_string(),
            "CoreML initialisation failed: provider not available"
        );

        let error = PupError::ModelLoadError(PathBuf::from("model.onnx"));
        assert!(error.to_string().contains("Model loading failed"));
        assert!(error.to_string().contains("model.onnx"));
    }

    #[test]
    fn test_structured_error_variants() {
        let error = PupError::WebcamNotFound { device_id: 0 };
        assert!(error.to_string().contains("device 0"));

        let error = PupError::InvalidConfigValue {
            field: "threshold".to_string(),
            value: "invalid".to_string(),
        };
        assert!(error.to_string().contains("threshold"));
        assert!(error.to_string().contains("invalid"));

        let error = PupError::PerformanceTarget {
            target_fps: 30.0,
            actual_fps: 15.0,
        };
        assert!(error.to_string().contains("30"));
        assert!(error.to_string().contains("15"));
    }

    #[test]
    fn test_error_conversion_from_io() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "test file");
        let pup_error: PupError = io_error.into();
        
        match pup_error {
            PupError::Unexpected(message) => {
                assert!(message.contains("File not found"));
            }
            _ => panic!("Expected Unexpected error variant"),
        }
    }
}

#[cfg(test)]
mod metrics_integration_tests {
    use super::*;

    #[test]
    fn test_metrics_basic_operations() {
        let metrics = Metrics::new();
        
        // Test initial state
        assert_eq!(metrics.get_fps(), 0.0);
        assert_eq!(metrics.get_memory_usage_mb(), 0);
        assert_eq!(metrics.get_dropped_frames(), 0);
        assert_eq!(metrics.get_total_frames(), 0);

        // Test updates
        metrics.update_fps(30.5);
        metrics.update_inference_latency(25.7);
        metrics.update_memory_usage(128);
        metrics.increment_total_frames();
        metrics.increment_dropped_frames();

        assert_eq!(metrics.get_fps(), 30.5);
        assert_eq!(metrics.get_inference_latency_ms(), 25.7);
        assert_eq!(metrics.get_memory_usage_mb(), 128);
        assert_eq!(metrics.get_peak_memory_mb(), 128);
        assert_eq!(metrics.get_total_frames(), 1);
        assert_eq!(metrics.get_dropped_frames(), 1);
        assert_eq!(metrics.get_frame_drop_rate(), 100.0);
    }

    #[test]
    fn test_metrics_peak_memory_tracking() {
        let metrics = Metrics::new();
        
        metrics.update_memory_usage(100);
        assert_eq!(metrics.get_peak_memory_mb(), 100);
        
        metrics.update_memory_usage(200);
        assert_eq!(metrics.get_peak_memory_mb(), 200);
        
        // Peak should not decrease
        metrics.update_memory_usage(150);
        assert_eq!(metrics.get_memory_usage_mb(), 150);
        assert_eq!(metrics.get_peak_memory_mb(), 200);
    }

    #[test]
    fn test_metrics_frame_drop_rate_calculation() {
        let metrics = Metrics::new();
        
        // No frames processed yet
        assert_eq!(metrics.get_frame_drop_rate(), 0.0);
        
        // Process some frames
        for _ in 0..10 {
            metrics.increment_total_frames();
        }
        
        // Drop some frames
        for _ in 0..3 {
            metrics.increment_dropped_frames();
        }
        
        // 3 dropped out of 10 total = 30%
        let drop_rate = metrics.get_frame_drop_rate();
        assert!((drop_rate - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_metrics_performance_targets() {
        let metrics = Metrics::new();
        
        // Set good performance
        metrics.update_fps(60.0);
        metrics.update_inference_latency(10.0);
        
        // Should pass targets
        assert!(metrics.check_performance_targets(30.0, 50.0).is_ok());
        
        // Should fail FPS target
        metrics.update_fps(20.0);
        assert!(metrics.check_performance_targets(30.0, 50.0).is_err());
        
        // Should fail latency target
        metrics.update_fps(60.0);
        metrics.update_inference_latency(100.0);
        assert!(metrics.check_performance_targets(30.0, 50.0).is_err());
    }

    #[test]
    fn test_metrics_reset_functionality() {
        let metrics = Metrics::new();
        
        // Set various metrics
        metrics.update_fps(30.0);
        metrics.update_inference_latency(50.0);
        metrics.update_memory_usage(256);
        metrics.increment_total_frames();
        metrics.increment_dropped_frames();
        metrics.update_cpu_usage(75.0);
        
        // Verify they're set
        assert_eq!(metrics.get_fps(), 30.0);
        assert_eq!(metrics.get_memory_usage_mb(), 256);
        assert_eq!(metrics.get_total_frames(), 1);
        
        // Reset
        metrics.reset();
        
        // Verify everything is reset
        assert_eq!(metrics.get_fps(), 0.0);
        assert_eq!(metrics.get_inference_latency_ms(), 0.0);
        assert_eq!(metrics.get_memory_usage_mb(), 0);
        assert_eq!(metrics.get_total_frames(), 0);
        assert_eq!(metrics.get_dropped_frames(), 0);
        assert_eq!(metrics.get_peak_memory_mb(), 0);
        assert_eq!(metrics.get_cpu_usage_percent(), 0.0);
    }

    #[test]
    fn test_performance_monitor_integration() {
        let mut monitor = PerformanceMonitor::new();
        
        // Add reporters
        monitor.add_reporter(Box::new(ConsoleReporter::new(1000)));
        
        // Get metrics reference
        let metrics = monitor.metrics();
        
        // Update some metrics
        metrics.update_fps(45.0);
        metrics.update_inference_latency(20.0);
        
        // Test system metrics update
        assert!(monitor.update_system_metrics().is_ok());
        
        // Test reporting
        assert!(monitor.report().is_ok());
        
        // Test performance target checking
        assert!(monitor.check_targets(30.0, 50.0).is_ok());
        assert!(monitor.check_targets(50.0, 15.0).is_err());
    }

    #[test]
    fn test_console_reporter() {
        let metrics = Metrics::new();
        metrics.update_fps(30.5);
        metrics.update_inference_latency(25.7);
        metrics.update_memory_usage(128);
        metrics.update_cpu_usage(45.0);
        metrics.update_gpu_usage(60.0);
        
        let reporter = ConsoleReporter::new(0); // No interval limit for testing
        assert!(reporter.report(&metrics).is_ok());
        assert_eq!(reporter.name(), "console");
        
        // Test summary format
        let summary = metrics.format_summary();
        assert!(summary.contains("FPS: 30.5"));
        assert!(summary.contains("Latency: 25.7ms"));
        assert!(summary.contains("Memory: 128MB"));
        assert!(summary.contains("CPU: 45.0%"));
        assert!(summary.contains("GPU: 60.0%"));
    }

    #[test]
    fn test_json_reporter() {
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_path_buf();
        
        let metrics = Metrics::new();
        metrics.update_fps(30.0);
        metrics.update_inference_latency(25.0);
        metrics.update_memory_usage(128);
        metrics.increment_total_frames();
        
        let reporter = JsonReporter::new(temp_path.clone(), 0); // No interval limit
        assert!(reporter.report(&metrics).is_ok());
        assert_eq!(reporter.name(), "json");
        
        // Verify file was written
        let content = std::fs::read_to_string(&temp_path).unwrap();
        assert!(content.contains("\"fps\":30"));
        assert!(content.contains("\"inference_latency_ms\":25"));
        assert!(content.contains("\"memory_usage_mb\":128"));
        assert!(content.contains("\"total_frames\":1"));
    }
}

#[cfg(test)]
mod configuration_validation_tests {
    use super::*;

    #[test]
    fn test_production_config_creation() {
        let config = AppConfig::production_example();
        
        assert_eq!(config.mode.mode_type, "production");
        assert_eq!(config.input.source, "webcam");
        assert_eq!(config.inference.backend, "ort");
        assert_eq!(config.inference.execution_providers, vec!["coreml", "cpu"]);
        assert_eq!(config.output.display_enabled, true);
        assert_eq!(config.output.recording_enabled, false);
    }

    #[test]
    fn test_config_validation_mode() {
        let mut config = AppConfig::default();
        
        // Valid modes should pass
        config.mode.mode_type = "production".to_string();
        assert!(config.validate().is_ok());
        
        config.mode.mode_type = "live".to_string();
        assert!(config.validate().is_ok());
        
        config.mode.mode_type = "detection".to_string();
        assert!(config.validate().is_ok());
        
        config.mode.mode_type = "benchmark".to_string();
        assert!(config.validate().is_ok());
        
        // Invalid mode should fail
        config.mode.mode_type = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_inference() {
        let mut config = AppConfig::production_example();
        
        // Valid configuration should pass
        assert!(config.validate().is_ok());
        
        // Invalid backend
        config.inference.backend = "invalid".to_string();
        assert!(config.validate().is_err());
        config.inference.backend = "ort".to_string();
        
        // Invalid execution provider
        config.inference.execution_providers = vec!["invalid".to_string()];
        assert!(config.validate().is_err());
        config.inference.execution_providers = vec!["coreml".to_string(), "cpu".to_string()];
        
        // Invalid confidence threshold
        config.inference.confidence_threshold = 1.5;
        assert!(config.validate().is_err());
        config.inference.confidence_threshold = 0.5;
        
        // Invalid batch size
        config.inference.batch_size = 0;
        assert!(config.validate().is_err());
        config.inference.batch_size = 100; // Too large
        assert!(config.validate().is_err());
        config.inference.batch_size = 1;
    }

    #[test]
    fn test_config_validation_input() {
        let mut config = AppConfig::production_example();
        
        // Valid webcam input
        config.input.source = "webcam".to_string();
        config.input.device_id = Some(0);
        assert!(config.validate().is_ok());
        
        // Invalid device ID
        config.input.device_id = Some(999);
        assert!(config.validate().is_err());
        config.input.device_id = Some(0);
        
        // Valid RTSP URL
        config.input.source = "rtsp://example.com/stream".to_string();
        assert!(config.validate().is_ok());
        
        // Invalid RTSP URL format
        config.input.source = "invalid-rtsp".to_string();
        assert!(config.validate().is_err());
        
        // Valid caps format
        config.input.source = "webcam".to_string();
        config.input.caps = Some("video/x-raw,width=640,height=480".to_string());
        assert!(config.validate().is_ok());
        
        // Invalid caps format
        config.input.caps = Some("invalid-caps".to_string());
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_output() {
        let mut config = AppConfig::production_example();
        
        // Valid output formats
        let valid_formats = ["mp4", "json", "rtmp", "avi", "mov"];
        for format in &valid_formats {
            config.output.output_format = format.to_string();
            assert!(config.validate().is_ok());
        }
        
        // Invalid output format
        config.output.output_format = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_preprocessing() {
        let mut config = AppConfig::production_example();
        
        // Valid preprocessing config
        assert!(config.validate().is_ok());
        
        // Invalid target size (zero dimension)
        if let Some(ref mut preprocessing) = config.preprocessing {
            preprocessing.target_size = [0, 640];
            assert!(config.validate().is_err());
            
            preprocessing.target_size = [640, 0];
            assert!(config.validate().is_err());
            
            // Too large dimensions
            preprocessing.target_size = [5000, 5000];
            assert!(config.validate().is_err());
            
            // Valid size
            preprocessing.target_size = [640, 640];
            assert!(config.validate().is_ok());
        }
    }

    #[test]
    fn test_config_backwards_compatibility() {
        let config = AppConfig::default();
        
        // Should have pipeline available for backward compatibility
        assert!(config.pipeline.is_some());
        
        // Test helper methods
        let pipeline = config.get_pipeline();
        assert_eq!(pipeline.video_source, config.input.source);
        assert_eq!(pipeline.display_enabled, config.output.display_enabled);
        
        // Test video source getter
        assert_eq!(config.video_source(), config.input.source);
        assert_eq!(config.input_source(), config.input.source);
    }

    #[test]
    fn test_config_convenience_methods() {
        let config = AppConfig::production_example();
        
        // Test model path access
        assert_eq!(config.model_path(), &PathBuf::from("models/yolov8n.onnx"));
        
        // Test input source access
        assert_eq!(config.input_source(), "webcam");
        
        // Test preprocessing access
        let preprocessing = config.get_preprocessing();
        assert_eq!(preprocessing.target_size, [640, 640]);
        assert!(preprocessing.letterbox);
        assert!(preprocessing.normalize);
    }
}

#[cfg(test)]
mod integration_workflow_tests {
    use super::*;

    #[test]
    fn test_error_to_metrics_integration() {
        let metrics = Metrics::new();
        let monitor = PerformanceMonitor::new();
        
        // Simulate performance degradation
        metrics.update_fps(15.0); // Below target
        metrics.update_inference_latency(100.0); // Above target
        
        // Check if performance targets fail
        let result = metrics.check_performance_targets(30.0, 50.0);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            PupError::PerformanceTarget { target_fps, actual_fps } => {
                assert_eq!(target_fps, 30.0);
                assert_eq!(actual_fps, 15.0);
            }
            _ => panic!("Expected PerformanceTarget error"),
        }
    }

    #[test]
    fn test_config_to_error_integration() {
        // Test invalid configuration generates appropriate errors
        let mut config = AppConfig::production_example();
        
        // Set invalid model path
        config.inference.model_path = PathBuf::from("nonexistent.onnx");
        
        let result = config.validate();
        assert!(result.is_err());
        
        match result.unwrap_err() {
            PupError::ModelLoadError(path) => {
                assert_eq!(path, PathBuf::from("nonexistent.onnx"));
            }
            _ => panic!("Expected ModelLoadError"),
        }
    }

    #[test]
    fn test_full_production_workflow() {
        // Test a complete production workflow integration
        
        // 1. Create production configuration
        let config = AppConfig::production_example();
        
        // 2. Initialize performance monitoring
        let mut monitor = PerformanceMonitor::new();
        monitor.add_reporter(Box::new(ConsoleReporter::new(1000)));
        
        // 3. Get metrics reference
        let metrics = monitor.metrics();
        
        // 4. Simulate good performance
        metrics.update_fps(60.0);
        metrics.update_inference_latency(15.0);
        metrics.update_memory_usage(256);
        
        // 5. Update system metrics
        assert!(monitor.update_system_metrics().is_ok());
        
        // 6. Check performance targets
        assert!(monitor.check_targets(30.0, 50.0).is_ok());
        
        // 7. Report metrics
        assert!(monitor.report().is_ok());
        
        // 8. Verify metrics are within expected ranges
        assert!(metrics.get_fps() >= 30.0);
        assert!(metrics.get_inference_latency_ms() <= 50.0);
        assert!(metrics.get_memory_usage_mb() > 0);
    }
}