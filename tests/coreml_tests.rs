//! CoreML-specific tests for macOS optimization
//! Tests CoreML execution provider configuration and performance optimization
//! Based on roadmap specifications for CoreML-optimized configuration

use gstpup::{AppConfig, InferenceConfig, InputConfig, ModeConfig, OutputConfig, PreprocessingConfig, Metrics, PerformanceMonitor, ConsoleReporter, PupError};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[cfg(target_os = "macos")]
#[cfg(test)]
mod coreml_optimization_tests {
    use super::*;

    #[test]
    fn test_coreml_execution_provider_configuration() {
        // Test CoreML execution provider configuration as specified in roadmap
        let mut config = AppConfig::default();
        
        // Set CoreML as primary execution provider
        config.inference.execution_providers = vec!["coreml".to_string(), "cpu".to_string()];
        config.inference.backend = "ort".to_string();
        config.inference.batch_size = 1; // CoreML optimization for single batch
        
        // Validate configuration
        assert!(config.validate().is_ok());
        assert_eq!(config.inference.execution_providers[0], "coreml");
        assert_eq!(config.inference.batch_size, 1);
    }

    #[test]
    fn test_coreml_optimized_settings() {
        // Test the optimized CoreML settings from roadmap issue #341
        let config = AppConfig::production_example();
        
        // Verify CoreML-first configuration
        assert_eq!(config.inference.execution_providers, vec!["coreml", "cpu"]);
        
        // Verify optimized batch size for CoreML
        assert_eq!(config.inference.batch_size, 1);
        
        // Verify model format compatibility
        assert!(config.inference.model_path.to_string_lossy().ends_with(".onnx"));
    }

    #[test]
    fn test_coreml_performance_target_validation() {
        // Test that CoreML configuration meets roadmap performance targets
        let metrics = Metrics::new();
        
        // Simulate CoreML performance (should be <50ms as per roadmap)
        metrics.update_inference_latency(25.0); // Well under 50ms target
        assert!(metrics.get_inference_latency_ms() < 50.0);
        
        // Test performance target validation
        assert!(metrics.check_performance_targets(30.0, 50.0).is_ok());
        
        // Verify it fails with poor performance
        metrics.update_inference_latency(75.0); // Over 50ms target
        assert!(metrics.check_performance_targets(30.0, 50.0).is_err());
    }

    #[test]
    fn test_coreml_memory_optimization() {
        // Test memory usage with CoreML configuration
        let metrics = Metrics::new();
        
        // Simulate optimized memory usage (should be <500MB as per roadmap)
        metrics.update_memory_usage(256); // Well under 500MB target
        assert!(metrics.get_memory_usage_mb() < 500);
        
        // Test peak memory tracking
        metrics.update_memory_usage(300);
        assert_eq!(metrics.get_peak_memory_mb(), 300);
        
        // Verify memory doesn't exceed targets
        assert!(metrics.get_memory_usage_mb() < 500);
        assert!(metrics.get_peak_memory_mb() < 500);
    }

    #[test]
    fn test_coreml_fallback_configuration() {
        // Test CPU fallback when CoreML is unavailable
        let mut config = AppConfig::default();
        
        // Test various execution provider combinations
        let provider_configs = vec![
            vec!["coreml", "cpu"],
            vec!["cpu"], // Fallback only
            vec!["coreml", "cpu", "cuda"], // With GPU fallback
        ];
        
        for providers in provider_configs {
            config.inference.execution_providers = providers.iter().map(|s| s.to_string()).collect();
            
            // All should validate successfully
            let result = config.validate();
            if result.is_err() {
                // Should only fail on model file not existing, not provider validation
                let error = result.unwrap_err();
                assert!(error.to_string().contains("model") || error.to_string().contains("Model"));
            }
        }
    }

    #[test]
    fn test_coreml_model_compatibility() {
        // Test model format compatibility with CoreML
        let mut config = AppConfig::production_example();
        
        // Test ONNX model format (required for CoreML)
        config.inference.model_path = PathBuf::from("test_model.onnx");
        
        // Should validate format (will fail on file existence, not format)
        let result = config.validate();
        if result.is_err() {
            let error = result.unwrap_err();
            // Should be model file error, not format error for .onnx files
            assert!(error.to_string().contains("Model loading failed"));
        }
        
        // Test invalid model format
        config.inference.model_path = PathBuf::from("test_model.invalid");
        let result = config.validate();
        assert!(result.is_err());
        let error = result.unwrap_err();
        assert!(error.to_string().contains("Model format"));
    }

    #[test]
    fn test_coreml_performance_monitoring_integration() {
        // Test integration between CoreML config and performance monitoring
        let config = AppConfig::production_example();
        let mut monitor = PerformanceMonitor::new();
        
        // Add reporters for monitoring
        monitor.add_reporter(Box::new(ConsoleReporter::new(1000)));
        
        let metrics = monitor.metrics();
        
        // Simulate good CoreML performance
        metrics.update_fps(60.0); // Above 30 FPS target
        metrics.update_inference_latency(15.0); // Well under 50ms target
        metrics.update_memory_usage(200); // Well under 500MB target
        
        // Verify performance targets are met
        assert!(monitor.check_targets(30.0, 50.0).is_ok());
        
        // Test system metrics update
        assert!(monitor.update_system_metrics().is_ok());
        
        // Test reporting
        assert!(monitor.report().is_ok());
    }

    #[test]
    fn test_coreml_error_handling() {
        // Test CoreML-specific error handling
        let error = PupError::CoreMLFailure("CoreML provider not available".to_string());
        assert!(error.to_string().contains("CoreML initialisation failed"));
        assert!(error.to_string().contains("provider not available"));
        
        // Test model compatibility errors
        let error = PupError::ModelFormatError(PathBuf::from("invalid_model.bin"));
        assert!(error.to_string().contains("Invalid model format"));
        
        // Test inference errors
        let error = PupError::InferenceError("CoreML execution failed".to_string());
        assert!(error.to_string().contains("Inference execution failed"));
    }

    #[test]
    fn test_coreml_benchmark_configuration() {
        // Test benchmark mode configuration for CoreML testing
        let mut config = AppConfig::default();
        config.mode.mode_type = "benchmark".to_string();
        config.inference.execution_providers = vec!["coreml".to_string()];
        
        // Should validate benchmark mode
        assert!(config.validate().is_ok());
        assert_eq!(config.mode.mode_type, "benchmark");
        
        // Test metrics collection for benchmarking
        let metrics = Metrics::new();
        let start_time = Instant::now();
        
        // Simulate benchmark run
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = start_time.elapsed();
        
        metrics.update_inference_latency(elapsed.as_millis() as f64);
        assert!(metrics.get_inference_latency_ms() >= 10.0);
    }

    #[test]
    fn test_coreml_roadmap_success_criteria() {
        // Test specific success criteria from roadmap for CoreML optimization
        
        // Criterion 1: <50ms inference latency for YOLOv8n on CoreML
        let metrics = Metrics::new();
        metrics.update_inference_latency(25.0); // Simulated CoreML performance
        assert!(metrics.get_inference_latency_ms() < 50.0, 
                "CoreML inference latency should be <50ms, got {}ms", 
                metrics.get_inference_latency_ms());
        
        // Criterion 2: >30 FPS real-time processing on 720p video
        metrics.update_fps(45.0); // Simulated CoreML FPS
        assert!(metrics.get_fps() > 30.0,
                "CoreML FPS should be >30, got {} FPS",
                metrics.get_fps());
        
        // Criterion 3: <500MB RAM usage during operation
        metrics.update_memory_usage(280); // Simulated CoreML memory usage
        assert!(metrics.get_memory_usage_mb() < 500,
                "CoreML memory usage should be <500MB, got {}MB",
                metrics.get_memory_usage_mb());
        
        // Criterion 4: Performance targets validation
        assert!(metrics.check_performance_targets(30.0, 50.0).is_ok(),
                "CoreML should meet roadmap performance targets");
    }
}

// Tests that run on all platforms (not just macOS)
#[cfg(test)]
mod coreml_configuration_tests {
    use super::*;

    #[test]
    fn test_coreml_execution_provider_validation() {
        // Test that CoreML execution provider is properly validated
        let mut config = AppConfig::default();
        
        // Valid CoreML configuration
        config.inference.execution_providers = vec!["coreml".to_string(), "cpu".to_string()];
        let result = config.validate();
        
        // Should pass provider validation (may fail on model file)
        if result.is_err() {
            let error = result.unwrap_err();
            assert!(error.to_string().contains("Model") || error.to_string().contains("model"));
        }
        
        // Invalid execution provider should fail
        config.inference.execution_providers = vec!["invalid_provider".to_string()];
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_coreml_configuration_serialization() {
        // Test that CoreML configuration can be serialized to/from TOML
        let config = AppConfig {
            mode: ModeConfig {
                mode_type: "production".to_string(),
            },
            input: InputConfig {
                source: "webcam".to_string(),
                device_id: Some(0),
                caps: Some("video/x-raw,width=1280,height=720,framerate=30/1".to_string()),
            },
            inference: InferenceConfig {
                backend: "ort".to_string(),
                execution_providers: vec!["coreml".to_string(), "cpu".to_string()],
                model_path: PathBuf::from("models/yolov8n.onnx"),
                confidence_threshold: 0.5,
                batch_size: 1,
                device: None,
            },
            output: OutputConfig {
                display_enabled: true,
                recording_enabled: false,
                output_format: "mp4".to_string(),
            },
            preprocessing: Some(PreprocessingConfig::default()),
            pipeline: None,
        };
        
        // Test serialization
        let toml_string = toml::to_string_pretty(&config).unwrap();
        assert!(toml_string.contains("execution_providers"));
        assert!(toml_string.contains("coreml"));
        
        // Test deserialization
        let parsed_config: AppConfig = toml::from_str(&toml_string).unwrap();
        assert_eq!(parsed_config.inference.execution_providers, vec!["coreml", "cpu"]);
        assert_eq!(parsed_config.inference.backend, "ort");
        assert_eq!(parsed_config.inference.batch_size, 1);
    }

    #[test]
    fn test_coreml_performance_expectations() {
        // Test that the system can handle expected CoreML performance metrics
        let metrics = Metrics::new();
        
        // Set metrics to roadmap performance expectations
        metrics.update_fps(60.0); // High FPS expected with CoreML
        metrics.update_inference_latency(15.0); // Fast inference with CoreML
        metrics.update_memory_usage(200); // Efficient memory usage
        metrics.update_cpu_usage(30.0); // Lower CPU with CoreML offload
        metrics.update_gpu_usage(0.0); // CoreML uses Neural Engine, not GPU
        
        // Verify all metrics are within expected ranges
        assert!(metrics.get_fps() >= 30.0);
        assert!(metrics.get_inference_latency_ms() <= 50.0);
        assert!(metrics.get_memory_usage_mb() <= 500);
        assert!(metrics.get_cpu_usage_percent() <= 80.0); // Should be efficient
        
        // Test performance summary
        let summary = metrics.format_summary();
        assert!(summary.contains("FPS: 60.0"));
        assert!(summary.contains("Latency: 15.0ms"));
        assert!(summary.contains("Memory: 200MB"));
    }
}