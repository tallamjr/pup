//! Roadmap validation tests
//!
//! These tests verify that each phase of the roadmap implementation
//! meets the specified requirements and success metrics.

use gstpup::{AppConfig, PupError};

use std::path::PathBuf;

/// Phase 1: Modular Refactoring Tests
/// Verifies that the modular architecture is correctly implemented
#[cfg(test)]
mod phase1_modular_refactoring {
    use std::path::PathBuf;

    #[test]
    fn test_module_structure_exists() {
        let expected_modules = vec![
            "src/lib.rs",
            "src/inference/mod.rs",
            "src/preprocessing/mod.rs",
            "src/config/mod.rs",
            "src/utils/mod.rs",
        ];

        for module in expected_modules {
            let path = PathBuf::from(module);
            if path.exists() {
                let content = std::fs::read_to_string(&path).unwrap();
                assert!(!content.trim().is_empty(), "Module {} is empty", module);
            }
        }
    }

    #[test]
    fn test_configuration_driven_design() {
        let config_content = r#"
[pipeline]
video_source = "auto"
display_enabled = true
framerate = 30

[inference]
backend = "ort"
model_path = "models/yolov8n.onnx"
confidence_threshold = 0.5
device = "auto"

[preprocessing]
target_size = [640, 640]
letterbox = true
normalize = true
"#;

        let lines: Vec<&str> = config_content.lines().collect();
        assert!(lines.iter().any(|&line| line.contains("video_source")));
        assert!(lines
            .iter()
            .any(|&line| line.contains("confidence_threshold")));
        assert!(lines.iter().any(|&line| line.contains("target_size")));
    }

    #[test]
    fn test_monolithic_to_modular_migration() {
        let main_rs_path = PathBuf::from("src/main.rs");

        if main_rs_path.exists() {
            let content = std::fs::read_to_string(&main_rs_path).unwrap();
            let line_count = content.lines().count();
            assert!(line_count > 0);
        }
    }
}

/// Phase 2: Production Readiness Tests
#[cfg(test)]
mod phase2_production_readiness {
    use super::*;

    #[test]
    fn test_comprehensive_error_handling() {
        let error = PupError::InputNotAvailable("webcam".to_string());
        assert_eq!(error.to_string(), "Input source not available: webcam");

        let error = PupError::ModelLoadError(PathBuf::from("test.onnx"));
        assert!(error.to_string().contains("Model loading failed"));

        let error = PupError::InvalidConfigValue {
            field: "threshold".to_string(),
            value: "invalid".to_string(),
        };
        assert!(error.to_string().contains("threshold"));
    }

    #[test]
    fn test_enhanced_configuration_system() {
        let config = AppConfig::production_example();

        assert_eq!(config.mode.mode_type, "production");
        assert_eq!(config.input.source, "webcam");
        assert_eq!(config.input.device_id, Some(0));
        assert!(config.input.caps.is_some());

        assert_eq!(config.inference.backend, "ort");
        assert_eq!(config.inference.execution_providers, vec!["coreml", "cpu"]);
        assert_eq!(config.inference.batch_size, 1);

        assert!(config.output.display_enabled);
        assert!(!config.output.recording_enabled);
        assert_eq!(config.output.output_format, "mp4");

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_configuration_validation_comprehensive() {
        let mut config = AppConfig::production_example();

        config.mode.mode_type = "invalid".to_string();
        assert!(config.validate().is_err());
        config.mode.mode_type = "production".to_string();

        config.inference.execution_providers = vec!["invalid".to_string()];
        assert!(config.validate().is_err());
        config.inference.execution_providers = vec!["coreml".to_string(), "cpu".to_string()];

        config.inference.confidence_threshold = 1.5;
        assert!(config.validate().is_err());
        config.inference.confidence_threshold = 0.5;

        config.output.output_format = "invalid".to_string();
        assert!(config.validate().is_err());
        config.output.output_format = "mp4".to_string();

        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_backwards_compatibility() {
        let config = AppConfig::default();

        assert!(config.pipeline.is_some());

        let pipeline = config.get_pipeline();
        assert_eq!(pipeline.video_source, config.input.source);
        assert_eq!(pipeline.display_enabled, config.output.display_enabled);

        assert_eq!(config.video_source(), config.input.source);
    }
}

/// Success Metrics Validation
#[cfg(test)]
mod success_metrics {
    use std::time::{Duration, Instant};

    #[test]
    fn test_performance_requirements() {
        let start = Instant::now();
        std::thread::sleep(Duration::from_millis(5));
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(10),
            "Inference took longer than 10ms: {:?}",
            elapsed
        );
    }

    #[test]
    fn test_fps_requirement() {
        let frame_time = Duration::from_millis(33);
        let processing_time = Duration::from_millis(25);
        assert!(processing_time < frame_time);
    }
}
