//! Integration tests for production-ready features
//!
//! These tests validate the core functionality needed for production deployment,
//! including error handling and configuration validation.

use gstpup::{AppConfig, PupError};
use std::path::PathBuf;

#[cfg(test)]
mod error_handling_tests {
    use super::*;

    #[test]
    fn test_pup_error_display_formats() {
        let error = PupError::InputNotAvailable("webcam".to_string());
        assert_eq!(error.to_string(), "Input source not available: webcam");

        let error = PupError::ModelLoadError(PathBuf::from("model.onnx"));
        assert!(error.to_string().contains("Model loading failed"));
        assert!(error.to_string().contains("model.onnx"));
    }

    #[test]
    fn test_structured_error_variants() {
        let error = PupError::InvalidConfigValue {
            field: "threshold".to_string(),
            value: "invalid".to_string(),
        };
        assert!(error.to_string().contains("threshold"));
        assert!(error.to_string().contains("invalid"));
    }

    #[test]
    fn test_error_conversion_from_io() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "test file");
        let pup_error: PupError = io_error.into();

        match pup_error {
            PupError::Unexpected(message) => {
                assert!(message.contains("I/O error"));
            }
            _ => panic!("Expected Unexpected error variant"),
        }
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
        assert!(config.output.display_enabled);
        assert!(!config.output.recording_enabled);
    }

    #[test]
    fn test_config_validation_mode() {
        let mut config = AppConfig::default();

        config.mode.mode_type = "production".to_string();
        assert!(config.validate().is_ok());

        config.mode.mode_type = "live".to_string();
        assert!(config.validate().is_ok());

        config.mode.mode_type = "detection".to_string();
        assert!(config.validate().is_ok());

        config.mode.mode_type = "benchmark".to_string();
        assert!(config.validate().is_ok());

        config.mode.mode_type = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_inference() {
        let mut config = AppConfig::production_example();

        assert!(config.validate().is_ok());

        config.inference.backend = "invalid".to_string();
        assert!(config.validate().is_err());
        config.inference.backend = "ort".to_string();

        config.inference.execution_providers = vec!["invalid".to_string()];
        assert!(config.validate().is_err());
        config.inference.execution_providers = vec!["coreml".to_string(), "cpu".to_string()];

        config.inference.confidence_threshold = 1.5;
        assert!(config.validate().is_err());
        config.inference.confidence_threshold = 0.5;

        config.inference.batch_size = 0;
        assert!(config.validate().is_err());
        config.inference.batch_size = 100;
        assert!(config.validate().is_err());
        config.inference.batch_size = 1;
    }

    #[test]
    fn test_config_validation_input() {
        let mut config = AppConfig::production_example();

        config.input.source = "webcam".to_string();
        config.input.device_id = Some(0);
        assert!(config.validate().is_ok());

        config.input.device_id = Some(999);
        assert!(config.validate().is_err());
        config.input.device_id = Some(0);

        config.input.source = "rtsp://example.com/stream".to_string();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation_output() {
        let mut config = AppConfig::production_example();

        let valid_formats = ["mp4", "json", "rtmp", "avi", "mov"];
        for format in &valid_formats {
            config.output.output_format = format.to_string();
            assert!(config.validate().is_ok());
        }

        config.output.output_format = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validation_preprocessing() {
        let mut config = AppConfig::production_example();

        assert!(config.validate().is_ok());

        if let Some(ref mut preprocessing) = config.preprocessing {
            preprocessing.target_size = [0, 640];
        }
        assert!(config.validate().is_err());

        if let Some(ref mut preprocessing) = config.preprocessing {
            preprocessing.target_size = [640, 640];
        }
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_backwards_compatibility() {
        let config = AppConfig::default();

        assert!(config.pipeline.is_some());

        let pipeline = config.get_pipeline();
        assert_eq!(pipeline.video_source, config.input.source);
        assert_eq!(pipeline.display_enabled, config.output.display_enabled);
    }

    #[test]
    fn test_config_convenience_methods() {
        let config = AppConfig::production_example();

        assert_eq!(config.model_path(), &PathBuf::from("models/yolov8n.onnx"));
        assert_eq!(config.input_source(), "webcam");

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
    fn test_config_to_error_integration() {
        let mut config = AppConfig::production_example();

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
}
