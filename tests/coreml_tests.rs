//! CoreML-specific tests for macOS optimization
//! Tests CoreML execution provider configuration and performance optimization

use gstpup::{
    AppConfig, InferenceConfig, InputConfig, ModeConfig, OutputConfig, PreprocessingConfig,
    PupError,
};
use std::path::PathBuf;

#[cfg(target_os = "macos")]
#[cfg(test)]
mod coreml_optimization_tests {
    use super::*;

    #[test]
    fn test_coreml_execution_provider_configuration() {
        let mut config = AppConfig::default();

        config.inference.execution_providers = vec!["coreml".to_string(), "cpu".to_string()];
        config.inference.backend = "ort".to_string();
        config.inference.batch_size = 1;

        assert!(config.validate().is_ok());
        assert_eq!(config.inference.execution_providers[0], "coreml");
        assert_eq!(config.inference.batch_size, 1);
    }

    #[test]
    fn test_coreml_optimized_settings() {
        let config = AppConfig::production_example();

        assert_eq!(config.inference.execution_providers, vec!["coreml", "cpu"]);
        assert_eq!(config.inference.batch_size, 1);
        assert!(config
            .inference
            .model_path
            .to_string_lossy()
            .ends_with(".onnx"));
    }

    #[test]
    fn test_coreml_fallback_configuration() {
        let mut config = AppConfig::default();

        let provider_configs = vec![
            vec!["coreml", "cpu"],
            vec!["cpu"],
            vec!["coreml", "cpu", "cuda"],
        ];

        for providers in provider_configs {
            config.inference.execution_providers =
                providers.iter().map(|s| s.to_string()).collect();

            let result = config.validate();
            if result.is_err() {
                let error = result.unwrap_err();
                assert!(error.to_string().contains("model") || error.to_string().contains("Model"));
            }
        }
    }

    #[test]
    fn test_coreml_model_compatibility() {
        let mut config = AppConfig::production_example();

        config.inference.model_path = PathBuf::from("test_model.onnx");

        let result = config.validate();
        if result.is_err() {
            let error = result.unwrap_err();
            assert!(error.to_string().contains("Model loading failed"));
        }

        // Test invalid model format
        config.inference.model_path = PathBuf::from("test_model.invalid");
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_coreml_error_handling() {
        let error = PupError::InferenceError("CoreML execution failed".to_string());
        assert!(error.to_string().contains("Inference execution failed"));
    }

    #[test]
    fn test_coreml_benchmark_configuration() {
        let mut config = AppConfig::default();
        config.mode.mode_type = "benchmark".to_string();
        config.inference.execution_providers = vec!["coreml".to_string()];

        assert!(config.validate().is_ok());
        assert_eq!(config.mode.mode_type, "benchmark");
    }
}

// Tests that run on all platforms
#[cfg(test)]
mod coreml_configuration_tests {
    use super::*;

    #[test]
    fn test_coreml_execution_provider_validation() {
        let mut config = AppConfig::default();

        config.inference.execution_providers = vec!["coreml".to_string(), "cpu".to_string()];
        let result = config.validate();

        if result.is_err() {
            let error = result.unwrap_err();
            assert!(error.to_string().contains("Model") || error.to_string().contains("model"));
        }

        config.inference.execution_providers = vec!["invalid_provider".to_string()];
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_coreml_configuration_serialization() {
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

        let toml_string = toml::to_string_pretty(&config).unwrap();
        assert!(toml_string.contains("execution_providers"));
        assert!(toml_string.contains("coreml"));

        let parsed_config: AppConfig = toml::from_str(&toml_string).unwrap();
        assert_eq!(
            parsed_config.inference.execution_providers,
            vec!["coreml", "cpu"]
        );
        assert_eq!(parsed_config.inference.backend, "ort");
        assert_eq!(parsed_config.inference.batch_size, 1);
    }
}
