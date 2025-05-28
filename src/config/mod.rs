//! Configuration management

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

/// Pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Video source: "auto", "webcam", or file path
    pub video_source: String,
    /// Whether to display video output
    pub display_enabled: bool,
    /// Target framerate
    pub framerate: u32,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            video_source: "auto".to_string(),
            display_enabled: true,
            framerate: 30,
        }
    }
}

/// Inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Backend type: "ort"
    pub backend: String,
    /// Path to the model file
    pub model_path: PathBuf,
    /// Confidence threshold for detections
    pub confidence_threshold: f32,
    /// Device: "auto", "cpu", "coreml", "cuda"
    pub device: String,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            backend: "ort".to_string(),
            model_path: PathBuf::from("models/yolov8n.onnx"),
            confidence_threshold: 0.5,
            device: "auto".to_string(),
        }
    }
}

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Target size [width, height]
    pub target_size: [u32; 2],
    /// Whether to use letterboxing
    pub letterbox: bool,
    /// Whether to normalize pixel values
    pub normalize: bool,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            target_size: [640, 640],
            letterbox: true,
            normalize: true,
        }
    }
}

/// Complete application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub pipeline: PipelineConfig,
    pub inference: InferenceConfig,
    pub preprocessing: PreprocessingConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            pipeline: PipelineConfig::default(),
            inference: InferenceConfig::default(),
            preprocessing: PreprocessingConfig::default(),
        }
    }
}

impl AppConfig {
    /// Load configuration from TOML file
    pub fn from_toml_file(path: &PathBuf) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::FileReadError(path.clone(), e))?;

        let config: AppConfig = toml::from_str(&content)
            .map_err(|e| ConfigError::ParseError(format!("TOML parse error: {}", e)))?;

        config.validate()?;
        Ok(config)
    }

    /// Save configuration to TOML file
    pub fn to_toml_file(&self, path: &PathBuf) -> Result<(), ConfigError> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| ConfigError::SerializeError(format!("TOML serialize error: {}", e)))?;

        std::fs::write(path, content).map_err(|e| ConfigError::FileWriteError(path.clone(), e))?;

        Ok(())
    }

    /// Create configuration from command line arguments
    pub fn from_args(model_path: Option<String>, video_path: Option<String>) -> Self {
        let mut config = Self::default();

        if let Some(model) = model_path {
            config.inference.model_path = PathBuf::from(model);
        }

        if let Some(video) = video_path {
            config.pipeline.video_source = video;
        }

        config
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Validate confidence threshold
        if self.inference.confidence_threshold < 0.0 || self.inference.confidence_threshold > 1.0 {
            return Err(ConfigError::InvalidValue(format!(
                "confidence_threshold must be between 0.0 and 1.0, got {}",
                self.inference.confidence_threshold
            )));
        }

        // Validate framerate
        if self.pipeline.framerate == 0 || self.pipeline.framerate > 120 {
            return Err(ConfigError::InvalidValue(format!(
                "framerate must be between 1 and 120, got {}",
                self.pipeline.framerate
            )));
        }

        // Validate target size
        if self.preprocessing.target_size[0] == 0 || self.preprocessing.target_size[1] == 0 {
            return Err(ConfigError::InvalidValue(
                "target_size dimensions must be greater than 0".to_string(),
            ));
        }

        // Validate backend
        if !["ort"].contains(&self.inference.backend.as_str()) {
            return Err(ConfigError::InvalidValue(format!(
                "unsupported backend: {}",
                self.inference.backend
            )));
        }

        // Validate device
        if !["auto", "cpu", "coreml", "cuda"].contains(&self.inference.device.as_str()) {
            return Err(ConfigError::InvalidValue(format!(
                "unsupported device: {}",
                self.inference.device
            )));
        }

        Ok(())
    }

    /// Get model file path
    pub fn model_path(&self) -> &PathBuf {
        &self.inference.model_path
    }

    /// Get video source
    pub fn video_source(&self) -> &str {
        &self.pipeline.video_source
    }

    /// Check if model file exists
    pub fn model_exists(&self) -> bool {
        self.inference.model_path.exists()
    }

    /// Check if video file exists (for file sources)
    pub fn video_exists(&self) -> bool {
        if self.pipeline.video_source == "auto" || self.pipeline.video_source == "webcam" {
            true // These don't need file validation
        } else {
            PathBuf::from(&self.pipeline.video_source).exists()
        }
    }
}

/// Configuration-related errors
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to read config file {0}: {1}")]
    FileReadError(PathBuf, std::io::Error),

    #[error("Failed to write config file {0}: {1}")]
    FileWriteError(PathBuf, std::io::Error),

    #[error("Config parse error: {0}")]
    ParseError(String),

    #[error("Config serialize error: {0}")]
    SerializeError(String),

    #[error("Invalid configuration value: {0}")]
    InvalidValue(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_configs() {
        let pipeline = PipelineConfig::default();
        assert_eq!(pipeline.video_source, "auto");
        assert!(pipeline.display_enabled);
        assert_eq!(pipeline.framerate, 30);

        let inference = InferenceConfig::default();
        assert_eq!(inference.backend, "ort");
        assert_eq!(inference.confidence_threshold, 0.5);
        assert_eq!(inference.device, "auto");

        let preprocessing = PreprocessingConfig::default();
        assert_eq!(preprocessing.target_size, [640, 640]);
        assert!(preprocessing.letterbox);
        assert!(preprocessing.normalize);
    }

    #[test]
    fn test_config_validation() {
        let mut config = AppConfig::default();

        // Valid config should pass
        assert!(config.validate().is_ok());

        // Invalid confidence threshold
        config.inference.confidence_threshold = 1.5;
        assert!(config.validate().is_err());
        config.inference.confidence_threshold = 0.5;

        // Invalid framerate
        config.pipeline.framerate = 0;
        assert!(config.validate().is_err());
        config.pipeline.framerate = 30;

        // Invalid target size
        config.preprocessing.target_size = [0, 640];
        assert!(config.validate().is_err());
        config.preprocessing.target_size = [640, 640];

        // Invalid backend
        config.inference.backend = "invalid".to_string();
        assert!(config.validate().is_err());
        config.inference.backend = "ort".to_string();

        // Invalid device
        config.inference.device = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_from_args() {
        let config = AppConfig::from_args(
            Some("custom_model.onnx".to_string()),
            Some("custom_video.mp4".to_string()),
        );

        assert_eq!(
            config.inference.model_path,
            PathBuf::from("custom_model.onnx")
        );
        assert_eq!(config.pipeline.video_source, "custom_video.mp4");
    }

    #[test]
    fn test_config_file_serialization() {
        let config = AppConfig::default();

        // Create a temporary file
        let temp_file = NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_path_buf();

        // Save config
        assert!(config.to_toml_file(&temp_path).is_ok());

        // Load config back
        let loaded_config = AppConfig::from_toml_file(&temp_path).unwrap();

        assert_eq!(
            config.pipeline.video_source,
            loaded_config.pipeline.video_source
        );
        assert_eq!(config.inference.backend, loaded_config.inference.backend);
        assert_eq!(
            config.preprocessing.target_size,
            loaded_config.preprocessing.target_size
        );
    }

    #[test]
    fn test_config_convenience_methods() {
        let config = AppConfig::default();

        assert_eq!(config.model_path(), &PathBuf::from("models/yolov8n.onnx"));
        assert_eq!(config.video_source(), "auto");

        // Test with non-existent model path
        let mut config_nonexistent = config.clone();
        config_nonexistent.inference.model_path = PathBuf::from("non_existent_model.onnx");
        assert!(!config_nonexistent.model_exists());
        assert!(config.video_exists()); // "auto" should return true
    }
}
