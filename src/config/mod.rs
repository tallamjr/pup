//! Configuration management

use crate::error::{PupError, PupResult};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

/// Application mode configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeConfig {
    /// Application mode: "production", "live", "detection", "benchmark"
    #[serde(rename = "type")]
    pub mode_type: String,
}

impl Default for ModeConfig {
    fn default() -> Self {
        Self {
            mode_type: "production".to_string(),
        }
    }
}

/// Input source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputConfig {
    /// Input source type: "webcam", file path, or "rtsp://url"
    pub source: String,
    /// Device ID for webcam (optional)
    pub device_id: Option<u32>,
    /// GStreamer caps specification (optional)
    pub caps: Option<String>,
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            source: "webcam".to_string(),
            device_id: Some(0),
            caps: Some("video/x-raw,width=1280,height=720,framerate=30/1".to_string()),
        }
    }
}

/// Pipeline configuration (kept for backwards compatibility)
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
    /// Execution providers in priority order: ["coreml", "cpu"]
    pub execution_providers: Vec<String>,
    /// Path to the model file
    pub model_path: PathBuf,
    /// Confidence threshold for detections
    pub confidence_threshold: f32,
    /// Batch size for inference
    pub batch_size: usize,
    /// Device: "auto", "cpu", "coreml", "cuda" (deprecated, use execution_providers)
    #[serde(default)]
    pub device: Option<String>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            backend: "ort".to_string(),
            execution_providers: vec!["coreml".to_string(), "cpu".to_string()],
            model_path: PathBuf::from("models/yolov8n.onnx"),
            confidence_threshold: 0.5,
            batch_size: 1,
            device: None,
        }
    }
}

/// Output configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Whether display is enabled
    pub display_enabled: bool,
    /// Whether recording is enabled
    pub recording_enabled: bool,
    /// Output format: "mp4", "json", "rtmp"
    pub output_format: String,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            display_enabled: true,
            recording_enabled: false,
            output_format: "mp4".to_string(),
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
    /// Application mode configuration
    #[serde(default)]
    pub mode: ModeConfig,
    /// Input source configuration
    #[serde(default)]
    pub input: InputConfig,
    /// Inference configuration
    pub inference: InferenceConfig,
    /// Output configuration
    #[serde(default)]
    pub output: OutputConfig,
    /// Preprocessing configuration (optional)
    #[serde(default)]
    pub preprocessing: Option<PreprocessingConfig>,
    /// Legacy pipeline configuration (backwards compatibility)
    /// This field provides direct access for backward compatibility
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pipeline: Option<PipelineConfig>,
}

impl Default for AppConfig {
    fn default() -> Self {
        let mut config = Self {
            mode: ModeConfig::default(),
            input: InputConfig::default(),
            inference: InferenceConfig::default(),
            output: OutputConfig::default(),
            preprocessing: Some(PreprocessingConfig::default()),
            pipeline: None,
        };
        
        // Ensure pipeline is available for backwards compatibility
        config.pipeline = Some(config.get_pipeline());
        config
    }
}

impl AppConfig {
    /// Load configuration from TOML file with comprehensive validation
    pub fn from_toml_file(path: &PathBuf) -> PupResult<Self> {
        if !path.exists() {
            return Err(PupError::ConfigNotFound(path.clone()));
        }

        let content = std::fs::read_to_string(path).map_err(|e| PupError::ConfigParseError(
            format!("Failed to read config file {}: {}", path.display(), e)
        ))?;

        let config: AppConfig = toml::from_str(&content).map_err(|e| PupError::ConfigParseError(
            format!("TOML parse error in {}: {}", path.display(), e)
        ))?;

        config.validate()?;
        Ok(config)
    }

    /// Load configuration from TOML file (legacy compatibility)
    pub fn from_toml_file_legacy(path: &PathBuf) -> Result<Self, ConfigError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| ConfigError::FileReadError(path.clone(), e))?;

        let config: AppConfig = toml::from_str(&content)
            .map_err(|e| ConfigError::ParseError(format!("TOML parse error: {}", e)))?;

        config.validate_legacy()?;
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
            config.input.source = video;
        }

        config
    }

    /// Comprehensive configuration validation
    pub fn validate(&self) -> PupResult<()> {
        // Validate mode
        self.validate_mode()?;

        // Validate input source
        self.validate_input()?;

        // Validate inference configuration
        self.validate_inference()?;

        // Validate output configuration
        self.validate_output()?;

        // Validate preprocessing if present
        if let Some(ref preprocessing) = self.preprocessing {
            self.validate_preprocessing(preprocessing)?;
        }

        // Validate legacy pipeline if present
        if let Some(ref pipeline) = self.pipeline {
            self.validate_pipeline_legacy(pipeline)?;
        }

        Ok(())
    }

    /// Validate mode configuration
    fn validate_mode(&self) -> PupResult<()> {
        const VALID_MODES: &[&str] = &["production", "live", "detection", "benchmark"];
        
        if !VALID_MODES.contains(&self.mode.mode_type.as_str()) {
            return Err(PupError::InvalidConfigValue {
                field: "mode.type".to_string(),
                value: self.mode.mode_type.clone(),
            });
        }
        
        Ok(())
    }

    /// Validate input configuration
    fn validate_input(&self) -> PupResult<()> {
        // Validate device ID if specified
        if let Some(device_id) = self.input.device_id {
            if device_id > 99 {
                return Err(PupError::InvalidConfigValue {
                    field: "input.device_id".to_string(),
                    value: device_id.to_string(),
                });
            }
        }

        // Validate source format
        if self.input.source.starts_with("rtsp://") {
            // Basic RTSP URL validation
            if !self.input.source.contains("://") {
                return Err(PupError::InvalidConfigValue {
                    field: "input.source".to_string(),
                    value: self.input.source.clone(),
                });
            }
        } else if self.input.source != "webcam" {
            // Check if file path exists for non-webcam sources
            let path = PathBuf::from(&self.input.source);
            if !path.exists() && !self.input.source.starts_with("/dev/video") {
                return Err(PupError::VideoFileError(path));
            }
        }

        // Validate caps format if specified
        if let Some(ref caps) = self.input.caps {
            if !caps.contains("video/") {
                return Err(PupError::InvalidConfigValue {
                    field: "input.caps".to_string(),
                    value: caps.clone(),
                });
            }
        }

        Ok(())
    }

    /// Validate inference configuration
    fn validate_inference(&self) -> PupResult<()> {
        // Validate backend
        const VALID_BACKENDS: &[&str] = &["ort"];
        if !VALID_BACKENDS.contains(&self.inference.backend.as_str()) {
            return Err(PupError::InvalidConfigValue {
                field: "inference.backend".to_string(),
                value: self.inference.backend.clone(),
            });
        }

        // Validate execution providers
        const VALID_PROVIDERS: &[&str] = &["coreml", "cpu", "cuda", "tensorrt", "openvino"];
        for provider in &self.inference.execution_providers {
            if !VALID_PROVIDERS.contains(&provider.as_str()) {
                return Err(PupError::InvalidConfigValue {
                    field: "inference.execution_providers".to_string(),
                    value: provider.clone(),
                });
            }
        }

        // Validate confidence threshold
        if self.inference.confidence_threshold < 0.0 || self.inference.confidence_threshold > 1.0 {
            return Err(PupError::InvalidConfigValue {
                field: "inference.confidence_threshold".to_string(),
                value: self.inference.confidence_threshold.to_string(),
            });
        }

        // Validate batch size
        if self.inference.batch_size == 0 || self.inference.batch_size > 32 {
            return Err(PupError::InvalidConfigValue {
                field: "inference.batch_size".to_string(),
                value: self.inference.batch_size.to_string(),
            });
        }

        // Validate model path exists
        if !self.inference.model_path.exists() {
            return Err(PupError::ModelLoadError(self.inference.model_path.clone()));
        }

        // Validate model file extension
        if let Some(extension) = self.inference.model_path.extension() {
            if extension != "onnx" {
                return Err(PupError::ModelFormatError(self.inference.model_path.clone()));
            }
        } else {
            return Err(PupError::ModelFormatError(self.inference.model_path.clone()));
        }

        Ok(())
    }

    /// Validate output configuration
    fn validate_output(&self) -> PupResult<()> {
        const VALID_FORMATS: &[&str] = &["mp4", "json", "rtmp", "avi", "mov"];
        
        if !VALID_FORMATS.contains(&self.output.output_format.as_str()) {
            return Err(PupError::InvalidConfigValue {
                field: "output.output_format".to_string(),
                value: self.output.output_format.clone(),
            });
        }

        Ok(())
    }

    /// Validate preprocessing configuration
    fn validate_preprocessing(&self, preprocessing: &PreprocessingConfig) -> PupResult<()> {
        // Validate target size
        if preprocessing.target_size[0] == 0 || preprocessing.target_size[1] == 0 {
            return Err(PupError::InvalidConfigValue {
                field: "preprocessing.target_size".to_string(),
                value: format!("{:?}", preprocessing.target_size),
            });
        }

        // Check reasonable size limits
        if preprocessing.target_size[0] > 4096 || preprocessing.target_size[1] > 4096 {
            return Err(PupError::InvalidConfigValue {
                field: "preprocessing.target_size".to_string(),
                value: format!("{:?}", preprocessing.target_size),
            });
        }

        Ok(())
    }

    /// Legacy validation for backwards compatibility
    pub fn validate_legacy(&self) -> Result<(), ConfigError> {
        if let Some(ref pipeline) = self.pipeline {
            self.validate_pipeline_legacy(pipeline).map_err(|e| {
                ConfigError::InvalidValue(e.to_string())
            })?;
        }

        Ok(())
    }

    /// Validate legacy pipeline configuration
    fn validate_pipeline_legacy(&self, pipeline: &PipelineConfig) -> PupResult<()> {
        // Validate framerate
        if pipeline.framerate == 0 || pipeline.framerate > 120 {
            return Err(PupError::InvalidConfigValue {
                field: "pipeline.framerate".to_string(),
                value: pipeline.framerate.to_string(),
            });
        }

        Ok(())
    }

    /// Get model file path
    pub fn model_path(&self) -> &PathBuf {
        &self.inference.model_path
    }

    /// Get input source
    pub fn input_source(&self) -> &str {
        &self.input.source
    }

    /// Get video source (legacy compatibility)
    pub fn video_source(&self) -> &str {
        if let Some(ref pipeline) = self.pipeline {
            &pipeline.video_source
        } else {
            &self.input.source
        }
    }

    /// Check if model file exists
    pub fn model_exists(&self) -> bool {
        self.inference.model_path.exists()
    }

    /// Check if video file exists (for file sources)
    pub fn video_exists(&self) -> bool {
        let source = self.input_source();
        if source == "webcam" || source.starts_with("rtsp://") || source.starts_with("/dev/video") {
            true // These don't need file validation
        } else {
            PathBuf::from(source).exists()
        }
    }

    /// Create a production configuration example
    pub fn production_example() -> Self {
        Self {
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
            preprocessing: Some(PreprocessingConfig {
                target_size: [640, 640],
                letterbox: true,
                normalize: true,
            }),
            pipeline: None,
        }
    }

    /// Get effective preprocessing configuration
    pub fn get_preprocessing(&self) -> PreprocessingConfig {
        self.preprocessing.clone().unwrap_or_default()
    }

    /// Get pipeline configuration (backwards compatibility helper)
    pub fn get_pipeline(&self) -> PipelineConfig {
        if let Some(ref pipeline) = self.pipeline {
            pipeline.clone()
        } else {
            // Convert new structure to legacy pipeline for compatibility
            PipelineConfig {
                video_source: self.input.source.clone(),
                display_enabled: self.output.display_enabled,
                framerate: 30, // Default framerate
            }
        }
    }

    /// Set pipeline configuration (backwards compatibility helper)
    pub fn set_pipeline(&mut self, pipeline: PipelineConfig) {
        self.pipeline = Some(pipeline.clone());
        // Also update the new structure
        self.input.source = pipeline.video_source;
        self.output.display_enabled = pipeline.display_enabled;
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
        assert_eq!(inference.device, Some("auto".to_string()));

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

        // Invalid framerate - using helper to access pipeline
        config.pipeline.as_mut().unwrap().framerate = 0;
        assert!(config.validate().is_err());
        config.pipeline.as_mut().unwrap().framerate = 30;

        // Invalid target size
        config.preprocessing.as_mut().unwrap().target_size = [0, 640];
        assert!(config.validate().is_err());
        config.preprocessing.as_mut().unwrap().target_size = [640, 640];

        // Invalid backend
        config.inference.backend = "invalid".to_string();
        assert!(config.validate().is_err());
        config.inference.backend = "ort".to_string();

        // Invalid device
        config.inference.device = Some("invalid".to_string());
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
        assert_eq!(config.input.source, "custom_video.mp4");
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
            config.input.source,
            loaded_config.input.source
        );
        assert_eq!(config.inference.backend, loaded_config.inference.backend);
        assert_eq!(
            config.preprocessing.as_ref().unwrap().target_size,
            loaded_config.preprocessing.as_ref().unwrap().target_size
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
