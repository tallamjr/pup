//! Error handling for the pup video processing system

use std::path::PathBuf;
use thiserror::Error;

/// Main error type for the pup video processing system
#[derive(Error, Debug, Clone)]
pub enum PupError {
    #[error("Input source not available: {0}")]
    InputNotAvailable(String),

    #[error("Video file not found or not readable: {0}")]
    VideoFileError(PathBuf),

    #[error("Model loading failed: {0}")]
    ModelLoadError(PathBuf),

    #[error("Inference execution failed: {0}")]
    InferenceError(String),

    #[error("GStreamer pipeline error: {0}")]
    PipelineError(String),

    #[error("GStreamer element creation failed: {0}")]
    ElementCreationFailed(String),

    #[error("Configuration file not found: {0}")]
    ConfigNotFound(PathBuf),

    #[error("Configuration parsing failed: {0}")]
    ConfigParseError(String),

    #[error("Invalid configuration value: {field} = {value}")]
    InvalidConfigValue { field: String, value: String },

    #[error("Required configuration field missing: {0}")]
    MissingConfigField(String),

    #[error("Unexpected error: {0}")]
    Unexpected(String),
}

/// Result type alias
pub type PupResult<T> = std::result::Result<T, PupError>;

impl From<crate::inference::InferenceError> for PupError {
    fn from(err: crate::inference::InferenceError) -> Self {
        match err {
            crate::inference::InferenceError::ModelLoadError(msg) => {
                PupError::ModelLoadError(PathBuf::from(msg))
            }
            crate::inference::InferenceError::InferenceFailed(msg) => {
                PupError::InferenceError(msg)
            }
            crate::inference::InferenceError::OrtError(msg) => {
                PupError::InferenceError(format!("ONNX Runtime error: {}", msg))
            }
            other => PupError::InferenceError(other.to_string()),
        }
    }
}

impl From<toml::de::Error> for PupError {
    fn from(err: toml::de::Error) -> Self {
        PupError::ConfigParseError(err.to_string())
    }
}

impl From<std::io::Error> for PupError {
    fn from(err: std::io::Error) -> Self {
        PupError::Unexpected(format!("I/O error: {}", err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let error = PupError::InputNotAvailable("webcam".to_string());
        assert_eq!(error.to_string(), "Input source not available: webcam");
    }

    #[test]
    fn test_io_error_conversion() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "test file");
        let pup_error: PupError = io_error.into();
        match pup_error {
            PupError::Unexpected(msg) => assert!(msg.contains("I/O error")),
            _ => panic!("Expected Unexpected variant"),
        }
    }

    #[test]
    fn test_inference_error_conversion() {
        let inference_error = crate::inference::InferenceError::ModelNotLoaded;
        let pup_error: PupError = inference_error.into();
        match pup_error {
            PupError::InferenceError(details) => assert_eq!(details, "Model not loaded"),
            _ => panic!("Expected InferenceError variant"),
        }
    }
}
