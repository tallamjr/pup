//! Comprehensive error handling for the Pup video processing system
//!
//! This module provides structured error types that cover all major failure modes
//! in the video processing pipeline, following the roadmap specifications.

use std::path::PathBuf;
use thiserror::Error;

/// Main error type for the Pup video processing system
#[derive(Error, Debug)]
pub enum PupError {
    // Input source errors
    #[error("Input source not available: {0}")]
    InputNotAvailable(String),

    #[error("Webcam device {device_id} not found or not accessible")]
    WebcamNotFound { device_id: u32 },

    #[error("Video file not found or not readable: {0}")]
    VideoFileError(PathBuf),

    #[error("RTSP stream connection failed: {0}")]
    RtspConnectionFailed(String),

    // CoreML and inference errors
    #[error("CoreML initialisation failed: {0}")]
    CoreMLFailure(String),

    #[error("Model loading failed: {0}")]
    ModelLoadError(PathBuf),

    #[error("Inference execution failed: {0}")]
    InferenceError(String),

    #[error("Invalid model format or compatibility: {0}")]
    ModelFormatError(PathBuf),

    // GStreamer pipeline errors
    #[error("GStreamer pipeline error: {0}")]
    PipelineError(String),

    #[error("GStreamer element creation failed: {0}")]
    ElementCreationFailed(String),

    #[error("GStreamer caps negotiation failed: {0}")]
    CapsNegotiationFailed(String),

    #[error("Video format not supported: {0}")]
    UnsupportedVideoFormat(String),

    // Configuration errors
    #[error("Configuration file not found: {0}")]
    ConfigNotFound(PathBuf),

    #[error("Configuration parsing failed: {0}")]
    ConfigParseError(String),

    #[error("Invalid configuration value: {field} = {value}")]
    InvalidConfigValue { field: String, value: String },

    #[error("Required configuration field missing: {0}")]
    MissingConfigField(String),

    // Performance and resource errors
    #[error("Insufficient memory: required {required_mb}MB, available {available_mb}MB")]
    InsufficientMemory {
        required_mb: usize,
        available_mb: usize,
    },

    #[error("Performance target not met: {target_fps} FPS, achieved {actual_fps} FPS")]
    PerformanceTarget {
        target_fps: f64,
        actual_fps: f64,
    },

    #[error("Frame processing timeout: {0}ms exceeded")]
    ProcessingTimeout(u64),

    #[error("Frame drop detected: dropped {0} frames")]
    FrameDropped(usize),

    // Platform-specific errors
    #[error("macOS NSApplication threading error: {0}")]
    MacOSThreadingError(String),

    #[error("OpenGL context creation failed: {0}")]
    OpenGLError(String),

    #[error("Platform not supported: {0}")]
    UnsupportedPlatform(String),

    // I/O and file system errors
    #[error("File permission denied: {0}")]
    PermissionDenied(PathBuf),

    #[error("Disk space insufficient: {0}MB required")]
    InsufficientDiskSpace(usize),

    #[error("Output directory creation failed: {0}")]
    OutputDirectoryError(PathBuf),

    // Network and streaming errors
    #[error("Network connection timeout: {0}")]
    NetworkTimeout(String),

    #[error("Stream encoding failed: {0}")]
    StreamEncodingError(String),

    #[error("RTMP server connection failed: {0}")]
    RtmpConnectionFailed(String),

    // Generic errors for compatibility
    #[error("Unexpected error: {0}")]
    Unexpected(String),

    #[error("Feature not implemented: {0}")]
    NotImplemented(String),
}

/// Result type alias for convenience
pub type PupResult<T> = std::result::Result<T, PupError>;

/// Helper trait for converting common error types to PupError
pub trait IntoPupError<T> {
    fn into_pup_error(self) -> PupResult<T>;
}

impl<T> IntoPupError<T> for Result<T, gstreamer::glib::Error> {
    fn into_pup_error(self) -> PupResult<T> {
        self.map_err(|e| PupError::PipelineError(e.to_string()))
    }
}

impl<T> IntoPupError<T> for Result<T, std::io::Error> {
    fn into_pup_error(self) -> PupResult<T> {
        self.map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => PupError::Unexpected(format!("File not found: {}", e)),
            std::io::ErrorKind::PermissionDenied => {
                PupError::PermissionDenied(PathBuf::from("unknown"))
            }
            _ => PupError::Unexpected(format!("I/O error: {}", e)),
        })
    }
}

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
        match err.kind() {
            std::io::ErrorKind::NotFound => PupError::Unexpected(format!("File not found: {}", err)),
            std::io::ErrorKind::PermissionDenied => {
                PupError::PermissionDenied(PathBuf::from("unknown"))
            }
            _ => PupError::Unexpected(format!("I/O error: {}", err)),
        }
    }
}

/// Error context builder for adding additional information
pub struct ErrorContext {
    base_error: PupError,
    context: Vec<String>,
}

impl ErrorContext {
    pub fn new(error: PupError) -> Self {
        Self {
            base_error: error,
            context: Vec::new(),
        }
    }

    pub fn with_context(mut self, context: &str) -> Self {
        self.context.push(context.to_string());
        self
    }

    pub fn build(self) -> PupError {
        if self.context.is_empty() {
            self.base_error
        } else {
            PupError::Unexpected(format!("{}: {}", self.context.join(" -> "), self.base_error))
        }
    }
}

/// Convenience macro for adding context to errors
#[macro_export]
macro_rules! pup_context {
    ($result:expr, $context:expr) => {
        $result.map_err(|e| {
            $crate::error::ErrorContext::new(e.into())
                .with_context($context)
                .build()
        })
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let error = PupError::InputNotAvailable("webcam".to_string());
        assert_eq!(error.to_string(), "Input source not available: webcam");

        let error = PupError::CoreMLFailure("provider not available".to_string());
        assert_eq!(
            error.to_string(),
            "CoreML initialisation failed: provider not available"
        );
    }

    #[test]
    fn test_error_conversion() {
        let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "test file");
        let pup_error: PupError = io_error.into();

        match pup_error {
            PupError::Unexpected(message) => {
                assert!(message.contains("File not found"));
            }
            _ => panic!("Expected Unexpected error variant"),
        }
    }

    #[test]
    fn test_error_context() {
        let base_error = PupError::ModelLoadError(PathBuf::from("test.onnx"));

        let contextual_error = ErrorContext::new(base_error)
            .with_context("loading configuration")
            .with_context("initialising inference")
            .build();

        match contextual_error {
            PupError::Unexpected(message) => {
                assert!(message.contains("loading configuration"));
                assert!(message.contains("initialising inference"));
                assert!(message.contains("Model loading failed"));
            }
            _ => panic!("Expected Unexpected error variant"),
        }
    }

    #[test]
    fn test_inference_error_conversion() {
        let inference_error = crate::inference::InferenceError::ModelNotLoaded;
        let pup_error: PupError = inference_error.into();

        match pup_error {
            PupError::InferenceError(details) => {
                assert_eq!(details, "Model not loaded");
            }
            _ => panic!("Expected InferenceError variant"),
        }
    }

    #[test]
    fn test_structured_errors() {
        let error = PupError::WebcamNotFound { device_id: 0 };
        assert!(error.to_string().contains("device 0"));

        let error = PupError::InvalidConfigValue {
            field: "threshold".to_string(),
            value: "invalid".to_string(),
        };
        assert!(error.to_string().contains("threshold"));
        assert!(error.to_string().contains("invalid"));
    }
}