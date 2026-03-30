//! Pup Video Processing Library
//!
//! Real-time object detection using GStreamer and ONNX Runtime.

pub mod common;
pub mod config;
pub mod error;
pub mod inference;
pub mod preprocessing;
pub mod utils;

pub mod gst_plugins;

pub use config::{
    AppConfig, InferenceConfig, InputConfig, ModeConfig, OutputConfig, PipelineConfig,
    PreprocessingConfig,
};
pub use error::{PupError, PupResult};
pub use inference::{InferenceBackend, InferenceError, OrtBackend, TaskOutput, TaskType};
pub use preprocessing::Preprocessor;
pub use utils::{Detection, DetectionError};

pub use common::run;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
