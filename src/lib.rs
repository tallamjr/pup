//! Pup Video Processing Library
//!
//! A modular video processing framework for real-time object detection
//! using GStreamer and ONNX Runtime.

pub mod common;
pub mod config;
pub mod error;
pub mod inference;
pub mod metrics;
pub mod pipeline;
pub mod preprocessing;
pub mod utils;

// GStreamer plugins
pub mod gst_plugins;

// Re-export commonly used types
pub use config::{
    AppConfig, InferenceConfig, InputConfig, ModeConfig, OutputConfig, PipelineConfig,
    PreprocessingConfig,
};
pub use error::{PupError, PupResult};
pub use inference::{InferenceBackend, InferenceError, OrtBackend, TaskOutput, TaskType};
pub use metrics::{
    ConsoleReporter, FrameTimer, JsonReporter, Metrics, MetricsReporter, PerformanceMonitor,
};
pub use preprocessing::Preprocessor;
pub use utils::{Detection, DetectionError};

/// Main application entry point
pub use common::run;

/// Current version of the library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Python bindings module
#[cfg(feature = "python")]
pub mod python;

// Python module definition for PyO3
#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn _pup(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_function(wrap_pyfunction!(python::benchmark_inference, m)?)?;
    m.add_class::<python::PyOrtBackend>()?;
    Ok(())
}
