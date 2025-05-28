//! Utility types and functions

pub mod detection;

pub use detection::{apply_nms, filter_by_confidence, Detection, DetectionError};
