//! Utility types and functions

pub mod detection;
pub mod coco_classes;

pub use detection::{apply_nms, filter_by_confidence, Detection, DetectionError};
