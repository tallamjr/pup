//! Utility types and functions

pub mod coco_classes;
pub mod detection;

pub use detection::{apply_nms, filter_by_confidence, Detection, DetectionError};
