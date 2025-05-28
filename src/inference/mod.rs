//! ML inference abstractions and implementations

use crate::utils::Detection;
use std::path::Path;
use thiserror::Error;

pub mod ort_backend;

pub use ort_backend::OrtBackend;

/// Types of computer vision tasks supported
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskType {
    ObjectDetection,
    KeypointDetection,
    PoseEstimation,
    ImageClassification,
    SemanticSegmentation,
    InstanceSegmentation,
    FacialRecognition,
    ActivityRecognition,
}

/// Output from different types of computer vision tasks
#[derive(Debug, Clone)]
pub enum TaskOutput {
    Detections(Vec<Detection>),
    // TODO: Add other task outputs as we implement them
    // Keypoints(Vec<Keypoint>),
    // Poses(Vec<Pose>),
    // Classifications(Vec<Classification>),
    // Segmentation(SegmentationMask),
    // Activities(Vec<Activity>),
}

/// Inference backend trait for different ML frameworks
pub trait InferenceBackend: Send + Sync {
    /// Load a model from the given path
    fn load_model(&mut self, path: &Path) -> Result<(), InferenceError>;

    /// Run inference on input tensor data
    fn infer(&self, input: &[f32]) -> Result<TaskOutput, InferenceError>;

    /// Get the expected input shape [batch, channels, height, width]
    fn get_input_shape(&self) -> &[usize];

    /// Get the task type this backend supports
    fn get_task_type(&self) -> TaskType;

    /// Get the model's confidence threshold
    fn get_confidence_threshold(&self) -> f32;

    /// Set the confidence threshold for filtering detections
    fn set_confidence_threshold(&mut self, threshold: f32);
}

/// Model post-processor trait for converting raw outputs to detections
pub trait ModelPostProcessor {
    /// Process raw model output into detections
    fn process_raw_output(
        &self,
        output: &[f32],
        input_shape: &[usize],
    ) -> Result<Vec<Detection>, InferenceError>;

    /// Apply Non-Maximum Suppression to detections
    fn apply_nms(&self, detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection>;

    /// Filter detections by confidence threshold
    fn filter_by_confidence(&self, detections: Vec<Detection>, threshold: f32) -> Vec<Detection>;
}

/// Inference-related errors
#[derive(Error, Debug)]
pub enum InferenceError {
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    #[error("Invalid input shape: expected {expected:?}, got {actual:?}")]
    InvalidInputShape {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Invalid output format: {0}")]
    InvalidOutputFormat(String),

    #[error("Model not loaded")]
    ModelNotLoaded,

    #[error("Invalid confidence threshold: {0} (must be between 0.0 and 1.0)")]
    InvalidConfidenceThreshold(f32),

    #[error("ORT runtime error: {0}")]
    OrtError(String),
}

/// Default YOLO post-processor implementation
pub struct YoloPostProcessor {
    iou_threshold: f32,
    num_classes: usize,
}

impl YoloPostProcessor {
    /// Create a new YOLO post-processor
    pub fn new(num_classes: usize, iou_threshold: f32) -> Self {
        Self {
            iou_threshold,
            num_classes,
        }
    }

    /// Default COCO dataset post-processor (80 classes)
    pub fn coco_default() -> Self {
        Self::new(80, 0.5)
    }
}

impl ModelPostProcessor for YoloPostProcessor {
    fn process_raw_output(
        &self,
        output: &[f32],
        input_shape: &[usize],
    ) -> Result<Vec<Detection>, InferenceError> {
        // YOLOv8 output format: [1, 84, 8400] -> flattened to [84 * 8400]
        // where 84 = 4 (bbox coords) + 80 (class scores)
        // Each of the 8400 predictions corresponds to an anchor point
        
        if output.len() != 84 * 8400 {
            return Err(InferenceError::InvalidOutputFormat(
                format!("Expected YOLOv8 output size 84*8400={}, got {}", 84 * 8400, output.len())
            ));
        }

        let num_predictions = 8400;
        let mut detections = Vec::new();

        // Get model input dimensions
        let model_width = if input_shape.len() >= 4 {
            input_shape[3] as f32
        } else {
            640.0
        };
        let model_height = if input_shape.len() >= 3 {
            input_shape[2] as f32
        } else {
            640.0
        };

        // YOLOv8 output is in format [84, 8400], so we need to transpose indexing
        for i in 0..num_predictions {
            // Get bbox coordinates: output[0..4, i]
            let cx = output[i];                    // center_x at index [0, i]
            let cy = output[num_predictions + i];  // center_y at index [1, i]  
            let w = output[2 * num_predictions + i]; // width at index [2, i]
            let h = output[3 * num_predictions + i]; // height at index [3, i]

            // Convert from center format to corner format
            let x1 = cx - w * 0.5;
            let y1 = cy - h * 0.5;
            let x2 = cx + w * 0.5;
            let y2 = cy + h * 0.5;

            // Skip detections with invalid coordinates
            if x1 < 0.0 || y1 < 0.0 || x2 > model_width || y2 > model_height || x1 >= x2 || y1 >= y2 {
                continue;
            }

            // Find best class and confidence score from class predictions
            let mut best_score = 0.0f32;
            let mut best_class = 0i32;

            for class_idx in 0..self.num_classes {
                // Class scores start at index [4, i] through [83, i]
                let score = output[(4 + class_idx) * num_predictions + i];
                if score > best_score {
                    best_score = score;
                    best_class = class_idx as i32;
                }
            }

            // Apply strict confidence threshold to reduce false positives
            if best_score > 0.5 { // Much higher threshold
                detections.push(Detection::new(x1, y1, x2, y2, best_score, best_class));
            }
        }

        // Apply Non-Maximum Suppression to remove overlapping detections
        let detections_after_nms = self.apply_nms(detections, self.iou_threshold);
        
        Ok(detections_after_nms)
    }

    fn apply_nms(&self, detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
        crate::utils::apply_nms(detections, iou_threshold)
    }

    fn filter_by_confidence(&self, detections: Vec<Detection>, threshold: f32) -> Vec<Detection> {
        crate::utils::filter_by_confidence(detections, threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yolo_postprocessor_creation() {
        let processor = YoloPostProcessor::new(80, 0.5);
        assert_eq!(processor.num_classes, 80);
        assert_eq!(processor.iou_threshold, 0.5);
    }

    #[test]
    fn test_yolo_postprocessor_coco_default() {
        let processor = YoloPostProcessor::coco_default();
        assert_eq!(processor.num_classes, 80);
        assert_eq!(processor.iou_threshold, 0.5);
    }

    #[test]
    fn test_inference_error_display() {
        let error = InferenceError::ModelNotLoaded;
        assert_eq!(error.to_string(), "Model not loaded");

        let error = InferenceError::InvalidConfidenceThreshold(1.5);
        assert!(error.to_string().contains("1.5"));
    }

    #[test]
    fn test_yolo_output_processing() {
        let processor = YoloPostProcessor::new(80, 0.5);

        // Create mock YOLOv8 output: one detection with high confidence for class 0
        let mut output = vec![0.0f32; 84];
        output[0] = 320.0; // cx in pixels (center at 320)
        output[1] = 320.0; // cy in pixels (center at 320)
        output[2] = 128.0; // width in pixels
        output[3] = 128.0; // height in pixels
        output[4] = 0.9;   // confidence for class 0

        let input_shape = vec![1, 3, 640, 640];
        let detections = processor.process_raw_output(&output, &input_shape).unwrap();

        assert_eq!(detections.len(), 1);
        let det = &detections[0];
        assert_eq!(det.class_id, 0);
        assert_eq!(det.score, 0.9);

        // Check bounding box conversion (center 320,320 with size 128x128 -> corners 256,256 to 448,448)
        assert!((det.x1 - 256.0).abs() < 1.0); // 320 - 128/2 = 256
        assert!((det.y1 - 256.0).abs() < 1.0); // 320 - 128/2 = 256
        assert!((det.x2 - 384.0).abs() < 1.0); // 320 + 128/2 = 384
        assert!((det.y2 - 384.0).abs() < 1.0); // 320 + 128/2 = 384
    }
}
