//! ONNX Runtime backend implementation

use super::{InferenceBackend, InferenceError, ModelPostProcessor, YoloPostProcessor, TaskType, TaskOutput};
use ort::{
    execution_providers::CoreMLExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use std::path::Path;
use std::sync::Arc;

/// ONNX Runtime inference backend
pub struct OrtBackend {
    session: Option<Arc<Session>>,
    input_shape: Vec<usize>,
    confidence_threshold: f32,
    post_processor: Box<dyn ModelPostProcessor + Send + Sync>,
}

impl OrtBackend {
    /// Create a new ORT backend with default settings
    pub fn new() -> Self {
        Self {
            session: None,
            input_shape: vec![1, 3, 640, 640], // Default YOLO input shape
            confidence_threshold: 0.5,
            post_processor: Box::new(YoloPostProcessor::coco_default()),
        }
    }

    /// Create a new ORT backend with custom post-processor
    pub fn with_post_processor(post_processor: Box<dyn ModelPostProcessor + Send + Sync>) -> Self {
        Self {
            session: None,
            input_shape: vec![1, 3, 640, 640],
            confidence_threshold: 0.5,
            post_processor,
        }
    }

    /// Create an optimized session with CoreML execution provider
    fn create_session(model_path: &Path) -> Result<Session, InferenceError> {
        let session = Session::builder()
            .map_err(|e| {
                InferenceError::OrtError(format!("Failed to create session builder: {}", e))
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| {
                InferenceError::OrtError(format!("Failed to set optimization level: {}", e))
            })?
            .with_intra_threads(8)
            .map_err(|e| InferenceError::OrtError(format!("Failed to set intra threads: {}", e)))?;

        // Try to use CoreML on Apple platforms
        #[cfg(target_vendor = "apple")]
        let session = session
            .with_execution_providers([CoreMLExecutionProvider::default().build()])
            .map_err(|e| {
                InferenceError::OrtError(format!("Failed to set execution providers: {}", e))
            })?;

        let session = session
            .commit_from_file(model_path)
            .map_err(|e| InferenceError::ModelLoadError(format!("Failed to load model: {}", e)))?;

        Ok(session)
    }

    /// Validate input tensor shape and data
    fn validate_input(&self, input: &[f32]) -> Result<(), InferenceError> {
        let expected_size: usize = self.input_shape.iter().product();
        if input.len() != expected_size {
            return Err(InferenceError::InvalidInputShape {
                expected: self.input_shape.clone(),
                actual: vec![input.len()],
            });
        }
        Ok(())
    }

    /// Convert input data to ONNX tensor
    fn create_input_tensor(&self, input: &[f32]) -> Result<Value, InferenceError> {
        let shape: Vec<i64> = self.input_shape.iter().map(|&x| x as i64).collect();

        Value::from_array((shape, input.to_vec()))
            .map_err(|e| InferenceError::OrtError(format!("Failed to create input tensor: {}", e)))
            .map(|v| v.into())
    }
}

impl Default for OrtBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InferenceBackend for OrtBackend {
    fn load_model(&mut self, path: &Path) -> Result<(), InferenceError> {
        if !path.exists() {
            return Err(InferenceError::ModelLoadError(format!(
                "Model file does not exist: {}",
                path.display()
            )));
        }

        let session = Self::create_session(path)?;
        self.session = Some(Arc::new(session));

        Ok(())
    }

    fn infer(&self, input: &[f32]) -> Result<TaskOutput, InferenceError> {
        let session = self
            .session
            .as_ref()
            .ok_or(InferenceError::ModelNotLoaded)?;

        // Validate input
        self.validate_input(input)?;

        // Create input tensor
        let input_tensor = self.create_input_tensor(input)?;

        // Run inference
        let outputs = session
            .run(vec![("images".to_owned(), input_tensor)])
            .map_err(|e| InferenceError::InferenceFailed(format!("Session run failed: {}", e)))?;

        // Extract output data
        let mut output_values = outputs.values();
        let first_output = output_values.next().ok_or_else(|| {
            InferenceError::InvalidOutputFormat("No outputs received from model".to_string())
        })?;

        let output_tensor = first_output.try_extract_tensor::<f32>().map_err(|e| {
            InferenceError::OrtError(format!("Failed to extract output tensor: {}", e))
        })?;

        let output_data: Vec<f32> = output_tensor.view().iter().copied().collect();

        // Process raw output to detections
        let mut detections = self
            .post_processor
            .process_raw_output(&output_data, &self.input_shape)?;

        // Filter by confidence threshold
        detections = self
            .post_processor
            .filter_by_confidence(detections, self.confidence_threshold);

        // Apply NMS
        detections = self.post_processor.apply_nms(detections, 0.5);

        Ok(TaskOutput::Detections(detections))
    }

    fn get_input_shape(&self) -> &[usize] {
        &self.input_shape
    }

    fn get_task_type(&self) -> TaskType {
        TaskType::ObjectDetection // Default to object detection for now
    }

    fn get_confidence_threshold(&self) -> f32 {
        self.confidence_threshold
    }

    fn set_confidence_threshold(&mut self, threshold: f32) {
        if threshold < 0.0 || threshold > 1.0 {
            eprintln!(
                "Warning: Invalid confidence threshold {}, keeping current value {}",
                threshold, self.confidence_threshold
            );
            return;
        }
        self.confidence_threshold = threshold;
    }
}

/// Builder for ORT backend configuration
pub struct OrtBackendBuilder {
    input_shape: Vec<usize>,
    confidence_threshold: f32,
    post_processor: Option<Box<dyn ModelPostProcessor + Send + Sync>>,
}

impl OrtBackendBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            input_shape: vec![1, 3, 640, 640],
            confidence_threshold: 0.5,
            post_processor: None,
        }
    }

    /// Set the input shape
    pub fn with_input_shape(mut self, shape: Vec<usize>) -> Self {
        self.input_shape = shape;
        self
    }

    /// Set the confidence threshold
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold;
        self
    }

    /// Set a custom post-processor
    pub fn with_post_processor(
        mut self,
        processor: Box<dyn ModelPostProcessor + Send + Sync>,
    ) -> Self {
        self.post_processor = Some(processor);
        self
    }

    /// Build the ORT backend
    pub fn build(self) -> OrtBackend {
        let post_processor = self
            .post_processor
            .unwrap_or_else(|| Box::new(YoloPostProcessor::coco_default()));

        OrtBackend {
            session: None,
            input_shape: self.input_shape,
            confidence_threshold: self.confidence_threshold,
            post_processor,
        }
    }
}

impl Default for OrtBackendBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_ort_backend_creation() {
        let backend = OrtBackend::new();
        assert_eq!(backend.get_input_shape(), &[1, 3, 640, 640]);
        assert_eq!(backend.get_confidence_threshold(), 0.5);
    }

    #[test]
    fn test_ort_backend_builder() {
        let backend = OrtBackendBuilder::new()
            .with_input_shape(vec![1, 3, 416, 416])
            .with_confidence_threshold(0.7)
            .build();

        assert_eq!(backend.get_input_shape(), &[1, 3, 416, 416]);
        assert_eq!(backend.get_confidence_threshold(), 0.7);
    }

    #[test]
    fn test_confidence_threshold_validation() {
        let mut backend = OrtBackend::new();

        // Valid threshold
        backend.set_confidence_threshold(0.8);
        assert_eq!(backend.get_confidence_threshold(), 0.8);

        // Invalid thresholds should be rejected
        backend.set_confidence_threshold(-0.1);
        assert_eq!(backend.get_confidence_threshold(), 0.8); // Should remain unchanged

        backend.set_confidence_threshold(1.5);
        assert_eq!(backend.get_confidence_threshold(), 0.8); // Should remain unchanged
    }

    #[test]
    fn test_model_loading_with_nonexistent_file() {
        let mut backend = OrtBackend::new();
        let result = backend.load_model(&PathBuf::from("nonexistent.onnx"));

        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::ModelLoadError(_) => {} // Expected
            _ => panic!("Expected ModelLoadError"),
        }
    }

    #[test]
    fn test_inference_without_loaded_model() {
        let backend = OrtBackend::new();
        let input = vec![0.0f32; 1 * 3 * 640 * 640];

        let result = backend.infer(&input);
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::ModelNotLoaded => {} // Expected
            _ => panic!("Expected ModelNotLoaded error"),
        }
    }

    #[test]
    fn test_input_validation() {
        let backend = OrtBackend::new();

        // Wrong size input
        let wrong_input = vec![0.0f32; 100];
        let result = backend.validate_input(&wrong_input);

        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::InvalidInputShape { .. } => {} // Expected
            _ => panic!("Expected InvalidInputShape error"),
        }

        // Correct size input
        let correct_input = vec![0.0f32; 1 * 3 * 640 * 640];
        let result = backend.validate_input(&correct_input);
        assert!(result.is_ok());
    }
}
