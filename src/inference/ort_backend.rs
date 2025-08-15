//! ONNX Runtime backend implementation

use super::{
    InferenceBackend, InferenceError, ModelPostProcessor, TaskOutput, TaskType, YoloPostProcessor,
};
use ort::{
    execution_providers::CoreMLExecutionProvider,
    session::{builder::GraphOptimizationLevel, Session},
    value::Value,
};
use std::path::Path;
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

/// ONNX Runtime inference backend
pub struct OrtBackend {
    session: Option<Arc<Mutex<Session>>>,
    input_shape: Vec<usize>,
    confidence_threshold: f32,
    post_processor: Box<dyn ModelPostProcessor + Send + Sync>,
    use_coreml: bool,
}

impl OrtBackend {
    /// Create a new ORT backend with default settings
    pub fn new() -> Self {
        Self {
            session: None,
            input_shape: vec![1, 3, 640, 640], // Default YOLO input shape
            confidence_threshold: 0.5,
            post_processor: Box::new(YoloPostProcessor::coco_default()),
            use_coreml: true, // Enable CoreML by default on Apple platforms
        }
    }

    /// Create a new ORT backend with custom post-processor
    pub fn with_post_processor(post_processor: Box<dyn ModelPostProcessor + Send + Sync>) -> Self {
        Self {
            session: None,
            input_shape: vec![1, 3, 640, 640],
            confidence_threshold: 0.5,
            post_processor,
            use_coreml: true,
        }
    }

    /// Create a new ORT backend with CoreML disabled
    pub fn with_cpu_only() -> Self {
        Self {
            session: None,
            input_shape: vec![1, 3, 640, 640],
            confidence_threshold: 0.5,
            post_processor: Box::new(YoloPostProcessor::coco_default()),
            use_coreml: false,
        }
    }

    /// Configure whether to use CoreML execution provider
    pub fn with_coreml(&mut self, use_coreml: bool) -> &mut Self {
        self.use_coreml = use_coreml;
        self
    }

    /// Create an optimized session with optional CoreML execution provider and CPU fallback
    fn create_session(model_path: &Path, use_coreml: bool) -> Result<Session, InferenceError> {
        let session_builder = Session::builder()
            .map_err(|e| {
                InferenceError::OrtError(format!("Failed to create session builder: {}", e))
            })?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| {
                InferenceError::OrtError(format!("Failed to set optimization level: {}", e))
            })?
            .with_intra_threads(8)
            .map_err(|e| InferenceError::OrtError(format!("Failed to set intra threads: {}", e)))?;

        // Try CoreML first on Apple platforms if enabled
        #[cfg(target_vendor = "apple")]
        if use_coreml {
            info!("Checking CoreML availability before creating session");

            // Check if we can safely use CoreML by testing a simple configuration
            let coreml_available = Self::check_coreml_availability();

            if coreml_available {
                info!("CoreML availability confirmed, attempting to create session with CoreML execution provider");

                let coreml_session = session_builder
                    .clone()
                    .with_execution_providers([CoreMLExecutionProvider::default().build()]);

                match coreml_session {
                    Ok(builder) => match builder.commit_from_file(model_path) {
                        Ok(session) => {
                            info!("Successfully created session with CoreML execution provider");
                            return Ok(session);
                        }
                        Err(e) => {
                            warn!("Failed to load model with CoreML execution provider: {}. Falling back to CPU", e);
                        }
                    },
                    Err(e) => {
                        warn!("Failed to configure CoreML execution provider: {}. Falling back to CPU", e);
                    }
                }
            } else {
                warn!("CoreML execution provider unavailable on this system. Using CPU execution provider");
            }
        }

        // Fallback to CPU execution provider
        info!("Creating session with CPU execution provider");
        let session = session_builder.commit_from_file(model_path).map_err(|e| {
            InferenceError::ModelLoadError(format!("Failed to load model with CPU provider: {}", e))
        })?;

        Ok(session)
    }

    /// Check if CoreML execution provider is available on this system
    #[cfg(target_vendor = "apple")]
    fn check_coreml_availability() -> bool {
        // Check if we're running on macOS 10.15+ (required for CoreML)
        use std::process::Command;

        if let Ok(output) = Command::new("sw_vers").arg("-productVersion").output() {
            if let Ok(version_str) = String::from_utf8(output.stdout) {
                let version_parts: Vec<&str> = version_str.trim().split('.').collect();
                if let Ok(major) = version_parts.first().unwrap_or(&"0").parse::<i32>() {
                    if let Ok(minor) = version_parts.get(1).unwrap_or(&"0").parse::<i32>() {
                        // macOS 10.15+ (Catalina and later) support CoreML
                        return major > 10 || (major == 10 && minor >= 15);
                    }
                }
            }
        }

        // If we can't determine the version, assume CoreML is available
        // on Apple platforms (safer default for modern systems)
        true
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
        let array = ndarray::Array::from_shape_vec(self.input_shape.clone(), input.to_vec())
            .map_err(|e| InferenceError::OrtError(format!("Failed to create ndarray: {}", e)))?;

        Value::from_array(array)
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

        let session = Self::create_session(path, self.use_coreml)?;
        self.session = Some(Arc::new(Mutex::new(session)));

        Ok(())
    }

    fn infer(&self, input: &[f32]) -> Result<TaskOutput, InferenceError> {
        use std::time::Instant;

        let inference_start = Instant::now();

        let session = self
            .session
            .as_ref()
            .ok_or(InferenceError::ModelNotLoaded)?;

        // Validate input
        self.validate_input(input)?;

        // Create input tensor
        let tensor_start = Instant::now();
        let input_tensor = self.create_input_tensor(input)?;
        let tensor_time = tensor_start.elapsed();

        // Run inference
        let session_start = Instant::now();
        let mut session_guard = session.lock().map_err(|e| {
            InferenceError::InferenceFailed(format!("Failed to acquire session lock: {}", e))
        })?;
        let outputs = session_guard
            .run(ort::inputs!["images" => input_tensor])
            .map_err(|e| InferenceError::InferenceFailed(format!("Session run failed: {}", e)))?;
        let session_time = session_start.elapsed();

        // Extract output data
        let mut output_values = outputs.values();
        let first_output = output_values.next().ok_or_else(|| {
            InferenceError::InvalidOutputFormat("No outputs received from model".to_string())
        })?;

        let output_array = first_output.try_extract_array::<f32>().map_err(|e| {
            InferenceError::OrtError(format!("Failed to extract output tensor: {}", e))
        })?;

        // Convert ndarray to Vec<f32> for processing
        let output_data: Vec<f32> = output_array.iter().copied().collect();

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

        let total_time = inference_start.elapsed();

        // Log detailed timing information
        tracing::debug!(
            "Inference timing - Total: {:.2}ms, Tensor: {:.2}ms, Session: {:.2}ms, Post-processing: {:.2}ms",
            total_time.as_secs_f64() * 1000.0,
            tensor_time.as_secs_f64() * 1000.0,
            session_time.as_secs_f64() * 1000.0,
            (total_time - tensor_time - session_time).as_secs_f64() * 1000.0
        );

        // Log inference time for benchmarking
        tracing::info!(
            "inference_time_ms={:.2} session_time_ms={:.2} execution_provider={}",
            total_time.as_secs_f64() * 1000.0,
            session_time.as_secs_f64() * 1000.0,
            if self.use_coreml { "CoreML" } else { "CPU" }
        );

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
        if !(0.0..=1.0).contains(&threshold) {
            warn!(
                "Invalid confidence threshold {}, keeping current value {}",
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
    use_coreml: bool,
}

impl OrtBackendBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            input_shape: vec![1, 3, 640, 640],
            confidence_threshold: 0.5,
            post_processor: None,
            use_coreml: true, // Enable CoreML by default on Apple platforms
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

    /// Enable or disable CoreML execution provider
    pub fn with_coreml(mut self, enable: bool) -> Self {
        self.use_coreml = enable;
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
            use_coreml: self.use_coreml,
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

    #[test]
    fn test_cpu_only_backend() {
        let backend = OrtBackend::with_cpu_only();
        assert_eq!(backend.use_coreml, false);
        assert_eq!(backend.get_input_shape(), &[1, 3, 640, 640]);
        assert_eq!(backend.get_confidence_threshold(), 0.5);
    }

    #[test]
    fn test_builder_with_coreml_disabled() {
        let backend = OrtBackendBuilder::new().with_coreml(false).build();

        assert_eq!(backend.use_coreml, false);
    }
}
