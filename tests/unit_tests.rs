//! Unit tests for individual components
//!
//! These tests verify specific functionality and will expand as we
//! implement the modular architecture from the roadmap.

use std::path::PathBuf;

/// Test letterbox function (from current main.rs)
#[cfg(test)]
mod letterbox_tests {
    use opencv::{core, imgproc, prelude::*};

    fn create_test_mat(width: i32, height: i32) -> opencv::Result<core::Mat> {
        core::Mat::new_rows_cols_with_default(
            height,
            width,
            core::CV_8UC3,
            core::Scalar::all(128.0),
        )
    }

    fn letterbox_to_640(src: &core::Mat) -> opencv::Result<core::Mat> {
        let src_size = src.size()?;
        let (orig_w, orig_h) = (src_size.width, src_size.height);

        let scale = if orig_w > orig_h {
            640.0 / orig_w as f64
        } else {
            640.0 / orig_h as f64
        };
        let new_w = (orig_w as f64 * scale).round() as i32;
        let new_h = (orig_h as f64 * scale).round() as i32;

        let mat_type = src.typ();
        let mut dst =
            core::Mat::new_rows_cols_with_default(640, 640, mat_type, core::Scalar::all(0.0))?;

        let mut resized = core::Mat::default();
        imgproc::resize(
            src,
            &mut resized,
            core::Size::new(new_w, new_h),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let x_offset = (640 - new_w) / 2;
        let y_offset = (640 - new_h) / 2;
        let roi_rect = core::Rect::new(x_offset, y_offset, new_w, new_h);
        let mut roi = dst.roi_mut(roi_rect)?;
        resized.copy_to(&mut roi)?;

        Ok(dst)
    }

    #[test]
    fn test_letterbox_square_image() {
        let src = create_test_mat(640, 640).unwrap();
        let result = letterbox_to_640(&src).unwrap();

        assert_eq!(result.size().unwrap().width, 640);
        assert_eq!(result.size().unwrap().height, 640);
    }

    #[test]
    fn test_letterbox_wide_image() {
        let src = create_test_mat(1280, 720).unwrap();
        let result = letterbox_to_640(&src).unwrap();

        assert_eq!(result.size().unwrap().width, 640);
        assert_eq!(result.size().unwrap().height, 640);

        // The scaled image should be 640x360, centered vertically
        // Check that borders are black (0,0,0)
        let top_pixel = result.at_2d::<core::Vec3b>(0, 320).unwrap();
        assert_eq!(top_pixel[0], 0); // Blue channel
        assert_eq!(top_pixel[1], 0); // Green channel
        assert_eq!(top_pixel[2], 0); // Red channel
    }

    #[test]
    fn test_letterbox_tall_image() {
        let src = create_test_mat(480, 800).unwrap();
        let result = letterbox_to_640(&src).unwrap();

        assert_eq!(result.size().unwrap().width, 640);
        assert_eq!(result.size().unwrap().height, 640);

        // The scaled image should be 384x640, centered horizontally
        // Check that borders are black
        let left_pixel = result.at_2d::<core::Vec3b>(320, 0).unwrap();
        assert_eq!(left_pixel[0], 0);
        assert_eq!(left_pixel[1], 0);
        assert_eq!(left_pixel[2], 0);
    }

    #[test]
    fn test_letterbox_small_image() {
        let src = create_test_mat(320, 240).unwrap();
        let result = letterbox_to_640(&src).unwrap();

        assert_eq!(result.size().unwrap().width, 640);
        assert_eq!(result.size().unwrap().height, 640);

        // Small image should be upscaled while maintaining aspect ratio
    }
}

/// Test detection data structure
#[cfg(test)]
mod detection_tests {
    #[derive(Clone, Debug, PartialEq)]
    struct Detection {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        score: f32,
        class_id: i32,
    }

    impl Detection {
        fn new(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class_id: i32) -> Self {
            Self {
                x1,
                y1,
                x2,
                y2,
                score,
                class_id,
            }
        }

        fn area(&self) -> f32 {
            (self.x2 - self.x1) * (self.y2 - self.y1)
        }

        fn center(&self) -> (f32, f32) {
            ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
        }

        fn width(&self) -> f32 {
            self.x2 - self.x1
        }

        fn height(&self) -> f32 {
            self.y2 - self.y1
        }
    }

    #[test]
    fn test_detection_creation() {
        let det = Detection::new(10.0, 20.0, 30.0, 40.0, 0.8, 1);

        assert_eq!(det.x1, 10.0);
        assert_eq!(det.y1, 20.0);
        assert_eq!(det.x2, 30.0);
        assert_eq!(det.y2, 40.0);
        assert_eq!(det.score, 0.8);
        assert_eq!(det.class_id, 1);
    }

    #[test]
    fn test_detection_area() {
        let det = Detection::new(0.0, 0.0, 10.0, 20.0, 0.9, 0);
        assert_eq!(det.area(), 200.0);
    }

    #[test]
    fn test_detection_center() {
        let det = Detection::new(0.0, 0.0, 10.0, 20.0, 0.9, 0);
        assert_eq!(det.center(), (5.0, 10.0));
    }

    #[test]
    fn test_detection_dimensions() {
        let det = Detection::new(5.0, 10.0, 15.0, 30.0, 0.7, 2);
        assert_eq!(det.width(), 10.0);
        assert_eq!(det.height(), 20.0);
    }
}

/// Test future modular architecture components
#[cfg(test)]
mod modular_architecture_tests {
    use std::error::Error;

    // Mock traits for future implementation testing
    trait InferenceBackend {
        type Error: Error;

        fn load_model(&mut self, path: &std::path::Path) -> Result<(), Self::Error>;
        fn get_input_shape(&self) -> &[usize];
    }

    trait ModelPostProcessor {
        fn apply_confidence_threshold(&self, detections: Vec<f32>, threshold: f32) -> Vec<f32>;
    }

    // Mock implementations for testing
    struct MockInferenceBackend {
        input_shape: Vec<usize>,
        model_loaded: bool,
    }

    #[derive(Debug)]
    struct MockError(String);

    impl std::fmt::Display for MockError {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "Mock error: {}", self.0)
        }
    }

    impl Error for MockError {}

    impl MockInferenceBackend {
        fn new() -> Self {
            Self {
                input_shape: vec![1, 3, 640, 640],
                model_loaded: false,
            }
        }
    }

    impl InferenceBackend for MockInferenceBackend {
        type Error = MockError;

        fn load_model(&mut self, path: &std::path::Path) -> Result<(), Self::Error> {
            if path.exists() {
                self.model_loaded = true;
                Ok(())
            } else {
                Err(MockError("Model file not found".to_string()))
            }
        }

        fn get_input_shape(&self) -> &[usize] {
            &self.input_shape
        }
    }

    struct MockPostProcessor;

    impl ModelPostProcessor for MockPostProcessor {
        fn apply_confidence_threshold(&self, detections: Vec<f32>, threshold: f32) -> Vec<f32> {
            detections.into_iter().filter(|&x| x >= threshold).collect()
        }
    }

    #[test]
    fn test_inference_backend_trait() {
        let mut backend = MockInferenceBackend::new();

        // Test input shape
        assert_eq!(backend.get_input_shape(), &[1, 3, 640, 640]);

        // Test model loading with non-existent file
        let result = backend.load_model(&std::path::PathBuf::from("nonexistent.onnx"));
        assert!(result.is_err());
        assert!(!backend.model_loaded);

        // Test model loading with existing file (use Cargo.toml as dummy)
        let result = backend.load_model(&std::path::PathBuf::from("Cargo.toml"));
        assert!(result.is_ok());
        assert!(backend.model_loaded);
    }

    #[test]
    fn test_post_processor_trait() {
        let processor = MockPostProcessor;

        let detections = vec![0.1, 0.3, 0.6, 0.8, 0.2, 0.9];
        let filtered = processor.apply_confidence_threshold(detections, 0.5);

        assert_eq!(filtered, vec![0.6, 0.8, 0.9]);
    }

    #[test]
    fn test_configuration_structure() {
        #[derive(Debug, Clone, PartialEq)]
        struct PipelineConfig {
            video_source: String,
            display_enabled: bool,
            framerate: u32,
        }

        #[derive(Debug, Clone, PartialEq)]
        struct InferenceConfig {
            backend: String,
            model_path: String,
            confidence_threshold: f32,
            device: String,
        }

        let pipeline_config = PipelineConfig {
            video_source: "auto".to_string(),
            display_enabled: true,
            framerate: 30,
        };

        let inference_config = InferenceConfig {
            backend: "ort".to_string(),
            model_path: "models/yolov8n.onnx".to_string(),
            confidence_threshold: 0.5,
            device: "auto".to_string(),
        };

        // Test default values
        assert_eq!(pipeline_config.video_source, "auto");
        assert!(pipeline_config.display_enabled);
        assert_eq!(pipeline_config.framerate, 30);

        assert_eq!(inference_config.backend, "ort");
        assert_eq!(inference_config.confidence_threshold, 0.5);
        assert_eq!(inference_config.device, "auto");
    }
}

/// Test ONNX model validation (if available)
#[cfg(test)]
mod onnx_tests {
    use std::path::PathBuf;

    #[test]
    fn test_onnx_model_file_structure() {
        let model_path = PathBuf::from("models/yolov8n.onnx");

        if !model_path.exists() {
            println!("ONNX model not available, skipping model validation tests");
            return;
        }

        // Basic file validation
        let metadata = std::fs::metadata(&model_path).unwrap();
        assert!(metadata.is_file());
        assert!(metadata.len() > 1024); // Should be at least 1KB

        // Check file signature (ONNX files start with specific bytes)
        let file_content = std::fs::read(&model_path).unwrap();
        if file_content.len() >= 4 {
            // ONNX files are Protocol Buffer format, but we can check basic structure
            assert!(file_content.len() > 1000000); // YOLOv8n should be > 1MB
        }
    }

    #[test]
    fn test_model_loading_performance() {
        let model_path = PathBuf::from("models/yolov8n.onnx");

        if !model_path.exists() {
            println!("ONNX model not available, skipping performance test");
            return;
        }

        use std::time::Instant;

        // This test will be expanded when we modularize the inference backend
        let start = Instant::now();

        // For now, just test file read time
        let _content = std::fs::read(&model_path).unwrap();
        let duration = start.elapsed();

        // File should be readable in under 1 second
        assert!(
            duration.as_secs() < 1,
            "Model file read took too long: {:?}",
            duration
        );
    }
}

/// Test image preprocessing pipeline
#[cfg(test)]
mod preprocessing_tests {
    #[test]
    fn test_rgb_normalization() {
        let input = vec![0u8, 128u8, 255u8];
        let normalized: Vec<f32> = input.iter().map(|&x| x as f32 / 255.0).collect();

        assert_eq!(normalized[0], 0.0);
        assert!((normalized[1] - 0.5019607843).abs() < 1e-6); // 128/255
        assert_eq!(normalized[2], 1.0);
    }

    #[test]
    fn test_tensor_shape_calculation() {
        let width = 640;
        let height = 640;
        let channels = 3;
        let batch_size = 1;

        let expected_shape = [batch_size, channels, height, width];
        let total_elements = expected_shape.iter().product::<usize>();

        assert_eq!(total_elements, 1 * 3 * 640 * 640);
        assert_eq!(total_elements, 1228800);
    }

    #[test]
    fn test_color_channel_ordering() {
        // Test that we handle RGB vs BGR correctly
        let rgb_pixel = [255u8, 0u8, 0u8]; // Red pixel
        let bgr_pixel = [0u8, 0u8, 255u8]; // Red pixel in BGR

        // RGB to float
        let rgb_float: Vec<f32> = rgb_pixel.iter().map(|&x| x as f32 / 255.0).collect();
        assert_eq!(rgb_float, vec![1.0, 0.0, 0.0]);

        // BGR to RGB conversion
        let bgr_to_rgb = [bgr_pixel[2], bgr_pixel[1], bgr_pixel[0]];
        assert_eq!(bgr_to_rgb, [255u8, 0u8, 0u8]);
    }
}

/// Test future GStreamer plugin architecture
#[cfg(test)]
mod gstreamer_plugin_tests {
    use std::collections::HashMap;

    // Mock GStreamer plugin structures for testing
    #[derive(Debug, Clone)]
    struct MockGstElement {
        name: String,
        properties: HashMap<String, String>,
    }

    impl MockGstElement {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                properties: HashMap::new(),
            }
        }

        fn set_property(&mut self, key: &str, value: &str) {
            self.properties.insert(key.to_string(), value.to_string());
        }

        fn get_property(&self, key: &str) -> Option<&String> {
            self.properties.get(key)
        }
    }

    #[test]
    fn test_plugin_property_management() {
        let mut element = MockGstElement::new("pupinference");

        element.set_property("model-path", "models/yolov8n.onnx");
        element.set_property("confidence-threshold", "0.6");
        element.set_property("device", "coreml");

        assert_eq!(
            element.get_property("model-path"),
            Some(&"models/yolov8n.onnx".to_string())
        );
        assert_eq!(
            element.get_property("confidence-threshold"),
            Some(&"0.6".to_string())
        );
        assert_eq!(element.get_property("device"), Some(&"coreml".to_string()));
        assert_eq!(element.get_property("nonexistent"), None);
    }

    #[test]
    fn test_plugin_pipeline_construction() {
        let elements = vec![
            "filesrc",
            "decodebin",
            "videoconvert",
            "pupinference",
            "pupoverlay",
            "autovideosink",
        ];

        // Test that pipeline elements are in correct order
        assert_eq!(elements[0], "filesrc");
        assert_eq!(elements[3], "pupinference");
        assert_eq!(elements[4], "pupoverlay");
        assert_eq!(elements[5], "autovideosink");
    }

    #[test]
    fn test_caps_negotiation_mock() {
        // Mock caps for testing
        #[derive(Debug, PartialEq)]
        struct VideoCaps {
            format: String,
            width: u32,
            height: u32,
            framerate: String,
        }

        let input_caps = VideoCaps {
            format: "RGB".to_string(),
            width: 1920,
            height: 1080,
            framerate: "30/1".to_string(),
        };

        let output_caps = VideoCaps {
            format: "RGB".to_string(),
            width: 1920,
            height: 1080,
            framerate: "30/1".to_string(),
        };

        // In our plugin, input and output caps should match (passthrough)
        assert_eq!(input_caps, output_caps);
        assert_eq!(input_caps.format, "RGB");
    }
}
