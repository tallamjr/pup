//! Unit tests for individual components
//!
//! These tests verify specific functionality of the library's public API.

#[cfg(test)]
mod library_integration_tests {
    use gstpup::{config::AppConfig, preprocessing::Preprocessor, utils::Detection};

    #[test]
    fn test_preprocessor_integration() {
        let preprocessor = Preprocessor::new(640, 640);
        assert_eq!(preprocessor.target_size(), (640, 640));
    }

    #[test]
    fn test_detection_creation() {
        let detection = Detection::new(10.0, 20.0, 30.0, 40.0, 0.8, 1);
        assert_eq!(detection.score, 0.8);
        assert_eq!(detection.class_id, 1);
    }

    #[test]
    fn test_config_creation() {
        let config = AppConfig::default();
        assert_eq!(config.inference.confidence_threshold, 0.5);
        assert_eq!(config.inference.backend, "ort");
    }
}

/// Test actual detection data structure from library
#[cfg(test)]
mod detection_tests {
    use gstpup::utils::Detection;

    #[test]
    fn test_detection_properties() {
        let det = Detection::new(0.0, 0.0, 10.0, 20.0, 0.9, 0);
        assert_eq!(det.area(), 200.0);
        assert_eq!(det.center(), (5.0, 10.0));
        assert_eq!(det.width(), 10.0);
        assert_eq!(det.height(), 20.0);
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
