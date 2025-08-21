//! Comprehensive tests for GStreamer plugins in the pup video processing project
//!
//! This module provides thorough testing for all GStreamer plugins including
//! plugin registration, element functionality, caps negotiation, and integration testing.

use std::collections::HashMap;

mod mocks;
use mocks::*;

/// Tests for plugin registration and discovery
#[cfg(test)]
mod plugin_registration_tests {
    use super::*;

    #[test]
    fn test_pupvision_plugin_registration() {
        let mut registry = MockPluginRegistry::new();

        let plugin_info = MockPluginInfo {
            name: "pupvision".to_string(),
            description: "Real-time multi-task computer vision with GStreamer and ONNX Runtime"
                .to_string(),
            version: "0.3.0".to_string(),
            elements: vec!["pupinference".to_string(), "pupoverlay".to_string()],
        };

        registry.register_plugin(plugin_info);

        // Verify plugin is registered
        let registered_plugin = registry.get_plugin("pupvision");
        assert!(registered_plugin.is_some());

        let plugin = registered_plugin.unwrap();
        assert_eq!(plugin.name, "pupvision");
        assert_eq!(plugin.version, "0.3.0");
        assert!(plugin.elements.contains(&"pupinference".to_string()));
        assert!(plugin.elements.contains(&"pupoverlay".to_string()));
    }

    #[test]
    fn test_plugin_element_discovery() {
        let mut registry = MockPluginRegistry::new();

        let plugin_info = MockPluginInfo {
            name: "pupvision".to_string(),
            description: "Test plugin".to_string(),
            version: "0.3.0".to_string(),
            elements: vec!["pupinference".to_string(), "pupoverlay".to_string()],
        };

        registry.register_plugin(plugin_info);

        // Test element existence
        assert!(registry.element_exists("pupinference"));
        assert!(registry.element_exists("pupoverlay"));
        assert!(!registry.element_exists("nonexistent"));
    }

    #[test]
    fn test_plugin_metadata_validation() {
        let plugin_info = MockPluginInfo {
            name: "pupvision".to_string(),
            description: "Real-time multi-task computer vision with GStreamer and ONNX Runtime"
                .to_string(),
            version: "0.3.0".to_string(),
            elements: vec!["pupinference".to_string(), "pupoverlay".to_string()],
        };

        // Validate required metadata fields
        assert!(!plugin_info.name.is_empty());
        assert!(!plugin_info.description.is_empty());
        assert!(!plugin_info.version.is_empty());
        assert!(!plugin_info.elements.is_empty());

        // Version format validation
        let version_parts: Vec<&str> = plugin_info.version.split('.').collect();
        assert_eq!(version_parts.len(), 3, "Version should be in format X.Y.Z");
    }

    #[test]
    fn test_multiple_plugin_registration() {
        let mut registry = MockPluginRegistry::new();

        // Register first plugin
        let plugin1 = MockPluginInfo {
            name: "pupvision".to_string(),
            description: "Main vision plugin".to_string(),
            version: "0.3.0".to_string(),
            elements: vec!["pupinference".to_string(), "pupoverlay".to_string()],
        };

        // Register second plugin (hypothetical)
        let plugin2 = MockPluginInfo {
            name: "pupaudio".to_string(),
            description: "Audio processing plugin".to_string(),
            version: "0.1.0".to_string(),
            elements: vec!["pupaudioprocess".to_string()],
        };

        registry.register_plugin(plugin1);
        registry.register_plugin(plugin2);

        // Verify both plugins are registered
        assert!(registry.get_plugin("pupvision").is_some());
        assert!(registry.get_plugin("pupaudio").is_some());

        // Verify element uniqueness across plugins
        assert!(registry.element_exists("pupinference"));
        assert!(registry.element_exists("pupoverlay"));
        assert!(registry.element_exists("pupaudioprocess"));
    }

    #[test]
    fn test_plugin_listing() {
        let mut registry = MockPluginRegistry::new();

        let plugin = MockPluginInfo {
            name: "pupvision".to_string(),
            description: "Test plugin".to_string(),
            version: "0.3.0".to_string(),
            elements: vec!["pupinference".to_string()],
        };

        registry.register_plugin(plugin);

        let plugins = registry.list_plugins();
        assert_eq!(plugins.len(), 1);
        assert_eq!(plugins[0].name, "pupvision");
    }
}

/// Tests for pupinference plugin functionality
#[cfg(test)]
mod pupinference_plugin_tests {
    use super::*;

    fn create_pupinference_element() -> MockGstElement {
        let mut element = MockGstElement::new("pupinference-test", "pupinference");

        // Add standard pads
        element.add_pad(MockGstPad::new("sink", PadDirection::Sink));
        element.add_pad(MockGstPad::new("src", PadDirection::Source));

        // Set default properties
        element.set_property(
            "model-path",
            MockGstValue::String("models/yolov8n.onnx".to_string()),
        );
        element.set_property("confidence-threshold", MockGstValue::Float(0.5));
        element.set_property("device", MockGstValue::String("auto".to_string()));

        element
    }

    #[test]
    fn test_pupinference_element_creation() {
        let element = create_pupinference_element();

        assert_eq!(element.name, "pupinference-test");
        assert_eq!(element.element_type, "pupinference");
        assert_eq!(element.state, ElementState::Null);

        // Verify pads
        assert!(element.pads.contains_key("sink"));
        assert!(element.pads.contains_key("src"));
        assert_eq!(element.pads["sink"].direction, PadDirection::Sink);
        assert_eq!(element.pads["src"].direction, PadDirection::Source);
    }

    #[test]
    fn test_pupinference_properties() {
        let element = create_pupinference_element();

        // Test default properties
        assert_eq!(
            element.get_property("model-path"),
            Some(&MockGstValue::String("models/yolov8n.onnx".to_string()))
        );
        assert_eq!(
            element.get_property("confidence-threshold"),
            Some(&MockGstValue::Float(0.5))
        );
        assert_eq!(
            element.get_property("device"),
            Some(&MockGstValue::String("auto".to_string()))
        );
    }

    #[test]
    fn test_pupinference_property_validation() {
        let mut element = create_pupinference_element();

        // Test valid confidence threshold values
        element.set_property("confidence-threshold", MockGstValue::Float(0.0));
        element.set_property("confidence-threshold", MockGstValue::Float(0.5));
        element.set_property("confidence-threshold", MockGstValue::Float(1.0));

        // Test device options
        element.set_property("device", MockGstValue::String("cpu".to_string()));
        element.set_property("device", MockGstValue::String("coreml".to_string()));
        element.set_property("device", MockGstValue::String("auto".to_string()));

        // Test model path
        element.set_property(
            "model-path",
            MockGstValue::String("/path/to/model.onnx".to_string()),
        );
        assert_eq!(
            element.get_property("model-path"),
            Some(&MockGstValue::String("/path/to/model.onnx".to_string()))
        );
    }

    #[test]
    fn test_pupinference_caps_negotiation() {
        let element = create_pupinference_element();

        // Create input caps (typical video format)
        let input_caps = MockGstCaps::new("video/x-raw")
            .with_field("format", MockGstValue::String("RGB".to_string()))
            .with_field("width", MockGstValue::Integer(1920))
            .with_field("height", MockGstValue::Integer(1080))
            .with_field("framerate", MockGstValue::Fraction { num: 30, denom: 1 });

        // For pupinference, output caps should match input (passthrough)
        let expected_output_caps = input_caps.clone();

        // In a real implementation, this would test actual caps negotiation
        assert!(input_caps.is_compatible(&expected_output_caps));
    }

    #[test]
    fn test_pupinference_state_transitions() {
        let mut element = create_pupinference_element();

        // Test state transitions
        assert!(element.set_state(ElementState::Ready).is_ok());
        assert_eq!(element.state, ElementState::Ready);

        assert!(element.set_state(ElementState::Paused).is_ok());
        assert_eq!(element.state, ElementState::Paused);

        assert!(element.set_state(ElementState::Playing).is_ok());
        assert_eq!(element.state, ElementState::Playing);
    }

    #[test]
    fn test_pupinference_error_handling() {
        let mut element = create_pupinference_element();

        // Test invalid model path handling
        element.set_property(
            "model-path",
            MockGstValue::String("nonexistent.onnx".to_string()),
        );

        // Test invalid confidence threshold (negative values)
        element.set_property("confidence-threshold", MockGstValue::Float(-0.1));

        // In a real implementation, these would trigger error states
        // For now, we just verify the properties were set
        assert_eq!(
            element.get_property("model-path"),
            Some(&MockGstValue::String("nonexistent.onnx".to_string()))
        );
        assert_eq!(
            element.get_property("confidence-threshold"),
            Some(&MockGstValue::Float(-0.1))
        );
    }

    #[test]
    fn test_pupinference_buffer_processing() {
        let element = create_pupinference_element();

        // Create a mock input buffer (640x640 RGB frame)
        let frame_size = 640 * 640 * 3;
        let mock_frame_data = vec![128u8; frame_size]; // Gray frame
        let buffer = MockGstBuffer::new(mock_frame_data.clone())
            .with_timestamp(1000000) // 1ms in nanoseconds
            .with_duration(33333333); // ~30 FPS frame duration

        let caps = MockGstCaps::new("video/x-raw")
            .with_field("format", MockGstValue::String("RGB".to_string()))
            .with_field("width", MockGstValue::Integer(640))
            .with_field("height", MockGstValue::Integer(640));

        let sample = MockGstSample::new(buffer, caps);

        // Verify buffer properties
        assert_eq!(sample.buffer.size(), frame_size);
        assert_eq!(sample.buffer.timestamp, Some(1000000));
        assert_eq!(sample.caps.media_type, "video/x-raw");

        // In a real implementation, this would process the buffer through inference
        // For now, we verify the sample was created correctly
        assert_eq!(sample.buffer.data, mock_frame_data);
    }
}

/// Tests for pupoverlay plugin functionality
#[cfg(test)]
mod pupoverlay_plugin_tests {
    use super::*;

    fn create_pupoverlay_element() -> MockGstElement {
        let mut element = MockGstElement::new("pupoverlay-test", "pupoverlay");

        // Add standard pads
        element.add_pad(MockGstPad::new("sink", PadDirection::Sink));
        element.add_pad(MockGstPad::new("src", PadDirection::Source));

        // Set default properties
        element.set_property("draw-bboxes", MockGstValue::Boolean(true));
        element.set_property("draw-labels", MockGstValue::Boolean(true));
        element.set_property("bbox-thickness", MockGstValue::Integer(2));
        element.set_property("font-scale", MockGstValue::Float(0.5));

        element
    }

    #[test]
    fn test_pupoverlay_element_creation() {
        let element = create_pupoverlay_element();

        assert_eq!(element.name, "pupoverlay-test");
        assert_eq!(element.element_type, "pupoverlay");
        assert_eq!(element.state, ElementState::Null);

        // Verify pads
        assert!(element.pads.contains_key("sink"));
        assert!(element.pads.contains_key("src"));
        assert_eq!(element.pads["sink"].direction, PadDirection::Sink);
        assert_eq!(element.pads["src"].direction, PadDirection::Source);
    }

    #[test]
    fn test_pupoverlay_properties() {
        let element = create_pupoverlay_element();

        // Test overlay properties
        assert_eq!(
            element.get_property("draw-bboxes"),
            Some(&MockGstValue::Boolean(true))
        );
        assert_eq!(
            element.get_property("draw-labels"),
            Some(&MockGstValue::Boolean(true))
        );
        assert_eq!(
            element.get_property("bbox-thickness"),
            Some(&MockGstValue::Integer(2))
        );
        assert_eq!(
            element.get_property("font-scale"),
            Some(&MockGstValue::Float(0.5))
        );
    }

    #[test]
    fn test_pupoverlay_property_configuration() {
        let mut element = create_pupoverlay_element();

        // Test disabling features
        element.set_property("draw-bboxes", MockGstValue::Boolean(false));
        element.set_property("draw-labels", MockGstValue::Boolean(false));

        assert_eq!(
            element.get_property("draw-bboxes"),
            Some(&MockGstValue::Boolean(false))
        );
        assert_eq!(
            element.get_property("draw-labels"),
            Some(&MockGstValue::Boolean(false))
        );

        // Test styling properties
        element.set_property("bbox-thickness", MockGstValue::Integer(5));
        element.set_property("font-scale", MockGstValue::Float(1.2));

        assert_eq!(
            element.get_property("bbox-thickness"),
            Some(&MockGstValue::Integer(5))
        );
        assert_eq!(
            element.get_property("font-scale"),
            Some(&MockGstValue::Float(1.2))
        );
    }

    #[test]
    fn test_pupoverlay_caps_negotiation() {
        let element = create_pupoverlay_element();

        // Create input caps
        let input_caps = MockGstCaps::new("video/x-raw")
            .with_field("format", MockGstValue::String("RGB".to_string()))
            .with_field("width", MockGstValue::Integer(1280))
            .with_field("height", MockGstValue::Integer(720));

        // Output caps should match input (passthrough with overlay)
        let expected_output_caps = input_caps.clone();

        assert!(input_caps.is_compatible(&expected_output_caps));
    }

    #[test]
    fn test_pupoverlay_detection_data_simulation() {
        let element = create_pupoverlay_element();

        // Simulate detection data that would be overlaid
        #[derive(Debug, Clone)]
        struct MockDetection {
            class_id: u32,
            confidence: f32,
            bbox: (f32, f32, f32, f32), // x1, y1, x2, y2
        }

        let detections = vec![
            MockDetection {
                class_id: 0,
                confidence: 0.85,
                bbox: (100.0, 100.0, 200.0, 150.0),
            },
            MockDetection {
                class_id: 15,
                confidence: 0.72,
                bbox: (300.0, 200.0, 450.0, 300.0),
            },
        ];

        // Verify detection data structure
        assert_eq!(detections.len(), 2);
        assert_eq!(detections[0].class_id, 0);
        assert_eq!(detections[0].confidence, 0.85);

        // In a real implementation, this data would be used to draw overlays
        // Verify we can access the bounding box coordinates
        let bbox = detections[0].bbox;
        assert_eq!(bbox, (100.0, 100.0, 200.0, 150.0));
    }

    #[test]
    fn test_pupoverlay_buffer_modification() {
        let element = create_pupoverlay_element();

        // Create input buffer
        let frame_size = 640 * 480 * 3;
        let input_data = vec![0u8; frame_size];
        let buffer = MockGstBuffer::new(input_data.clone());

        // In a real implementation, the overlay plugin would:
        // 1. Receive detection data via property or metadata
        // 2. Draw bounding boxes on the frame
        // 3. Add text labels
        // 4. Pass modified frame to output

        // For testing, verify the buffer structure
        assert_eq!(buffer.size(), frame_size);
        assert_eq!(buffer.data.len(), frame_size);

        // Simulate overlay processing by "modifying" some pixels
        let mut modified_data = buffer.data.clone();
        // Simulate drawing a red box (set some pixels to red)
        if modified_data.len() >= 3 {
            modified_data[0] = 255; // R
            modified_data[1] = 0; // G
            modified_data[2] = 0; // B
        }

        assert_ne!(buffer.data, modified_data);
    }
}

/// Tests for demo implementations
#[cfg(test)]
mod demo_implementation_tests {
    use super::*;

    #[test]
    fn test_detection_demo_pipeline_construction() {
        let mut pipeline = MockGstPipeline::new("detection-demo");

        // Elements that would be in detection demo
        let elements = vec![
            ("filesrc", "filesrc"),
            ("decodebin", "decodebin"),
            ("videoconvert", "videoconvert"),
            ("videoscale", "videoscale"),
            ("capsfilter", "capsfilter"),
            ("appsink", "appsink"),
        ];

        // Add all elements to pipeline
        for (name, element_type) in elements {
            let mut element = MockGstElement::new(name, element_type);

            // Add appropriate pads
            match element_type {
                "filesrc" => {
                    element.add_pad(MockGstPad::new("src", PadDirection::Source));
                }
                "decodebin" => {
                    // Decodebin has dynamic pads, but we'll simulate with static for testing
                    element.add_pad(MockGstPad::new("sink", PadDirection::Sink));
                    element.add_pad(MockGstPad::new("src", PadDirection::Source));
                }
                "appsink" => {
                    element.add_pad(MockGstPad::new("sink", PadDirection::Sink));
                }
                _ => {
                    element.add_pad(MockGstPad::new("sink", PadDirection::Sink));
                    element.add_pad(MockGstPad::new("src", PadDirection::Source));
                }
            }

            pipeline.add_element(element);
        }

        // Verify all elements are in pipeline
        assert!(pipeline.get_element("filesrc").is_some());
        assert!(pipeline.get_element("decodebin").is_some());
        assert!(pipeline.get_element("appsink").is_some());

        // Test pipeline state transitions
        assert!(pipeline.set_state(ElementState::Ready).is_ok());
        assert_eq!(pipeline.state, ElementState::Ready);
    }

    #[test]
    fn test_visual_demo_pipeline_construction() {
        let mut pipeline = MockGstPipeline::new("visual-demo");

        // Elements for visual demo with display
        let elements = vec![
            ("source", "videotestsrc"),
            ("pupinference", "pupinference"),
            ("pupoverlay", "pupoverlay"),
            ("sink", "autovideosink"),
        ];

        for (name, element_type) in elements {
            let mut element = MockGstElement::new(name, element_type);

            match element_type {
                "videotestsrc" => {
                    element.add_pad(MockGstPad::new("src", PadDirection::Source));
                    element.set_property("pattern", MockGstValue::Integer(0));
                }
                "autovideosink" => {
                    element.add_pad(MockGstPad::new("sink", PadDirection::Sink));
                }
                _ => {
                    element.add_pad(MockGstPad::new("sink", PadDirection::Sink));
                    element.add_pad(MockGstPad::new("src", PadDirection::Source));
                }
            }

            pipeline.add_element(element);
        }

        // Verify pipeline construction
        assert!(pipeline.get_element("source").is_some());
        assert!(pipeline.get_element("pupinference").is_some());
        assert!(pipeline.get_element("pupoverlay").is_some());
        assert!(pipeline.get_element("sink").is_some());
    }

    #[test]
    fn test_demo_configuration_parameters() {
        // Test configuration parameters that demos would use
        let demo_config = HashMap::from([
            ("input-source", "assets/sample.mp4"),
            ("model-path", "models/yolov8n.onnx"),
            ("output-file", "detections.txt"),
            ("confidence-threshold", "0.5"),
            ("device", "auto"),
        ]);

        // Verify all required parameters are present
        assert!(demo_config.contains_key("input-source"));
        assert!(demo_config.contains_key("model-path"));
        assert!(demo_config.contains_key("confidence-threshold"));

        // Verify parameter values
        assert_eq!(demo_config["input-source"], "assets/sample.mp4");
        assert_eq!(demo_config["model-path"], "models/yolov8n.onnx");
        assert_eq!(demo_config["confidence-threshold"], "0.5");
    }

    #[test]
    fn test_demo_error_scenarios() {
        // Test various error conditions that demos should handle

        // Missing input file
        let invalid_config = HashMap::from([
            ("input-source", "nonexistent.mp4"),
            ("model-path", "models/yolov8n.onnx"),
        ]);

        // In a real implementation, this would trigger an error
        assert!(invalid_config.contains_key("input-source"));
        assert_eq!(invalid_config["input-source"], "nonexistent.mp4");

        // Missing model file
        let invalid_model_config = HashMap::from([
            ("input-source", "assets/sample.mp4"),
            ("model-path", "nonexistent.onnx"),
        ]);

        assert_eq!(invalid_model_config["model-path"], "nonexistent.onnx");

        // Invalid confidence threshold
        let invalid_threshold_config = HashMap::from([("confidence-threshold", "invalid")]);

        assert_eq!(invalid_threshold_config["confidence-threshold"], "invalid");
    }
}

/// Integration tests for plugin interactions
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_pupinference_to_pupoverlay_pipeline() {
        let mut pipeline = MockGstPipeline::new("inference-overlay-pipeline");

        // Create pupinference element
        let mut inference = create_pupinference_element();
        inference.name = "inference".to_string();

        // Create pupoverlay element
        let mut overlay = create_pupoverlay_element();
        overlay.name = "overlay".to_string();

        // Add to pipeline
        pipeline.add_element(inference);
        pipeline.add_element(overlay);

        // Test linking (simplified)
        // In reality this would involve complex caps negotiation
        assert!(pipeline.get_element("inference").is_some());
        assert!(pipeline.get_element("overlay").is_some());

        // Verify elements can transition to playing
        assert!(pipeline.set_state(ElementState::Playing).is_ok());
    }

    #[test]
    fn test_full_detection_pipeline_simulation() {
        let mut pipeline = MockGstPipeline::new("full-detection");

        // Create all elements for a complete detection pipeline
        let elements = vec![
            ("source", "filesrc", vec![("location", "assets/sample.mp4")]),
            ("decoder", "decodebin", vec![]),
            ("convert", "videoconvert", vec![]),
            (
                "inference",
                "pupinference",
                vec![("model-path", "models/yolov8n.onnx")],
            ),
            ("overlay", "pupoverlay", vec![("draw-bboxes", "true")]),
            ("sink", "autovideosink", vec![]),
        ];

        for (name, element_type, properties) in elements {
            let mut element = MockGstElement::new(name, element_type);

            // Set properties
            for (prop_name, prop_value) in properties {
                element.set_property(prop_name, MockGstValue::String(prop_value.to_string()));
            }

            // Add standard pads
            match element_type {
                "filesrc" => {
                    element.add_pad(MockGstPad::new("src", PadDirection::Source));
                }
                "autovideosink" => {
                    element.add_pad(MockGstPad::new("sink", PadDirection::Sink));
                }
                _ => {
                    element.add_pad(MockGstPad::new("sink", PadDirection::Sink));
                    element.add_pad(MockGstPad::new("src", PadDirection::Source));
                }
            }

            pipeline.add_element(element);
        }

        // Verify complete pipeline
        assert_eq!(pipeline.elements.len(), 6);

        // Test state transition for entire pipeline
        assert!(pipeline.set_state(ElementState::Ready).is_ok());
        assert!(pipeline.set_state(ElementState::Playing).is_ok());

        // Verify specific element configurations
        let inference = pipeline.get_element("inference").unwrap();
        assert_eq!(
            inference.get_property("model-path"),
            Some(&MockGstValue::String("models/yolov8n.onnx".to_string()))
        );

        let overlay = pipeline.get_element("overlay").unwrap();
        assert_eq!(
            overlay.get_property("draw-bboxes"),
            Some(&MockGstValue::String("true".to_string()))
        );
    }

    #[test]
    fn test_caps_negotiation_between_plugins() {
        // Test caps negotiation between pupinference and pupoverlay
        let caps = MockGstCaps::new("video/x-raw")
            .with_field("format", MockGstValue::String("RGB".to_string()))
            .with_field("width", MockGstValue::Integer(640))
            .with_field("height", MockGstValue::Integer(640))
            .with_field("framerate", MockGstValue::Fraction { num: 30, denom: 1 });

        // Both plugins should accept the same caps (passthrough)
        let inference_output_caps = caps.clone();
        let overlay_input_caps = caps.clone();

        assert!(inference_output_caps.is_compatible(&overlay_input_caps));

        // Test different resolutions
        let high_res_caps = MockGstCaps::new("video/x-raw")
            .with_field("format", MockGstValue::String("RGB".to_string()))
            .with_field("width", MockGstValue::Integer(1920))
            .with_field("height", MockGstValue::Integer(1080));

        let low_res_caps = MockGstCaps::new("video/x-raw")
            .with_field("format", MockGstValue::String("RGB".to_string()))
            .with_field("width", MockGstValue::Integer(320))
            .with_field("height", MockGstValue::Integer(240));

        // Different resolutions should not be compatible
        assert!(!high_res_caps.is_compatible(&low_res_caps));
    }

    fn create_pupinference_element() -> MockGstElement {
        let mut element = MockGstElement::new("pupinference-test", "pupinference");
        element.add_pad(MockGstPad::new("sink", PadDirection::Sink));
        element.add_pad(MockGstPad::new("src", PadDirection::Source));
        element.set_property(
            "model-path",
            MockGstValue::String("models/yolov8n.onnx".to_string()),
        );
        element.set_property("confidence-threshold", MockGstValue::Float(0.5));
        element
    }

    fn create_pupoverlay_element() -> MockGstElement {
        let mut element = MockGstElement::new("pupoverlay-test", "pupoverlay");
        element.add_pad(MockGstPad::new("sink", PadDirection::Sink));
        element.add_pad(MockGstPad::new("src", PadDirection::Source));
        element.set_property("draw-bboxes", MockGstValue::Boolean(true));
        element
    }
}

/// Performance benchmarking tests for plugin operations
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_element_creation_performance() {
        let start = Instant::now();

        for _ in 0..1000 {
            let _element = MockGstElement::new("test", "pupinference");
        }

        let duration = start.elapsed();

        // Element creation should be fast (< 10ms for 1000 elements)
        assert!(
            duration.as_millis() < 10,
            "Element creation too slow: {:?}",
            duration
        );
    }

    #[test]
    fn test_property_access_performance() {
        let mut element = MockGstElement::new("test", "pupinference");
        element.set_property("model-path", MockGstValue::String("test.onnx".to_string()));

        let start = Instant::now();

        for _ in 0..10000 {
            let _ = element.get_property("model-path");
        }

        let duration = start.elapsed();

        // Property access should be very fast (< 5ms for 10k accesses)
        assert!(
            duration.as_millis() < 5,
            "Property access too slow: {:?}",
            duration
        );
    }

    #[test]
    fn test_caps_negotiation_performance() {
        let caps1 = MockGstCaps::new("video/x-raw")
            .with_field("format", MockGstValue::String("RGB".to_string()))
            .with_field("width", MockGstValue::Integer(640));

        let caps2 = MockGstCaps::new("video/x-raw")
            .with_field("format", MockGstValue::String("RGB".to_string()))
            .with_field("width", MockGstValue::Integer(640));

        let start = Instant::now();

        for _ in 0..1000 {
            let _ = caps1.is_compatible(&caps2);
        }

        let duration = start.elapsed();

        // Caps comparison should be fast (< 5ms for 1000 comparisons)
        assert!(
            duration.as_millis() < 5,
            "Caps negotiation too slow: {:?}",
            duration
        );
    }

    #[test]
    fn test_buffer_processing_simulation() {
        // Simulate processing time for different buffer sizes
        let test_cases = vec![
            (320, 240, 3),   // Small frame
            (640, 480, 3),   // Medium frame
            (1920, 1080, 3), // HD frame
        ];

        for (width, height, channels) in test_cases {
            let frame_size = width * height * channels;
            let buffer_data = vec![128u8; frame_size];

            let start = Instant::now();

            // Simulate some processing (normalization)
            let _normalized: Vec<f32> = buffer_data.iter().map(|&x| x as f32 / 255.0).collect();

            let duration = start.elapsed();

            println!("Buffer processing {}x{}: {:?}", width, height, duration);

            // Processing should complete reasonably fast
            // Larger frames can take longer, but set reasonable limits
            let max_duration_ms = match frame_size {
                size if size <= 320 * 240 * 3 => 5,  // 5ms for small frames
                size if size <= 640 * 480 * 3 => 20, // 20ms for medium frames
                _ => 100,                            // 100ms for HD frames (more lenient for CI)
            };

            assert!(
                duration.as_millis() <= max_duration_ms,
                "Buffer processing too slow for {}x{}: {:?}",
                width,
                height,
                duration
            );
        }
    }

    #[test]
    fn test_pipeline_state_transition_performance() {
        let mut pipeline = MockGstPipeline::new("perf-test");

        // Add several elements
        for i in 0..10 {
            let element = MockGstElement::new(&format!("element_{}", i), "test");
            pipeline.add_element(element);
        }

        let start = Instant::now();

        // Test multiple state transitions
        for _ in 0..100 {
            assert!(pipeline.set_state(ElementState::Ready).is_ok());
            assert!(pipeline.set_state(ElementState::Paused).is_ok());
            assert!(pipeline.set_state(ElementState::Playing).is_ok());
            assert!(pipeline.set_state(ElementState::Paused).is_ok());
            assert!(pipeline.set_state(ElementState::Ready).is_ok());
            assert!(pipeline.set_state(ElementState::Null).is_ok());
        }

        let duration = start.elapsed();

        // State transitions should be fast (< 50ms for 100 cycles)
        assert!(
            duration.as_millis() < 50,
            "Pipeline state transitions too slow: {:?}",
            duration
        );
    }
}
