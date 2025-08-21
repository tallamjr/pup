//! Test helpers for pipeline module

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::config::{AppConfig, PipelineConfig};

    // Mock inference backend for testing
    pub struct MockInferenceBackend {
        should_fail: bool,
    }

    impl MockInferenceBackend {
        pub fn new() -> Self {
            Self { should_fail: false }
        }

        pub fn new_failing() -> Self {
            Self { should_fail: true }
        }
    }

    impl crate::inference::InferenceBackend for MockInferenceBackend {
        fn load_model(
            &mut self,
            _path: &std::path::Path,
        ) -> Result<(), crate::inference::InferenceError> {
            if self.should_fail {
                Err(crate::inference::InferenceError::ModelLoadError(
                    "Mock failure".to_string(),
                ))
            } else {
                Ok(())
            }
        }

        fn infer(
            &self,
            _input: &[f32],
        ) -> Result<crate::inference::TaskOutput, crate::inference::InferenceError> {
            if self.should_fail {
                Err(crate::inference::InferenceError::InferenceFailed(
                    "Mock inference failure".to_string(),
                ))
            } else {
                Ok(crate::inference::TaskOutput::Detections(vec![]))
            }
        }

        fn get_input_shape(&self) -> &[usize] {
            &[1, 3, 640, 640]
        }

        fn get_task_type(&self) -> crate::inference::TaskType {
            crate::inference::TaskType::ObjectDetection
        }

        fn get_confidence_threshold(&self) -> f32 {
            0.5
        }

        fn set_confidence_threshold(&mut self, _threshold: f32) {}
    }

    #[test]
    fn test_pipeline_string_generation() {
        let mut config = AppConfig::default();

        // Test webcam pipeline
        config.input.source = "webcam".to_string();
        config.output.display_enabled = true;
        // Update pipeline config to reflect the new input source
        config.pipeline = None; // Clear the existing pipeline
        config.pipeline = Some(config.get_pipeline());
        let pipeline_str =
            super::VideoPipeline::build_pipeline_string(&config.get_pipeline()).unwrap();
        assert!(pipeline_str.contains("avfvideosrc"));
        assert!(pipeline_str.contains("autovideosink"));
        assert!(pipeline_str.contains("appsink name=sink"));

        // Test file pipeline
        config.input.source = "test.mp4".to_string();
        config.output.display_enabled = false;
        // Force recreation of pipeline config to reflect the new input source
        config.pipeline = None; // Clear the existing pipeline
        config.pipeline = Some(config.get_pipeline()); // Recreate with new input
        let pipeline_config = config.get_pipeline();
        let pipeline_str = super::VideoPipeline::build_pipeline_string(&pipeline_config).unwrap();
        assert!(pipeline_str.contains("filesrc location=\"test.mp4\""));
        assert!(pipeline_str.contains("appsink name=sink"));
        assert!(!pipeline_str.contains("autovideosink"));
    }

    #[test]
    fn test_frame_processor_creation() {
        let preprocessor = super::Preprocessor::default();
        let backend = Box::new(MockInferenceBackend::new());
        let _processor = super::FrameProcessor::new(preprocessor, backend);
        // Just testing creation doesn't panic
    }

    #[test]
    fn test_pipeline_stats() {
        use gstreamer as gst;

        let stats = super::PipelineStats {
            is_running: true,
            current_state: gst::State::Playing,
        };

        assert!(stats.is_running);
        assert_eq!(stats.current_state, gst::State::Playing);
    }
}
