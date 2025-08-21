//! Simple pipeline tests that compile and run
#[cfg(test)]
mod pipeline_tests {
    use super::super::*;
    use crate::config::PipelineConfig;

    #[test]
    fn test_pipeline_string_basic() {
        let config = PipelineConfig {
            video_source: "webcam".to_string(),
            display_enabled: true,
            framerate: 30,
        };

        let result = VideoPipeline::build_pipeline_string(&config);
        assert!(result.is_ok());
        let pipeline_str = result.unwrap();
        assert!(pipeline_str.contains("appsink name=sink"));
    }

    #[test]
    fn test_pipeline_string_file_source() {
        let config = PipelineConfig {
            video_source: "test.mp4".to_string(),
            display_enabled: false,
            framerate: 30,
        };

        let result = VideoPipeline::build_pipeline_string(&config);
        assert!(result.is_ok());
        let pipeline_str = result.unwrap();
        assert!(pipeline_str.contains("filesrc"));
        assert!(pipeline_str.contains("appsink name=sink"));
    }

    #[test]
    fn test_pipeline_stats() {
        use gstreamer as gst;

        let stats = PipelineStats {
            is_running: true,
            current_state: gst::State::Playing,
        };

        assert!(stats.is_running);
        assert_eq!(stats.current_state, gst::State::Playing);
    }

    #[test]
    fn test_pipeline_stats_clone() {
        use gstreamer as gst;

        let stats = PipelineStats {
            is_running: false,
            current_state: gst::State::Null,
        };

        let cloned = stats.clone();
        assert_eq!(stats.is_running, cloned.is_running);
        assert_eq!(stats.current_state, cloned.current_state);
    }
}
