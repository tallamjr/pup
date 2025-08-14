//! GStreamer pipeline management
//!
//! This module provides video pipeline management using GStreamer,
//! including frame processing, inference integration, and output handling.
use crate::config::{AppConfig, PipelineConfig};
use crate::error::{PupError, PupResult};
use crate::inference::InferenceBackend;
use crate::preprocessing::Preprocessor;
use crate::utils::Detection;
use anyhow::Result;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app::AppSink;
use gstreamer_video::{VideoFrameRef, VideoInfo};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tracing::{debug, error, info, warn};

/// Video processing pipeline using GStreamer
pub struct VideoPipeline {
    pipeline: gst::Pipeline,
    appsink: AppSink,
    _config: PipelineConfig,
    is_running: bool,
}

impl VideoPipeline {
    /// Create a new video pipeline
    pub fn new(config: &AppConfig) -> PupResult<Self> {
        // Initialize GStreamer
        gst::init()
            .map_err(|e| PupError::PipelineError(format!("GStreamer init failed: {}", e)))?;

        let pipeline_config = config
            .pipeline
            .as_ref()
            .ok_or_else(|| PupError::MissingConfigField("pipeline".to_string()))?;
        let pipeline_str = Self::build_pipeline_string(pipeline_config).map_err(|e| {
            PupError::PipelineError(format!("Pipeline string creation failed: {}", e))
        })?;
        info!("Creating pipeline: {}", pipeline_str);

        let pipeline = gst::parse::launch(&pipeline_str)
            .map_err(|e| PupError::PipelineError(format!("Pipeline launch failed: {}", e)))?
            .dynamic_cast::<gst::Pipeline>()
            .map_err(|_| {
                PupError::PipelineError("Failed to cast pipeline to gst::Pipeline".to_string())
            })?;

        let appsink = pipeline
            .by_name("sink")
            .ok_or_else(|| PupError::ElementCreationFailed("No element named 'sink'".to_string()))?
            .dynamic_cast::<AppSink>()
            .map_err(|_| PupError::ElementCreationFailed("'sink' is not an AppSink".to_string()))?;

        appsink.set_property("emit-signals", true);

        Ok(Self {
            pipeline,
            appsink,
            _config: pipeline_config.clone(),
            is_running: false,
        })
    }

    /// Build the GStreamer pipeline string based on configuration
    fn build_pipeline_string(config: &PipelineConfig) -> Result<String> {
        let pipeline_str = if config.video_source == "auto" || config.video_source == "webcam" {
            // Use webcam
            if config.display_enabled {
                "avfvideosrc ! videoconvert ! tee name=t \
                     t. ! queue ! autovideosink \
                     t. ! queue ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink"
                    .to_string()
            } else {
                "avfvideosrc ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink"
                    .to_string()
            }
        } else {
            // Use video file
            if config.display_enabled {
                format!(
                    "filesrc location=\"{}\" ! decodebin name=d \
                     d. ! queue ! videoconvert ! tee name=t \
                         t. ! queue ! autovideosink \
                         t. ! queue ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink \
                     d. ! queue ! audioconvert ! audioresample ! autoaudiosink",
                    config.video_source
                )
            } else {
                format!(
                    "filesrc location=\"{}\" ! decodebin ! videoconvert ! \
                     video/x-raw,format=RGB ! appsink name=sink",
                    config.video_source
                )
            }
        };

        Ok(pipeline_str)
    }

    /// Start the pipeline
    pub fn start(&mut self) -> PupResult<()> {
        self.pipeline
            .set_state(gst::State::Playing)
            .map_err(|e| PupError::PipelineError(format!("Failed to start pipeline: {}", e)))?;
        self.is_running = true;
        info!("Pipeline started");
        Ok(())
    }

    /// Stop the pipeline
    pub fn stop(&mut self) -> PupResult<()> {
        self.pipeline
            .set_state(gst::State::Null)
            .map_err(|e| PupError::PipelineError(format!("Failed to stop pipeline: {}", e)))?;
        self.is_running = false;
        info!("Pipeline stopped");
        Ok(())
    }

    /// Check if pipeline is running
    pub fn is_running(&self) -> bool {
        self.is_running
    }

    /// Set up frame processing callback
    pub fn set_frame_processor<F>(&self, processor: F) -> Result<()>
    where
        F: Fn(&VideoFrameRef<&gst::BufferRef>, &VideoInfo) -> Vec<Detection>
            + Send
            + Sync
            + 'static,
    {
        self.appsink.connect("new-sample", false, move |vals| {
            let sink = match vals[0].get::<AppSink>() {
                Ok(sink) => sink,
                Err(e) => {
                    error!("Failed to get AppSink from callback: {:?}", e);
                    return Some(gst::FlowReturn::Error.to_value());
                }
            };

            if let Ok(sample) = sink.pull_sample() {
                if let Some(buffer) = sample.buffer() {
                    if let Some(caps) = sample.caps() {
                        if let Ok(info) = VideoInfo::from_caps(caps) {
                            match VideoFrameRef::from_buffer_ref_readable(buffer, &info) {
                                Ok(frame) => {
                                    let _detections = processor(&frame, &info);
                                    // Detections are processed in the callback
                                }
                                Err(e) => {
                                    error!("Failed to create video frame: {:?}", e);
                                }
                            }
                        }
                    }
                }
            }
            Some(gst::FlowReturn::Ok.to_value())
        });

        Ok(())
    }

    /// Process messages from the pipeline bus
    pub fn process_messages(&self, timeout: Option<Duration>) -> PupResult<bool> {
        let bus = self
            .pipeline
            .bus()
            .ok_or_else(|| PupError::PipelineError("Failed to get pipeline bus".to_string()))?;
        let timeout_ns = timeout.map(|d| gst::ClockTime::from_nseconds(d.as_nanos() as u64));

        if let Some(msg) = bus.timed_pop(timeout_ns.unwrap_or(gst::ClockTime::ZERO)) {
            match msg.view() {
                gst::MessageView::Eos(_) => {
                    info!("End of stream");
                    return Ok(false);
                }
                gst::MessageView::Error(e) => {
                    error!("Pipeline error: {}", e.error());
                    error!("Debug info: {:?}", e.debug());
                    return Ok(false);
                }
                gst::MessageView::Warning(w) => {
                    warn!("Pipeline warning: {}", w.error());
                    warn!("Debug info: {:?}", w.debug());
                }
                gst::MessageView::Info(i) => {
                    info!("Pipeline info: {}", i.error());
                }
                _ => {}
            }
        }

        Ok(true)
    }

    /// Get pipeline statistics
    pub fn get_stats(&self) -> PipelineStats {
        // This is a basic implementation - could be expanded with actual metrics
        PipelineStats {
            is_running: self.is_running,
            current_state: self.pipeline.current_state(),
        }
    }
}

impl Drop for VideoPipeline {
    fn drop(&mut self) {
        if self.is_running {
            let _ = self.stop();
        }
    }
}

/// Pipeline statistics
#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub is_running: bool,
    pub current_state: gst::State,
}

/// Frame processor that combines preprocessing and inference
pub struct FrameProcessor {
    preprocessor: Preprocessor,
    inference_backend: Arc<Mutex<Box<dyn InferenceBackend + Send>>>,
}

impl FrameProcessor {
    /// Create a new frame processor
    pub fn new(
        preprocessor: Preprocessor,
        inference_backend: Box<dyn InferenceBackend + Send>,
    ) -> Self {
        Self {
            preprocessor,
            inference_backend: Arc::new(Mutex::new(inference_backend)),
        }
    }

    /// Process a video frame and return detections
    pub fn process_frame(
        &self,
        frame: &VideoFrameRef<&gst::BufferRef>,
        info: &VideoInfo,
    ) -> Vec<Detection> {
        // Extract frame data
        let frame_data = match frame.plane_data(0) {
            Ok(data) => data,
            Err(e) => {
                error!("Failed to get frame data: {:?}", e);
                return Vec::new();
            }
        };

        let width = info.width() as usize;
        let height = info.height() as usize;
        let stride = info.stride()[0] as usize;

        // Convert frame data to OpenCV Mat
        let mat = match self.gstreamer_to_opencv_mat(frame_data, width, height, stride) {
            Ok(mat) => mat,
            Err(e) => {
                error!("Failed to convert frame to Mat: {:?}", e);
                return Vec::new();
            }
        };

        // Preprocess the frame
        let tensor_data = match self.preprocessor.process(&mat) {
            Ok(data) => data,
            Err(e) => {
                error!("Preprocessing failed: {:?}", e);
                return Vec::new();
            }
        };

        // Run inference
        let inference_backend = self.inference_backend.lock().unwrap();
        match inference_backend.infer(&tensor_data) {
            Ok(task_output) => match task_output {
                crate::inference::TaskOutput::Detections(detections) => {
                    debug!("Found {} detections", detections.len());
                    for detection in &detections {
                        debug!("  {}", detection);
                    }
                    detections
                }
            },
            Err(e) => {
                error!("Inference failed: {:?}", e);
                Vec::new()
            }
        }
    }

    /// Convert GStreamer frame data to OpenCV Mat
    fn gstreamer_to_opencv_mat(
        &self,
        frame_data: &[u8],
        width: usize,
        height: usize,
        stride: usize,
    ) -> Result<opencv::core::Mat> {
        use opencv::{core, prelude::*};

        // Create OpenCV Mat
        let mat_type = core::CV_8UC3;
        let mut mat = core::Mat::new_rows_cols_with_default(
            height as i32,
            width as i32,
            mat_type,
            core::Scalar::all(0.0),
        )?;

        // Copy row by row to handle stride
        let mat_bytes = mat.data_bytes_mut()?;
        let row_size = width * 3; // RGB

        for row in 0..height {
            let src_start = row * stride;
            let src_end = src_start + row_size;
            let dst_start = row * row_size;
            let dst_end = dst_start + row_size;

            if src_end <= frame_data.len() && dst_end <= mat_bytes.len() {
                mat_bytes[dst_start..dst_end].copy_from_slice(&frame_data[src_start..src_end]);
            }
        }

        Ok(mat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AppConfig;

    #[test]
    fn test_pipeline_string_generation() {
        let mut config = AppConfig::default();

        // Test webcam pipeline
        config.input.source = "webcam".to_string();
        config.output.display_enabled = true;
        // Update pipeline config to reflect the new input source
        config.pipeline = None; // Clear the existing pipeline
        config.pipeline = Some(config.get_pipeline());
        let pipeline_str = VideoPipeline::build_pipeline_string(&config.get_pipeline()).unwrap();
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
        let pipeline_str = VideoPipeline::build_pipeline_string(&pipeline_config).unwrap();
        assert!(pipeline_str.contains("filesrc location=\"test.mp4\""));
        assert!(pipeline_str.contains("appsink name=sink"));
        assert!(!pipeline_str.contains("autovideosink"));
    }

    #[test]
    fn test_pipeline_stats() {
        let stats = PipelineStats {
            is_running: true,
            current_state: gst::State::Playing,
        };

        assert!(stats.is_running);
        assert_eq!(stats.current_state, gst::State::Playing);
    }

    #[test]
    fn test_frame_processor_creation() {
        use crate::inference::OrtBackend;

        let preprocessor = Preprocessor::default();
        let backend = Box::new(OrtBackend::new());
        let _processor = FrameProcessor::new(preprocessor, backend);

        // Just testing creation doesn't panic
    }
}
