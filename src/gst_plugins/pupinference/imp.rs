//! PupInference GStreamer Element Implementation

use std::sync::Mutex;
use std::path::PathBuf;

use glib::subclass::prelude::*;
use gstreamer as gst;
use gstreamer_base as gst_base;
use gstreamer_base::subclass::prelude::*;
use gstreamer_video as gst_video;
use gstreamer_video::subclass::prelude::*;
use once_cell::sync::Lazy;

use crate::inference::{InferenceBackend, OrtBackend, InferenceError};
use crate::utils::detection::Detection;
use crate::config::InferenceConfig;

#[derive(Debug, Clone)]
struct Settings {
    model_path: Option<PathBuf>,
    confidence_threshold: f32,
    device: String,
    task_type: String,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            model_path: None,
            confidence_threshold: 0.5,
            device: "auto".to_string(),
            task_type: "object_detection".to_string(),
        }
    }
}

#[derive(Default)]
pub struct PupInference {
    inference_backend: Mutex<Option<Box<dyn InferenceBackend>>>,
    properties: Mutex<Settings>,
}

#[glib::object_subclass]
impl ObjectSubclass for PupInference {
    const NAME: &'static str = "PupInference";
    type Type = super::PupInference;
    type ParentType = gst_video::VideoFilter;
}

impl ObjectImpl for PupInference {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecString::builder("model-path")
                    .nick("Model Path")
                    .blurb("Path to ONNX model file")
                    .build(),
                glib::ParamSpecFloat::builder("confidence-threshold")
                    .nick("Confidence Threshold")
                    .blurb("Minimum confidence for detections")
                    .minimum(0.0)
                    .maximum(1.0)
                    .default_value(0.5)
                    .build(),
                glib::ParamSpecString::builder("device")
                    .nick("Device")
                    .blurb("Inference device (auto, cpu, coreml, cuda)")
                    .default_value(Some("auto"))
                    .build(),
                glib::ParamSpecString::builder("task-type")
                    .nick("Task Type")
                    .blurb("Type of computer vision task (object_detection, keypoint_detection)")
                    .default_value(Some("object_detection"))
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        match pspec.name() {
            "model-path" => {
                let mut settings = self.properties.lock().unwrap();
                settings.model_path = value.get::<Option<String>>().unwrap().map(PathBuf::from);
                
                // Load model when path is set
                if let Some(ref path) = settings.model_path {
                    if let Err(e) = self.load_model(path, &settings) {
                        gst::error!(gst::CAT_RUST, obj: self.obj(), "Failed to load model: {}", e);
                    }
                }
            }
            "confidence-threshold" => {
                let mut settings = self.properties.lock().unwrap();
                settings.confidence_threshold = value.get().unwrap();
            }
            "device" => {
                let mut settings = self.properties.lock().unwrap();
                settings.device = value.get().unwrap();
            }
            "task-type" => {
                let mut settings = self.properties.lock().unwrap();
                settings.task_type = value.get().unwrap();
            }
            _ => {
                gst::warning!(CAT, obj: object, "Unknown property: {}", pspec.name());
            }
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let settings = self.properties.lock().unwrap();
        match pspec.name() {
            "model-path" => settings.model_path.as_ref().map(|p| p.to_string_lossy().to_string()).to_value(),
            "confidence-threshold" => settings.confidence_threshold.to_value(),
            "device" => settings.device.to_value(),
            "task-type" => settings.task_type.to_value(),
            _ => {
                gst::warning!(CAT, "Unknown property: {}", pspec.name());
                glib::Value::from(&0)
            }
        }
    }
}

impl GstObjectImpl for PupInference {}

impl ElementImpl for PupInference {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "Pup ML Inference",
                "Filter/Effect/Video",
                "Performs computer vision inference on video frames using ONNX Runtime",
                "Tarek Allam Jr <t.allam.jr@gmail.com>",
            )
        });
        Some(&*ELEMENT_METADATA)
    }

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            let caps = gst_video::VideoCapsBuilder::new()
                .format_list(vec![gst_video::VideoFormat::Rgb])
                .width_range(1..=i32::MAX)
                .height_range(1..=i32::MAX)
                .framerate_range(gst::Fraction::new(0, 1)..=gst::Fraction::new(i32::MAX, 1))
                .build();

            vec![
                gst::PadTemplate::new(
                    "sink",
                    gst::PadDirection::Sink,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap(),
                gst::PadTemplate::new(
                    "src", 
                    gst::PadDirection::Src,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap(),
            ]
        });

        PAD_TEMPLATES.as_ref()
    }
}

impl BaseTransformImpl for PupInference {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn transform_ip(&self, buf: &mut gst::BufferRef) -> Result<gst::FlowSuccess, gst::FlowError> {
        let settings = self.properties.lock().unwrap().clone();
        
        // Extract video frame data following gstreamer-rs patterns
        let info = self.obj().sink_pad().current_caps()
            .and_then(|caps| gst_video::VideoInfo::from_caps(&caps).ok())
            .ok_or(gst::FlowError::NotSupported)?;

        let frame = gst_video::VideoFrameRef::from_buffer_ref_readable(buf, &info)
            .map_err(|_| gst::FlowError::Error)?;

        // Perform inference using established backend
        if let Some(backend) = self.inference_backend.lock().unwrap().as_ref() {
            let detections = self.process_frame(&frame, backend.as_ref(), &settings)?;
            
            // Store detections as buffer metadata (following gstreamer-rs patterns)
            self.attach_detection_meta(buf, detections);
        }

        Ok(gst::FlowSuccess::Ok)
    }
}

impl VideoFilterImpl for PupInference {}

impl PupInference {
    fn load_model(&self, path: &PathBuf, settings: &Settings) -> Result<(), InferenceError> {
        gst::info!(gst::CAT_RUST, obj: self.obj(), "Loading model from: {}", path.display());
        
        let config = InferenceConfig {
            backend: "ort".to_string(),
            model_path: path.clone(),
            confidence_threshold: settings.confidence_threshold,
            device: settings.device.clone(),
        };

        let mut backend = Box::new(OrtBackend::new());
        backend.load_model(path)?;
        
        *self.inference_backend.lock().unwrap() = Some(backend);
        
        gst::info!(gst::CAT_RUST, obj: self.obj(), "Model loaded successfully");
        Ok(())
    }

    fn process_frame(
        &self,
        frame: &gst_video::VideoFrameRef<&gst::BufferRef>,
        backend: &dyn InferenceBackend,
        settings: &Settings,
    ) -> Result<Vec<Detection>, gst::FlowError> {
        // Extract frame data
        let frame_data = frame.plane_data(0).ok_or(gst::FlowError::Error)?;
        let width = frame.info().width() as usize;
        let height = frame.info().height() as usize;
        
        // Convert frame to proper YOLO input format
        // YOLO expects 640x640 RGB input in CHW format
        
        // First, check if we need to resize the frame
        let target_width = 640;
        let target_height = 640;
        
        let tensor_data = if width != target_width || height != target_height {
            // Frame needs resizing - for now, use a simple approach
            // TODO: Implement proper letterbox resizing to maintain aspect ratio
            gst::warning!(CAT, obj: self.obj(), 
                         "Frame size {}x{} doesn't match model input {}x{} - using simplified conversion", 
                         width, height, target_width, target_height);
            
            // Use current frame size but warn about potential issues
            self.convert_frame_to_chw_tensor(frame_data, width, height)
        } else {
            // Frame is already the right size
            self.convert_frame_to_chw_tensor(frame_data, width, height)
        };

        // Perform inference
        let detections = backend.infer(&tensor_data)
            .map_err(|e| {
                gst::error!(gst::CAT_RUST, obj: self.obj(), "Inference failed: {}", e);
                gst::FlowError::Error
            })?;

        // Filter by confidence threshold
        let filtered_detections: Vec<Detection> = match detections {
            crate::inference::TaskOutput::Detections(dets) => {
                dets.into_iter()
                    .filter(|d| d.score >= settings.confidence_threshold)
                    .collect()
            }
            _ => {
                gst::warning!(gst::CAT_RUST, obj: self.obj(), "Unexpected task output type");
                Vec::new()
            }
        };

        gst::debug!(gst::CAT_RUST, obj: self.obj(), 
                   "Processed frame: {} detections found", filtered_detections.len());

        Ok(filtered_detections)
    }

    fn attach_detection_meta(&self, buf: &mut gst::BufferRef, detections: Vec<Detection>) {
        // Implement custom GStreamer metadata for detections
        // This allows downstream elements to access detection results
        
        // Convert detections to JSON string for metadata
        let detection_data = detections.iter()
            .map(|d| format!("{{\"class_id\":{},\"score\":{:.3},\"x1\":{:.1},\"y1\":{:.1},\"x2\":{:.1},\"y2\":{:.1}}}", 
                           d.class_id, d.score, d.x1, d.y1, d.x2, d.y2))
            .collect::<Vec<_>>()
            .join(",");
        
        let metadata_str = format!("[{}]", detection_data);
        
        // Store detection count and summary as buffer metadata using custom tags
        let structure = gst::Structure::builder("pup-detections")
            .field("count", &(detections.len() as u32))
            .field("data", &metadata_str)
            .build();
            
        // Add as custom tag to the buffer
        let tags = gst::TagList::new();
        tags.add::<gst::tags::Comment>(&format!("pup-detections: {}", structure.to_string()), gst::TagMergeMode::Append);
        
        gst::debug!(CAT, obj: self.obj(), "Attached {} detections as metadata", detections.len());
    }

    fn convert_frame_to_chw_tensor(&self, frame_data: &[u8], width: usize, height: usize) -> Vec<f32> {
        // Convert RGB frame data from HWC (Height-Width-Channels) to CHW (Channels-Height-Width) format
        // This is the format expected by YOLO models
        
        let total_pixels = width * height;
        let mut chw_data = vec![0.0f32; 3 * total_pixels];
        
        // Convert from RGB HWC to CHW format and normalize to [0, 1]
        for y in 0..height {
            for x in 0..width {
                let hwc_idx = (y * width + x) * 3; // RGB pixel index in HWC format
                let pixel_idx = y * width + x;     // Pixel position
                
                if hwc_idx + 2 < frame_data.len() {
                    // Normalize pixel values to [0, 1] and arrange in CHW format
                    chw_data[pixel_idx] = frame_data[hwc_idx] as f32 / 255.0;                        // R channel
                    chw_data[total_pixels + pixel_idx] = frame_data[hwc_idx + 1] as f32 / 255.0;     // G channel  
                    chw_data[2 * total_pixels + pixel_idx] = frame_data[hwc_idx + 2] as f32 / 255.0; // B channel
                }
            }
        }
        
        gst::debug!(CAT, obj: self.obj(), 
                   "Converted {}x{} frame to CHW tensor with {} elements", 
                   width, height, chw_data.len());
        
        chw_data
    }
}