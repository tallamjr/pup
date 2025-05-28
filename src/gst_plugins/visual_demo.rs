//! Visual demo application with bounding box overlay
//! Shows video with YOLO detections rendered as bounding boxes with labels

use crate::config::AppConfig;
use crate::inference::{InferenceBackend, OrtBackend, TaskOutput};
use crate::preprocessing::Preprocessor;
use crate::utils::Detection;
use crate::utils::coco_classes::NAMES as COCO_NAMES;
use anyhow::Result;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use std::sync::{Arc, Mutex};
//use cairo_rs::{Context, FontSlant, FontWeight};

pub struct VisualVideoProcessor {
    pipeline: gst::Pipeline,
    _inference_backend: Arc<Mutex<Box<dyn InferenceBackend>>>,
    _preprocessor: Preprocessor,
    _detections: Arc<Mutex<Vec<Detection>>>,
}

impl VisualVideoProcessor {
    pub fn new(config: &AppConfig) -> Result<Self> {
        // Initialize GStreamer
        gst::init()?;

        // Create pipeline
        let pipeline = gst::Pipeline::new();

        // Create elements
        let filesrc = gst::ElementFactory::make("filesrc")
            .property("location", &config.pipeline.video_source)
            .build()?;

        let decodebin = gst::ElementFactory::make("decodebin").build()?;
        let videoconvert1 = gst::ElementFactory::make("videoconvert").build()?;
        let videoscale = gst::ElementFactory::make("videoscale").build()?;
        
        // Create capsfilter for standardized format
        let capsfilter = gst::ElementFactory::make("capsfilter")
            .property(
                "caps",
                gst::Caps::builder("video/x-raw")
                    .field("format", "RGB")
                    .field("width", 640i32)
                    .field("height", 640i32)
                    .build(),
            )
            .build()?;

        // Create tee to split the stream
        let tee = gst::ElementFactory::make("tee").build()?;
        
        // Create queue for inference path
        let queue1 = gst::ElementFactory::make("queue").build()?;
        let appsink = gst_app::AppSink::builder()
            .caps(
                &gst::Caps::builder("video/x-raw")
                    .field("format", "RGB")
                    .field("width", 640i32)
                    .field("height", 640i32)
                    .build(),
            )
            .build();

        // Create queue for display path (simple playback for now)
        let queue2 = gst::ElementFactory::make("queue").build()?;
        let videoconvert2 = gst::ElementFactory::make("videoconvert").build()?;
        
        // Use fakesink for now to avoid macOS video window issues
        // This allows the demo to run and show detection output in terminal
        let videosink = gst::ElementFactory::make("fakesink")
            .property("sync", true)
            .build()?;

        // Add elements to pipeline
        pipeline.add_many(&[
            &filesrc, &decodebin, &videoconvert1, &videoscale, &capsfilter, &tee,
            &queue1, appsink.upcast_ref(), &queue2, &videoconvert2, &videosink
        ])?;

        // Link static elements
        filesrc.link(&decodebin)?;
        gst::Element::link_many(&[&videoconvert1, &videoscale, &capsfilter, &tee])?;
        
        // Link inference path
        gst::Element::link_many(&[&tee, &queue1, appsink.upcast_ref()])?;
        
        // Link display path
        gst::Element::link_many(&[&tee, &queue2, &videoconvert2, &videosink])?;

        // Setup dynamic linking for decodebin
        let videoconvert1_clone = videoconvert1.clone();
        decodebin.connect_pad_added(move |_, src_pad| {
            let caps = src_pad.current_caps().unwrap();
            let structure = caps.structure(0).unwrap();
            
            if structure.name().starts_with("video/") {
                let sink_pad = videoconvert1_clone.static_pad("sink").unwrap();
                if src_pad.link(&sink_pad).is_err() {
                    eprintln!("Failed to link decodebin to videoconvert");
                }
            }
        });

        // Setup inference backend
        let mut backend = Box::new(OrtBackend::new());
        backend.load_model(&config.inference.model_path)?;
        
        let inference_backend: Arc<Mutex<Box<dyn InferenceBackend>>> = Arc::new(Mutex::new(backend));
        let preprocessor = Preprocessor::default();
        let detections = Arc::new(Mutex::new(Vec::new()));

        // Setup appsink callback for processing frames
        let inference_clone = inference_backend.clone();
        let preprocessor_clone = preprocessor.clone();
        let detections_clone = detections.clone();
        appsink.set_callbacks(
            gst_app::AppSinkCallbacks::builder()
                .new_sample(move |appsink| {
                    if let Ok(sample) = appsink.pull_sample() {
                        if let Some(buffer) = sample.buffer() {
                            if let Some(caps) = sample.caps() {
                                if let Ok(info) = gst_video::VideoInfo::from_caps(&caps) {
                                    match Self::process_sample_static(
                                        &buffer,
                                        &info,
                                        &inference_clone,
                                        &preprocessor_clone,
                                    ) {
                                        Ok(new_detections) => {
                                            println!("Found {} detections", new_detections.len());
                                            for detection in &new_detections {
                                                let class_name = if detection.class_id < COCO_NAMES.len() as i32 {
                                                    COCO_NAMES[detection.class_id as usize]
                                                } else {
                                                    "unknown"
                                                };
                                                println!("  {}: {:.1}% at ({:.1}, {:.1}, {:.1}, {:.1})", 
                                                    class_name, detection.score, 
                                                    detection.x1, detection.y1, detection.x2, detection.y2);
                                            }
                                            // Update shared detections for future overlay
                                            if let Ok(mut detections_guard) = detections_clone.lock() {
                                                *detections_guard = new_detections;
                                            }
                                        }
                                        Err(e) => eprintln!("Processing error: {}", e),
                                    }
                                }
                            }
                        }
                    }
                    Ok(gst::FlowSuccess::Ok)
                })
                .build(),
        );

        // Note: For now, we're just displaying the video and printing detections
        // Future enhancement: implement cairo overlay for bounding box rendering

        Ok(Self {
            pipeline,
            _inference_backend: inference_backend,
            _preprocessor: preprocessor,
            _detections: detections,
        })
    }

    pub fn run(&self) -> Result<()> {
        println!("Starting visual processing with YOLO detection...");
        println!("Video frames will be processed and detections shown in terminal");
        
        // Start playing
        self.pipeline.set_state(gst::State::Playing)?;

        // Wait for EOS or error
        let bus = self.pipeline.bus().unwrap();
        for msg in bus.iter_timed(gst::ClockTime::NONE) {
            match msg.view() {
                gst::MessageView::Eos(..) => {
                    println!("End of stream - video processing complete");
                    break;
                }
                gst::MessageView::Error(err) => {
                    eprintln!("Error: {}", err.error());
                    if let Some(debug) = err.debug() {
                        eprintln!("Debug info: {}", debug);
                    }
                    break;
                }
                gst::MessageView::StateChanged(state_changed) => {
                    if state_changed.src().map(|s| s == &self.pipeline).unwrap_or(false) {
                        println!("Pipeline state changed from {:?} to {:?}", 
                            state_changed.old(), state_changed.current());
                    }
                }
                _ => {}
            }
        }

        // Stop pipeline
        self.pipeline.set_state(gst::State::Null)?;
        println!("Visual demo completed");
        Ok(())
    }

    fn process_sample_static(
        buffer: &gst::BufferRef,
        info: &gst_video::VideoInfo,
        inference_backend: &Arc<Mutex<Box<dyn InferenceBackend>>>,
        _preprocessor: &Preprocessor,
    ) -> Result<Vec<Detection>> {
        // Map buffer for reading
        let map = buffer.map_readable().map_err(|e| anyhow::anyhow!("Failed to map buffer: {}", e))?;
        let data = map.as_slice();

        // Convert RGB data to f32 tensor
        let width = info.width() as usize;
        let height = info.height() as usize;
        let expected_size = width * height * 3;

        if data.len() < expected_size {
            return Err(anyhow::anyhow!(
                "Buffer too small: got {}, expected {}",
                data.len(),
                expected_size
            ));
        }

        // Simple RGB to tensor conversion (normalize to 0-1)
        let tensor_data: Vec<f32> = data
            .chunks(3)
            .take(width * height)
            .flat_map(|pixel| {
                [
                    pixel[0] as f32 / 255.0,
                    pixel[1] as f32 / 255.0,
                    pixel[2] as f32 / 255.0,
                ]
            })
            .collect();

        // Reshape to CHW format (channels, height, width)
        let mut chw_data = vec![0.0f32; 3 * width * height];
        for y in 0..height {
            for x in 0..width {
                let hwc_idx = (y * width + x) * 3;
                let chw_r_idx = y * width + x;
                let chw_g_idx = width * height + y * width + x;
                let chw_b_idx = 2 * width * height + y * width + x;
                
                chw_data[chw_r_idx] = tensor_data[hwc_idx];
                chw_data[chw_g_idx] = tensor_data[hwc_idx + 1];
                chw_data[chw_b_idx] = tensor_data[hwc_idx + 2];
            }
        }

        // Run inference
        let backend = inference_backend.lock().unwrap();
        let result = backend.infer(&chw_data)?;

        match result {
            TaskOutput::Detections(detections) => {
                // Filter detections with reasonable confidence and normalize coordinates
                let filtered_detections: Vec<Detection> = detections
                    .into_iter()
                    .filter(|d| d.score > 50.0) // Filter by confidence
                    .collect();
                
                Ok(filtered_detections)
            }
        }
    }

    // TODO: Implement overlay rendering with cairo in future enhancement
}

pub fn run_visual_demo() -> Result<()> {
    // Use sample video file
    let mut config = AppConfig::default();
    config.pipeline.video_source = "assets/sample.mp4".to_string();
    config.inference.model_path = "models/yolov8n.onnx".into();

    println!("Starting visual YOLO demo with bounding boxes on {}", config.pipeline.video_source);
    
    let processor = VisualVideoProcessor::new(&config)?;
    processor.run()?;

    Ok(())
}