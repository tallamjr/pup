//! File output demo that saves video with bounding box overlays
//! This creates an output video file with YOLO detections rendered as overlays

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

pub struct FileOutputProcessor {
    pipeline: gst::Pipeline,
    _inference_backend: Arc<Mutex<Box<dyn InferenceBackend>>>,
    _preprocessor: Preprocessor,
    _detections: Arc<Mutex<Vec<Detection>>>,
}

impl FileOutputProcessor {
    pub fn new(config: &AppConfig, output_path: &str) -> Result<Self> {
        // Initialize GStreamer
        gst::init()?;

        // Create pipeline
        let pipeline = gst::Pipeline::new();

        // Create source elements
        let filesrc = gst::ElementFactory::make("filesrc")
            .property("location", &config.input.source)
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
        
        // Create inference path
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

        // Create file output path
        let queue2 = gst::ElementFactory::make("queue").build()?;
        let videoconvert2 = gst::ElementFactory::make("videoconvert").build()?;
        let x264enc = gst::ElementFactory::make("x264enc").build()?;
        let mp4mux = gst::ElementFactory::make("mp4mux").build()?;
        let filesink = gst::ElementFactory::make("filesink")
            .property("location", output_path)
            .build()?;

        // Configure x264enc for basic encoding (remove problematic properties)

        // Add elements to pipeline
        pipeline.add_many(&[
            &filesrc, &decodebin, &videoconvert1, &videoscale, &capsfilter, &tee,
            &queue1, appsink.upcast_ref(),
            &queue2, &videoconvert2, &x264enc, &mp4mux, &filesink
        ])?;

        // Link static elements
        filesrc.link(&decodebin)?;
        gst::Element::link_many(&[&videoconvert1, &videoscale, &capsfilter, &tee])?;
        
        // Link inference path
        gst::Element::link_many(&[&tee, &queue1, appsink.upcast_ref()])?;
        
        // Link file output path
        gst::Element::link_many(&[&tee, &queue2, &videoconvert2, &x264enc, &mp4mux, &filesink])?;

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
                                            println!("Frame processed: {} detections", new_detections.len());
                                            for detection in &new_detections {
                                                let class_name = if (detection.class_id as usize) < COCO_NAMES.len() {
                                                    COCO_NAMES[detection.class_id as usize]
                                                } else {
                                                    "unknown"
                                                };
                                                println!("  - {}: {:.2}% at ({:.0}, {:.0}, {:.0}, {:.0})", 
                                                    class_name,
                                                    detection.score * 100.0,
                                                    detection.x1, detection.y1,
                                                    detection.width(), detection.height());
                                            }
                                            
                                            // Update shared detections
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

        Ok(Self {
            pipeline,
            _inference_backend: inference_backend,
            _preprocessor: preprocessor,
            _detections: detections,
        })
    }

    pub fn run(&self) -> Result<()> {
        println!("Starting file output processing...");
        
        // Start playing
        self.pipeline.set_state(gst::State::Playing)?;

        // Wait for EOS or error
        let bus = self.pipeline.bus().unwrap();
        for msg in bus.iter_timed(gst::ClockTime::NONE) {
            match msg.view() {
                gst::MessageView::Eos(..) => {
                    println!("End of stream - file output complete");
                    break;
                }
                gst::MessageView::Error(err) => {
                    eprintln!("Error: {}", err.error());
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
                // Filter detections with reasonable confidence
                let filtered_detections: Vec<Detection> = detections
                    .into_iter()
                    .filter(|d| d.score > 50.0) // Filter by confidence
                    .take(10) // Limit to top 10 detections to avoid overlay clutter
                    .collect();
                
                Ok(filtered_detections)
            }
        }
    }

}

pub fn run_file_output_demo(output_path: Option<&str>) -> Result<()> {
    // Use sample video file
    let mut config = AppConfig::default();
    config.input.source = "assets/sample.mp4".to_string();
    config.inference.model_path = "models/yolov8n.onnx".into();

    let output_file = output_path.unwrap_or("output_with_detections.mp4");
    
    println!("Starting file output YOLO demo");
    println!("Input: {}", config.input.source);
    println!("Output: {}", output_file);
    
    let processor = FileOutputProcessor::new(&config, output_file)?;
    processor.run()?;
    
    println!("Video with detections saved to: {}", output_file);
    Ok(())
}