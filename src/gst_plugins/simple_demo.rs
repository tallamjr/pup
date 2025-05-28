//! Simple demo application to test YOLO inference on sample.mp4
//! This bypasses the complex plugin system and focuses on getting a working demo

use crate::config::AppConfig;
use crate::inference::{InferenceBackend, OrtBackend, TaskOutput};
use crate::preprocessing::Preprocessor;
use crate::utils::Detection;
use anyhow::Result;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use std::sync::{Arc, Mutex};

pub struct SimpleVideoProcessor {
    pipeline: gst::Pipeline,
    inference_backend: Arc<Mutex<Box<dyn InferenceBackend>>>,
    preprocessor: Preprocessor,
}

impl SimpleVideoProcessor {
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
        let videoconvert = gst::ElementFactory::make("videoconvert").build()?;
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

        let appsink = gst_app::AppSink::builder()
            .caps(
                &gst::Caps::builder("video/x-raw")
                    .field("format", "RGB")
                    .field("width", 640i32)
                    .field("height", 640i32)
                    .build(),
            )
            .build();

        // Add elements to pipeline
        pipeline.add_many(&[&filesrc, &decodebin, &videoconvert, &videoscale, &capsfilter])?;
        pipeline.add(&appsink)?;

        // Link static elements
        filesrc.link(&decodebin)?;
        gst::Element::link_many(&[&videoconvert, &videoscale, &capsfilter, appsink.upcast_ref()])?;

        // Setup dynamic linking for decodebin
        let pipeline_clone = pipeline.clone();
        let videoconvert_clone = videoconvert.clone();
        decodebin.connect_pad_added(move |_, src_pad| {
            let caps = src_pad.current_caps().unwrap();
            let structure = caps.structure(0).unwrap();
            
            if structure.name().starts_with("video/") {
                let sink_pad = videoconvert_clone.static_pad("sink").unwrap();
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

        // Setup appsink callback for processing frames
        let inference_clone = inference_backend.clone();
        let preprocessor_clone = preprocessor.clone();
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
                                        Ok(detections) => {
                                            println!("Found {} detections", detections.len());
                                            for detection in detections {
                                                println!("  Detection: {}", detection);
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
            inference_backend,
            preprocessor,
        })
    }

    pub fn run(&self) -> Result<()> {
        // Start playing
        self.pipeline.set_state(gst::State::Playing)?;

        // Wait for EOS or error
        let bus = self.pipeline.bus().unwrap();
        for msg in bus.iter_timed(gst::ClockTime::NONE) {
            match msg.view() {
                gst::MessageView::Eos(..) => {
                    println!("End of stream");
                    break;
                }
                gst::MessageView::Error(err) => {
                    eprintln!("Error: {}", err.error());
                    break;
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
            TaskOutput::Detections(detections) => Ok(detections),
        }
    }
}

pub fn run_simple_demo() -> Result<()> {
    // Use sample video file
    let mut config = AppConfig::default();
    config.pipeline.video_source = "assets/sample.mp4".to_string();
    config.inference.model_path = "models/yolov8n.onnx".into();

    println!("Starting simple YOLO demo on {}", config.pipeline.video_source);
    
    let processor = SimpleVideoProcessor::new(&config)?;
    processor.run()?;

    Ok(())
}