//! Detection demo that processes video and saves detection results to a file
//! This processes frames and saves detected objects to a text file

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
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};

pub struct DetectionProcessor {
    pipeline: gst::Pipeline,
    inference_backend: Arc<Mutex<Box<dyn InferenceBackend>>>,
    preprocessor: Preprocessor,
    detections: Arc<Mutex<Vec<Detection>>>,
    output_file: Arc<Mutex<File>>,
    frame_count: Arc<Mutex<u32>>,
}

impl DetectionProcessor {
    pub fn new(config: &AppConfig, output_path: &str) -> Result<Self> {
        // Initialize GStreamer
        gst::init()?;

        let pipeline = gst::Pipeline::builder().name("detection-pipeline").build();

        // Create elements
        let filesrc = gst::ElementFactory::make("filesrc")
            .property("location", &config.pipeline.video_source)
            .build()?;
        
        let decodebin = gst::ElementFactory::make("decodebin").build()?;
        let videoconvert = gst::ElementFactory::make("videoconvert").build()?;
        let videoscale = gst::ElementFactory::make("videoscale").build()?;
        let capsfilter = gst::ElementFactory::make("capsfilter")
            .property("caps", &gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", 640)
                .field("height", 480)
                .build())
            .build()?;
        
        let appsink = gst_app::AppSink::builder()
            .caps(&gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", 640)
                .field("height", 480)
                .build())
            .build();

        // Add elements to pipeline
        pipeline.add_many(&[
            &filesrc, &decodebin, &videoconvert, &videoscale, &capsfilter, 
            appsink.upcast_ref()
        ])?;

        // Link static elements
        filesrc.link(&decodebin)?;
        gst::Element::link_many(&[&videoconvert, &videoscale, &capsfilter, appsink.upcast_ref()])?;

        // Connect decodebin pad-added signal
        let videoconvert_clone = videoconvert.clone();
        decodebin.connect("pad-added", false, move |values| {
            let pad = values[1].get::<gst::Pad>().unwrap();
            if let Some(caps) = pad.current_caps() {
                let structure = caps.structure(0).unwrap();
                if structure.name().starts_with("video/") {
                    let sink_pad = videoconvert_clone.static_pad("sink").unwrap();
                    if !sink_pad.is_linked() {
                        let _ = pad.link(&sink_pad);
                    }
                }
            }
            None
        });

        // Setup inference backend
        let mut backend = Box::new(OrtBackend::new());
        backend.load_model(&config.inference.model_path)?;
        
        let inference_backend: Arc<Mutex<Box<dyn InferenceBackend>>> = Arc::new(Mutex::new(backend));
        let preprocessor = Preprocessor::default();
        let detections = Arc::new(Mutex::new(Vec::new()));
        
        // Create output file
        let output_file = Arc::new(Mutex::new(File::create(output_path)?));
        let frame_count = Arc::new(Mutex::new(0u32));

        // Setup appsink callback for processing frames
        let inference_clone = inference_backend.clone();
        let preprocessor_clone = preprocessor.clone();
        let detections_clone = detections.clone();
        let output_file_clone = output_file.clone();
        let frame_count_clone = frame_count.clone();
        
        appsink.set_callbacks(
            gst_app::AppSinkCallbacks::builder()
                .new_sample(move |appsink| {
                    if let Ok(sample) = appsink.pull_sample() {
                        if let Some(buffer) = sample.buffer() {
                            if let Some(caps) = sample.caps() {
                                if let Ok(info) = gst_video::VideoInfo::from_caps(&caps) {
                                    // Increment frame count
                                    let current_frame = {
                                        let mut count = frame_count_clone.lock().unwrap();
                                        *count += 1;
                                        *count
                                    };

                                    match Self::process_sample_static(
                                        &buffer,
                                        &info,
                                        &inference_clone,
                                        &preprocessor_clone,
                                    ) {
                                        Ok(new_detections) => {
                                            println!("Frame {}: {} detections", current_frame, new_detections.len());
                                            
                                            // Write detections to file
                                            if let Ok(mut file) = output_file_clone.lock() {
                                                let _ = writeln!(file, "Frame {}: {} detections", current_frame, new_detections.len());
                                                
                                                for detection in &new_detections {
                                                    let class_name = if (detection.class_id as usize) < COCO_NAMES.len() {
                                                        COCO_NAMES[detection.class_id as usize]
                                                    } else {
                                                        "unknown"
                                                    };
                                                    let detection_line = format!(
                                                        "  - {}: {:.2}% at ({:.0}, {:.0}, {:.0}, {:.0})",
                                                        class_name,
                                                        detection.score * 100.0,
                                                        detection.x1, detection.y1,
                                                        detection.width(), detection.height()
                                                    );
                                                    println!("{}", detection_line);
                                                    let _ = writeln!(file, "{}", detection_line);
                                                }
                                                let _ = file.flush();
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
            inference_backend,
            preprocessor,
            detections,
            output_file,
            frame_count,
        })
    }

    pub fn run(&self) -> Result<()> {
        println!("Starting detection processing...");
        
        // Start playing
        self.pipeline.set_state(gst::State::Playing)?;

        // Wait for EOS or error
        let bus = self.pipeline.bus().unwrap();
        for msg in bus.iter_timed(gst::ClockTime::NONE) {
            match msg.view() {
                gst::MessageView::Eos(..) => {
                    println!("End of stream - processing complete");
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
        
        // Final stats
        let total_frames = *self.frame_count.lock().unwrap();
        println!("Processed {} frames total", total_frames);
        
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
                Ok(detections.into_iter().filter(|d| d.score > 0.5).collect())
            }
            _ => Ok(Vec::new()),
        }
    }
}

pub fn run_detection_demo(output_path: Option<&str>) -> Result<()> {
    // Use sample video file
    let mut config = AppConfig::default();
    config.pipeline.video_source = "assets/sample.mp4".to_string();
    config.inference.model_path = "models/yolov8n.onnx".into();

    let output_file = output_path.unwrap_or("assets/detections.txt");
    
    println!("Starting YOLO detection demo");
    println!("Input: {}", config.pipeline.video_source);
    println!("Output: {}", output_file);
    
    let processor = DetectionProcessor::new(&config, output_file)?;
    processor.run()?;
    
    println!("Detection results saved to: {}", output_file);
    
    Ok(())
}