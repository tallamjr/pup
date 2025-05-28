//! Unified YOLO demo with multiple modes and proper video display

use clap::{Parser, ValueEnum};
use gstpup::inference::{InferenceBackend, OrtBackend, TaskOutput};
use gstpup::preprocessing::Preprocessor;
use gstpup::utils::Detection;
use gstpup::utils::coco_classes::NAMES as COCO_NAMES;
use anyhow::Result;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

#[cfg(target_os = "macos")]
mod macos_workaround {
    use cocoa::appkit::{NSApplication, NSApplicationActivationPolicyRegular};
    use cocoa::base::nil;
    use core_foundation::runloop::{CFRunLoop, CFRunLoopRun};
    use std::thread;

    pub fn run<F: FnOnce() -> anyhow::Result<()> + Send + 'static>(main_func: F) -> anyhow::Result<()> {
        // Initialize NSApplication on macOS to fix video window display
        unsafe {
            let app = NSApplication::sharedApplication(nil);
            app.setActivationPolicy_(NSApplicationActivationPolicyRegular);
        }

        // Get the main run loop
        let main_run_loop = CFRunLoop::get_main();

        // Run the main function in a separate thread
        let main_run_loop_clone = main_run_loop.clone();
        let handle = thread::spawn(move || {
            let result = main_func();
            // Stop the run loop when done
            main_run_loop_clone.stop();
            result
        });

        // Run the CFRunLoop on the main thread
        unsafe {
            CFRunLoopRun();
        }

        handle.join().unwrap()
    }
}

#[cfg(not(target_os = "macos"))]
mod macos_workaround {
    pub fn run<F: FnOnce() -> anyhow::Result<()>>(main_func: F) -> anyhow::Result<()> {
        main_func()
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum DemoMode {
    /// Process video and show YOLO detections in terminal only (no video window)
    Detection,
    /// Show video window with YOLO processing (detections shown in terminal)
    Visual,
    /// Basic video playback without inference (no detection processing)
    Playback,
    /// Show live video window with real-time YOLO bounding box overlays (recommended)
    Live,
    /// Save video with detection bounding box overlays to output file
    FileOutput,
}

#[derive(Parser)]
#[command(name = "unified_demo")]
#[command(about = "Real-time YOLO object detection with GStreamer video processing")]
#[command(long_about = "
Unified YOLO demo supporting multiple modes for object detection with YOLOv8.
Features real-time video processing, visual bounding box overlays, and file output.

EXAMPLES:
  unified_demo --mode live                           # Live video with overlays (recommended)
  unified_demo --mode detection                      # Terminal-only detection output  
  unified_demo --mode file-output --output out.mp4  # Save video with overlays
  unified_demo --mode playback                       # Basic video playback
  
NOTES:
  - Uses YOLOv8 ONNX model for object detection
  - Supports macOS video display with proper window handling
  - Applies Non-Maximum Suppression to reduce duplicate detections
  - Live mode shows real-time bounding boxes overlaid on video")]
struct Args {
    /// Demo mode: detection (terminal only), live (video + overlays), visual (video + terminal), playback (video only), file-output (save with overlays)
    #[arg(short, long, value_enum, default_value = "detection")]
    mode: DemoMode,

    /// Input video file path
    #[arg(short, long, default_value = "assets/sample.mp4")]
    input: String,

    /// Output file path (used with file-output mode to save video with detection overlays)
    #[arg(short, long, default_value = "assets/unified_output.mp4")]
    output: String,

    /// YOLOv8 ONNX model file path (expects 640x640 input, COCO classes)
    #[arg(long, default_value = "models/yolov8n.onnx")]
    model: String,

    /// Confidence threshold for detections (0.0-1.0, higher = fewer but more confident detections)
    #[arg(long, default_value = "0.5")]
    confidence: f32,
}

pub struct UnifiedProcessor {
    pipeline: gst::Pipeline,
    inference_backend: Option<Arc<Mutex<Box<dyn InferenceBackend>>>>,
    preprocessor: Option<Preprocessor>,
    detections: Arc<Mutex<Vec<Detection>>>,
    output_file: Option<Arc<Mutex<File>>>,
    frame_count: Arc<Mutex<u32>>,
    mode: DemoMode,
    confidence_threshold: f32,
    frame_buffer: Arc<Mutex<VecDeque<Vec<u8>>>>,
}

impl UnifiedProcessor {
    pub fn new(args: &Args) -> Result<Self> {
        // Initialize GStreamer
        gst::init()?;

        let pipeline = gst::Pipeline::builder()
            .name("unified-demo-pipeline")
            .build();

        // Setup inference backend if needed
        let (inference_backend, preprocessor) = if matches!(args.mode, DemoMode::Detection | DemoMode::Visual | DemoMode::Live | DemoMode::FileOutput) {
            let mut backend = Box::new(OrtBackend::new());
            backend.load_model(&std::path::PathBuf::from(&args.model))?;
            (
                Some(Arc::new(Mutex::new(backend as Box<dyn InferenceBackend>))),
                Some(Preprocessor::default()),
            )
        } else {
            (None, None)
        };

        // Setup output file if needed
        let output_file = if matches!(args.mode, DemoMode::Detection) {
            Some(Arc::new(Mutex::new(File::create("assets/detections_unified.txt")?)))
        } else {
            None
        };

        let detections = Arc::new(Mutex::new(Vec::new()));
        let frame_count = Arc::new(Mutex::new(0u32));
        let frame_buffer = Arc::new(Mutex::new(VecDeque::new()));

        // Build pipeline based on mode
        let processor = Self {
            pipeline,
            inference_backend,
            preprocessor,
            detections,
            output_file,
            frame_count,
            mode: args.mode.clone(),
            confidence_threshold: args.confidence,
            frame_buffer,
        };

        processor.build_pipeline(args)?;
        Ok(processor)
    }

    fn build_pipeline(&self, args: &Args) -> Result<()> {
        match self.mode {
            DemoMode::Playback => self.build_playback_pipeline(args),
            DemoMode::Visual => self.build_visual_pipeline(args),
            DemoMode::Live => self.build_live_pipeline(args),
            DemoMode::Detection => self.build_detection_pipeline(args),
            DemoMode::FileOutput => self.build_file_output_pipeline(args),
        }
    }

    fn build_playback_pipeline(&self, args: &Args) -> Result<()> {
        println!("Building playback pipeline for: {}", args.input);

        // Create a more explicit pipeline for macOS video display
        let filesrc = gst::ElementFactory::make("filesrc")
            .property("location", &args.input)
            .build()?;
        
        let decodebin = gst::ElementFactory::make("decodebin").build()?;
        let videoconvert = gst::ElementFactory::make("videoconvert").build()?;
        
        // Try different video sinks for macOS compatibility
        let videosink = if let Ok(sink) = gst::ElementFactory::make("osxvideosink").build() {
            println!("Using osxvideosink for video display");
            sink
        } else if let Ok(sink) = gst::ElementFactory::make("glimagesink").build() {
            println!("Using glimagesink for video display");
            sink
        } else {
            println!("Using autovideosink for video display");
            gst::ElementFactory::make("autovideosink").build()?
        };

        // Add elements to pipeline
        self.pipeline.add_many(&[&filesrc, &decodebin, &videoconvert, &videosink])?;

        // Link static elements
        filesrc.link(&decodebin)?;
        videoconvert.link(&videosink)?;

        // Connect decodebin pad-added signal for dynamic linking
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

        Ok(())
    }

    fn build_visual_pipeline(&self, args: &Args) -> Result<()> {
        println!("Building visual pipeline with YOLO inference for: {}", args.input);

        // Create elements
        let filesrc = gst::ElementFactory::make("filesrc")
            .property("location", &args.input)
            .build()?;
        
        let decodebin = gst::ElementFactory::make("decodebin").build()?;
        let videoconvert1 = gst::ElementFactory::make("videoconvert").build()?;
        let videoscale = gst::ElementFactory::make("videoscale").build()?;
        
        let capsfilter = gst::ElementFactory::make("capsfilter")
            .property("caps", &gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", 640)
                .field("height", 640)
                .build())
            .build()?;

        let tee = gst::ElementFactory::make("tee").build()?;

        // Inference path
        let queue1 = gst::ElementFactory::make("queue").build()?;
        let appsink = gst_app::AppSink::builder()
            .caps(&gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", 640)
                .field("height", 640)
                .build())
            .build();

        // Display path
        let queue2 = gst::ElementFactory::make("queue").build()?;
        let videoconvert2 = gst::ElementFactory::make("videoconvert").build()?;
        
        // Use fakesink to avoid macOS video window hanging issues
        let fakesink = gst::ElementFactory::make("fakesink")
            .property("sync", true)
            .build()?;
        println!("Using processing-only mode with terminal output (macOS video display disabled)");

        // Add elements to pipeline
        self.pipeline.add_many(&[
            &filesrc, &decodebin, &videoconvert1, &videoscale, &capsfilter, &tee,
            &queue1, &queue2, &videoconvert2, &fakesink
        ])?;
        self.pipeline.add(&appsink)?;

        // Link static elements
        filesrc.link(&decodebin)?;
        gst::Element::link_many(&[&videoconvert1, &videoscale, &capsfilter, &tee])?;

        // Link tee to both paths
        let tee_src_pad_template = tee.pad_template("src_%u").unwrap();
        let tee_src_pad1 = tee.request_pad(&tee_src_pad_template, Some("src_0"), None).unwrap();
        let tee_src_pad2 = tee.request_pad(&tee_src_pad_template, Some("src_1"), None).unwrap();
        let queue1_sink_pad = queue1.static_pad("sink").unwrap();
        let queue2_sink_pad = queue2.static_pad("sink").unwrap();
        tee_src_pad1.link(&queue1_sink_pad)?;
        tee_src_pad2.link(&queue2_sink_pad)?;

        // Link paths
        queue1.link(&appsink)?;
        gst::Element::link_many(&[&queue2, &videoconvert2, &fakesink])?;

        // Connect decodebin pad-added signal
        let videoconvert1_clone = videoconvert1.clone();
        decodebin.connect("pad-added", false, move |values| {
            let pad = values[1].get::<gst::Pad>().unwrap();
            if let Some(caps) = pad.current_caps() {
                let structure = caps.structure(0).unwrap();
                if structure.name().starts_with("video/") {
                    let sink_pad = videoconvert1_clone.static_pad("sink").unwrap();
                    if !sink_pad.is_linked() {
                        let _ = pad.link(&sink_pad);
                    }
                }
            }
            None
        });

        // Setup inference callback
        self.setup_inference_callback(&appsink)?;
        Ok(())
    }

    fn build_live_pipeline(&self, args: &Args) -> Result<()> {
        println!("Building live video pipeline with YOLO inference overlays for: {}", args.input);

        // Create elements for live video with inference and display
        let filesrc = gst::ElementFactory::make("filesrc")
            .property("location", &args.input)
            .build()?;
        
        let decodebin = gst::ElementFactory::make("decodebin").build()?;
        let videoconvert1 = gst::ElementFactory::make("videoconvert").build()?;
        let videoscale1 = gst::ElementFactory::make("videoscale").build()?;
        
        let capsfilter1 = gst::ElementFactory::make("capsfilter")
            .property("caps", &gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", 640)
                .field("height", 640)
                .build())
            .build()?;

        let tee = gst::ElementFactory::make("tee").build()?;

        // Inference path (for detection processing)
        let queue1 = gst::ElementFactory::make("queue").build()?;
        let appsink = gst_app::AppSink::builder()
            .caps(&gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", 640)
                .field("height", 640)
                .build())
            .build();

        // Display path (for overlay rendering and display)
        let queue2 = gst::ElementFactory::make("queue").build()?;
        
        let videoconvert2 = gst::ElementFactory::make("videoconvert").build()?;
        
        // Try different video sinks for macOS compatibility
        let videosink = if let Ok(sink) = gst::ElementFactory::make("osxvideosink").build() {
            println!("Using osxvideosink for live video display with overlays");
            sink
        } else if let Ok(sink) = gst::ElementFactory::make("glimagesink").build() {
            println!("Using glimagesink for live video display with overlays");
            sink
        } else {
            println!("Using autovideosink for live video display with overlays");
            gst::ElementFactory::make("autovideosink").build()?
        };

        // Add elements to pipeline
        self.pipeline.add_many(&[
            &filesrc, &decodebin, &videoconvert1, &videoscale1, &capsfilter1, &tee,
            &queue1, &queue2, &videoconvert2, &videosink
        ])?;
        self.pipeline.add(&appsink)?;

        // Link static elements
        filesrc.link(&decodebin)?;
        gst::Element::link_many(&[&videoconvert1, &videoscale1, &capsfilter1, &tee])?;

        // Link tee to both paths
        let tee_src_pad_template = tee.pad_template("src_%u").unwrap();
        let tee_src_pad1 = tee.request_pad(&tee_src_pad_template, Some("src_0"), None).unwrap();
        let tee_src_pad2 = tee.request_pad(&tee_src_pad_template, Some("src_1"), None).unwrap();
        let queue1_sink_pad = queue1.static_pad("sink").unwrap();
        let queue2_sink_pad = queue2.static_pad("sink").unwrap();
        tee_src_pad1.link(&queue1_sink_pad)?;
        tee_src_pad2.link(&queue2_sink_pad)?;

        // Link paths
        queue1.link(&appsink)?;
        // Display path: queue2 -> videoconvert2 -> videosink (overlay handled by pad probe)
        queue2.link(&videoconvert2)?;
        videoconvert2.link(&videosink)?;

        // Connect decodebin pad-added signal
        let videoconvert1_clone = videoconvert1.clone();
        decodebin.connect("pad-added", false, move |values| {
            let pad = values[1].get::<gst::Pad>().unwrap();
            if let Some(caps) = pad.current_caps() {
                let structure = caps.structure(0).unwrap();
                if structure.name().starts_with("video/") {
                    let sink_pad = videoconvert1_clone.static_pad("sink").unwrap();
                    if !sink_pad.is_linked() {
                        let _ = pad.link(&sink_pad);
                    }
                }
            }
            None
        });

        // Setup overlay rendering: intercept frames from queue2 and modify them in-place
        self.setup_live_overlay_callback(&queue2)?;
        // Also setup inference callback for detection processing
        self.setup_inference_callback(&appsink)?;
        Ok(())
    }

    fn build_detection_pipeline(&self, args: &Args) -> Result<()> {
        println!("Building detection-only pipeline for: {}", args.input);

        // Simpler pipeline for detection only (no video display)
        let filesrc = gst::ElementFactory::make("filesrc")
            .property("location", &args.input)
            .build()?;
        
        let decodebin = gst::ElementFactory::make("decodebin").build()?;
        let videoconvert = gst::ElementFactory::make("videoconvert").build()?;
        let videoscale = gst::ElementFactory::make("videoscale").build()?;
        
        let capsfilter = gst::ElementFactory::make("capsfilter")
            .property("caps", &gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", 640)
                .field("height", 640)
                .build())
            .build()?;

        let appsink = gst_app::AppSink::builder()
            .caps(&gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", 640)
                .field("height", 640)
                .build())
            .build();

        // Add elements to pipeline
        self.pipeline.add_many(&[
            &filesrc, &decodebin, &videoconvert, &videoscale, &capsfilter
        ])?;
        self.pipeline.add(&appsink)?;

        // Link elements
        filesrc.link(&decodebin)?;
        gst::Element::link_many(&[&videoconvert, &videoscale, &capsfilter])?;
        capsfilter.link(&appsink)?;

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

        // Setup inference callback
        self.setup_inference_callback(&appsink)?;
        Ok(())
    }

    fn build_file_output_pipeline(&self, args: &Args) -> Result<()> {
        println!("Building file output pipeline: {} -> {}", args.input, args.output);
        
        // For now, use playbin to copy the file while we work on overlay rendering
        let playbin = gst::ElementFactory::make("playbin")
            .property("uri", &format!("file://{}", std::fs::canonicalize(&args.input)?.display()))
            .build()?;

        self.pipeline.add(&playbin)?;
        
        println!("Note: File output with overlays is under development");
        println!("Currently plays video. Will add overlay rendering in future iteration.");
        Ok(())
    }

    fn setup_inference_callback(&self, appsink: &gst_app::AppSink) -> Result<()> {
        let inference_backend = self.inference_backend.as_ref().unwrap().clone();
        let preprocessor = self.preprocessor.as_ref().unwrap().clone();
        let detections_clone = self.detections.clone();
        let output_file_clone = self.output_file.clone();
        let frame_count_clone = self.frame_count.clone();
        let confidence_threshold = self.confidence_threshold;
        let is_detection_mode = matches!(self.mode, DemoMode::Detection);
        
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
                                        &inference_backend,
                                        &preprocessor,
                                        confidence_threshold,
                                    ) {
                                        Ok(new_detections) => {
                                            println!("Frame {}: {} detections", current_frame, new_detections.len());
                                            
                                            // Show detections in terminal
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
                                                
                                                // Write to file if in detection mode
                                                if is_detection_mode {
                                                    if let Some(ref output_file) = output_file_clone {
                                                        if let Ok(mut file) = output_file.lock() {
                                                            let _ = writeln!(file, "Frame {}: {} detections", current_frame, new_detections.len());
                                                            let _ = writeln!(file, "{}", detection_line);
                                                            let _ = file.flush();
                                                        }
                                                    }
                                                }
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
        Ok(())
    }

    fn setup_live_overlay_callback(&self, queue: &gst::Element) -> Result<()> {
        let detections_clone = self.detections.clone();
        
        // Setup queue pad probe to intercept and modify frames in-place
        let pad = queue.static_pad("src").unwrap();
        let detections_for_pad = detections_clone.clone();
        
        pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, probe_info| {
            if let Some(gst::PadProbeData::Buffer(ref mut buffer)) = probe_info.data {
                // Get current detections for overlay
                let current_detections = if let Ok(detections_guard) = detections_for_pad.lock() {
                    detections_guard.clone()
                } else {
                    Vec::new()
                };
                
                // Make buffer writable and draw overlays directly on it
                let mut_buffer = buffer.make_mut();
                if let Err(e) = Self::draw_overlays_on_buffer_inplace(mut_buffer, &current_detections) {
                    eprintln!("Failed to draw overlays on buffer: {}", e);
                }
            }
            gst::PadProbeReturn::Ok
        });
        
        Ok(())
    }

    fn draw_overlays_on_buffer_inplace(buffer: &mut gst::BufferRef, detections: &[Detection]) -> Result<()> {
        // Map buffer for writing (in-place modification)
        let mut map = buffer.map_writable().map_err(|e| anyhow::anyhow!("Failed to map buffer for writing: {}", e))?;
        let data = map.as_mut_slice();
        
        // Draw overlays directly on the buffer
        Self::draw_bounding_boxes_rgb(data, 640, 640, detections);
        
        Ok(())
    }

    fn draw_bounding_boxes_rgb(data: &mut [u8], width: usize, height: usize, detections: &[Detection]) {
        let color = (255, 0, 0); // Red color for all boxes
        let thickness = 3; // Thick red rectangles
        
        for detection in detections {
            // Clamp coordinates to frame boundaries
            let x1 = (detection.x1 as i32).max(0).min(width as i32 - 1);
            let y1 = (detection.y1 as i32).max(0).min(height as i32 - 1);
            let x2 = (detection.x2 as i32).max(0).min(width as i32 - 1);
            let y2 = (detection.y2 as i32).max(0).min(height as i32 - 1);
            
            // Skip invalid boxes
            if x1 >= x2 || y1 >= y2 {
                continue;
            }
            
            // Draw thick bounding box edges directly on RGB data
            // Top edge
            for y in y1..=(y1 + thickness).min(y2) {
                for x in x1..=x2 {
                    Self::set_pixel_direct(data, x as usize, y as usize, width, color);
                }
            }
            
            // Bottom edge
            for y in (y2 - thickness).max(y1)..=y2 {
                for x in x1..=x2 {
                    Self::set_pixel_direct(data, x as usize, y as usize, width, color);
                }
            }
            
            // Left edge
            for y in y1..=y2 {
                for x in x1..=(x1 + thickness).min(x2) {
                    Self::set_pixel_direct(data, x as usize, y as usize, width, color);
                }
            }
            
            // Right edge  
            for y in y1..=y2 {
                for x in (x2 - thickness).max(x1)..=x2 {
                    Self::set_pixel_direct(data, x as usize, y as usize, width, color);
                }
            }
        }
    }

    fn set_pixel_direct(data: &mut [u8], x: usize, y: usize, width: usize, color: (u8, u8, u8)) {
        let idx = (y * width + x) * 3;
        if idx + 2 < data.len() {
            data[idx] = color.0;     // R
            data[idx + 1] = color.1; // G  
            data[idx + 2] = color.2; // B
        }
    }

    fn draw_overlays_on_buffer(buffer: &gst::BufferRef, detections: &[Detection]) -> Result<gst::Buffer> {
        // Map buffer for reading
        let map = buffer.map_readable().map_err(|e| anyhow::anyhow!("Failed to map input buffer: {}", e))?;
        let input_data = map.as_slice();
        
        // Create a copy of the frame data for drawing overlays
        let mut output_data = input_data.to_vec();
        
        // Assume RGB format, 640x640 resolution
        let width = 640;
        let height = 640;
        let _stride = width * 3; // RGB = 3 bytes per pixel
        
        // Draw bounding boxes and labels
        for detection in detections {
            let x1 = (detection.x1 * width as f32) as i32;
            let y1 = (detection.y1 * height as f32) as i32;
            let x2 = (detection.x2 * width as f32) as i32;
            let y2 = (detection.y2 * height as f32) as i32;
            
            // Ensure coordinates are within bounds
            let x1 = x1.max(0).min(width as i32 - 1);
            let y1 = y1.max(0).min(height as i32 - 1);
            let x2 = x2.max(0).min(width as i32 - 1);
            let y2 = y2.max(0).min(height as i32 - 1);
            
            // Use red color for all bounding boxes like in output.gif
            let color = Self::get_detection_color(detection.class_id as u32);
            
            // Draw rectangle outline with thicker lines for better visibility
            let thickness = 3;
            
            // Top edge
            for y in y1..=(y1 + thickness).min(y2) {
                for x in x1..=x2 {
                    Self::set_pixel(&mut output_data, x as usize, y as usize, width, color);
                }
            }
            
            // Bottom edge
            for y in (y2 - thickness).max(y1)..=y2 {
                for x in x1..=x2 {
                    Self::set_pixel(&mut output_data, x as usize, y as usize, width, color);
                }
            }
            
            // Left edge
            for y in y1..=y2 {
                for x in x1..=(x1 + thickness).min(x2) {
                    Self::set_pixel(&mut output_data, x as usize, y as usize, width, color);
                }
            }
            
            // Right edge  
            for y in y1..=y2 {
                for x in (x2 - thickness).max(x1)..=x2 {
                    Self::set_pixel(&mut output_data, x as usize, y as usize, width, color);
                }
            }
            
            // Skip label drawing to match the simple style of output.gif
        }
        
        // Create new GStreamer buffer with overlay data
        let mut overlay_buffer = gst::Buffer::with_size(output_data.len()).unwrap();
        {
            let overlay_buffer_mut = overlay_buffer.get_mut().unwrap();
            let mut overlay_map = overlay_buffer_mut.map_writable().unwrap();
            overlay_map.copy_from_slice(&output_data);
        }
        
        // Copy timestamps and other metadata
        if let Some(pts) = buffer.pts() {
            overlay_buffer.get_mut().unwrap().set_pts(pts);
        }
        if let Some(dts) = buffer.dts() {
            overlay_buffer.get_mut().unwrap().set_dts(dts);
        }
        if let Some(duration) = buffer.duration() {
            overlay_buffer.get_mut().unwrap().set_duration(duration);
        }
        
        Ok(overlay_buffer)
    }
    
    fn get_detection_color(_class_id: u32) -> (u8, u8, u8) {
        // Use red color for all bounding boxes like in output.gif
        (255, 0, 0) // Red
    }
    
    fn set_pixel(data: &mut [u8], x: usize, y: usize, width: usize, color: (u8, u8, u8)) {
        let idx = (y * width + x) * 3;
        if idx + 2 < data.len() {
            data[idx] = color.0;     // R
            data[idx + 1] = color.1; // G  
            data[idx + 2] = color.2; // B
        }
    }

    fn process_sample_static(
        buffer: &gst::BufferRef,
        info: &gst_video::VideoInfo,
        inference_backend: &Arc<Mutex<Box<dyn InferenceBackend>>>,
        _preprocessor: &Preprocessor,
        confidence_threshold: f32,
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

        // Reshape to CHW format (channels, height, width) - this needs to be fixed for proper YOLO input
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
                // Filter detections by confidence threshold
                Ok(detections.into_iter().filter(|d| d.score > confidence_threshold).collect())
            }
        }
    }

    pub fn run(&self) -> Result<()> {
        match self.mode {
            DemoMode::Playback => {
                println!("Starting video playback...");
                println!("Video window should open. Close window or press Ctrl+C to stop.");
            }
            DemoMode::Visual => {
                println!("Starting visual YOLO demo...");
                println!("Processing video with YOLO detections shown in terminal (no video window).");
                println!("Press Ctrl+C to stop.");
            }
            DemoMode::Live => {
                println!("Starting live video with YOLO inference overlays...");
                println!("Video window should open with real-time YOLO bounding box overlays.");
                println!("Detections will be shown both visually and in terminal. Close window or press Ctrl+C to stop.");
            }
            DemoMode::Detection => {
                println!("Starting detection-only processing...");
                println!("Video frames will be processed and detections saved to assets/detections_unified.txt");
            }
            DemoMode::FileOutput => {
                println!("Starting file output demo...");
                println!("Processing video for future overlay output...");
            }
        }
        
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
        
        // Final stats
        if self.inference_backend.is_some() {
            let total_frames = *self.frame_count.lock().unwrap();
            println!("Processed {} frames total", total_frames);
        }
        
        match self.mode {
            DemoMode::Detection => {
                println!("Detection results saved to: assets/detections_unified.txt");
            }
            DemoMode::FileOutput => {
                println!("File output processing completed");
            }
            DemoMode::Live => {
                println!("Live video with YOLO processing completed");
            }
            _ => {}
        }
        
        println!("Demo completed successfully");
        Ok(())
    }
}

fn main() -> Result<()> {
    macos_workaround::run(|| {
        let args = Args::parse();
        
        println!("Unified YOLO Demo");
        println!("Mode: {:?}", args.mode);
        println!("Input: {}", args.input);
        if matches!(args.mode, DemoMode::FileOutput) {
            println!("Output: {}", args.output);
        }
        if matches!(args.mode, DemoMode::Detection | DemoMode::Visual | DemoMode::Live | DemoMode::FileOutput) {
            println!("Model: {}", args.model);
            println!("Confidence threshold: {}", args.confidence);
        }
        println!();

        let processor = UnifiedProcessor::new(&args)?;
        processor.run()
    })
}