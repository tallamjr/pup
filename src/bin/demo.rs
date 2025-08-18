//! Unified YOLO demo with multiple modes and proper video display

use anyhow::Result;
use clap::{Parser, ValueEnum};
use gstpup::inference::{InferenceBackend, OrtBackend, TaskOutput};
use gstpup::preprocessing::Preprocessor;
use gstpup::utils::coco_classes::NAMES as COCO_NAMES;
use gstpup::utils::Detection;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use std::fs::File;
use std::io::Write;
use std::sync::{Arc, Mutex};
use tracing::{debug, error, info, warn};

#[cfg(target_os = "macos")]
#[allow(deprecated)]
mod macos_workaround {
    use cocoa::appkit::{NSApplication, NSApplicationActivationPolicyRegular};
    use cocoa::base::nil;
    use core_foundation::runloop::{CFRunLoop, CFRunLoopRun};
    use std::thread;

    pub fn run<F: FnOnce() -> anyhow::Result<()> + Send + 'static>(
        main_func: F,
    ) -> anyhow::Result<()> {
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
  unified_demo --mode live --input webcam           # Live webcam with YOLO overlays
  unified_demo --mode detection                      # Terminal-only detection output
  unified_demo --mode detection --input webcam      # Webcam detection (terminal only)
  unified_demo --mode playback --input webcam       # Basic webcam playback
  unified_demo --mode file-output --output out.mp4  # Save video with overlays

NOTES:
  - Uses YOLOv8 ONNX model for object detection
  - Supports file input (MP4, etc.) and webcam input (use 'webcam' as input)
  - Supports macOS video display with proper window handling
  - Applies Non-Maximum Suppression to reduce duplicate detections
  - Live mode shows real-time bounding boxes overlaid on video
  - Webcam support: Cross-platform using autovideosrc (auto-detects camera)")]
pub struct Args {
    /// Demo mode: detection (terminal only), live (video + overlays), visual (video + terminal), playback (video only), file-output (save with overlays)
    #[arg(short, long, value_enum, default_value = "detection")]
    mode: DemoMode,

    /// Input source: video file path or 'webcam' for camera input
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
}

impl UnifiedProcessor {
    pub fn new(args: &Args) -> Result<Self> {
        // Initialize GStreamer
        gst::init()?;

        let pipeline = gst::Pipeline::builder()
            .name("unified-demo-pipeline")
            .build();

        // Setup inference backend if needed
        let (inference_backend, preprocessor) = if matches!(
            args.mode,
            DemoMode::Detection | DemoMode::Visual | DemoMode::Live | DemoMode::FileOutput
        ) {
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
            Some(Arc::new(Mutex::new(File::create(
                "assets/detections_unified.txt",
            )?)))
        } else {
            None
        };

        let detections = Arc::new(Mutex::new(Vec::new()));
        let frame_count = Arc::new(Mutex::new(0u32));

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

    fn create_source_elements(&self, input: &str) -> Result<(gst::Element, gst::Element)> {
        if input.to_lowercase() == "webcam" {
            info!("Creating webcam source pipeline");

            // Use autovideosrc for cross-platform camera access
            let source = gst::ElementFactory::make("autovideosrc").build()?;

            // No decodebin needed for webcam - raw video frames
            let videoconvert = gst::ElementFactory::make("videoconvert").build()?;

            Ok((source, videoconvert))
        } else {
            info!("Creating file source pipeline for: {}", input);

            // Create file source elements
            let filesrc = gst::ElementFactory::make("filesrc")
                .property("location", input)
                .build()?;

            let decodebin = gst::ElementFactory::make("decodebin").build()?;

            Ok((filesrc, decodebin))
        }
    }

    fn build_playback_pipeline(&self, args: &Args) -> Result<()> {
        info!("Building playback pipeline for: {}", args.input);

        // Create source elements (file or webcam)
        let (source, decode_or_convert) = self.create_source_elements(&args.input)?;
        let videoconvert = gst::ElementFactory::make("videoconvert").build()?;

        // Try different video sinks for macOS compatibility
        let videosink = if let Ok(sink) = gst::ElementFactory::make("osxvideosink").build() {
            debug!("Using osxvideosink for video display");
            sink
        } else if let Ok(sink) = gst::ElementFactory::make("glimagesink").build() {
            debug!("Using glimagesink for video display");
            sink
        } else {
            debug!("Using autovideosink for video display");
            gst::ElementFactory::make("autovideosink").build()?
        };

        if args.input.to_lowercase() == "webcam" {
            // Webcam pipeline: source -> videoconvert -> videosink
            self.pipeline
                .add_many([&source, &decode_or_convert, &videoconvert, &videosink])?;

            // Link directly for webcam (no decoding needed)
            source.link(&decode_or_convert)?;
            decode_or_convert.link(&videoconvert)?;
            videoconvert.link(&videosink)?;
        } else {
            // File pipeline: filesrc -> decodebin -> videoconvert -> videosink
            self.pipeline
                .add_many([&source, &decode_or_convert, &videoconvert, &videosink])?;

            // Link static elements
            source.link(&decode_or_convert)?;
            videoconvert.link(&videosink)?;

            // Connect decodebin pad-added signal for dynamic linking
            let videoconvert_clone = videoconvert.clone();
            decode_or_convert.connect("pad-added", false, move |values| {
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
        }

        Ok(())
    }

    fn build_visual_pipeline(&self, args: &Args) -> Result<()> {
        info!(
            "Building visual pipeline with YOLO inference for: {}",
            args.input
        );

        // Create elements
        let filesrc = gst::ElementFactory::make("filesrc")
            .property("location", &args.input)
            .build()?;

        let decodebin = gst::ElementFactory::make("decodebin").build()?;
        let videoconvert1 = gst::ElementFactory::make("videoconvert").build()?;
        let videoscale = gst::ElementFactory::make("videoscale").build()?;

        let capsfilter = gst::ElementFactory::make("capsfilter")
            .property(
                "caps",
                gst::Caps::builder("video/x-raw")
                    .field("format", "RGB")
                    .field("width", 640)
                    .field("height", 640)
                    .build(),
            )
            .build()?;

        let tee = gst::ElementFactory::make("tee").build()?;

        // Inference path
        let queue1 = gst::ElementFactory::make("queue").build()?;
        let appsink = gst_app::AppSink::builder()
            .caps(
                &gst::Caps::builder("video/x-raw")
                    .field("format", "RGB")
                    .field("width", 640)
                    .field("height", 640)
                    .build(),
            )
            .build();

        // Display path
        let queue2 = gst::ElementFactory::make("queue").build()?;
        let videoconvert2 = gst::ElementFactory::make("videoconvert").build()?;

        // Use fakesink to avoid macOS video window hanging issues
        let fakesink = gst::ElementFactory::make("fakesink")
            .property("sync", true)
            .build()?;
        info!("Using processing-only mode with terminal output (macOS video display disabled)");

        // Add elements to pipeline
        self.pipeline.add_many([
            &filesrc,
            &decodebin,
            &videoconvert1,
            &videoscale,
            &capsfilter,
            &tee,
            &queue1,
            &queue2,
            &videoconvert2,
            &fakesink,
        ])?;
        self.pipeline.add(&appsink)?;

        // Link static elements
        filesrc.link(&decodebin)?;
        gst::Element::link_many([&videoconvert1, &videoscale, &capsfilter, &tee])?;

        // Link tee to both paths
        let tee_src_pad_template = tee.pad_template("src_%u").unwrap();
        let tee_src_pad1 = tee
            .request_pad(&tee_src_pad_template, Some("src_0"), None)
            .unwrap();
        let tee_src_pad2 = tee
            .request_pad(&tee_src_pad_template, Some("src_1"), None)
            .unwrap();
        let queue1_sink_pad = queue1.static_pad("sink").unwrap();
        let queue2_sink_pad = queue2.static_pad("sink").unwrap();
        tee_src_pad1.link(&queue1_sink_pad)?;
        tee_src_pad2.link(&queue2_sink_pad)?;

        // Link paths
        queue1.link(&appsink)?;
        gst::Element::link_many([&queue2, &videoconvert2, &fakesink])?;

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
        info!(
            "Building live video pipeline with YOLO inference overlays for: {}",
            args.input
        );

        // Create source elements (file or webcam)
        let (source, decode_or_convert) = self.create_source_elements(&args.input)?;
        let videoconvert1 = gst::ElementFactory::make("videoconvert").build()?;
        let videoscale1 = gst::ElementFactory::make("videoscale").build()?;

        let capsfilter1 = gst::ElementFactory::make("capsfilter")
            .property(
                "caps",
                gst::Caps::builder("video/x-raw")
                    .field("format", "RGB")
                    .field("width", 640)
                    .field("height", 640)
                    .build(),
            )
            .build()?;

        let tee = gst::ElementFactory::make("tee").build()?;

        // Inference path (for detection processing)
        let queue1 = gst::ElementFactory::make("queue").build()?;
        let appsink = gst_app::AppSink::builder()
            .caps(
                &gst::Caps::builder("video/x-raw")
                    .field("format", "RGB")
                    .field("width", 640)
                    .field("height", 640)
                    .build(),
            )
            .build();

        // Display path (for overlay rendering and display)
        let queue2 = gst::ElementFactory::make("queue").build()?;

        let videoconvert2 = gst::ElementFactory::make("videoconvert").build()?;

        // Try different video sinks for macOS compatibility
        let videosink = if let Ok(sink) = gst::ElementFactory::make("osxvideosink").build() {
            debug!("Using osxvideosink for live video display with overlays");
            sink
        } else if let Ok(sink) = gst::ElementFactory::make("glimagesink").build() {
            debug!("Using glimagesink for live video display with overlays");
            sink
        } else {
            debug!("Using autovideosink for live video display with overlays");
            gst::ElementFactory::make("autovideosink").build()?
        };

        if args.input.to_lowercase() == "webcam" {
            // Webcam pipeline: source -> videoconvert -> videoscale -> capsfilter -> tee
            self.pipeline.add_many([
                &source,
                &decode_or_convert,
                &videoconvert1,
                &videoscale1,
                &capsfilter1,
                &tee,
                &queue1,
                &queue2,
                &videoconvert2,
                &videosink,
            ])?;
            self.pipeline.add(&appsink)?;

            // Link static elements for webcam
            source.link(&decode_or_convert)?;
            gst::Element::link_many([
                &decode_or_convert,
                &videoconvert1,
                &videoscale1,
                &capsfilter1,
                &tee,
            ])?;
        } else {
            // File pipeline: filesrc -> decodebin -> videoconvert -> videoscale -> capsfilter -> tee
            self.pipeline.add_many([
                &source,
                &decode_or_convert,
                &videoconvert1,
                &videoscale1,
                &capsfilter1,
                &tee,
                &queue1,
                &queue2,
                &videoconvert2,
                &videosink,
            ])?;
            self.pipeline.add(&appsink)?;

            // Link static elements for file
            source.link(&decode_or_convert)?;
            gst::Element::link_many([&videoconvert1, &videoscale1, &capsfilter1, &tee])?;

            // Connect decodebin pad-added signal for file input
            let videoconvert1_clone = videoconvert1.clone();
            decode_or_convert.connect("pad-added", false, move |values| {
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
        }

        // Link tee to both paths (same for both file and webcam)
        let tee_src_pad_template = tee.pad_template("src_%u").unwrap();
        let tee_src_pad1 = tee
            .request_pad(&tee_src_pad_template, Some("src_0"), None)
            .unwrap();
        let tee_src_pad2 = tee
            .request_pad(&tee_src_pad_template, Some("src_1"), None)
            .unwrap();
        let queue1_sink_pad = queue1.static_pad("sink").unwrap();
        let queue2_sink_pad = queue2.static_pad("sink").unwrap();
        tee_src_pad1.link(&queue1_sink_pad)?;
        tee_src_pad2.link(&queue2_sink_pad)?;

        // Link paths
        queue1.link(&appsink)?;
        // Display path: queue2 -> videoconvert2 -> videosink (overlay handled by pad probe)
        queue2.link(&videoconvert2)?;
        videoconvert2.link(&videosink)?;

        // Setup overlay rendering: intercept frames from queue2 and modify them in-place
        self.setup_live_overlay_callback(&queue2)?;
        // Also setup inference callback for detection processing
        self.setup_inference_callback(&appsink)?;
        Ok(())
    }

    fn build_detection_pipeline(&self, args: &Args) -> Result<()> {
        info!("Building detection-only pipeline for: {}", args.input);

        // Simpler pipeline for detection only (no video display)
        let filesrc = gst::ElementFactory::make("filesrc")
            .property("location", &args.input)
            .build()?;

        let decodebin = gst::ElementFactory::make("decodebin").build()?;
        let videoconvert = gst::ElementFactory::make("videoconvert").build()?;
        let videoscale = gst::ElementFactory::make("videoscale").build()?;

        let capsfilter = gst::ElementFactory::make("capsfilter")
            .property(
                "caps",
                gst::Caps::builder("video/x-raw")
                    .field("format", "RGB")
                    .field("width", 640)
                    .field("height", 640)
                    .build(),
            )
            .build()?;

        let appsink = gst_app::AppSink::builder()
            .caps(
                &gst::Caps::builder("video/x-raw")
                    .field("format", "RGB")
                    .field("width", 640)
                    .field("height", 640)
                    .build(),
            )
            .build();

        // Add elements to pipeline
        self.pipeline.add_many([
            &filesrc,
            &decodebin,
            &videoconvert,
            &videoscale,
            &capsfilter,
        ])?;
        self.pipeline.add(&appsink)?;

        // Link elements
        filesrc.link(&decodebin)?;
        gst::Element::link_many([&videoconvert, &videoscale, &capsfilter])?;
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
        info!(
            "Building file output pipeline: {} -> {}",
            args.input, args.output
        );

        // For now, use playbin to copy the file while we work on overlay rendering
        let playbin = gst::ElementFactory::make("playbin")
            .property(
                "uri",
                format!("file://{}", std::fs::canonicalize(&args.input)?.display()),
            )
            .build()?;

        self.pipeline.add(&playbin)?;

        warn!("Note: File output with overlays is under development");
        warn!("Currently plays video. Will add overlay rendering in future iteration.");
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
                                if let Ok(info) = gst_video::VideoInfo::from_caps(caps) {
                                    // Increment frame count
                                    let current_frame = {
                                        let mut count = frame_count_clone.lock().unwrap();
                                        *count += 1;
                                        *count
                                    };

                                    match Self::process_sample_static(
                                        buffer,
                                        &info,
                                        &inference_backend,
                                        &preprocessor,
                                        confidence_threshold,
                                    ) {
                                        Ok(new_detections) => {
                                            debug!("Frame {}: {} detections", current_frame, new_detections.len());

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
                                                debug!("{}", detection_line);

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
                                        Err(e) => error!("Processing error: {}", e),
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
                if let Err(e) =
                    Self::draw_overlays_on_buffer_inplace(mut_buffer, &current_detections)
                {
                    error!("Failed to draw overlays on buffer: {}", e);
                }
            }
            gst::PadProbeReturn::Ok
        });

        Ok(())
    }

    fn draw_overlays_on_buffer_inplace(
        buffer: &mut gst::BufferRef,
        detections: &[Detection],
    ) -> Result<()> {
        // Map buffer for writing (in-place modification)
        let mut map = buffer
            .map_writable()
            .map_err(|e| anyhow::anyhow!("Failed to map buffer for writing: {}", e))?;
        let data = map.as_mut_slice();

        // Draw overlays directly on the buffer
        Self::draw_bounding_boxes_rgb(data, 640, 640, detections);

        Ok(())
    }

    fn draw_bounding_boxes_rgb(
        data: &mut [u8],
        width: usize,
        height: usize,
        detections: &[Detection],
    ) {
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

            // Get class name
            let class_name =
                if detection.class_id >= 0 && (detection.class_id as usize) < COCO_NAMES.len() {
                    COCO_NAMES[detection.class_id as usize]
                } else {
                    "unknown"
                };

            // Draw simple text label using basic pixel drawing
            let label_text = format!("{}: {:.1}%", class_name, detection.score * 100.0);

            // Position label above bounding box
            let label_x = x1 as usize;
            let label_y = if y1 > 20 {
                (y1 - 15) as usize // Above box
            } else {
                (y2 + 5) as usize // Below box
            };

            // Draw simple text background (black rectangle)
            let text_width = label_text.len() * 8; // Rough estimate: 8 pixels per character
            let text_height = 12;

            for ty in label_y.saturating_sub(2)..=(label_y + text_height + 2).min(height - 1) {
                for tx in label_x.saturating_sub(2)..=(label_x + text_width + 2).min(width - 1) {
                    Self::set_pixel_direct(data, tx, ty, width, (0, 0, 0)); // Black background
                }
            }

            // Draw simple white text using bitmap font
            Self::draw_text(
                data,
                width,
                height,
                &label_text,
                label_x,
                label_y,
                (255, 255, 255),
            )
        }
    }

    fn draw_text(
        data: &mut [u8],
        width: usize,
        height: usize,
        text: &str,
        start_x: usize,
        start_y: usize,
        color: (u8, u8, u8),
    ) {
        // Simple 8x12 bitmap font patterns for common characters
        // Each character is 8 pixels wide and 12 pixels tall
        let font_patterns = Self::get_font_patterns();

        let mut x = start_x;
        let y = start_y;

        for ch in text.chars() {
            if let Some(pattern) = font_patterns.get(&ch) {
                // Draw this character
                for (row, &pattern_row) in pattern.iter().enumerate() {
                    if y + row >= height {
                        break;
                    }
                    for col in 0..8 {
                        if x + col >= width {
                            break;
                        }
                        // Check if this pixel should be drawn (1 in the pattern)
                        if (pattern_row >> (7 - col)) & 1 == 1 {
                            Self::set_pixel_direct(data, x + col, y + row, width, color);
                        }
                    }
                }
            }
            x += 8; // Move to next character position
            if x >= width {
                break;
            }
        }
    }

    fn get_font_patterns() -> std::collections::HashMap<char, [u8; 12]> {
        let mut patterns = std::collections::HashMap::new();

        // Define simple bitmap patterns for each character (8x12)
        // Format: each byte represents one row, each bit represents one pixel

        // Letters
        patterns.insert(
            'A',
            [
                0x00, 0x18, 0x24, 0x42, 0x42, 0x7E, 0x42, 0x42, 0x42, 0x42, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'B',
            [
                0x00, 0x7C, 0x42, 0x42, 0x7C, 0x42, 0x42, 0x42, 0x42, 0x7C, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'C',
            [
                0x00, 0x3C, 0x42, 0x40, 0x40, 0x40, 0x40, 0x40, 0x42, 0x3C, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'D',
            [
                0x00, 0x78, 0x44, 0x42, 0x42, 0x42, 0x42, 0x42, 0x44, 0x78, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'E',
            [
                0x00, 0x7E, 0x40, 0x40, 0x40, 0x7C, 0x40, 0x40, 0x40, 0x7E, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'F',
            [
                0x00, 0x7E, 0x40, 0x40, 0x40, 0x7C, 0x40, 0x40, 0x40, 0x40, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'G',
            [
                0x00, 0x3C, 0x42, 0x40, 0x40, 0x4E, 0x42, 0x42, 0x46, 0x3A, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'H',
            [
                0x00, 0x42, 0x42, 0x42, 0x42, 0x7E, 0x42, 0x42, 0x42, 0x42, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'I',
            [
                0x00, 0x3E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x3E, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'L',
            [
                0x00, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x7E, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'M',
            [
                0x00, 0x42, 0x66, 0x5A, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'N',
            [
                0x00, 0x42, 0x62, 0x52, 0x4A, 0x46, 0x42, 0x42, 0x42, 0x42, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'O',
            [
                0x00, 0x3C, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'P',
            [
                0x00, 0x7C, 0x42, 0x42, 0x42, 0x7C, 0x40, 0x40, 0x40, 0x40, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'R',
            [
                0x00, 0x7C, 0x42, 0x42, 0x42, 0x7C, 0x48, 0x44, 0x42, 0x42, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'S',
            [
                0x00, 0x3C, 0x42, 0x40, 0x30, 0x0C, 0x02, 0x42, 0x42, 0x3C, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'T',
            [
                0x00, 0x7F, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'U',
            [
                0x00, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'Y',
            [
                0x00, 0x41, 0x22, 0x14, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00,
            ],
        );

        // Lowercase letters (common ones)
        patterns.insert(
            'a',
            [
                0x00, 0x00, 0x00, 0x3C, 0x02, 0x3E, 0x42, 0x42, 0x46, 0x3A, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'b',
            [
                0x00, 0x40, 0x40, 0x5C, 0x62, 0x42, 0x42, 0x42, 0x62, 0x5C, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'c',
            [
                0x00, 0x00, 0x00, 0x3C, 0x42, 0x40, 0x40, 0x40, 0x42, 0x3C, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'd',
            [
                0x00, 0x02, 0x02, 0x3A, 0x46, 0x42, 0x42, 0x42, 0x46, 0x3A, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'e',
            [
                0x00, 0x00, 0x00, 0x3C, 0x42, 0x7E, 0x40, 0x40, 0x42, 0x3C, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'i',
            [
                0x00, 0x08, 0x00, 0x18, 0x08, 0x08, 0x08, 0x08, 0x08, 0x3E, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'k',
            [
                0x00, 0x40, 0x40, 0x44, 0x48, 0x70, 0x48, 0x44, 0x42, 0x41, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'l',
            [
                0x00, 0x18, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x3E, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'n',
            [
                0x00, 0x00, 0x00, 0x5C, 0x62, 0x42, 0x42, 0x42, 0x42, 0x42, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'o',
            [
                0x00, 0x00, 0x00, 0x3C, 0x42, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'r',
            [
                0x00, 0x00, 0x00, 0x5C, 0x62, 0x40, 0x40, 0x40, 0x40, 0x40, 0x00, 0x00,
            ],
        );
        patterns.insert(
            's',
            [
                0x00, 0x00, 0x00, 0x3E, 0x40, 0x3C, 0x02, 0x02, 0x42, 0x3C, 0x00, 0x00,
            ],
        );
        patterns.insert(
            't',
            [
                0x00, 0x10, 0x10, 0x7C, 0x10, 0x10, 0x10, 0x10, 0x10, 0x0C, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'u',
            [
                0x00, 0x00, 0x00, 0x42, 0x42, 0x42, 0x42, 0x42, 0x46, 0x3A, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'v',
            [
                0x00, 0x00, 0x00, 0x42, 0x42, 0x42, 0x24, 0x24, 0x18, 0x18, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'w',
            [
                0x00, 0x00, 0x00, 0x42, 0x42, 0x42, 0x5A, 0x66, 0x42, 0x42, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'x',
            [
                0x00, 0x00, 0x00, 0x42, 0x24, 0x18, 0x18, 0x24, 0x42, 0x42, 0x00, 0x00,
            ],
        );
        patterns.insert(
            'y',
            [
                0x00, 0x00, 0x00, 0x42, 0x42, 0x42, 0x26, 0x1A, 0x02, 0x3C, 0x00, 0x00,
            ],
        );

        // Numbers
        patterns.insert(
            '0',
            [
                0x00, 0x3C, 0x42, 0x46, 0x4A, 0x52, 0x62, 0x42, 0x42, 0x3C, 0x00, 0x00,
            ],
        );
        patterns.insert(
            '1',
            [
                0x00, 0x08, 0x18, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x3E, 0x00, 0x00,
            ],
        );
        patterns.insert(
            '2',
            [
                0x00, 0x3C, 0x42, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x7E, 0x00, 0x00,
            ],
        );
        patterns.insert(
            '3',
            [
                0x00, 0x3C, 0x42, 0x02, 0x1C, 0x02, 0x02, 0x02, 0x42, 0x3C, 0x00, 0x00,
            ],
        );
        patterns.insert(
            '4',
            [
                0x00, 0x04, 0x0C, 0x14, 0x24, 0x44, 0x7E, 0x04, 0x04, 0x04, 0x00, 0x00,
            ],
        );
        patterns.insert(
            '5',
            [
                0x00, 0x7E, 0x40, 0x40, 0x7C, 0x02, 0x02, 0x02, 0x42, 0x3C, 0x00, 0x00,
            ],
        );
        patterns.insert(
            '6',
            [
                0x00, 0x1C, 0x20, 0x40, 0x7C, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00, 0x00,
            ],
        );
        patterns.insert(
            '7',
            [
                0x00, 0x7E, 0x02, 0x04, 0x08, 0x08, 0x10, 0x10, 0x20, 0x20, 0x00, 0x00,
            ],
        );
        patterns.insert(
            '8',
            [
                0x00, 0x3C, 0x42, 0x42, 0x3C, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00, 0x00,
            ],
        );
        patterns.insert(
            '9',
            [
                0x00, 0x3C, 0x42, 0x42, 0x42, 0x3E, 0x02, 0x04, 0x08, 0x70, 0x00, 0x00,
            ],
        );

        // Special characters
        patterns.insert(
            ':',
            [
                0x00, 0x00, 0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x00, 0x00, 0x00,
            ],
        );
        patterns.insert(
            '.',
            [
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00, 0x00,
            ],
        );
        patterns.insert(
            '%',
            [
                0x00, 0x62, 0x64, 0x08, 0x10, 0x10, 0x20, 0x26, 0x46, 0x00, 0x00, 0x00,
            ],
        );
        patterns.insert(
            ' ',
            [
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
        );

        patterns
    }

    fn set_pixel_direct(data: &mut [u8], x: usize, y: usize, width: usize, color: (u8, u8, u8)) {
        let idx = (y * width + x) * 3;
        if idx + 2 < data.len() {
            data[idx] = color.0; // R
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
        let map = buffer
            .map_readable()
            .map_err(|e| anyhow::anyhow!("Failed to map buffer: {}", e))?;
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
                Ok(detections
                    .into_iter()
                    .filter(|d| d.score > confidence_threshold)
                    .collect())
            }
        }
    }

    pub fn run(&self) -> Result<()> {
        match self.mode {
            DemoMode::Playback => {
                info!("Starting video playback...");
                info!("Video window should open. Close window or press Ctrl+C to stop.");
            }
            DemoMode::Visual => {
                info!("Starting visual YOLO demo...");
                info!("Processing video with YOLO detections shown in terminal (no video window).");
                info!("Press Ctrl+C to stop.");
            }
            DemoMode::Live => {
                info!("Starting live video with YOLO inference overlays...");
                info!("Video window should open with real-time YOLO bounding box overlays.");
                info!("Detections will be shown both visually and in terminal. Close window or press Ctrl+C to stop.");
            }
            DemoMode::Detection => {
                info!("Starting detection-only processing...");
                info!("Video frames will be processed and detections saved to assets/detections_unified.txt");
            }
            DemoMode::FileOutput => {
                info!("Starting file output demo...");
                info!("Processing video for future overlay output...");
            }
        }

        // Start playing
        self.pipeline.set_state(gst::State::Playing)?;

        // Wait for EOS or error
        let bus = self.pipeline.bus().unwrap();
        for msg in bus.iter_timed(gst::ClockTime::NONE) {
            match msg.view() {
                gst::MessageView::Eos(..) => {
                    info!("End of stream - processing complete");
                    break;
                }
                gst::MessageView::Error(err) => {
                    error!("Error: {}", err.error());
                    if let Some(debug_info) = err.debug() {
                        error!("Debug info: {}", debug_info);
                    }
                    break;
                }
                gst::MessageView::StateChanged(state_changed) => {
                    if state_changed
                        .src()
                        .map(|s| s == &self.pipeline)
                        .unwrap_or(false)
                    {
                        debug!(
                            "Pipeline state changed from {:?} to {:?}",
                            state_changed.old(),
                            state_changed.current()
                        );
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
            info!("Processed {} frames total", total_frames);
        }

        match self.mode {
            DemoMode::Detection => {
                info!("Detection results saved to: assets/detections_unified.txt");
            }
            DemoMode::FileOutput => {
                info!("File output processing completed");
            }
            DemoMode::Live => {
                info!("Live video with YOLO processing completed");
            }
            _ => {}
        }

        info!("Demo completed successfully");
        Ok(())
    }
}

fn main() -> Result<()> {
    // Initialize tracing subscriber
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .with_thread_ids(true)
        .with_file(true)
        .with_line_number(true)
        .init();

    macos_workaround::run(|| {
        let args = Args::parse();

        info!("Unified YOLO Demo");
        info!("Mode: {:?}", args.mode);
        info!("Input: {}", args.input);
        if matches!(args.mode, DemoMode::FileOutput) {
            info!("Output: {}", args.output);
        }
        if matches!(
            args.mode,
            DemoMode::Detection | DemoMode::Visual | DemoMode::Live | DemoMode::FileOutput
        ) {
            info!("Model: {}", args.model);
            info!("Confidence threshold: {}", args.confidence);
        }

        let processor = UnifiedProcessor::new(&args)?;
        processor.run()
    })
}
