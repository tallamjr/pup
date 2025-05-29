//! Pup video processing application
//!
//! Real-time object detection using GStreamer and ONNX Runtime.

use clap::{Parser, ValueEnum};
use gstpup::{
    config::AppConfig,
    inference::{InferenceBackend, OrtBackend},
    pipeline::{FrameProcessor, VideoPipeline},
    preprocessing::Preprocessor,
    run,
};
use gstreamer as gst;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

mod live_processor;
use live_processor::LiveVideoProcessor;

#[derive(Debug, Clone, ValueEnum)]
enum ProcessingMode {
    /// Process video and show detections in terminal only (no video window)
    Detection,
    /// Show video window with processing (detections shown in terminal)
    Visual,
    /// Basic video playback without inference (no detection processing)
    Playback,
    /// Show live video window with real-time bounding box overlays (recommended)
    Live,
    /// Production mode with configuration-driven processing
    Production,
}

/// Command line arguments
#[derive(Parser, Debug)]
#[command(author, version, about)]
#[command(long_about = "
Real-time object detection with YOLOv8 and GStreamer video processing.
Supports live video overlays, webcam input, and configuration-driven processing.

EXAMPLES:
  pup --mode live --model models/yolov8n.onnx --video assets/sample.mp4
  pup --mode live --model models/yolov8n.onnx --video webcam
  pup --mode detection --model models/yolov8n.onnx --video assets/sample.mp4
  pup --config config.toml

")]
struct Args {
    /// Processing mode: production (config-driven), live (video + overlays), visual (video + terminal), detection (terminal only), playback (video only)
    #[arg(short, long, value_enum, default_value = "production")]
    mode: ProcessingMode,

    /// Path to ONNX model (optional if using --config)
    #[arg(long)]
    model: Option<String>,

    /// Video source: file path, 'webcam', or auto-detection
    #[arg(long)]
    video: Option<String>,

    /// Confidence threshold for detections (0.0 to 1.0)
    #[arg(long, default_value = "0.5")]
    confidence: f32,

    /// Whether to disable display output
    #[arg(long)]
    no_display: bool,

    /// Whether to show bounding box overlays (for live mode)
    #[arg(long, default_value = "true")]
    show_overlays: bool,

    /// Whether to show class labels on bounding boxes
    #[arg(long, default_value = "true")]
    show_labels: bool,

    /// Whether to show confidence scores on bounding boxes
    #[arg(long, default_value = "true")]
    show_confidence: bool,

    /// Path to configuration file (optional)
    #[arg(long)]
    config: Option<String>,
}

fn gst_main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Parse command line arguments
    println!("Parsing command-line arguments...");
    let args = Args::parse();
    println!("Parsed arguments: {:?}", args);
    
    // Handle different modes
    match args.mode {
        ProcessingMode::Production => run_production_mode(&args),
        ProcessingMode::Live => run_live_mode(&args),
        ProcessingMode::Visual => run_visual_mode(&args),
        ProcessingMode::Detection => run_detection_mode(&args),
        ProcessingMode::Playback => run_playback_mode(&args),
    }
}

fn run_production_mode(args: &Args) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {

    // Load or create configuration
    let config = if let Some(config_path) = &args.config {
        println!("Loading configuration from: {}", config_path);
        AppConfig::from_toml_file(&PathBuf::from(config_path))?
    } else {
        // Create config from command line arguments
        if args.model.is_none() {
            return Err("--model is required when not using --config".into());
        }
        let mut config = AppConfig::from_args(args.model.clone(), args.video.clone());
        config.inference.confidence_threshold = args.confidence;
        config.output.display_enabled = !args.no_display;
        config
    };

    // Validate configuration
    println!("Validating configuration...");
    config.validate()?;

    // Check if model file exists
    if !config.model_exists() {
        eprintln!(
            "Error: ONNX model file '{}' not found.",
            config.model_path().display()
        );
        return Err("Model file not found.".into());
    }

    // Check if video file exists (for file sources)
    if !config.video_exists()
        && config.video_source() != "auto"
        && config.video_source() != "webcam"
    {
        eprintln!("Error: Video file '{}' not found.", config.video_source());
        return Err("Video file not found.".into());
    }

    println!("Configuration validated successfully");

    // Initialize inference backend
    println!("Loading ONNX model from: {}", config.model_path().display());
    let mut inference_backend = OrtBackend::new();
    inference_backend.load_model(config.model_path())?;
    inference_backend.set_confidence_threshold(config.inference.confidence_threshold);
    println!("ONNX model loaded successfully");

    // Initialize preprocessor
    let target_size = config.preprocessing.as_ref().unwrap().target_size;
    let preprocessor = Preprocessor::new(target_size[0] as i32, target_size[1] as i32);
    println!(
        "Preprocessor initialized with target size: {}x{}",
        target_size[0], target_size[1]
    );

    // Create frame processor
    let frame_processor = FrameProcessor::new(preprocessor, Box::new(inference_backend));
    let frame_processor = Arc::new(frame_processor);

    // Create and configure video pipeline
    println!("Setting up GStreamer pipeline...");
    let mut pipeline = VideoPipeline::new(&config)?;
    println!("GStreamer pipeline created successfully");

    // Set up frame processing callback
    let frame_processor_clone = Arc::clone(&frame_processor);
    pipeline
        .set_frame_processor(move |frame, info| frame_processor_clone.process_frame(frame, info))?;

    // Start the pipeline
    println!("Starting the pipeline...");
    pipeline.start()?;

    // Main processing loop
    println!("Processing video... Press Ctrl+C to stop");
    loop {
        // Process pipeline messages with a timeout
        let continue_processing = pipeline.process_messages(Some(Duration::from_millis(100)))?;

        if !continue_processing {
            println!("Pipeline finished or encountered an error");
            break;
        }

        // Check if pipeline is still running
        if !pipeline.is_running() {
            println!("Pipeline stopped");
            break;
        }
    }

    // Stop the pipeline
    println!("Stopping pipeline...");
    pipeline.stop()?;
    println!("Pipeline stopped successfully");

    Ok(())
}

fn run_live_mode(args: &Args) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Starting live video mode with YOLO inference overlays...");
    
    if args.model.is_none() {
        return Err("--model is required for live mode".into());
    }
    
    let model_path = PathBuf::from(args.model.as_ref().unwrap());
    let video_source = args.video.as_deref().unwrap_or("webcam");
    
    // Initialize GStreamer
    gst::init()?;
    
    let processor = LiveVideoProcessor::new(&model_path, video_source, args)?;
    processor.run()
}

fn run_visual_mode(_args: &Args) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Starting visual mode with detection output...");
    println!("Visual mode not fully implemented yet. Use --mode live for overlay functionality.");
    Ok(())
}

fn run_detection_mode(_args: &Args) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Starting detection-only mode...");
    println!("Detection mode not fully implemented yet. Use --mode live for overlay functionality.");
    Ok(())
}

fn run_playback_mode(_args: &Args) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Starting playback mode (no inference)...");
    println!("Playback mode not fully implemented yet. Use --mode live for overlay functionality.");
    Ok(())
}

fn main() {
    // Use the platform-specific run function from common module
    let result = run(gst_main);

    match result {
        Ok(()) => println!("Application completed successfully"),
        Err(e) => {
            eprintln!("Application error: {}", e);
            std::process::exit(1);
        }
    }
}
