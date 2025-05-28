//! Pup video processing application
//!
//! Real-time object detection using GStreamer and ONNX Runtime.

use clap::Parser;
use gstpup::{
    config::AppConfig,
    inference::{InferenceBackend, OrtBackend},
    pipeline::{FrameProcessor, VideoPipeline},
    preprocessing::Preprocessor,
    run,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// Command line arguments
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to ONNX model
    #[arg(long)]
    model: String,

    /// Optional path to a video file (if unset, uses webcam by default)
    #[arg(long)]
    video: Option<String>,

    /// Confidence threshold for detections (0.0 to 1.0)
    #[arg(long, default_value = "0.5")]
    confidence: f32,

    /// Whether to disable display output
    #[arg(long)]
    no_display: bool,

    /// Path to configuration file (optional)
    #[arg(long)]
    config: Option<String>,
}

fn gst_main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Parse command line arguments
    println!("Parsing command-line arguments...");
    let args = Args::parse();
    println!("Parsed arguments: {:?}", args);

    // Load or create configuration
    let config = if let Some(config_path) = &args.config {
        println!("Loading configuration from: {}", config_path);
        AppConfig::from_toml_file(&PathBuf::from(config_path))?
    } else {
        // Create config from command line arguments
        let mut config = AppConfig::from_args(Some(args.model.clone()), args.video.clone());
        config.inference.confidence_threshold = args.confidence;
        config.pipeline.display_enabled = !args.no_display;
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
    let target_size = config.preprocessing.target_size;
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
