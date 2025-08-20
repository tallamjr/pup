//! Unified Pup - Real-time object detection with YOLOv8 and GStreamer
//!
//! Combines the best features from the production and demo implementations
//! into a clean subcommand-based architecture.

use clap::{Parser, Subcommand};
use gstpup::{config::AppConfig, run};
use std::path::PathBuf;
use tracing::{error, info};

mod subcommands;
use subcommands::{OutputFormat, VideoFormat};

#[derive(Parser, Debug)]
#[command(name = "pup")]
#[command(
    version,
    about = "Real-time object detection with YOLOv8 and GStreamer"
)]
#[command(long_about = "
High-performance video processing application for real-time object detection.
Combines YOLOv8 ONNX models with GStreamer for efficient video processing.

EXAMPLES:
  # Live webcam with overlays (recommended)
  pup live --input webcam

  # Live video file with overlays
  pup live --input video.mp4 --model models/yolov8n.onnx

  # Detection-only processing
  pup detect --input video.mp4 --output detections.txt

  # Production processing with config
  pup process --config configs/production.toml

  # Simple video playback
  pup play --input video.mp4

  # Record video with overlays
  pup record --input webcam --output output.mp4
")]
struct Args {
    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Configuration file (can override subcommand options)
    #[arg(short, long, global = true)]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Live video with real-time YOLO overlays (webcam or file)
    Live {
        /// Input source: file path, 'webcam', or camera device ID
        #[arg(short, long, default_value = "webcam")]
        input: String,

        /// ONNX model file path
        #[arg(short, long, default_value = "models/yolov8n.onnx")]
        model: PathBuf,

        /// Confidence threshold (0.0-1.0)
        #[arg(long, default_value = "0.5")]
        confidence: f32,

        /// Disable overlay rendering
        #[arg(long)]
        no_overlays: bool,

        /// Hide class labels on bounding boxes
        #[arg(long)]
        no_labels: bool,

        /// Hide confidence scores on bounding boxes
        #[arg(long)]
        no_confidence: bool,
    },

    /// Detection-only processing (no video display)
    Detect {
        /// Input source: file path, 'webcam', or camera device ID
        #[arg(short, long)]
        input: String,

        /// Output file for detection results
        #[arg(short, long, default_value = "detections.txt")]
        output: PathBuf,

        /// ONNX model file path
        #[arg(short, long, default_value = "models/yolov8n.onnx")]
        model: PathBuf,

        /// Confidence threshold (0.0-1.0)
        #[arg(long, default_value = "0.5")]
        confidence: f32,

        /// Output format
        #[arg(long, value_enum, default_value = "text")]
        format: OutputFormat,
    },

    /// Production/batch processing with configuration files
    Process {
        /// Configuration file path (required)
        #[arg(short, long)]
        config: PathBuf,

        /// Override input source from config
        #[arg(long)]
        input: Option<String>,

        /// Override output settings
        #[arg(long)]
        no_display: bool,
    },

    /// Simple video playback without inference
    Play {
        /// Input source: file path, 'webcam', or camera device ID
        #[arg(short, long)]
        input: String,
    },

    /// Process and record video with overlays
    Record {
        /// Input source: file path, 'webcam', or camera device ID
        #[arg(short, long)]
        input: String,

        /// Output video file
        #[arg(short, long)]
        output: PathBuf,

        /// ONNX model file path
        #[arg(short, long, default_value = "models/yolov8n.onnx")]
        model: PathBuf,

        /// Confidence threshold (0.0-1.0)
        #[arg(long, default_value = "0.5")]
        confidence: f32,

        /// Output video format
        #[arg(long, value_enum, default_value = "mp4")]
        format: VideoFormat,

        /// Video quality (1-10, higher = better)
        #[arg(long, default_value = "5")]
        quality: u8,
    },
}

fn gst_main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();

    // Setup logging based on verbosity
    if args.verbose {
        std::env::set_var("RUST_LOG", "debug");
    } else if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }

    // Load global config if provided
    let global_config = if let Some(config_path) = &args.config {
        info!(
            "Loading global configuration from: {}",
            config_path.display()
        );
        Some(AppConfig::from_toml_file(config_path)?)
    } else {
        None
    };

    match args.command {
        Commands::Live {
            input,
            model,
            confidence,
            no_overlays,
            no_labels,
            no_confidence,
        } => subcommands::live::run(
            input,
            model,
            confidence,
            !no_overlays,
            !no_labels,
            !no_confidence,
            global_config,
        ),
        Commands::Detect {
            input,
            output,
            model,
            confidence,
            format,
        } => subcommands::detect::run(input, output, model, confidence, format, global_config),
        Commands::Process {
            config,
            input,
            no_display,
        } => subcommands::process::run(config, input, !no_display, global_config),
        Commands::Play { input } => subcommands::play::run(input, global_config),
        Commands::Record {
            input,
            output,
            model,
            confidence,
            format,
            quality,
        } => subcommands::record::run(
            input,
            output,
            model,
            confidence,
            format,
            quality,
            global_config,
        ),
    }
}

fn main() {
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

    // Use platform-specific run function
    let result = run(gst_main);

    match result {
        Ok(()) => info!("Application completed successfully"),
        Err(e) => {
            error!("Application error: {}", e);
            std::process::exit(1);
        }
    }
}
