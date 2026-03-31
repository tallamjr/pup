//! pup - Real-time object detection with YOLOv8 and GStreamer

use clap::Parser;
use gstpup::run;
use std::path::PathBuf;
use tracing::{error, info};

mod live;

#[derive(Parser, Debug)]
#[command(name = "pup")]
#[command(
    version,
    about = "Real-time object detection with YOLOv8 and GStreamer"
)]
struct Args {
    /// Input source: file path or 'webcam'
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

    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
}

fn gst_main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();

    if args.verbose {
        std::env::set_var("RUST_LOG", "debug");
    } else if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }

    live::run(
        args.input,
        args.model,
        args.confidence,
        !args.no_overlays,
        !args.no_labels,
        !args.no_confidence,
    )
}

fn main() {
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

    let result = run(gst_main);

    match result {
        Ok(()) => info!("Application completed successfully"),
        Err(e) => {
            error!("Application error: {}", e);
            std::process::exit(1);
        }
    }
}
