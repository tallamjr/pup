mod common;

use clap::Parser;
use gstreamer::parse::launch;
use gstreamer::prelude::*;
use gstreamer::{FlowReturn, MessageView, State};
use gstreamer_app::AppSink;
use gstreamer_video::{VideoFrameRef, VideoInfo};
use ndarray::{Array4, ArrayView3};
use opencv::prelude::*;
use opencv::{core, highgui, imgproc};
use ort::execution_providers::CoreMLExecutionProvider;
use ort::inputs;
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::sync::{Arc, Mutex};

/// Struct to store bounding box detections
#[derive(Clone, Debug)]
struct Detection {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
    class_id: i32,
}

// Shared storage for detections
type DetectionList = Arc<Mutex<Vec<Detection>>>;

/// Command-line arguments
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to the YOLO ONNX model
    #[arg(long)]
    model: String,
}

fn gst_main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Parsing command-line arguments...");
    let args = Args::parse();
    println!("Parsed arguments: {:?}", args);

    if !std::path::Path::new(&args.model).exists() {
        eprintln!("Error: ONNX model file '{}' not found.", args.model);
        return Err(Box::from("Model file not found."));
    }

    println!("Initializing GStreamer...");
    gstreamer::init()?;
    println!("GStreamer initialized successfully.");

    println!("Loading ONNX model from: {}", args.model);
    let session_obj = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .with_execution_providers([CoreMLExecutionProvider::default().build()])?
        .commit_from_file(&args.model)?;
    println!("ONNX model loaded successfully.");

    let session: Arc<Mutex<Session>> = Arc::new(Mutex::new(session_obj));

    println!("Setting up GStreamer pipeline...");
    let pipeline_str = format!(
        "avfvideosrc ! videoconvert ! videoscale \
         ! video/x-raw,format=RGB,width=640,height=640 \
         ! queue ! autovideosink" // Change this to glimagesink if preferred
    );

    let pipeline = launch(&pipeline_str)?
        .dynamic_cast::<gstreamer::Pipeline>()
        .expect("Failed to cast pipeline to gstreamer::Pipeline");

    println!("Starting the pipeline...");
    pipeline.set_state(State::Playing)?;

    let bus = pipeline.bus().unwrap();
    for msg in bus.iter_timed(gstreamer::ClockTime::NONE) {
        match msg.view() {
            MessageView::Eos(_) => {
                println!("End of stream.");
                break;
            }
            MessageView::Error(err) => {
                eprintln!("GStreamer error: {}", err.error());
                break;
            }
            _ => (),
        }
    }

    pipeline.set_state(State::Null)?;
    println!("Pipeline stopped.");
    Ok(())
}

fn main() {
    common::run(gst_main);
}
