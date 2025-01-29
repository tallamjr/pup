mod common;

use clap::Parser;
use gstreamer::parse::launch;
use gstreamer::prelude::*;
use gstreamer::{FlowReturn, MessageView, State};
use gstreamer_app::AppSink;
use gstreamer_video::{VideoFrameRef, VideoInfo};
use ndarray::Array4;
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

use gstreamer::ClockTime;

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

    /// Optional path to a video file (if unset, uses webcam by default)
    #[arg(long)]
    video: Option<String>,
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
        .with_intra_threads(8)?
        .with_execution_providers([CoreMLExecutionProvider::default().build()])?
        .commit_from_file(&args.model)?;
    println!("ONNX model loaded successfully.");

    let session: Arc<Mutex<Session>> = Arc::new(Mutex::new(session_obj));
    let detections: DetectionList = Arc::new(Mutex::new(Vec::new()));

    // Build the pipeline string differently depending on --video
    println!("Setting up GStreamer pipeline...");
    let pipeline_str = if let Some(video_path) = &args.video {
        // Use a video file pipeline
        format!(
            "filesrc location=\"{}\" ! decodebin ! videoconvert ! videoscale \
             ! video/x-raw,format=RGB,width=640,height=640 \
             ! tee name=t \
                 t. ! queue ! appsink name=sink \
                 t. ! queue ! autovideosink",
            video_path
        )
    } else {
        // Use the webcam (avfvideosrc) by default
        format!(
            "avfvideosrc ! videoconvert ! videoscale \
             ! video/x-raw,format=RGB,width=640,height=640 \
             ! tee name=t \
                 t. ! queue ! appsink name=sink \
                 t. ! queue ! autovideosink"
        )
    };

    println!("Using pipeline: {}", pipeline_str);

    let pipeline = launch(&pipeline_str)?
        .dynamic_cast::<gstreamer::Pipeline>()
        .expect("Failed to cast pipeline to gstreamer::Pipeline");

    let appsink = pipeline
        .by_name("sink")
        .expect("Sink element 'sink' not found")
        .dynamic_cast::<AppSink>()
        .expect("'sink' is not an appsink");

    appsink.set_property("emit-signals", &true);

    let session_clone = Arc::clone(&session);
    let detections_clone = Arc::clone(&detections);

    println!("Connecting to new-sample signal...");
    appsink.connect("new-sample", false, move |vals| {
        let appsink = vals[0]
            .get::<AppSink>()
            .expect("Failed to retrieve AppSink from signal vals");

        if let Ok(sample) = appsink.pull_sample() {
            if let Some(buffer) = sample.buffer() {
                if let Some(caps) = sample.caps() {
                    if let Ok(info) = VideoInfo::from_caps(&caps) {
                        let (width, height) = (info.width() as usize, info.height() as usize);

                        // Acquire the frame data
                        if let Ok(frame) = VideoFrameRef::from_buffer_ref_readable(&buffer, &info) {
                            if let Ok(frame_data) = frame.plane_data(0) {
                                println!("Processing frame: {}x{}", width, height);

                                // 1) Allocate a buffer for pixel data, normalized to [0..1]
                                let total_elems = 1 * 3 * height * width;
                                let mut buffer = vec![0f32; total_elems];

                                // 2) Fill `buffer` in parallel
                                buffer.par_iter_mut().enumerate().for_each(|(i, px)| {
                                    let pixel = frame_data[i];
                                    *px = pixel as f32 / 255.0;
                                });

                                // 3) Convert `buffer` to an Array4<f32>
                                let input_tensor = match Array4::from_shape_vec(
                                    (1, 3, height, width),
                                    buffer,
                                ) {
                                    Ok(t) => t,
                                    Err(e) => {
                                        eprintln!("Failed to reshape buffer into Array4: {:?}", e);
                                        return Some(FlowReturn::Ok.to_value());
                                    }
                                };

                                println!("Running inference...");

                                // 4) Flatten for ONNX input
                                let flattened_data = input_tensor.as_slice().unwrap().to_vec();
                                let shape = [1_i64, 3, height as i64, width as i64];

                                // 5) Build the ONNX input tensor
                                let mut value = match Value::from_array((shape, flattened_data)) {
                                    Ok(v) => v,
                                    Err(e) => {
                                        eprintln!("Failed to create ONNX tensor: {:?}", e);
                                        return Some(FlowReturn::Ok.to_value());
                                    }
                                };

                                // 6) Convert to dynamic if the model expects Value<DynValueTypeMarker>
                                value = value.into();

                                // 7) Create input list
                                let input_values = vec![("images".to_string(), value)];

                                // 8) Run inference
                                let locked_session = session_clone.lock().unwrap();
                                let outputs = match locked_session.run(input_values) {
                                    Ok(o) => o,
                                    Err(e) => {
                                        eprintln!("Session run failed: {:?}", e);
                                        return Some(FlowReturn::Ok.to_value());
                                    }
                                };

                                let output_tensor = match outputs["output0"].try_extract_tensor::<f32>()
                                {
                                    Ok(t) => t,
                                    Err(e) => {
                                        eprintln!("Failed to extract output tensor: {:?}", e);
                                        return Some(FlowReturn::Ok.to_value());
                                    }
                                };

                                println!("YOLO output shape: {:?}", output_tensor.shape());
                                let data = output_tensor.view();
                                let num_boxes = data.shape()[1];

                                // 9) Detection filtering
                                let dets: Vec<Detection> = (0..num_boxes)
                                    .into_par_iter()
                                    .filter_map(|box_i| {
                                        let x1 = data[[0, box_i, 0]] * width as f32;
                                        let y1 = data[[0, box_i, 1]] * height as f32;
                                        let x2 = data[[0, box_i, 2]] * width as f32;
                                        let y2 = data[[0, box_i, 3]] * height as f32;
                                        let score = data[[0, box_i, 4]];
                                        let class_id = data[[0, box_i, 5]] as i32;

                                        if score > 0.5 {
                                            Some(Detection {
                                                x1,
                                                y1,
                                                x2,
                                                y2,
                                                score,
                                                class_id,
                                            })
                                        } else {
                                            None
                                        }
                                    })
                                    .collect();

                                // Print each detection
                                for d in &dets {
                                    println!(
                                        "Detected class={} score={:.2} bbox=({:.2}, {:.2}, {:.2}, {:.2})",
                                        d.class_id,
                                        d.score,
                                        d.x1,
                                        d.y1,
                                        d.x2 - d.x1,
                                        d.y2 - d.y1
                                    );
                                }

                                // Update detection list
                                *detections_clone.lock().unwrap() = dets;
                            }
                        }
                    }
                }
            }
        }

        Some(FlowReturn::Ok.to_value())
    });

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
