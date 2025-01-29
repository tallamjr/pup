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
    score: f32,    // best class score
    class_id: i32, // best class index
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
    // 1) Parse CLI
    println!("Parsing command-line arguments...");
    let args = Args::parse();
    println!("Parsed arguments: {:?}", args);

    // Verify model path
    if !std::path::Path::new(&args.model).exists() {
        eprintln!("Error: ONNX model file '{}' not found.", args.model);
        return Err("Model file not found.".into());
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

    // 2) Build pipeline string based on --video
    println!("Setting up GStreamer pipeline...");
    let pipeline_str = if let Some(video_path) = &args.video {
        // Use a video file pipeline with two branches:
        //   1) Original aspect ratio -> autovideosink
        //   2) Resized forcibly to 640×640 -> appsink
        format!(
            "filesrc location=\"{video_path}\" ! decodebin name=d \
       d. ! queue ! videoconvert ! tee name=t \
         t. ! queue ! autovideosink \
         t. ! queue ! videoscale ! video/x-raw,format=RGB,width=640,height=640 ! appsink name=sink \
       d. ! queue ! audioconvert ! audioresample ! autoaudiosink"
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

    // We'll pull frames from "sink"
    appsink.set_property("emit-signals", &true);

    let session_clone = Arc::clone(&session);
    let detections_clone = Arc::clone(&detections);

    println!("Connecting to new-sample signal...");
    appsink.connect("new-sample", false, move |vals| {
        let appsink = vals[0]
            .get::<AppSink>()
            .expect("Failed to retrieve AppSink from signal vals");

        // Pull the latest sample
        if let Ok(sample) = appsink.pull_sample() {
            if let Some(buffer) = sample.buffer() {
                if let Some(caps) = sample.caps() {
                    if let Ok(info) = VideoInfo::from_caps(&caps) {
                        let (width, height) = (info.width() as usize, info.height() as usize);

                        // Acquire the raw frame data
                        if let Ok(frame) = VideoFrameRef::from_buffer_ref_readable(&buffer, &info) {
                            if let Ok(frame_data) = frame.plane_data(0) {
                                println!("Processing frame: {}x{}", width, height);

                                // 1) Convert each pixel to [0..1]
                                let total_elems = 3 * height * width; // 1 batch
                                if frame_data.len() < total_elems {
                                    eprintln!("Frame data is too small, skipping frame");
                                    return Some(FlowReturn::Ok.to_value());
                                }

                                let mut float_buffer = vec![0f32; total_elems];
                                float_buffer
                                    .par_iter_mut()
                                    .enumerate()
                                    .for_each(|(i, px)| {
                                        *px = frame_data[i] as f32 / 255.0;
                                    });

                                // 2) Build an Array4
                                let arr = match Array4::from_shape_vec((1, 3, height, width), float_buffer) {
                                    Ok(a) => a,
                                    Err(e) => {
                                        eprintln!("Failed to reshape buffer: {:?}", e);
                                        return Some(FlowReturn::Ok.to_value());
                                    }
                                };

                                // Flatten
                                let flattened = arr.as_slice().unwrap().to_vec();
                                let shape = [1_i64, 3, height as i64, width as i64];

                                // 3) Create ONNX input
                                let mut val = match Value::from_array((shape, flattened)) {
                                    Ok(v) => v,
                                    Err(e) => {
                                        eprintln!("Failed to create ONNX tensor: {:?}", e);
                                        return Some(FlowReturn::Ok.to_value());
                                    }
                                };
                                val = val.into(); // dynamic if needed

                                let input_values = vec![("images".to_owned(), val)];

                                // 4) Inference
                                let locked_session = session_clone.lock().unwrap();
                                let outs = match locked_session.run(input_values) {
                                    Ok(o) => o,
                                    Err(e) => {
                                        eprintln!("Session run failed: {:?}", e);
                                        return Some(FlowReturn::Ok.to_value());
                                    }
                                };

                                // YOLOv8 shape usually [1, 84, #boxes]
                                let output_tensor = match outs["output0"].try_extract_tensor::<f32>() {
                                    Ok(t) => t,
                                    Err(e) => {
                                        eprintln!("Failed extracting output: {:?}", e);
                                        return Some(FlowReturn::Ok.to_value());
                                    }
                                };

                                println!("YOLO output shape: {:?}", output_tensor.shape());

                                // Suppose shape is [1, 84, num_boxes]
                                let data = output_tensor.view();
                                let channels = data.shape()[1];
                                let num_boxes = data.shape()[2]; // if the layout is [1, 84, #boxes]

                                // We interpret:
                                //   data[[0, 0, i]] => cx
                                //   data[[0, 1, i]] => cy
                                //   data[[0, 2, i]] => w
                                //   data[[0, 3, i]] => h
                                //   data[[0, 4..84, i]] => 80 class scores
                                // If your model is different, adjust indexing.

                                // YOLOv8 shape => [1, 84, num_boxes]
                                // channels=84 => [0..4] = [cx,cy,w,h], [4..84] = class scores
                                let mut dets = Vec::new();

                                // Iterate over boxes
                                for i in 0..num_boxes {
                                    // 1) decode bounding box
                                    let cx = data[[0, 0, i]] * width as f32;
                                    let cy = data[[0, 1, i]] * height as f32;
                                    let w  = data[[0, 2, i]] * width as f32;
                                    let h  = data[[0, 3, i]] * height as f32;

                                    let x1 = cx - w*0.5;
                                    let y1 = cy - h*0.5;
                                    let x2 = cx + w*0.5;
                                    let y2 = cy + h*0.5;

                                    // 2) find best class by scanning [4..channels]
                                    let mut best_score = -f32::MAX;
                                        let mut best_class = -1;
                                        for c in 4..84 {
                                            let cls_score = data[[0, c, i]];
                                            if cls_score > best_score {
                                                best_score = cls_score;
                                                best_class = (c - 4) as i32;
                                            }
                                        }

                                    // 3) filter by score
                                    if best_score > 0.0 {
                                        println!("DEBUG: box {} best_score={:.3} best_class={}", i, best_score, best_class);
                                            dets.push(Detection {
                                                x1, y1, x2, y2,
                                                score: best_score,
                                                class_id: best_class,
                                            });
                                        }
                                    }

                                // Print each detection
                                for d in &dets {
                                    let w = d.x2 - d.x1;
                                    let h = d.y2 - d.y1;
                                    println!(
                                        "Detected class={} score={:.2} bbox=({:.2}, {:.2}, {:.2}, {:.2})",
                                        d.class_id, d.score,
                                        d.x1, d.y1, w, h
                                    );
                                }

                                // Save
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
    for msg in bus.iter_timed(ClockTime::NONE) {
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
