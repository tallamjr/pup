mod common;

use clap::Parser;
use gstreamer::parse::launch;
use gstreamer::prelude::*;
use gstreamer::{ClockTime, FlowReturn, MessageView, State};
use gstreamer_app::AppSink;
use gstreamer_video::{VideoFrameRef, VideoInfo};

use ndarray::Array4;
use ort::execution_providers::CoreMLExecutionProvider;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Value;

use rayon::prelude::*;
use std::sync::{Arc, Mutex};

// OpenCV
use opencv::{
    core::{self, Scalar},
    imgproc,
    prelude::*,
    Result as CvResult,
};

/// Letterbox an OpenCV Mat to 640x640 preserving aspect ratio.
fn letterbox_to_640(src: &Mat) -> CvResult<Mat> {
    let src_size = src.size()?;
    let (orig_w, orig_h) = (src_size.width, src_size.height);

    // 1) scale so the longer side is 640
    let scale = if orig_w > orig_h {
        640.0 / orig_w as f64
    } else {
        640.0 / orig_h as f64
    };
    let new_w = (orig_w as f64 * scale).round() as i32;
    let new_h = (orig_h as f64 * scale).round() as i32;

    // 2) black 640x640 Mat
    let mat_type = src.typ();
    let mut dst = Mat::new_rows_cols_with_default(640, 640, mat_type, Scalar::all(0.0))?;

    // 3) resize src
    let mut resized = Mat::default();
    imgproc::resize(
        src,
        &mut resized,
        core::Size::new(new_w, new_h),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )?;

    // 4) copy resized into dst center
    let x_offset = (640 - new_w) / 2;
    let y_offset = (640 - new_h) / 2;
    let roi_rect = core::Rect::new(x_offset, y_offset, new_w, new_h);
    let mut roi = dst.roi_mut(roi_rect)?;
    resized.copy_to(&mut roi)?;

    Ok(dst)
}

/// Detection structure
#[derive(Clone, Debug)]
struct Detection {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
    class_id: i32,
}

// Shared detection list
type DetectionList = Arc<Mutex<Vec<Detection>>>;

/// CLI
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to ONNX model
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
    let session = Arc::new(Mutex::new(session_obj));
    let detections: DetectionList = Arc::new(Mutex::new(Vec::new()));

    // 2) Build pipeline string based on --video
    println!("Setting up GStreamer pipeline...");
    let pipeline_str = if let Some(video_path) = &args.video {
        // Use a video file pipeline with two branches:
        //   1) Original aspect ratio -> autovideosink
        //   2) Convert explicitly to RGB before appsink
        format!(
            "filesrc location=\"{video_path}\" ! decodebin name=d \
            d. ! queue ! videoconvert ! tee name=t \
                t. ! queue ! autovideosink \
                t. ! queue ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink \
            d. ! queue ! audioconvert ! audioresample ! autoaudiosink"
        )
    } else {
        // Use the webcam (avfvideosrc) by default
        format!(
            "avfvideosrc ! videoconvert ! tee name=t \
           t. ! queue ! autovideosink \
           t. ! queue ! videoconvert ! video/x-raw,format=RGB ! appsink name=sink"
        )
    };

    println!("Pipeline: {pipeline_str}");
    let pipeline = launch(&pipeline_str)?
        .dynamic_cast::<gstreamer::Pipeline>()
        .expect("Failed to cast pipeline to gstreamer::Pipeline");

    let appsink = pipeline
        .by_name("sink")
        .expect("No element named 'sink'")
        .dynamic_cast::<AppSink>()
        .expect("'sink' is not an AppSink");
    // We'll pull frames from "sink"
    appsink.set_property("emit-signals", &true);

    let session_clone = Arc::clone(&session);
    let detections_clone = Arc::clone(&detections);

    println!("Connecting to new-sample signal...");
    appsink.connect("new-sample", false, move |vals| {
        let sink = vals[0].get::<AppSink>().unwrap();
        // Pull the latest sample
        if let Ok(sample) = sink.pull_sample() {
            if let Some(buffer) = sample.buffer() {
                if let Some(caps) = sample.caps() {
                    if let Ok(info) = VideoInfo::from_caps(&caps) {
                        let native_w = info.width() as usize;
                        let native_h = info.height() as usize;

                        // Acquire GStreamer frame
                        if let Ok(frame) = VideoFrameRef::from_buffer_ref_readable(&buffer, &info) {
                            if let Ok(frame_data) = frame.plane_data(0) {
                                let stride0 = info.stride()[0] as usize;
                                println!("Processing frame: {native_w}x{native_h}, stride0={stride0}, data.len()={}", frame_data.len());
                                println!("Video format: {:?}", info.format());

                                // row-by-row copy to an OpenCV Mat
                                let mat_type = core::CV_8UC3;
                                let mut mat = match Mat::new_rows_cols_with_default(
                                    native_h as i32, native_w as i32, mat_type,
                                    Scalar::all(0.0)
                                ) {
                                    Ok(m) => m,
                                    Err(e) => {
                                        eprintln!("Failed to create Mat: {e:?}");
                                        return Some(FlowReturn::Ok.to_value());
                                    }
                                };

                                // We must do row-wise copy using stride
                                let mat_bytes = mat.data_bytes_mut().unwrap();
                                // Each row has exactly native_w * 3 bytes to fill
                                let row_size = native_w * 3;
                                if stride0 < row_size {
                                    eprintln!("Stride {stride0} < row_size {row_size}, skipping");
                                    return Some(FlowReturn::Ok.to_value());
                                }
                                if frame_data.len() < stride0 * native_h {
                                    eprintln!("Frame data is smaller than (stride*height), skip!");
                                    return Some(FlowReturn::Ok.to_value());
                                }

                                for row in 0..native_h {
                                    let src_start = row * stride0;
                                    let src_end = src_start + row_size;
                                    if src_end > frame_data.len() {
                                        eprintln!("Src end beyond frame_data, skipping row");
                                        return Some(FlowReturn::Ok.to_value());
                                    }
                                    let dst_start = row * row_size;
                                    let dst_end = dst_start + row_size;
                                    let src_slice = &frame_data[src_start..src_end];
                                    let dst_slice = &mut mat_bytes[dst_start..dst_end];
                                    dst_slice.copy_from_slice(src_slice);
                                }

                                // letterbox to 640x640
                                let letterboxed = match letterbox_to_640(&mat) {
                                    Ok(lb) => lb,
                                    Err(e) => {
                                        eprintln!("letterbox error: {e:?}");
                                        return Some(FlowReturn::Ok.to_value());
                                    }
                                };

                                // Now we have a 640x640 3-channel mat
                                let (width, height) = (640, 640);
                                let total_elems = (width * height * 3) as usize;
                                let src_data = letterboxed.data_bytes().unwrap();
                                if src_data.len() < total_elems {
                                    eprintln!("Letterboxed data is too small, skip");
                                    return Some(FlowReturn::Ok.to_value());
                                }

                                // convert to float [0..1]
                                let mut float_buffer = vec![0f32; total_elems];
                                float_buffer.par_iter_mut().enumerate().for_each(|(i, px)| {
                                    *px = src_data[i] as f32 / 255.0;
                                });

                                println!("First 10 pixel values: {:?}", &float_buffer[..10]);

                                // shape = [1, 3, 640, 640]
                                let arr = match Array4::from_shape_vec((1, 3, height, width), float_buffer) {
                                    Ok(a) => a,
                                    Err(e) => {
                                        eprintln!("Failed to build Array4: {e:?}");
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
                                        eprintln!("Failed to create ONNX tensor: {e:?}");
                                        return Some(FlowReturn::Ok.to_value());
                                    }
                                };
                                val = val.into();

                                // 4) Run Inference
                                let locked_session = session_clone.lock().unwrap();
                                let outs = match locked_session.run(vec![("images".to_owned(), val)]) {
                                    Ok(o) => o,
                                    Err(e) => {
                                        eprintln!("Session run failed: {e:?}");
                                        return Some(FlowReturn::Ok.to_value());
                                    }
                                };

                                // shape [1, 84, #boxes]
                                let output_tensor = match outs["output0"].try_extract_tensor::<f32>() {
                                    Ok(t) => t,
                                    Err(e) => {
                                        eprintln!("Failed extracting output: {e:?}");
                                        return Some(FlowReturn::Ok.to_value());
                                    }
                                };
                                println!("YOLO output shape: {:?}", output_tensor.shape());

                                let data = output_tensor.view();
                                let channels = data.shape()[1];
                                let num_boxes = data.shape()[2];
                                let mut dets = Vec::new();

                                for i in 0..num_boxes {
                                    let cx = data[[0, 0, i]] * 640.0;
                                    let cy = data[[0, 1, i]] * 640.0;
                                    let w  = data[[0, 2, i]] * 640.0;
                                    let h  = data[[0, 3, i]] * 640.0;
                                    let x1 = cx - w*0.5;
                                    let y1 = cy - h*0.5;
                                    let x2 = cx + w*0.5;
                                    let y2 = cy + h*0.5;

                                    let mut best_score = -f32::MAX;
                                    let mut best_class = -1;
                                    for c in 4..channels {
                                        let cls_score = data[[0, c, i]];
                                        if cls_score > best_score {
                                            best_score = cls_score;
                                            best_class = (c - 4) as i32;
                                        }
                                    }
                                    if best_score > 0.0 {
                                        println!("DEBUG: box{i} best_score={:.2} class={}", best_score, best_class);
                                        dets.push(Detection { x1,y1,x2,y2, score:best_score, class_id:best_class });
                                    }
                                }

                                for d in &dets {
                                    let w = d.x2 - d.x1;
                                    let h = d.y2 - d.y1;
                                    println!(
                                        "Detected class={} score={:.2} bbox=({:.2}, {:.2}, {:.2}, {:.2})",
                                        d.class_id, d.score, d.x1,d.y1, w,h
                                    );
                                }

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
            MessageView::Error(e) => {
                eprintln!("GStreamer error: {}", e.error());
                break;
            }
            _ => {}
        }
    }
    pipeline.set_state(State::Null)?;
    println!("Pipeline stopped.");
    Ok(())
}

fn main() {
    common::run(gst_main);
}
