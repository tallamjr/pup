mod common;

use clap::Parser;
use gstreamer::parse::launch;
use gstreamer::prelude::*;
use gstreamer::{FlowReturn, MessageView, State};
use gstreamer_app::AppSink;
use gstreamer_video::{VideoFrameRef, VideoInfo};
use ndarray::{Array4, ArrayView3};
use ort::execution_providers::CoreMLExecutionProvider;
use ort::inputs; // The macro is here in 0.17
use ort::session::builder::SessionBuilder; // For older builder pattern
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::sync::{Arc, Mutex};

/// Command-line arguments
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to the YOLO ONNX model
    #[arg(long)]
    model: String,

    /// Path to an MP4 file
    #[arg(long)]
    video: String,
}

fn gst_main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // 1) Parse CLI
    let args = Args::parse();

    // 2) Init GStreamer
    gstreamer::init()?;

    // 3) Build an ORT session (old 0.17 style)
    //    - SessionBuilder::new() or Session::builder()
    //    - with_optimization_level(Level3)
    //    - with_intra_threads(...)
    //    - commit_from_file(...)
    let session_obj: Session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .with_execution_providers([CoreMLExecutionProvider::default().build()])?
        .commit_from_file(&args.model)?;

    // Wrap it in Arc<Mutex<...>> with explicit type annotation
    let session: Arc<Mutex<Session>> = Arc::new(Mutex::new(session_obj));

    // 4) GStreamer pipeline from MP4 => decode => 224x224 RGB => appsink
    let pipeline_str = format!(
        "filesrc location={} ! decodebin ! videoconvert ! videoscale \
         ! video/x-raw,format=RGB,width=640,height=640 \
         ! appsink name=sink",
        args.video
    );

    let pipeline = launch(&pipeline_str)?
        .dynamic_cast::<gstreamer::Pipeline>()
        .expect("Failed to cast pipeline to gstreamer::Pipeline");

    // 5) Retrieve appsink
    let appsink = pipeline
        .by_name("sink")
        .expect("Sink element 'sink' not found")
        .dynamic_cast::<AppSink>()
        .expect("'sink' is not an appsink");

    appsink.set_property("emit-signals", &true);

    // Clone session for callback
    let session_clone = Arc::clone(&session);

    // 6) Connect "new-sample"
    appsink.connect("new-sample", false, move |args| {
        let appsink = match args[0].get::<AppSink>() {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Failed to retrieve AppSink from signal args");
                return Some(FlowReturn::Error.to_value());
            }
        };

        // pull_sample => Result<Sample, BoolError>
        let sample = match appsink.pull_sample() {
            Ok(s) => s,
            Err(_) => {
                eprintln!("No sample (pull_sample failed). Returning Eos.");
                return Some(FlowReturn::Eos.to_value());
            }
        };

        // sample.buffer => Option<BufferRef>
        let buffer = match sample.buffer() {
            Some(b) => b,
            None => {
                eprintln!("No buffer found in sample");
                return Some(FlowReturn::Error.to_value());
            }
        };

        // sample.caps => Option<CapsRef>
        let caps = match sample.caps() {
            Some(c) => c,
            None => {
                eprintln!("No caps found in sample");
                return Some(FlowReturn::Error.to_value());
            }
        };

        // VideoInfo::from_caps => Result<VideoInfo, BoolError>
        let info = match VideoInfo::from_caps(&caps) {
            Ok(i) => i,
            Err(_) => {
                eprintln!("VideoInfo::from_caps failed");
                return Some(FlowReturn::Error.to_value());
            }
        };

        // from_buffer_ref_readable => Result<VideoFrameRef, BoolError>
        let frame = match VideoFrameRef::from_buffer_ref_readable(&buffer, &info) {
            Ok(f) => f,
            Err(_) => {
                eprintln!("Failed to create VideoFrameRef");
                return Some(FlowReturn::Error.to_value());
            }
        };

        let (width, height) = (info.width() as usize, info.height() as usize);

        // plane_data(0) => Result<&[u8], BoolError>
        let frame_data = match frame.plane_data(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("No plane data found");
                return Some(FlowReturn::Error.to_value());
            }
        };

        // Make ndarray => shape (height, width, 3)
        let array = match ArrayView3::from_shape((height, width, 3), frame_data) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("ArrayView3 creation error: {}", e);
                return Some(FlowReturn::Error.to_value());
            }
        };

        // Build a 4D array => (1, 3, H, W)
        let mut input_tensor = Array4::<f32>::zeros((1, 3, height, width));
        for ((y, x, c), &pixel) in array.indexed_iter() {
            input_tensor[[0, c, y, x]] = pixel as f32 / 255.0;
        }

        {
            // Lock the ORT session
            let mut locked_session = session_clone.lock().unwrap();

            // Flatten into a slice
            let input_data = input_tensor.as_slice().unwrap();

            // The model's input name might be "input", "images", etc.
            let input_name = "images";
            // The macro expects shape + data if you’re not passing a single slice
            // E.g. ( [1,3,height,width], data ), or just data if you have an impl for it.
            //
            // Use the shape + data approach:
            let shape = [1, 3, height, width];

            // Build input map
            let inputs_map = match inputs![input_name => (shape, input_data)] {
                Ok(i) => i,
                Err(e) => {
                    eprintln!("Failed to build ort inputs: {}", e);
                    return Some(FlowReturn::Error.to_value());
                }
            };

            // Run inference
            let outputs = match locked_session.run(inputs_map) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("Inference error: {}", e);
                    return Some(FlowReturn::Error.to_value());
                }
            };

            // Suppose your YOLO model's output is named "output0"
            let output_tensor = match outputs["output0"].try_extract_tensor::<f32>() {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Failed to extract YOLO output: {}", e);
                    return Some(FlowReturn::Error.to_value());
                }
            };

            // Parse bounding boxes
            let data = output_tensor.view();
            let shape = data.shape();
            println!("YOLO output shape: {:?}", shape);

            // e.g. shape [1, N, 6] => [x1,y1,x2,y2,score,class]
            if shape.len() == 3 && shape[2] >= 6 {
                let (batch, num_boxes, _dims) = (shape[0], shape[1], shape[2]);
                for b in 0..batch {
                    for box_i in 0..num_boxes {
                        let x1 = data[[b, box_i, 0]];
                        let y1 = data[[b, box_i, 1]];
                        let x2 = data[[b, box_i, 2]];
                        let y2 = data[[b, box_i, 3]];
                        let score = data[[b, box_i, 4]];
                        let class_id = data[[b, box_i, 5]] as i32;

                        if score > 0.5 {
                            println!(
                                "Detected class={} score={:.2} bbox=({}, {}, {}, {})",
                                class_id, score, x1, y1, x2, y2
                            );
                        }
                    }
                }
            } else {
                println!("Unexpected shape for YOLO output");
            }
        }

        // Return FlowReturn::Ok
        Some(FlowReturn::Ok.to_value())
    });

    // 7) Play pipeline
    pipeline.set_state(State::Playing)?;

    // 8) Wait for EOS or error
    let bus = pipeline.bus().unwrap();
    for msg in bus.iter_timed(gstreamer::ClockTime::NONE) {
        match msg.view() {
            MessageView::Eos(..) => {
                println!("End of stream");
                break;
            }
            MessageView::Error(e) => {
                eprintln!(
                    "GStreamer error from {:?}: {} ({:?})",
                    e.src().map(|s| s.path_string()),
                    e.error(),
                    e.debug()
                );
                break;
            }
            _ => (),
        }
    }

    // 9) Cleanup
    pipeline.set_state(State::Null)?;
    println!("Pipeline stopped.");
    Ok(())
}

fn main() {
    // tutorials_common::run is only required to set up the application environment on macOS
    // (but not necessary in normal Cocoa applications where this is set up automatically)
    common::run(gst_main);
}
