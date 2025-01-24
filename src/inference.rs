use std::sync::{Arc, Mutex};

use gstreamer::prelude::{ElementExtManual, GstObjectExtManual}; // if you needed them, but likely not
use gstreamer_app::{AppSink, AppSrc};
use gstreamer_video::{VideoFrameRef, VideoInfo};

use ndarray::{Array4, ArrayView3};
use ort::session::Session;
// Only keep GraphOptimizationLevel if you use it below:
use ort::inputs;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};

use crate::overlay::draw_rectangle;

/// Hard-coded bounding-box threshold
const SCORE_THRESHOLD: f32 = 0.5;

/// Create the inference pipeline:
///   filesrc -> decodebin -> videoconvert -> video/x-raw,format=RGB -> appsink name=infer_sink
/// We'll connect a "new-sample" callback to run YOLO, then push annotated frames into `AppSrc`.
pub fn create_inference_pipeline(
    video_path: &str,
    session: Arc<Mutex<Session>>,
    display_appsrc: AppSrc,
) -> Pipeline {
    let pipeline_str = format!(
        "filesrc location={} ! decodebin ! videoconvert ! videoscale \
         ! video/x-raw,format=RGB \
         ! appsink name=infer_sink",
        video_path
    );

    let pipeline = gstreamer::parse_launch(&pipeline_str)
        .unwrap()
        .dynamic_cast::<Pipeline>()
        .unwrap();

    // Retrieve the appsink
    let infer_sink = pipeline
        .by_name("infer_sink")
        .expect("No element named infer_sink")
        .dynamic_cast::<AppSink>()
        .unwrap();

    // Turn on signals
    infer_sink.set_property("emit-signals", &true);

    // Connect the "new-sample" signal in the old style
    infer_sink.connect("new-sample", false, move |args| {
        let appsink = match args[0].get::<AppSink>() {
            Ok(s) => s,
            Err(_) => {
                eprintln!("Failed to get AppSink from signal");
                return Some(FlowReturn::Error.to_value());
            }
        };

        // pull_sample => Result<Sample, BoolError>
        let sample = match appsink.pull_sample() {
            Ok(s) => s,
            Err(_) => {
                eprintln!("No sample, returning Eos");
                return Some(FlowReturn::Eos.to_value());
            }
        };

        let buffer = match sample.buffer() {
            Some(b) => b,
            None => {
                eprintln!("No buffer in sample");
                return Some(FlowReturn::Error.to_value());
            }
        };

        let caps = match sample.caps() {
            Some(c) => c,
            None => {
                eprintln!("No caps in sample");
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

        let frame = match VideoFrameRef::from_buffer_ref_readable(&buffer, &info) {
            Ok(f) => f,
            Err(_) => {
                eprintln!("Can't create VideoFrameRef");
                return Some(FlowReturn::Error.to_value());
            }
        };

        let (width, height) = (info.width() as usize, info.height() as usize);

        let plane_data = match frame.plane_data(0) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("No plane data found");
                return Some(FlowReturn::Error.to_value());
            }
        };

        let array = match ArrayView3::from_shape((height, width, 3), plane_data) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("Failed to create ArrayView3: {}", e);
                return Some(FlowReturn::Error.to_value());
            }
        };

        // Build (1,3,H,W) for YOLO
        let mut input_tensor = Array4::<f32>::zeros((1, 3, height, width));
        for ((y, x, c), &pixel) in array.indexed_iter() {
            input_tensor[[0, c, y, x]] = pixel as f32 / 255.0;
        }

        {
            let mut locked_session = session.lock().unwrap();

            let input_name = "images";
            let shape = [1, 3, height, width];
            let input_data = input_tensor.as_slice().unwrap();

            // Build inputs
            let input_map = match inputs![input_name => (shape, input_data)] {
                Ok(i) => i,
                Err(e) => {
                    eprintln!("Failed building ort inputs: {}", e);
                    return Some(FlowReturn::Error.to_value());
                }
            };

            let outputs = match locked_session.run(input_map) {
                Ok(o) => o,
                Err(e) => {
                    eprintln!("Inference error: {}", e);
                    return Some(FlowReturn::Error.to_value());
                }
            };

            let output_tensor = match outputs["output0"].try_extract_tensor::<f32>() {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("Failed to extract YOLO output: {}", e);
                    return Some(FlowReturn::Error.to_value());
                }
            };

            // Convert plane_data to a mutable Vec for drawing
            let mut annotated_pixels = plane_data.to_vec();

            let detections = output_tensor.view();
            let out_shape = detections.shape();

            if out_shape.len() == 3 && out_shape[2] >= 6 {
                let (batch, num_boxes, _dims) = (out_shape[0], out_shape[1], out_shape[2]);
                for b in 0..batch {
                    for i in 0..num_boxes {
                        let x1 = detections[[b, i, 0]];
                        let y1 = detections[[b, i, 1]];
                        let x2 = detections[[b, i, 2]];
                        let y2 = detections[[b, i, 3]];
                        let score = detections[[b, i, 4]];

                        if score > SCORE_THRESHOLD {
                            draw_rectangle(
                                &mut annotated_pixels,
                                width,
                                height,
                                x1 as i32,
                                y1 as i32,
                                x2 as i32,
                                y2 as i32,
                                [255, 0, 0], // red
                            );
                        }
                    }
                }
            }

            // Now we push the annotated image to display_appsrc
            let mut out_buf = match gstreamer::Buffer::with_size(annotated_pixels.len()) {
                Ok(b) => b,
                Err(_) => {
                    eprintln!("Failed to create GStreamer buffer");
                    return Some(FlowReturn::Error.to_value());
                }
            };

            {
                let mut map = match out_buf.make_mut().map_writable() {
                    Ok(m) => m,
                    Err(_) => {
                        eprintln!("Failed to map buffer writable");
                        return Some(FlowReturn::Error.to_value());
                    }
                };
                map.copy_from_slice(&annotated_pixels);
            }

            // We set new caps for the appsrc matching the original width/height, but still in RGB
            let outcaps = gstreamer::Caps::builder("video/x-raw")
                .field("format", &"RGB")
                .field("width", &(width as i32))
                .field("height", &(height as i32))
                .build();
            display_appsrc.set_caps(Some(&outcaps));

            // Finally push
            let _ = display_appsrc.push_buffer(out_buf);
        }

        // Done
        Some(FlowReturn::Ok.to_value())
    });

    pipeline
}

/// Build the Session in the older `ort` 0.17 style
pub fn create_onnx_session(model_path: &str) -> Session {
    Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap()
        .with_intra_threads(4)
        .unwrap()
        .commit_from_file(model_path)
        .unwrap()
}

pub fn play_inference_pipeline(pipeline: &Pipeline) {
    pipeline
        .set_state(State::Playing)
        .expect("Couldn't set inference pipeline to Playing");
}
