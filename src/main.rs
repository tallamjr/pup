use gstreamer::prelude::*;
use gstreamer_video::{VideoFrameRef, VideoInfo};
use ndarray::{Array4, ArrayView3};
use ort::{init, session::builder::SessionBuilder};
use std::sync::{Arc, Mutex};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize GStreamer
    gstreamer::init()?;

    // Initialize the ONNX Runtime environment
    init().commit()?;

    // Load the ONNX model
    let session = SessionBuilder::new()?.commit_from_file("model.onnx")?;

    let session = Arc::new(Mutex::new(session));

    // Create the GStreamer pipeline
    let pipeline = gstreamer::parse_launch(
        "videotestsrc ! videoconvert ! videoscale ! video/x-raw,format=RGB,width=224,height=224 ! appsink name=sink",
    )?;

    let pipeline = pipeline
        .dynamic_cast::<gstreamer::Pipeline>()
        .expect("Failed to cast pipeline");

    // Retrieve the appsink element
    let appsink = pipeline
        .by_name("sink")
        .expect("Sink element not found")
        .dynamic_cast::<gstreamer_app::AppSink>()
        .expect("Sink element is expected to be an appsink");

    // Configure appsink
    appsink.set_caps(Some(
        &gstreamer::Caps::builder("video/x-raw")
            .field("format", &"RGB")
            .field("width", &224)
            .field("height", &224)
            .build(),
    ));
    appsink.set_property("emit-signals", &true);

    // Clone the session for use in the sample handler
    let session_clone: Arc<Mutex<_>> = Arc::clone(&session);

    // Connect to the new-sample signal
    appsink.connect("new-sample", false, move |args| {
        let appsink = args[0]
            .get::<gstreamer_app::AppSink>()
            .expect("Failed to get appsink");

        let sample = appsink.pull_sample().ok()?;

        let buffer = sample.buffer()?;
        let caps = sample.caps()?;

        let info = VideoInfo::from_caps(&caps).ok()?;

        let frame = VideoFrameRef::from_buffer_ref_readable(&buffer, &info).ok()?;

        let width = info.width() as usize;
        let height = info.height() as usize;

        let frame_data = frame.plane_data(0).ok()?;

        let array = ArrayView3::from_shape((height, width, 3), frame_data).ok()?;

        let mut input_tensor = Array4::<f32>::zeros((1, 3, height, width));
        for ((y, x, c), pixel) in array.indexed_iter() {
            input_tensor[[0, c, y, x]] = *pixel as f32 / 255.0;
        }
        let input_shape = input_tensor.shape().to_vec();
        let input_data = input_tensor.iter().cloned().collect::<Vec<f32>>();
        let inputs = ort::inputs!("input" => (input_shape, input_data)).ok()?;

        // let inputs = ort::inputs!("input" => &input_tensor).ok()?;
        // The error suggests that the ort::inputs! macro does not accept a reference to the ndarray::ArrayBase type. Instead, it requires the input to be owned.
        // Solution Instead of passing a reference &input_tensor, try converting it to an owned type that ONNX Runtime can accept.
        // let inputs = ort::inputs!("input" => input_tensor.clone()).ok()?;
        // Or alternatively, use into_owned() to transfer ownership:
        // let inputs = ort::inputs!("input" => input_tensor.into_owned()).ok()?;
        //
        // If the ONNX model expects a dynamic shape, use .into_dyn() before converting:
        // let inputs = ort::inputs!("input" => input_tensor.into_dyn().into_owned())?;
        //

        let mut session = session_clone.lock().unwrap();

        let start = Instant::now();
        let outputs = session.run(inputs).ok()?;
        let duration = start.elapsed();

        println!("Inference time: {:?}", duration);

        let output_tensor = outputs[0].try_extract_tensor::<f32>().ok()?;

        println!("Model output: {:?}", output_tensor);

        Some(gstreamer::FlowReturn::Ok.to_value())
    });

    // Start the pipeline
    pipeline.set_state(gstreamer::State::Playing)?;

    // Wait until an error or EOS
    let bus = pipeline.bus().unwrap();
    for msg in bus.iter_timed(gstreamer::ClockTime::NONE) {
        use gstreamer::MessageView;

        match msg.view() {
            MessageView::Eos(..) => break,
            MessageView::Error(err) => {
                eprintln!(
                    "Error from {}: {} ({:?})",
                    err.src()
                        .map(|s| s.path_string())
                        .unwrap_or_else(|| "unknown".into()),
                    err.error(),
                    err.debug()
                );
                break;
            }
            _ => (),
        }
    }

    // Shutdown the pipeline
    pipeline.set_state(gstreamer::State::Null)?;

    Ok(())
}
