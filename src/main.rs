use gstreamer::prelude::*;
use gstreamer::{FlowReturn, MessageView, State};
use gstreamer_video::{VideoFrameRef, VideoInfo};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    gstreamer::init()?;

    let pipeline = gstreamer::parse_launch(
        "videotestsrc ! videoconvert ! videoscale \
         ! video/x-raw,format=RGB,width=224,height=224 \
         ! appsink name=sink",
    )?
    .dynamic_cast::<gstreamer::Pipeline>()
    .unwrap();

    let appsink = pipeline
        .by_name("sink")
        .unwrap()
        .dynamic_cast::<gstreamer_app::AppSink>()
        .unwrap();

    // Make sure "emit-signals" is set:
    appsink.set_property("emit-signals", &true);

    appsink.connect("new-sample", false, move |args| {
        let appsink = args[0]
            .get::<gstreamer_app::AppSink>()
            .expect("Failed to get appsink from signal");

        // 1) pull_sample() => Result<Sample, BoolError>
        let sample = match appsink.pull_sample() {
            Ok(s) => s,
            Err(_) => {
                eprintln!("No sample (pull_sample failed), returning Eos");
                return Some(FlowReturn::Eos.to_value());
            }
        };

        // sample.buffer() => Option<BufferRef>, so 'Some(...)' is correct
        let buffer = match sample.buffer() {
            Some(b) => b,
            None => {
                eprintln!("No buffer found, returning Error");
                return Some(FlowReturn::Error.to_value());
            }
        };

        // sample.caps() => Option<CapsRef>, so 'Some(...)' is correct
        let caps = match sample.caps() {
            Some(c) => c,
            None => {
                eprintln!("No caps found, returning Error");
                return Some(FlowReturn::Error.to_value());
            }
        };

        // 2) VideoInfo::from_caps(...) => Result<VideoInfo, BoolError>
        let info = match VideoInfo::from_caps(&caps) {
            Ok(i) => i,
            Err(_) => {
                eprintln!("VideoInfo::from_caps failed, returning Error");
                return Some(FlowReturn::Error.to_value());
            }
        };

        // 3) from_buffer_ref_readable(...) => Result<VideoFrameRef<...>, BoolError>
        let frame = match VideoFrameRef::from_buffer_ref_readable(&buffer, &info) {
            Ok(f) => f,
            Err(_) => {
                eprintln!("Can't create VideoFrameRef, returning Error");
                return Some(FlowReturn::Error.to_value());
            }
        };

        // Do your inference logic here...
        println!(
            "Processing a new frame with resolution {}x{}",
            info.width(),
            info.height()
        );

        // Must always return Some(FlowReturn::...)
        Some(FlowReturn::Ok.to_value())
    });

    // Play pipeline, watch bus
    pipeline.set_state(State::Playing)?;
    let bus = pipeline.bus().unwrap();
    for msg in bus.iter_timed(gstreamer::ClockTime::NONE) {
        match msg.view() {
            MessageView::Eos(..) => break,
            MessageView::Error(err) => {
                eprintln!("Error! {}", err.error());
                break;
            }
            _ => (),
        }
    }

    pipeline.set_state(State::Null)?;
    Ok(())
}
