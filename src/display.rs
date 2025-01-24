use gstreamer::prelude::*;
use gstreamer::{Pipeline, State};
use gstreamer_app::{AppSrc, AppStreamType};

/// Build a pipeline:
///     appsrc name=frame_src is-live=true stream-type=stream
///       ! videoconvert
///       ! autovideosink
/// We'll push frames from inference into `frame_src`.
pub fn create_display_pipeline() -> (Pipeline, AppSrc) {
    let pipeline_str = "\
        appsrc name=frame_src is-live=true stream-type=stream \
        ! videoconvert \
        ! autovideosink";

    let pipeline = gstreamer::parse_launch(pipeline_str)
        .unwrap()
        .dynamic_cast::<Pipeline>()
        .unwrap();

    // Retrieve the appsrc
    let frame_src = pipeline
        .by_name("frame_src")
        .expect("No appsrc named frame_src found")
        .dynamic_cast::<AppSrc>()
        .unwrap();

    // Set the stream type for appsrc
    frame_src.set_stream_type(AppStreamType::Stream);
    frame_src.set_is_live(true);

    (pipeline, frame_src)
}

/// Start playing the display pipeline (non-blocking).
pub fn play_display_pipeline(pipeline: &Pipeline) {
    pipeline
        .set_state(State::Playing)
        .expect("Failed to set display pipeline to Playing");
}
