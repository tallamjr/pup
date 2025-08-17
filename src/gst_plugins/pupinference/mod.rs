//! PupInference GStreamer Element
//!
//! Performs ML inference on video frames using ONNX Runtime

use gstreamer as gst;
use gstreamer::prelude::*;

mod simple_imp;
use simple_imp as imp;

glib::wrapper! {
    pub struct PupInference(ObjectSubclass<imp::PupInference>) @extends gst::Element, gst::Object;
}

// Registers the type within the plugin
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "pupinference",
        gst::Rank::NONE,
        PupInference::static_type(),
    )
}
