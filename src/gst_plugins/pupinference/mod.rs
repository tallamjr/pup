//! PupInference GStreamer Element
//! 
//! Performs ML inference on video frames using ONNX Runtime

use glib::subclass::prelude::*;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_base as gst_base;
use gstreamer_video as gst_video;

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