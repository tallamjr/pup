//! PupOverlay GStreamer Element
//! 
//! Renders computer vision results (bounding boxes, keypoints, etc.) on video frames

use glib::subclass::prelude::*;
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_base as gst_base;
use gstreamer_video as gst_video;

mod simple_imp;
use simple_imp as imp;

glib::wrapper! {
    pub struct PupOverlay(ObjectSubclass<imp::PupOverlay>) @extends gst::Element, gst::Object;
}

// Registers the type within the plugin
pub fn register(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    gst::Element::register(
        Some(plugin),
        "pupoverlay",
        gst::Rank::NONE,
        PupOverlay::static_type(),
    )
}