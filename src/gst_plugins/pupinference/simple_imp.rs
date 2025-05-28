//! Simplified PupInference implementation for initial testing

use std::sync::Mutex;

use glib::subclass::prelude::*;
use gstreamer as gst;
use gstreamer::subclass::prelude::*;
use once_cell::sync::Lazy;

#[derive(Default)]
pub struct PupInference {
    properties: Mutex<()>,
}

#[glib::object_subclass]
impl ObjectSubclass for PupInference {
    const NAME: &'static str = "PupInference";
    type Type = super::PupInference;
    type ParentType = gst::Element;
}

impl ObjectImpl for PupInference {}
impl GstObjectImpl for PupInference {}

impl ElementImpl for PupInference {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "Pup ML Inference",
                "Filter/Effect/Video",
                "Performs computer vision inference on video frames using ONNX Runtime",
                "Tarek Allam Jr <t.allam.jr@gmail.com>",
            )
        });
        Some(&*ELEMENT_METADATA)
    }
}