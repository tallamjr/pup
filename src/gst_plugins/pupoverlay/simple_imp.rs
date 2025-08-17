//! Simplified PupOverlay implementation for initial testing

use std::sync::Mutex;

use glib::subclass::prelude::*;
use gstreamer as gst;
use gstreamer::subclass::prelude::*;
use once_cell::sync::Lazy;

#[derive(Default)]
pub struct PupOverlay {
    _properties: Mutex<()>,
}

#[glib::object_subclass]
impl ObjectSubclass for PupOverlay {
    const NAME: &'static str = "PupOverlay";
    type Type = super::PupOverlay;
    type ParentType = gst::Element;
}

impl ObjectImpl for PupOverlay {}
impl GstObjectImpl for PupOverlay {}

impl ElementImpl for PupOverlay {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "Pup Vision Overlay",
                "Filter/Effect/Video",
                "Renders computer vision results on video frames",
                "Tarek Allam Jr <t.allam.jr@gmail.com>",
            )
        });
        Some(&*ELEMENT_METADATA)
    }
}