//! GStreamer plugins for pup computer vision framework

use gstreamer as gst;
use gst::glib;

pub mod pupinference;
pub mod pupoverlay;
pub mod simple_demo;
pub mod visual_demo;

// Plugin registration following gstreamer-rs patterns
gst::plugin_define!(
    pupvision,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    env!("CARGO_PKG_VERSION"),
    "MIT",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    "2024-01-01"
);

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    pupinference::register(plugin)?;
    pupoverlay::register(plugin)?;
    Ok(())
}

// Export the plugin entry point
pub fn plugin_entry() -> gst::PluginFlags {
    gst::PluginFlags::empty()
}