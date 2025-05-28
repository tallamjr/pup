//! Visual demo binary for testing YOLO with bounding box overlay

use gstpup::gst_plugins::visual_demo::run_visual_demo;

fn main() {
    if let Err(e) = run_visual_demo() {
        eprintln!("Visual demo failed: {}", e);
        std::process::exit(1);
    }
}