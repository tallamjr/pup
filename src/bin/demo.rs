//! Simple demo binary for testing YOLO on sample video

use gstpup::gst_plugins::simple_demo::run_simple_demo;

fn main() {
    if let Err(e) = run_simple_demo() {
        eprintln!("Demo failed: {}", e);
        std::process::exit(1);
    }
}