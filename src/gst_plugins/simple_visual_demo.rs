//! Simplified visual demo for macOS compatibility
//! Basic video playback without complex inference processing

use crate::config::AppConfig;
use anyhow::Result;
use gstreamer as gst;
use gstreamer::prelude::*;

pub struct SimpleVisualProcessor {
    pipeline: gst::Pipeline,
}

impl SimpleVisualProcessor {
    pub fn new(config: &AppConfig) -> Result<Self> {
        // Initialize GStreamer
        gst::init()?;

        let pipeline = gst::Pipeline::builder()
            .name("simple-visual-pipeline")
            .build();

        // Create a simple playback pipeline using playbin for better compatibility
        let playbin = gst::ElementFactory::make("playbin")
            .property(
                "uri",
                format!(
                    "file://{}",
                    std::fs::canonicalize(&config.input.source)?.display()
                ),
            )
            .build()?;

        // Add playbin to pipeline
        pipeline.add(&playbin)?;

        Ok(Self { pipeline })
    }

    pub fn run(&self) -> Result<()> {
        println!("Starting simple video playback demo...");
        println!("Video should open in a new window");
        println!("Press Ctrl+C to stop when done");

        // Start playing
        self.pipeline.set_state(gst::State::Playing)?;

        // Wait for EOS or error
        let bus = self.pipeline.bus().unwrap();
        for msg in bus.iter_timed(gst::ClockTime::NONE) {
            match msg.view() {
                gst::MessageView::Eos(..) => {
                    println!("End of stream - video finished");
                    break;
                }
                gst::MessageView::Error(err) => {
                    eprintln!("Error: {}", err.error());
                    if let Some(debug) = err.debug() {
                        eprintln!("Debug info: {}", debug);
                    }
                    break;
                }
                gst::MessageView::StateChanged(state_changed) => {
                    if state_changed
                        .src()
                        .map(|s| s == &self.pipeline)
                        .unwrap_or(false)
                    {
                        println!(
                            "Pipeline state changed from {:?} to {:?}",
                            state_changed.old(),
                            state_changed.current()
                        );
                    }
                }
                _ => {}
            }
        }

        // Stop pipeline
        self.pipeline.set_state(gst::State::Null)?;
        println!("Simple visual demo completed");
        Ok(())
    }
}

pub fn run_simple_visual_demo() -> Result<()> {
    let mut config = AppConfig::default();
    config.input.source = "assets/sample.mp4".to_string();

    println!("Starting simple visual YOLO demo");
    println!("Input: {}", config.input.source);

    let processor = SimpleVisualProcessor::new(&config)?;
    processor.run()?;

    Ok(())
}
