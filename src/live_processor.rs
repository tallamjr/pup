//! Live video processor with overlay rendering

use gstpup::inference::{InferenceBackend, OrtBackend, TaskOutput};
use gstpup::utils::{Detection, coco_classes::NAMES as COCO_NAMES};
use gstreamer as gst;
use gstreamer::prelude::*;
use gstreamer_app as gst_app;
use gstreamer_video as gst_video;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

pub struct LiveVideoProcessor {
    pipeline: gst::Pipeline,
    inference_backend: Arc<Mutex<Box<dyn InferenceBackend>>>,
    detections: Arc<Mutex<Vec<Detection>>>,
    frame_count: Arc<Mutex<u32>>,
    confidence_threshold: f32,
    show_overlays: bool,
    show_labels: bool,
    show_confidence: bool,
}

impl LiveVideoProcessor {
    pub fn new(
        model_path: &PathBuf,
        video_source: &str,
        args: &super::Args,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let pipeline = gst::Pipeline::builder()
            .name("live-video-pipeline")
            .build();
        
        // Setup inference backend
        let mut backend = Box::new(OrtBackend::new());
        backend.load_model(model_path)?;
        backend.set_confidence_threshold(args.confidence);
        let inference_backend = Arc::new(Mutex::new(backend as Box<dyn InferenceBackend>));
        
        let detections = Arc::new(Mutex::new(Vec::new()));
        let frame_count = Arc::new(Mutex::new(0u32));
        
        let processor = Self {
            pipeline,
            inference_backend,
            detections,
            frame_count,
            confidence_threshold: args.confidence,
            show_overlays: args.show_overlays,
            show_labels: args.show_labels,
            show_confidence: args.show_confidence,
        };
        
        processor.build_pipeline(video_source)?;
        Ok(processor)
    }
    
    fn build_pipeline(&self, video_source: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let (source, decode_or_convert) = self.create_source_elements(video_source)?;
        let videoconvert1 = gst::ElementFactory::make("videoconvert").build()?;
        let videoscale1 = gst::ElementFactory::make("videoscale").build()?;
        
        let capsfilter1 = gst::ElementFactory::make("capsfilter")
            .property("caps", &gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", 640)
                .field("height", 640)
                .build())
            .build()?;
        
        let tee = gst::ElementFactory::make("tee").build()?;
        
        // Inference path
        let queue1 = gst::ElementFactory::make("queue").build()?;
        let appsink = gst_app::AppSink::builder()
            .caps(&gst::Caps::builder("video/x-raw")
                .field("format", "RGB")
                .field("width", 640)
                .field("height", 640)
                .build())
            .build();
        
        // Display path
        let queue2 = gst::ElementFactory::make("queue").build()?;
        let videoconvert2 = gst::ElementFactory::make("videoconvert").build()?;
        
        // Try different video sinks for best compatibility
        let videosink = if let Ok(sink) = gst::ElementFactory::make("osxvideosink").build() {
            println!("Using osxvideosink for live video display");
            sink
        } else if let Ok(sink) = gst::ElementFactory::make("glimagesink").build() {
            println!("Using glimagesink for live video display");
            sink
        } else {
            println!("Using autovideosink for live video display");
            gst::ElementFactory::make("autovideosink").build()?
        };
        
        if video_source.to_lowercase() == "webcam" {
            // Webcam pipeline
            self.pipeline.add_many(&[
                &source, &decode_or_convert, &videoconvert1, &videoscale1, &capsfilter1, &tee,
                &queue1, &queue2, &videoconvert2, &videosink
            ])?;
            self.pipeline.add(&appsink)?;
            
            source.link(&decode_or_convert)?;
            gst::Element::link_many(&[&decode_or_convert, &videoconvert1, &videoscale1, &capsfilter1, &tee])?;
        } else {
            // File pipeline
            self.pipeline.add_many(&[
                &source, &decode_or_convert, &videoconvert1, &videoscale1, &capsfilter1, &tee,
                &queue1, &queue2, &videoconvert2, &videosink
            ])?;
            self.pipeline.add(&appsink)?;
            
            source.link(&decode_or_convert)?;
            gst::Element::link_many(&[&videoconvert1, &videoscale1, &capsfilter1, &tee])?;
            
            // Connect decodebin pad-added signal
            let videoconvert1_clone = videoconvert1.clone();
            decode_or_convert.connect("pad-added", false, move |values| {
                let pad = values[1].get::<gst::Pad>().unwrap();
                if let Some(caps) = pad.current_caps() {
                    let structure = caps.structure(0).unwrap();
                    if structure.name().starts_with("video/") {
                        let sink_pad = videoconvert1_clone.static_pad("sink").unwrap();
                        if !sink_pad.is_linked() {
                            let _ = pad.link(&sink_pad);
                        }
                    }
                }
                None
            });
        }
        
        // Link tee to both paths
        let tee_src_pad_template = tee.pad_template("src_%u").unwrap();
        let tee_src_pad1 = tee.request_pad(&tee_src_pad_template, Some("src_0"), None).unwrap();
        let tee_src_pad2 = tee.request_pad(&tee_src_pad_template, Some("src_1"), None).unwrap();
        let queue1_sink_pad = queue1.static_pad("sink").unwrap();
        let queue2_sink_pad = queue2.static_pad("sink").unwrap();
        tee_src_pad1.link(&queue1_sink_pad)?;
        tee_src_pad2.link(&queue2_sink_pad)?;
        
        // Link paths
        queue1.link(&appsink)?;
        queue2.link(&videoconvert2)?;
        videoconvert2.link(&videosink)?;
        
        // Setup overlay rendering
        if self.show_overlays {
            self.setup_live_overlay_callback(&queue2)?;
        }
        
        // Setup inference callback
        self.setup_inference_callback(&appsink)?;
        
        Ok(())
    }
    
    fn create_source_elements(&self, input: &str) -> Result<(gst::Element, gst::Element), Box<dyn std::error::Error + Send + Sync>> {
        if input.to_lowercase() == "webcam" {
            println!("Creating webcam source pipeline");
            let source = gst::ElementFactory::make("autovideosrc").build()?;
            let videoconvert = gst::ElementFactory::make("videoconvert").build()?;
            Ok((source, videoconvert))
        } else {
            println!("Creating file source pipeline for: {}", input);
            let filesrc = gst::ElementFactory::make("filesrc")
                .property("location", input)
                .build()?;
            let decodebin = gst::ElementFactory::make("decodebin").build()?;
            Ok((filesrc, decodebin))
        }
    }
    
    fn setup_inference_callback(&self, appsink: &gst_app::AppSink) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let inference_backend = self.inference_backend.clone();
        let detections_clone = self.detections.clone();
        let frame_count_clone = self.frame_count.clone();
        let confidence_threshold = self.confidence_threshold;
        
        appsink.set_callbacks(
            gst_app::AppSinkCallbacks::builder()
                .new_sample(move |appsink| {
                    if let Ok(sample) = appsink.pull_sample() {
                        if let Some(buffer) = sample.buffer() {
                            if let Some(caps) = sample.caps() {
                                if let Ok(info) = gst_video::VideoInfo::from_caps(&caps) {
                                    let current_frame = {
                                        let mut count = frame_count_clone.lock().unwrap();
                                        *count += 1;
                                        *count
                                    };
                                    
                                    match Self::process_sample_static(
                                        &buffer,
                                        &info,
                                        &inference_backend,
                                        confidence_threshold,
                                    ) {
                                        Ok(new_detections) => {
                                            if !new_detections.is_empty() {
                                                println!("Frame {}: {} detections", current_frame, new_detections.len());
                                                for detection in &new_detections {
                                                    let class_name = if (detection.class_id as usize) < COCO_NAMES.len() {
                                                        COCO_NAMES[detection.class_id as usize]
                                                    } else {
                                                        "unknown"
                                                    };
                                                    println!("  - {}: {:.1}% at ({:.0}, {:.0}, {:.0}, {:.0})",
                                                        class_name,
                                                        detection.score * 100.0,
                                                        detection.x1, detection.y1,
                                                        detection.x2, detection.y2
                                                    );
                                                }
                                            }
                                            
                                            if let Ok(mut detections_guard) = detections_clone.lock() {
                                                *detections_guard = new_detections;
                                            }
                                        }
                                        Err(e) => eprintln!("Processing error: {}", e),
                                    }
                                }
                            }
                        }
                    }
                    Ok(gst::FlowSuccess::Ok)
                })
                .build(),
        );
        Ok(())
    }
    
    fn setup_live_overlay_callback(&self, queue: &gst::Element) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let detections_clone = self.detections.clone();
        let show_labels = self.show_labels;
        let show_confidence = self.show_confidence;
        
        let pad = queue.static_pad("src").unwrap();
        
        pad.add_probe(gst::PadProbeType::BUFFER, move |_pad, probe_info| {
            if let Some(gst::PadProbeData::Buffer(ref mut buffer)) = probe_info.data {
                let current_detections = if let Ok(detections_guard) = detections_clone.lock() {
                    detections_guard.clone()
                } else {
                    Vec::new()
                };
                
                let mut_buffer = buffer.make_mut();
                if let Err(e) = Self::draw_overlays_on_buffer_inplace(mut_buffer, &current_detections, show_labels, show_confidence) {
                    eprintln!("Failed to draw overlays: {}", e);
                }
            }
            gst::PadProbeReturn::Ok
        });
        
        Ok(())
    }
    
    fn draw_overlays_on_buffer_inplace(
        buffer: &mut gst::BufferRef, 
        detections: &[Detection],
        show_labels: bool,
        show_confidence: bool,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut map = buffer.map_writable().map_err(|e| format!("Failed to map buffer: {}", e))?;
        let data = map.as_mut_slice();
        
        Self::draw_bounding_boxes_rgb(data, 640, 640, detections, show_labels, show_confidence);
        
        Ok(())
    }
    
    fn draw_bounding_boxes_rgb(
        data: &mut [u8], 
        width: usize, 
        height: usize, 
        detections: &[Detection],
        show_labels: bool,
        show_confidence: bool,
    ) {
        let color = (255, 0, 0); // Red color
        let thickness = 3;
        
        for detection in detections {
            let x1 = (detection.x1 as i32).max(0).min(width as i32 - 1);
            let y1 = (detection.y1 as i32).max(0).min(height as i32 - 1);
            let x2 = (detection.x2 as i32).max(0).min(width as i32 - 1);
            let y2 = (detection.y2 as i32).max(0).min(height as i32 - 1);
            
            if x1 >= x2 || y1 >= y2 {
                continue;
            }
            
            // Draw bounding box
            for y in y1..=(y1 + thickness).min(y2) {
                for x in x1..=x2 {
                    Self::set_pixel_direct(data, x as usize, y as usize, width, color);
                }
            }
            
            for y in (y2 - thickness).max(y1)..=y2 {
                for x in x1..=x2 {
                    Self::set_pixel_direct(data, x as usize, y as usize, width, color);
                }
            }
            
            for y in y1..=y2 {
                for x in x1..=(x1 + thickness).min(x2) {
                    Self::set_pixel_direct(data, x as usize, y as usize, width, color);
                }
            }
            
            for y in y1..=y2 {
                for x in (x2 - thickness).max(x1)..=x2 {
                    Self::set_pixel_direct(data, x as usize, y as usize, width, color);
                }
            }
            
            // Draw label if enabled
            if show_labels || show_confidence {
                let class_name = if detection.class_id >= 0 && (detection.class_id as usize) < COCO_NAMES.len() {
                    COCO_NAMES[detection.class_id as usize]
                } else {
                    "unknown"
                };
                
                let label_text = if show_labels && show_confidence {
                    format!("{}: {:.1}%", class_name, detection.score * 100.0)
                } else if show_labels {
                    class_name.to_string()
                } else {
                    format!("{:.1}%", detection.score * 100.0)
                };
                
                let label_x = x1 as usize;
                let label_y = if y1 > 20 { (y1 - 15) as usize } else { (y2 + 5) as usize };
                
                // Draw text background
                let text_width = label_text.len() * 8;
                let text_height = 12;
                
                for ty in label_y.saturating_sub(2)..=(label_y + text_height + 2).min(height - 1) {
                    for tx in label_x.saturating_sub(2)..=(label_x + text_width + 2).min(width - 1) {
                        Self::set_pixel_direct(data, tx, ty, width, (0, 0, 0));
                    }
                }
                
                Self::draw_text(data, width, height, &label_text, label_x, label_y, (255, 255, 255));
            }
        }
    }
    
    fn draw_text(data: &mut [u8], width: usize, height: usize, text: &str, start_x: usize, start_y: usize, color: (u8, u8, u8)) {
        let font_patterns = Self::get_font_patterns();
        
        let mut x = start_x;
        let y = start_y;
        
        for ch in text.chars() {
            if let Some(pattern) = font_patterns.get(&ch) {
                for row in 0..12 {
                    if y + row >= height { break; }
                    for col in 0..8 {
                        if x + col >= width { break; }
                        if (pattern[row] >> (7 - col)) & 1 == 1 {
                            Self::set_pixel_direct(data, x + col, y + row, width, color);
                        }
                    }
                }
            }
            x += 8;
            if x >= width { break; }
        }
    }
    
    fn get_font_patterns() -> HashMap<char, [u8; 12]> {
        let mut patterns = HashMap::new();
        
        // Letters
        patterns.insert('A', [0x00, 0x18, 0x24, 0x42, 0x42, 0x7E, 0x42, 0x42, 0x42, 0x42, 0x00, 0x00]);
        patterns.insert('B', [0x00, 0x7C, 0x42, 0x42, 0x7C, 0x42, 0x42, 0x42, 0x42, 0x7C, 0x00, 0x00]);
        patterns.insert('C', [0x00, 0x3C, 0x42, 0x40, 0x40, 0x40, 0x40, 0x40, 0x42, 0x3C, 0x00, 0x00]);
        patterns.insert('D', [0x00, 0x78, 0x44, 0x42, 0x42, 0x42, 0x42, 0x42, 0x44, 0x78, 0x00, 0x00]);
        patterns.insert('E', [0x00, 0x7E, 0x40, 0x40, 0x40, 0x7C, 0x40, 0x40, 0x40, 0x7E, 0x00, 0x00]);
        patterns.insert('F', [0x00, 0x7E, 0x40, 0x40, 0x40, 0x7C, 0x40, 0x40, 0x40, 0x40, 0x00, 0x00]);
        patterns.insert('G', [0x00, 0x3C, 0x42, 0x40, 0x40, 0x4E, 0x42, 0x42, 0x46, 0x3A, 0x00, 0x00]);
        patterns.insert('H', [0x00, 0x42, 0x42, 0x42, 0x42, 0x7E, 0x42, 0x42, 0x42, 0x42, 0x00, 0x00]);
        patterns.insert('I', [0x00, 0x3E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x3E, 0x00, 0x00]);
        patterns.insert('L', [0x00, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x7E, 0x00, 0x00]);
        patterns.insert('M', [0x00, 0x42, 0x66, 0x5A, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x00, 0x00]);
        patterns.insert('N', [0x00, 0x42, 0x62, 0x52, 0x4A, 0x46, 0x42, 0x42, 0x42, 0x42, 0x00, 0x00]);
        patterns.insert('O', [0x00, 0x3C, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00, 0x00]);
        patterns.insert('P', [0x00, 0x7C, 0x42, 0x42, 0x42, 0x7C, 0x40, 0x40, 0x40, 0x40, 0x00, 0x00]);
        patterns.insert('R', [0x00, 0x7C, 0x42, 0x42, 0x42, 0x7C, 0x48, 0x44, 0x42, 0x42, 0x00, 0x00]);
        patterns.insert('S', [0x00, 0x3C, 0x42, 0x40, 0x30, 0x0C, 0x02, 0x42, 0x42, 0x3C, 0x00, 0x00]);
        patterns.insert('T', [0x00, 0x7F, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00]);
        patterns.insert('U', [0x00, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00, 0x00]);
        patterns.insert('Y', [0x00, 0x41, 0x22, 0x14, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00]);
        
        // Lowercase
        patterns.insert('a', [0x00, 0x00, 0x00, 0x3C, 0x02, 0x3E, 0x42, 0x42, 0x46, 0x3A, 0x00, 0x00]);
        patterns.insert('b', [0x00, 0x40, 0x40, 0x5C, 0x62, 0x42, 0x42, 0x42, 0x62, 0x5C, 0x00, 0x00]);
        patterns.insert('c', [0x00, 0x00, 0x00, 0x3C, 0x42, 0x40, 0x40, 0x40, 0x42, 0x3C, 0x00, 0x00]);
        patterns.insert('d', [0x00, 0x02, 0x02, 0x3A, 0x46, 0x42, 0x42, 0x42, 0x46, 0x3A, 0x00, 0x00]);
        patterns.insert('e', [0x00, 0x00, 0x00, 0x3C, 0x42, 0x7E, 0x40, 0x40, 0x42, 0x3C, 0x00, 0x00]);
        patterns.insert('g', [0x00, 0x00, 0x00, 0x3A, 0x46, 0x42, 0x46, 0x3A, 0x02, 0x3C, 0x00, 0x00]);
        patterns.insert('h', [0x00, 0x40, 0x40, 0x5C, 0x62, 0x42, 0x42, 0x42, 0x42, 0x42, 0x00, 0x00]);
        patterns.insert('i', [0x00, 0x08, 0x00, 0x18, 0x08, 0x08, 0x08, 0x08, 0x08, 0x3E, 0x00, 0x00]);
        patterns.insert('k', [0x00, 0x40, 0x40, 0x44, 0x48, 0x70, 0x48, 0x44, 0x42, 0x41, 0x00, 0x00]);
        patterns.insert('l', [0x00, 0x18, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x3E, 0x00, 0x00]);
        patterns.insert('m', [0x00, 0x00, 0x00, 0x76, 0x49, 0x49, 0x49, 0x49, 0x49, 0x49, 0x00, 0x00]);
        patterns.insert('n', [0x00, 0x00, 0x00, 0x5C, 0x62, 0x42, 0x42, 0x42, 0x42, 0x42, 0x00, 0x00]);
        patterns.insert('o', [0x00, 0x00, 0x00, 0x3C, 0x42, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00, 0x00]);
        patterns.insert('p', [0x00, 0x00, 0x00, 0x5C, 0x62, 0x42, 0x62, 0x5C, 0x40, 0x40, 0x00, 0x00]);
        patterns.insert('r', [0x00, 0x00, 0x00, 0x5C, 0x62, 0x40, 0x40, 0x40, 0x40, 0x40, 0x00, 0x00]);
        patterns.insert('s', [0x00, 0x00, 0x00, 0x3E, 0x40, 0x3C, 0x02, 0x02, 0x42, 0x3C, 0x00, 0x00]);
        patterns.insert('t', [0x00, 0x10, 0x10, 0x7C, 0x10, 0x10, 0x10, 0x10, 0x10, 0x0C, 0x00, 0x00]);
        patterns.insert('u', [0x00, 0x00, 0x00, 0x42, 0x42, 0x42, 0x42, 0x42, 0x46, 0x3A, 0x00, 0x00]);
        patterns.insert('v', [0x00, 0x00, 0x00, 0x42, 0x42, 0x42, 0x24, 0x24, 0x18, 0x18, 0x00, 0x00]);
        patterns.insert('w', [0x00, 0x00, 0x00, 0x42, 0x42, 0x42, 0x5A, 0x66, 0x42, 0x42, 0x00, 0x00]);
        patterns.insert('x', [0x00, 0x00, 0x00, 0x42, 0x24, 0x18, 0x18, 0x24, 0x42, 0x42, 0x00, 0x00]);
        patterns.insert('y', [0x00, 0x00, 0x00, 0x42, 0x42, 0x42, 0x26, 0x1A, 0x02, 0x3C, 0x00, 0x00]);
        
        // Numbers
        patterns.insert('0', [0x00, 0x3C, 0x42, 0x46, 0x4A, 0x52, 0x62, 0x42, 0x42, 0x3C, 0x00, 0x00]);
        patterns.insert('1', [0x00, 0x08, 0x18, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x3E, 0x00, 0x00]);
        patterns.insert('2', [0x00, 0x3C, 0x42, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x7E, 0x00, 0x00]);
        patterns.insert('3', [0x00, 0x3C, 0x42, 0x02, 0x1C, 0x02, 0x02, 0x02, 0x42, 0x3C, 0x00, 0x00]);
        patterns.insert('4', [0x00, 0x04, 0x0C, 0x14, 0x24, 0x44, 0x7E, 0x04, 0x04, 0x04, 0x00, 0x00]);
        patterns.insert('5', [0x00, 0x7E, 0x40, 0x40, 0x7C, 0x02, 0x02, 0x02, 0x42, 0x3C, 0x00, 0x00]);
        patterns.insert('6', [0x00, 0x1C, 0x20, 0x40, 0x7C, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00, 0x00]);
        patterns.insert('7', [0x00, 0x7E, 0x02, 0x04, 0x08, 0x08, 0x10, 0x10, 0x20, 0x20, 0x00, 0x00]);
        patterns.insert('8', [0x00, 0x3C, 0x42, 0x42, 0x3C, 0x42, 0x42, 0x42, 0x42, 0x3C, 0x00, 0x00]);
        patterns.insert('9', [0x00, 0x3C, 0x42, 0x42, 0x42, 0x3E, 0x02, 0x04, 0x08, 0x70, 0x00, 0x00]);
        
        // Special characters
        patterns.insert(':', [0x00, 0x00, 0x00, 0x18, 0x18, 0x00, 0x00, 0x18, 0x18, 0x00, 0x00, 0x00]);
        patterns.insert('.', [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00, 0x00]);
        patterns.insert('%', [0x00, 0x62, 0x64, 0x08, 0x10, 0x10, 0x20, 0x26, 0x46, 0x00, 0x00, 0x00]);
        patterns.insert(' ', [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]);
        
        patterns
    }
    
    fn set_pixel_direct(data: &mut [u8], x: usize, y: usize, width: usize, color: (u8, u8, u8)) {
        let idx = (y * width + x) * 3;
        if idx + 2 < data.len() {
            data[idx] = color.0;
            data[idx + 1] = color.1;
            data[idx + 2] = color.2;
        }
    }
    
    fn process_sample_static(
        buffer: &gst::BufferRef,
        info: &gst_video::VideoInfo,
        inference_backend: &Arc<Mutex<Box<dyn InferenceBackend>>>,
        confidence_threshold: f32,
    ) -> Result<Vec<Detection>, Box<dyn std::error::Error + Send + Sync>> {
        let map = buffer.map_readable().map_err(|e| format!("Failed to map buffer: {}", e))?;
        let data = map.as_slice();
        
        let width = info.width() as usize;
        let height = info.height() as usize;
        let expected_size = width * height * 3;
        
        if data.len() < expected_size {
            return Err(format!("Buffer too small: got {}, expected {}", data.len(), expected_size).into());
        }
        
        // Convert RGB to tensor
        let tensor_data: Vec<f32> = data
            .chunks(3)
            .take(width * height)
            .flat_map(|pixel| {
                [
                    pixel[0] as f32 / 255.0,
                    pixel[1] as f32 / 255.0,
                    pixel[2] as f32 / 255.0,
                ]
            })
            .collect();
        
        // Reshape to CHW format
        let mut chw_data = vec![0.0f32; 3 * width * height];
        for y in 0..height {
            for x in 0..width {
                let hwc_idx = (y * width + x) * 3;
                let chw_r_idx = y * width + x;
                let chw_g_idx = width * height + y * width + x;
                let chw_b_idx = 2 * width * height + y * width + x;
                
                chw_data[chw_r_idx] = tensor_data[hwc_idx];
                chw_data[chw_g_idx] = tensor_data[hwc_idx + 1];
                chw_data[chw_b_idx] = tensor_data[hwc_idx + 2];
            }
        }
        
        let backend = inference_backend.lock().map_err(|_| "Failed to lock inference backend")?;
        let result = backend.infer(&chw_data)?;
        
        match result {
            TaskOutput::Detections(detections) => {
                Ok(detections.into_iter().filter(|d| d.score > confidence_threshold).collect())
            }
        }
    }
    
    pub fn run(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        println!("Starting live video with YOLO inference overlays...");
        println!("Video window should open with real-time bounding box overlays.");
        println!("Close window or press Ctrl+C to stop.");
        
        self.pipeline.set_state(gst::State::Playing)?;
        
        let bus = self.pipeline.bus().unwrap();
        for msg in bus.iter_timed(gst::ClockTime::NONE) {
            match msg.view() {
                gst::MessageView::Eos(..) => {
                    println!("End of stream");
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
                    if state_changed.src().map(|s| s == &self.pipeline).unwrap_or(false) {
                        println!("Pipeline state changed from {:?} to {:?}", 
                            state_changed.old(), state_changed.current());
                    }
                }
                _ => {}
            }
        }
        
        self.pipeline.set_state(gst::State::Null)?;
        
        let total_frames = *self.frame_count.lock().unwrap();
        println!("Processed {} frames total", total_frames);
        println!("Live video processing completed");
        Ok(())
    }
}