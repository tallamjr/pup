//! PupOverlay GStreamer Element Implementation

use std::sync::Mutex;

use glib::subclass::prelude::*;
use gstreamer as gst;
use gstreamer_base as gst_base;
use gstreamer_base::subclass::prelude::*;
use gstreamer_video as gst_video;
use gstreamer_video::subclass::prelude::*;
use gstreamer_video::VideoFrameExt;
use once_cell::sync::Lazy;

use crate::utils::detection::Detection;

#[derive(Debug, Clone)]
struct Settings {
    show_labels: bool,
    show_confidence: bool,
    bbox_color: String,
    font_size: u32,
    line_thickness: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            show_labels: true,
            show_confidence: true,
            bbox_color: "#FF0000".to_string(),
            font_size: 12,
            line_thickness: 2,
        }
    }
}

#[derive(Default)]
pub struct PupOverlay {
    properties: Mutex<Settings>,
}

#[glib::object_subclass]
impl ObjectSubclass for PupOverlay {
    const NAME: &'static str = "PupOverlay";
    type Type = super::PupOverlay;
    type ParentType = gst_video::VideoFilter;
}

impl ObjectImpl for PupOverlay {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecBoolean::builder("show-labels")
                    .nick("Show Labels")
                    .blurb("Whether to show class labels")
                    .default_value(true)
                    .build(),
                glib::ParamSpecBoolean::builder("show-confidence")
                    .nick("Show Confidence")
                    .blurb("Whether to show confidence scores")
                    .default_value(true)
                    .build(),
                glib::ParamSpecString::builder("bbox-color")
                    .nick("Bounding Box Color")
                    .blurb("Color for bounding boxes (hex format)")
                    .default_value(Some("#FF0000"))
                    .build(),
                glib::ParamSpecUInt::builder("font-size")
                    .nick("Font Size")
                    .blurb("Font size for labels")
                    .minimum(8)
                    .maximum(72)
                    .default_value(12)
                    .build(),
                glib::ParamSpecUInt::builder("line-thickness")
                    .nick("Line Thickness")
                    .blurb("Thickness of bounding box lines")
                    .minimum(1)
                    .maximum(10)
                    .default_value(2)
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        match pspec.name() {
            "show-labels" => {
                let mut settings = self.properties.lock().unwrap();
                settings.show_labels = value.get().unwrap();
            }
            "show-confidence" => {
                let mut settings = self.properties.lock().unwrap();
                settings.show_confidence = value.get().unwrap();
            }
            "bbox-color" => {
                let mut settings = self.properties.lock().unwrap();
                settings.bbox_color = value.get().unwrap();
            }
            "font-size" => {
                let mut settings = self.properties.lock().unwrap();
                settings.font_size = value.get().unwrap();
            }
            "line-thickness" => {
                let mut settings = self.properties.lock().unwrap();
                settings.line_thickness = value.get().unwrap();
            }
            _ => {
                gst::warning!(CAT, obj: object, "Unknown property: {}", pspec.name());
            }
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let settings = self.properties.lock().unwrap();
        match pspec.name() {
            "show-labels" => settings.show_labels.to_value(),
            "show-confidence" => settings.show_confidence.to_value(),
            "bbox-color" => settings.bbox_color.to_value(),
            "font-size" => settings.font_size.to_value(),
            "line-thickness" => settings.line_thickness.to_value(),
            _ => {
                gst::warning!(CAT, "Unknown property: {}", pspec.name());
                glib::Value::from(&0)
            }
        }
    }
}

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

    fn pad_templates() -> &'static [gst::PadTemplate] {
        static PAD_TEMPLATES: Lazy<Vec<gst::PadTemplate>> = Lazy::new(|| {
            let caps = gst_video::VideoCapsBuilder::new()
                .format_list(vec![gst_video::VideoFormat::Rgb])
                .width_range(1..=i32::MAX)
                .height_range(1..=i32::MAX)
                .framerate_range(gst::Fraction::new(0, 1)..=gst::Fraction::new(i32::MAX, 1))
                .build();

            vec![
                gst::PadTemplate::new(
                    "sink",
                    gst::PadDirection::Sink,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap(),
                gst::PadTemplate::new(
                    "src",
                    gst::PadDirection::Src,
                    gst::PadPresence::Always,
                    &caps,
                )
                .unwrap(),
            ]
        });

        PAD_TEMPLATES.as_ref()
    }
}

impl BaseTransformImpl for PupOverlay {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn transform_ip(&self, buf: &mut gst::BufferRef) -> Result<gst::FlowSuccess, gst::FlowError> {
        let settings = self.properties.lock().unwrap().clone();

        // Extract video frame data
        let info = self
            .obj()
            .sink_pad()
            .current_caps()
            .and_then(|caps| gst_video::VideoInfo::from_caps(&caps).ok())
            .ok_or(gst::FlowError::NotSupported)?;

        let mut frame = gst_video::VideoFrameRef::from_buffer_ref_writable(buf, &info)
            .map_err(|_| gst::FlowError::Error)?;

        // Get detections from metadata (TODO: implement custom metadata)
        let detections = self.extract_detection_meta(buf);

        // Render overlay
        self.render_overlay(&mut frame, &detections, &settings)?;

        Ok(gst::FlowSuccess::Ok)
    }
}

impl VideoFilterImpl for PupOverlay {}

impl PupOverlay {
    fn extract_detection_meta(&self, buf: &gst::BufferRef) -> Vec<Detection> {
        // Extract detections from custom GStreamer metadata
        // Look for custom metadata attached by PupInference

        // Try to find custom tags with detection data
        if let Some(tags) = buf.meta::<gst::meta::TagsMeta>() {
            let tag_list = tags.tags();
            if let Some(comment) = tag_list.get::<gst::tags::Comment>() {
                if let Some(comment_str) = comment.get() {
                    if comment_str.starts_with("pup-detections:") {
                        // Parse the detection metadata
                        if let Some(structure_str) = comment_str.strip_prefix("pup-detections: ") {
                            gst::debug!(CAT, obj: self.obj(), "Found detection metadata: {}", structure_str);
                            // TODO: Implement full JSON parsing of detection metadata
                            return Vec::new();
                        }
                    }
                }
            }
        }

        // Fallback: return empty vector if no metadata found
        Vec::new()
    }

    fn render_overlay(
        &self,
        frame: &mut gst_video::VideoFrameRef<&mut gst::BufferRef>,
        detections: &[Detection],
        settings: &Settings,
    ) -> Result<(), gst::FlowError> {
        if detections.is_empty() {
            return Ok(());
        }

        let width = frame.info().width() as f32;
        let height = frame.info().height() as f32;

        gst::debug!(gst::CAT_RUST, obj: self.obj(),
                   "Rendering overlay for {} detections on {}x{} frame",
                   detections.len(), width, height);

        // Get frame data for direct pixel manipulation
        let frame_data = frame.plane_data_mut(0).ok_or(gst::FlowError::Error)?;
        let stride = frame.info().stride()[0] as usize;

        for detection in detections {
            // Convert normalized coordinates to pixel coordinates
            let x1 = (detection.x1 * width) as usize;
            let y1 = (detection.y1 * height) as usize;
            let x2 = (detection.x2 * width) as usize;
            let y2 = (detection.y2 * height) as usize;

            // Draw bounding box
            self.draw_rectangle(
                frame_data,
                stride,
                x1,
                y1,
                x2,
                y2,
                settings.line_thickness as usize,
            );

            // Add text rendering for labels and confidence
            if settings.show_labels || settings.show_confidence {
                let label = if settings.show_labels && settings.show_confidence {
                    format!(
                        "Class {}: {:.1}%",
                        detection.class_id,
                        detection.score * 100.0
                    )
                } else if settings.show_labels {
                    format!("Class {}", detection.class_id)
                } else {
                    format!("{:.1}%", detection.score * 100.0)
                };

                // Render simple text above bounding box
                // Note: This is a basic implementation - more sophisticated text rendering
                // would require a proper font rendering library
                self.render_simple_text(
                    frame_data,
                    stride,
                    x1,
                    y1.saturating_sub(15), // Position text above bbox
                    &label,
                    settings.font_size as usize,
                );
            }
        }

        Ok(())
    }

    fn draw_rectangle(
        &self,
        frame_data: &mut [u8],
        stride: usize,
        x1: usize,
        y1: usize,
        x2: usize,
        y2: usize,
        thickness: usize,
    ) {
        let (r, g, b) = (255u8, 0u8, 0u8); // Red color for now

        // Draw horizontal lines (top and bottom)
        for t in 0..thickness {
            if y1 + t < frame_data.len() / stride
                && y2.saturating_sub(t) < frame_data.len() / stride
            {
                self.draw_horizontal_line(frame_data, stride, x1, x2, y1 + t, r, g, b);
                if y2 > t {
                    self.draw_horizontal_line(frame_data, stride, x1, x2, y2 - t, r, g, b);
                }
            }
        }

        // Draw vertical lines (left and right)
        for t in 0..thickness {
            self.draw_vertical_line(frame_data, stride, x1 + t, y1, y2, r, g, b);
            if x2 > t {
                self.draw_vertical_line(frame_data, stride, x2 - t, y1, y2, r, g, b);
            }
        }
    }

    fn draw_horizontal_line(
        &self,
        frame_data: &mut [u8],
        stride: usize,
        x1: usize,
        x2: usize,
        y: usize,
        r: u8,
        g: u8,
        b: u8,
    ) {
        let row_start = y * stride;
        for x in x1..=x2.min((stride / 3).saturating_sub(1)) {
            let pixel_start = row_start + x * 3;
            if pixel_start + 2 < frame_data.len() {
                frame_data[pixel_start] = r;
                frame_data[pixel_start + 1] = g;
                frame_data[pixel_start + 2] = b;
            }
        }
    }

    fn draw_vertical_line(
        &self,
        frame_data: &mut [u8],
        stride: usize,
        x: usize,
        y1: usize,
        y2: usize,
        r: u8,
        g: u8,
        b: u8,
    ) {
        for y in y1..=y2 {
            let pixel_start = y * stride + x * 3;
            if pixel_start + 2 < frame_data.len() {
                frame_data[pixel_start] = r;
                frame_data[pixel_start + 1] = g;
                frame_data[pixel_start + 2] = b;
            }
        }
    }

    fn render_simple_text(
        &self,
        frame_data: &mut [u8],
        stride: usize,
        x: usize,
        y: usize,
        text: &str,
        font_size: usize,
    ) {
        // Simple text rendering using basic pixel drawing
        // TODO: Replace with proper font rendering library like cairo or freetype

        let char_width = font_size.max(8);
        let char_height = font_size.max(12);

        for (i, ch) in text.chars().enumerate() {
            let char_x = x + i * char_width;

            // Draw a simple rectangular background for each character
            for py in y..=(y + char_height).min(frame_data.len() / stride) {
                for px in char_x..=(char_x + char_width - 1) {
                    let pixel_start = py * stride + px * 3;
                    if pixel_start + 2 < frame_data.len() {
                        // Semi-transparent black background
                        frame_data[pixel_start] = frame_data[pixel_start] / 2; // R
                        frame_data[pixel_start + 1] = frame_data[pixel_start + 1] / 2; // G
                        frame_data[pixel_start + 2] = frame_data[pixel_start + 2] / 2;
                        // B
                    }
                }
            }

            // Draw simple character representation (basic blocks for now)
            // A full implementation would use actual font data
            if ch.is_ascii_alphanumeric() {
                let mid_x = char_x + char_width / 2;
                let mid_y = y + char_height / 2;

                // Draw a simple cross pattern to represent the character
                self.draw_horizontal_line(
                    frame_data,
                    stride,
                    char_x,
                    char_x + char_width - 1,
                    mid_y,
                    255,
                    255,
                    255,
                );
                self.draw_vertical_line(
                    frame_data,
                    stride,
                    mid_x,
                    y,
                    y + char_height,
                    255,
                    255,
                    255,
                );
            }
        }
    }
}
