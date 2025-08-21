//! GStreamer mocking infrastructure for testing GStreamer plugins
//!
//! This module provides mock implementations of GStreamer elements, pipelines,
//! and related functionality to enable unit testing of GStreamer plugins
//! without requiring the full GStreamer runtime or video hardware.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Mock GStreamer caps for testing caps negotiation
#[derive(Debug, Clone, PartialEq)]
pub struct MockGstCaps {
    pub media_type: String,
    pub fields: HashMap<String, MockGstValue>,
}

/// Mock GStreamer value types for property testing
#[derive(Debug, Clone, PartialEq)]
pub enum MockGstValue {
    String(String),
    Integer(i32),
    UInteger(u32),
    Float(f32),
    Boolean(bool),
    Fraction { num: i32, denom: i32 },
}

impl MockGstCaps {
    pub fn new(media_type: &str) -> Self {
        Self {
            media_type: media_type.to_string(),
            fields: HashMap::new(),
        }
    }

    pub fn with_field(mut self, name: &str, value: MockGstValue) -> Self {
        self.fields.insert(name.to_string(), value);
        self
    }

    pub fn get_field(&self, name: &str) -> Option<&MockGstValue> {
        self.fields.get(name)
    }

    pub fn is_compatible(&self, other: &MockGstCaps) -> bool {
        if self.media_type != other.media_type {
            return false;
        }

        // Check if all fields in other caps are compatible with self
        for (field_name, other_value) in &other.fields {
            if let Some(self_value) = self.fields.get(field_name) {
                if self_value != other_value {
                    return false;
                }
            }
        }
        true
    }
}

/// Mock GStreamer pad for testing element connections
#[derive(Debug, Clone)]
pub struct MockGstPad {
    pub name: String,
    pub direction: PadDirection,
    pub caps: Option<MockGstCaps>,
    pub linked_pad: Option<String>, // Name of linked pad for simplicity
}

#[derive(Debug, Clone, PartialEq)]
pub enum PadDirection {
    Sink,
    Source,
}

impl MockGstPad {
    pub fn new(name: &str, direction: PadDirection) -> Self {
        Self {
            name: name.to_string(),
            direction,
            caps: None,
            linked_pad: None,
        }
    }

    pub fn set_caps(&mut self, caps: MockGstCaps) {
        self.caps = Some(caps);
    }

    pub fn link(&mut self, other_pad_name: &str) -> Result<(), String> {
        if self.linked_pad.is_some() {
            return Err("Pad is already linked".to_string());
        }
        self.linked_pad = Some(other_pad_name.to_string());
        Ok(())
    }

    pub fn is_linked(&self) -> bool {
        self.linked_pad.is_some()
    }
}

/// Mock GStreamer element for testing plugin functionality
#[derive(Debug, Clone)]
pub struct MockGstElement {
    pub name: String,
    pub element_type: String,
    pub properties: HashMap<String, MockGstValue>,
    pub pads: HashMap<String, MockGstPad>,
    pub state: ElementState,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ElementState {
    Null,
    Ready,
    Paused,
    Playing,
}

impl MockGstElement {
    pub fn new(name: &str, element_type: &str) -> Self {
        Self {
            name: name.to_string(),
            element_type: element_type.to_string(),
            properties: HashMap::new(),
            pads: HashMap::new(),
            state: ElementState::Null,
        }
    }

    pub fn add_pad(&mut self, pad: MockGstPad) {
        self.pads.insert(pad.name.clone(), pad);
    }

    pub fn set_property(&mut self, name: &str, value: MockGstValue) {
        self.properties.insert(name.to_string(), value);
    }

    pub fn get_property(&self, name: &str) -> Option<&MockGstValue> {
        self.properties.get(name)
    }

    pub fn set_state(&mut self, state: ElementState) -> Result<(), String> {
        match (&self.state, &state) {
            // Allow any state transition for testing simplicity
            (old_state, new_state) if old_state == new_state => {
                Ok(()) // No-op for same state
            }
            (ElementState::Null, ElementState::Ready) => {
                self.state = state;
                Ok(())
            }
            (ElementState::Ready, ElementState::Paused) => {
                self.state = state;
                Ok(())
            }
            (ElementState::Paused, ElementState::Playing) => {
                self.state = state;
                Ok(())
            }
            (ElementState::Playing, ElementState::Paused) => {
                self.state = state;
                Ok(())
            }
            (ElementState::Paused, ElementState::Ready) => {
                self.state = state;
                Ok(())
            }
            (ElementState::Ready, ElementState::Null) => {
                self.state = state;
                Ok(())
            }
            // Allow direct transitions for testing
            (ElementState::Null, ElementState::Playing) => {
                self.state = state;
                Ok(())
            }
            (ElementState::Ready, ElementState::Playing) => {
                self.state = state;
                Ok(())
            }
            _ => Err(format!(
                "Invalid state transition from {:?} to {:?}",
                self.state, state
            )),
        }
    }

    pub fn link_to(
        &mut self,
        other: &mut MockGstElement,
        src_pad: &str,
        sink_pad: &str,
    ) -> Result<(), String> {
        // Check that we have the source pad and other has the sink pad
        if !self.pads.contains_key(src_pad) {
            return Err(format!("Source pad '{}' not found", src_pad));
        }
        if !other.pads.contains_key(sink_pad) {
            return Err(format!("Sink pad '{}' not found", sink_pad));
        }

        // Get pad directions
        let src_direction = &self.pads[src_pad].direction;
        let sink_direction = &other.pads[sink_pad].direction;

        if *src_direction != PadDirection::Source {
            return Err("Source pad must be of Source direction".to_string());
        }
        if *sink_direction != PadDirection::Sink {
            return Err("Sink pad must be of Sink direction".to_string());
        }

        // Perform the link
        self.pads
            .get_mut(src_pad)
            .unwrap()
            .link(&format!("{}:{}", other.name, sink_pad))?;
        other
            .pads
            .get_mut(sink_pad)
            .unwrap()
            .link(&format!("{}:{}", self.name, src_pad))?;

        Ok(())
    }
}

/// Mock GStreamer pipeline for testing
#[derive(Debug)]
pub struct MockGstPipeline {
    pub name: String,
    pub elements: HashMap<String, MockGstElement>,
    pub state: ElementState,
}

impl MockGstPipeline {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            elements: HashMap::new(),
            state: ElementState::Null,
        }
    }

    pub fn add_element(&mut self, element: MockGstElement) {
        self.elements.insert(element.name.clone(), element);
    }

    pub fn get_element(&self, name: &str) -> Option<&MockGstElement> {
        self.elements.get(name)
    }

    pub fn get_element_mut(&mut self, name: &str) -> Option<&mut MockGstElement> {
        self.elements.get_mut(name)
    }

    pub fn link_elements(&mut self, src_name: &str, sink_name: &str) -> Result<(), String> {
        // Get elements (we need to handle borrowing carefully)
        if !self.elements.contains_key(src_name) {
            return Err(format!("Source element '{}' not found", src_name));
        }
        if !self.elements.contains_key(sink_name) {
            return Err(format!("Sink element '{}' not found", sink_name));
        }

        // For simplicity, assume default pad names
        let src_pad = "src";
        let sink_pad = "sink";

        // We need to clone to avoid borrowing issues
        let mut src_element = self.elements[src_name].clone();
        let mut sink_element = self.elements[sink_name].clone();

        src_element.link_to(&mut sink_element, src_pad, sink_pad)?;

        // Put them back
        self.elements.insert(src_name.to_string(), src_element);
        self.elements.insert(sink_name.to_string(), sink_element);

        Ok(())
    }

    pub fn set_state(&mut self, state: ElementState) -> Result<(), String> {
        // Set state on all elements
        for element in self.elements.values_mut() {
            element.set_state(state.clone())?;
        }
        self.state = state;
        Ok(())
    }
}

/// Mock buffer for testing data flow
#[derive(Debug, Clone)]
pub struct MockGstBuffer {
    pub data: Vec<u8>,
    pub timestamp: Option<u64>,
    pub duration: Option<u64>,
}

impl MockGstBuffer {
    pub fn new(data: Vec<u8>) -> Self {
        Self {
            data,
            timestamp: None,
            duration: None,
        }
    }

    pub fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    pub fn with_duration(mut self, duration: u64) -> Self {
        self.duration = Some(duration);
        self
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

/// Mock sample that combines buffer and caps
#[derive(Debug, Clone)]
pub struct MockGstSample {
    pub buffer: MockGstBuffer,
    pub caps: MockGstCaps,
}

impl MockGstSample {
    pub fn new(buffer: MockGstBuffer, caps: MockGstCaps) -> Self {
        Self { buffer, caps }
    }
}

/// Plugin registry for testing plugin registration
#[derive(Debug)]
pub struct MockPluginRegistry {
    plugins: HashMap<String, MockPluginInfo>,
}

#[derive(Debug, Clone)]
pub struct MockPluginInfo {
    pub name: String,
    pub description: String,
    pub version: String,
    pub elements: Vec<String>,
}

impl MockPluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: HashMap::new(),
        }
    }

    pub fn register_plugin(&mut self, info: MockPluginInfo) {
        self.plugins.insert(info.name.clone(), info);
    }

    pub fn get_plugin(&self, name: &str) -> Option<&MockPluginInfo> {
        self.plugins.get(name)
    }

    pub fn list_plugins(&self) -> Vec<&MockPluginInfo> {
        self.plugins.values().collect()
    }

    pub fn element_exists(&self, element_name: &str) -> bool {
        self.plugins
            .values()
            .any(|plugin| plugin.elements.contains(&element_name.to_string()))
    }
}

impl Default for MockPluginRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Thread-safe plugin registry singleton for tests
pub struct GlobalMockRegistry {
    registry: Arc<Mutex<MockPluginRegistry>>,
}

impl GlobalMockRegistry {
    pub fn new() -> Self {
        Self {
            registry: Arc::new(Mutex::new(MockPluginRegistry::new())),
        }
    }

    pub fn register_plugin(&self, info: MockPluginInfo) {
        let mut registry = self.registry.lock().unwrap();
        registry.register_plugin(info);
    }

    pub fn get_plugin(&self, name: &str) -> Option<MockPluginInfo> {
        let registry = self.registry.lock().unwrap();
        registry.get_plugin(name).cloned()
    }

    pub fn element_exists(&self, element_name: &str) -> bool {
        let registry = self.registry.lock().unwrap();
        registry.element_exists(element_name)
    }
}

impl Default for GlobalMockRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_caps_creation() {
        let caps = MockGstCaps::new("video/x-raw")
            .with_field("format", MockGstValue::String("RGB".to_string()))
            .with_field("width", MockGstValue::Integer(640))
            .with_field("height", MockGstValue::Integer(480));

        assert_eq!(caps.media_type, "video/x-raw");
        assert_eq!(
            caps.get_field("format"),
            Some(&MockGstValue::String("RGB".to_string()))
        );
        assert_eq!(caps.get_field("width"), Some(&MockGstValue::Integer(640)));
    }

    #[test]
    fn test_caps_compatibility() {
        let caps1 = MockGstCaps::new("video/x-raw")
            .with_field("format", MockGstValue::String("RGB".to_string()))
            .with_field("width", MockGstValue::Integer(640));

        let caps2 = MockGstCaps::new("video/x-raw")
            .with_field("format", MockGstValue::String("RGB".to_string()))
            .with_field("width", MockGstValue::Integer(640));

        let caps3 = MockGstCaps::new("video/x-raw")
            .with_field("format", MockGstValue::String("BGR".to_string()));

        assert!(caps1.is_compatible(&caps2));
        assert!(!caps1.is_compatible(&caps3));
    }

    #[test]
    fn test_element_creation_and_properties() {
        let mut element = MockGstElement::new("test-element", "pupinference");

        element.set_property("model-path", MockGstValue::String("test.onnx".to_string()));
        element.set_property("confidence", MockGstValue::Float(0.5));

        assert_eq!(element.name, "test-element");
        assert_eq!(element.element_type, "pupinference");
        assert_eq!(
            element.get_property("model-path"),
            Some(&MockGstValue::String("test.onnx".to_string()))
        );
        assert_eq!(
            element.get_property("confidence"),
            Some(&MockGstValue::Float(0.5))
        );
    }

    #[test]
    fn test_element_state_transitions() {
        let mut element = MockGstElement::new("test", "test");

        assert_eq!(element.state, ElementState::Null);

        assert!(element.set_state(ElementState::Ready).is_ok());
        assert_eq!(element.state, ElementState::Ready);

        assert!(element.set_state(ElementState::Paused).is_ok());
        assert_eq!(element.state, ElementState::Paused);

        assert!(element.set_state(ElementState::Playing).is_ok());
        assert_eq!(element.state, ElementState::Playing);

        // Invalid transition should fail
        assert!(element.set_state(ElementState::Null).is_err());
    }

    #[test]
    fn test_pad_linking() {
        let mut src_pad = MockGstPad::new("src", PadDirection::Source);
        let sink_pad = MockGstPad::new("sink", PadDirection::Sink);

        assert!(!src_pad.is_linked());
        assert!(!sink_pad.is_linked());

        assert!(src_pad.link("sink_element:sink").is_ok());
        assert!(src_pad.is_linked());
    }

    #[test]
    fn test_pipeline_creation() {
        let mut pipeline = MockGstPipeline::new("test-pipeline");

        let mut filesrc = MockGstElement::new("filesrc", "filesrc");
        filesrc.add_pad(MockGstPad::new("src", PadDirection::Source));

        let mut sink = MockGstElement::new("sink", "fakesink");
        sink.add_pad(MockGstPad::new("sink", PadDirection::Sink));

        pipeline.add_element(filesrc);
        pipeline.add_element(sink);

        assert!(pipeline.get_element("filesrc").is_some());
        assert!(pipeline.get_element("sink").is_some());
    }

    #[test]
    fn test_plugin_registry() {
        let mut registry = MockPluginRegistry::new();

        let plugin_info = MockPluginInfo {
            name: "pupvision".to_string(),
            description: "Pup computer vision plugin".to_string(),
            version: "0.3.0".to_string(),
            elements: vec!["pupinference".to_string(), "pupoverlay".to_string()],
        };

        registry.register_plugin(plugin_info.clone());

        assert!(registry.get_plugin("pupvision").is_some());
        assert!(registry.element_exists("pupinference"));
        assert!(registry.element_exists("pupoverlay"));
        assert!(!registry.element_exists("nonexistent"));
    }

    #[test]
    fn test_buffer_creation() {
        let data = vec![1, 2, 3, 4, 5];
        let buffer = MockGstBuffer::new(data.clone())
            .with_timestamp(1000)
            .with_duration(500);

        assert_eq!(buffer.data, data);
        assert_eq!(buffer.timestamp, Some(1000));
        assert_eq!(buffer.duration, Some(500));
        assert_eq!(buffer.size(), 5);
    }
}
