# Roadmap: Pup Video Processing Architecture

This roadmap outlines the restructuring and optimisation of the pup codebase to create a modular, extensible video processing framework with GStreamer and ORT inference.

## Current State Analysis

### Issues with Current Architecture
- **Monolithic design**: All functionality crammed into `main.rs` (~350 lines)
- **Tight coupling**: GStreamer pipeline, ONNX inference, and image processing are intertwined
- **Hard-coded configuration**: Model paths, pipeline parameters, and thresholds are embedded in code
- **Limited extensibility**: Adding new models or processing steps requires code modification
- **Testing challenges**: Monolithic structure makes unit testing difficult
- **Memory inefficiency**: Redundant data copies between OpenCV and GStreamer

### Inspiration from Reference Projects
- **gstreamed_rust_inference**: Demonstrates modular backend support (Candle, ONNX Runtime)
- **gst-inference**: Shows plugin-based architecture with abstraction layers
- **GStreamer Rust examples**: Illustrate proper pipeline management patterns

## Phase 1: Modular Refactoring (Weeks 1-2)

### 1.1 Core Module Separation
```
src/
├── lib.rs                    # Public API and re-exports
├── pipeline/                 # GStreamer pipeline management
│   ├── mod.rs
│   ├── builder.rs           # Pipeline construction
│   ├── elements.rs          # Custom GStreamer elements
│   └── manager.rs           # Pipeline lifecycle management
├── inference/               # ML inference abstractions
│   ├── mod.rs
│   ├── ort_backend.rs       # ONNX Runtime implementation
│   ├── traits.rs           # Inference traits and interfaces
│   └── postprocess.rs       # Detection post-processing
├── preprocessing/           # Image processing pipeline
│   ├── mod.rs
│   ├── letterbox.rs         # Aspect-ratio preserving resize
│   ├── colorspace.rs        # Format conversions
│   └── normalization.rs     # Pixel value normalization
├── models/                  # Model management
│   ├── mod.rs
│   ├── yolo.rs             # YOLO-specific implementations
│   ├── registry.rs         # Model registration and discovery
│   └── metadata.rs         # Model metadata handling
├── config/                  # Configuration management
│   ├── mod.rs
│   ├── pipeline.rs         # Pipeline configuration
│   └── inference.rs        # Inference configuration
└── utils/                   # Shared utilities
    ├── mod.rs
    ├── detection.rs        # Detection data structures
    └── metrics.rs          # Performance metrics
```

### 1.2 Configuration-Driven Design
```toml
# config/default.toml
[pipeline]
video_source = "auto"  # "auto", "webcam", or file path
display_enabled = true
framerate = 30

[inference]
backend = "ort"
model_path = "models/yolov8n.onnx"
confidence_threshold = 0.5
device = "auto"  # "cpu", "coreml", "cuda"

[preprocessing]
target_size = [640, 640]
letterbox = true
normalize = true
```

### 1.3 Trait-Based Inference System
```rust
pub trait InferenceBackend {
    type Error;
    
    fn load_model(&mut self, path: &Path) -> Result<(), Self::Error>;
    fn infer(&self, input: &Tensor) -> Result<Vec<Detection>, Self::Error>;
    fn get_input_shape(&self) -> &[usize];
}

pub trait ModelPostProcessor {
    fn process_raw_output(&self, output: RawInferenceOutput) -> Vec<Detection>;
    fn apply_nms(&self, detections: Vec<Detection>) -> Vec<Detection>;
}
```

## Phase 2: GStreamer-RS Plugin Architecture (Weeks 3-4)

### 2.1 Official GStreamer Plugin Development
Following `gstreamer-rs` and `gst-plugins-rs` patterns for proper plugin registration:

```rust
// src/gst_plugins/mod.rs
use gstreamer as gst;
use gst::glib;

// Plugin registration following gstreamer-rs patterns
gst::plugin_define!(
    pupinference,
    env!("CARGO_PKG_DESCRIPTION"),
    plugin_init,
    concat!(env!("CARGO_PKG_VERSION"), "-", env!("COMMIT_ID")),
    "MIT/X11",
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_NAME"),
    env!("CARGO_PKG_REPOSITORY"),
    env!("BUILD_REL_DATE")
);

fn plugin_init(plugin: &gst::Plugin) -> Result<(), glib::BoolError> {
    pupinference::register(plugin)?;
    pupoverlay::register(plugin)?;
    Ok(())
}

// src/gst_plugins/pupinference/imp.rs
use gstreamer_base::subclass::prelude::*;
use gstreamer_video::{self as gst_video, subclass::prelude::*};

#[derive(Default)]
pub struct PupInference {
    inference_backend: Mutex<Option<Box<dyn InferenceBackend>>>,
    properties: Mutex<Settings>,
}

#[derive(Clone, Debug)]
struct Settings {
    model_path: Option<PathBuf>,
    confidence_threshold: f32,
    device: String,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            model_path: None,
            confidence_threshold: 0.5,
            device: "auto".to_string(),
        }
    }
}

#[glib::object_subclass]
impl ObjectSubclass for PupInference {
    const NAME: &'static str = "PupInference";
    type Type = super::PupInference;
    type ParentType = gst_video::VideoFilter;
}

impl ObjectImpl for PupInference {
    fn properties() -> &'static [glib::ParamSpec] {
        static PROPERTIES: Lazy<Vec<glib::ParamSpec>> = Lazy::new(|| {
            vec![
                glib::ParamSpecString::builder("model-path")
                    .nick("Model Path")
                    .blurb("Path to ONNX model file")
                    .build(),
                glib::ParamSpecFloat::builder("confidence-threshold")
                    .nick("Confidence Threshold")
                    .blurb("Minimum confidence for detections")
                    .minimum(0.0)
                    .maximum(1.0)
                    .default_value(0.5)
                    .build(),
                glib::ParamSpecString::builder("device")
                    .nick("Device")
                    .blurb("Inference device (cpu, coreml, cuda)")
                    .default_value(Some("auto"))
                    .build(),
            ]
        });
        PROPERTIES.as_ref()
    }

    fn set_property(&self, _id: usize, value: &glib::Value, pspec: &glib::ParamSpec) {
        match pspec.name() {
            "model-path" => {
                let mut settings = self.properties.lock().unwrap();
                settings.model_path = value.get::<Option<String>>().unwrap().map(PathBuf::from);
            }
            "confidence-threshold" => {
                let mut settings = self.properties.lock().unwrap();
                settings.confidence_threshold = value.get().unwrap();
            }
            "device" => {
                let mut settings = self.properties.lock().unwrap();
                settings.device = value.get().unwrap();
            }
            _ => unimplemented!(),
        }
    }

    fn property(&self, _id: usize, pspec: &glib::ParamSpec) -> glib::Value {
        let settings = self.properties.lock().unwrap();
        match pspec.name() {
            "model-path" => settings.model_path.as_ref().map(|p| p.to_string_lossy().as_ref()).to_value(),
            "confidence-threshold" => settings.confidence_threshold.to_value(),
            "device" => settings.device.to_value(),
            _ => unimplemented!(),
        }
    }
}

impl GstObjectImpl for PupInference {}
impl ElementImpl for PupInference {
    fn metadata() -> Option<&'static gst::subclass::ElementMetadata> {
        static ELEMENT_METADATA: Lazy<gst::subclass::ElementMetadata> = Lazy::new(|| {
            gst::subclass::ElementMetadata::new(
                "Pup ML Inference",
                "Filter/Effect/Video",
                "Performs object detection inference on video frames",
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

impl BaseTransformImpl for PupInference {
    const MODE: gst_base::subclass::BaseTransformMode =
        gst_base::subclass::BaseTransformMode::AlwaysInPlace;
    const PASSTHROUGH_ON_SAME_CAPS: bool = false;
    const TRANSFORM_IP_ON_PASSTHROUGH: bool = false;

    fn transform_ip(&self, buf: &mut gst::BufferRef) -> Result<gst::FlowSuccess, gst::FlowError> {
        let settings = self.properties.lock().unwrap().clone();
        
        // Extract video frame data following gstreamer-rs patterns
        let info = self.instance().sink_pad().current_caps()
            .and_then(|caps| gst_video::VideoInfo::from_caps(&caps).ok())
            .ok_or(gst::FlowError::NotSupported)?;

        let frame = gst_video::VideoFrameRef::from_buffer_ref_readable(buf, &info)
            .map_err(|_| gst::FlowError::Error)?;

        // Perform inference using established backend
        if let Some(backend) = self.inference_backend.lock().unwrap().as_ref() {
            let detections = self.process_frame(&frame, backend, &settings)?;
            
            // Store detections as buffer metadata (following gstreamer-rs patterns)
            self.attach_detection_meta(buf, detections);
        }

        Ok(gst::FlowSuccess::Ok)
    }
}

impl VideoFilterImpl for PupInference {}

// Pipeline usage becomes:
// filesrc location=video.mp4 ! decodebin ! videoconvert ! pupinference model-path=model.onnx confidence-threshold=0.6 ! pupoverlay ! autovideosink
```

### 2.2 Model Plugin System
```rust
// src/models/plugin_manager.rs
pub struct ModelPluginManager {
    registered_models: HashMap<String, Box<dyn ModelPlugin>>,
}

pub trait ModelPlugin {
    fn name(&self) -> &str;
    fn supported_tasks(&self) -> Vec<TaskType>;
    fn create_backend(&self, config: &InferenceConfig) -> Box<dyn InferenceBackend>;
    fn create_postprocessor(&self) -> Box<dyn ModelPostProcessor>;
}
```

### 2.3 Dynamic Model Loading
Support for Hugging Face model hub integration:
```rust
// src/models/hub_integration.rs
pub struct HuggingFaceModelLoader {
    cache_dir: PathBuf,
    api_client: HfApiClient,
}

impl HuggingFaceModelLoader {
    pub async fn download_model(&self, repo_id: &str, revision: Option<&str>) -> Result<ModelBundle>;
    pub fn load_from_cache(&self, model_id: &str) -> Result<ModelBundle>;
}
```

## Phase 3: Advanced Pipeline Management (Weeks 5-6)

### 3.1 Declarative Pipeline Configuration
```yaml
# pipelines/webcam_detection.yml
name: "webcam_object_detection"
description: "Real-time object detection from webcam"

sources:
  - type: "webcam"
    device: 0
    caps: "video/x-raw,width=1280,height=720"

processing:
  - element: "videoconvert"
  - element: "pup_inference"
    model: "yolov8n"
    device: "coreml"
  - element: "pup_overlay"
    show_labels: true
    show_confidence: true

outputs:
  - type: "display"
    sink: "autovideosink"
  - type: "rtmp"
    url: "rtmp://localhost/live/stream"
    optional: true
```

### 3.2 Performance Monitoring
```rust
// src/monitoring/metrics.rs
pub struct PipelineMetrics {
    fps: AtomicF64,
    inference_time: AtomicF64,
    preprocessing_time: AtomicF64,
    memory_usage: AtomicUsize,
}

pub struct MetricsCollector {
    metrics: Arc<PipelineMetrics>,
    reporter: Box<dyn MetricsReporter>,
}
```

### 3.3 Multi-Stream Support
```rust
// src/pipeline/multi_stream.rs
pub struct MultiStreamPipeline {
    streams: HashMap<StreamId, StreamPipeline>,
    shared_inference: Arc<dyn InferenceBackend>,
    scheduler: Box<dyn InferenceScheduler>,
}
```

## Phase 4: Performance Optimisation (Weeks 7-8)

### 4.1 GStreamer-RS Memory Management
Leverage official gstreamer-rs memory management patterns:
```rust
// src/buffers/gst_memory.rs
use gstreamer as gst;
use gstreamer_video as gst_video;

pub struct GstManagedBuffer {
    buffer: gst::Buffer,
    frame: gst_video::VideoFrame<gst_video::video_frame::Readable>,
    tensor_view: TensorView<f32>,
}

impl GstManagedBuffer {
    pub fn from_gst_buffer(
        buffer: gst::Buffer, 
        info: &gst_video::VideoInfo
    ) -> Result<Self, gst::FlowError> {
        // Use gstreamer-rs VideoFrame for safe memory access
        let frame = gst_video::VideoFrame::from_buffer_readable(buffer.clone(), info)
            .map_err(|_| gst::FlowError::Error)?;
        
        // Create tensor view without copying data
        let plane_data = frame.plane_data(0).ok_or(gst::FlowError::Error)?;
        let tensor_view = TensorView::from_slice(plane_data, frame.info().format_info().pixel_stride());
        
        Ok(Self { buffer, frame, tensor_view })
    }
    
    pub fn as_tensor(&self) -> &TensorView<f32> {
        &self.tensor_view
    }
    
    // Proper cleanup following gstreamer-rs patterns
    pub fn into_buffer(self) -> gst::Buffer {
        self.buffer
    }
}

// Memory pool using gstreamer-rs allocator
pub struct GstMemoryPool {
    allocator: gst::Allocator,
    params: gst::AllocationParams,
}

impl GstMemoryPool {
    pub fn with_allocator(allocator: gst::Allocator) -> Self {
        Self {
            allocator,
            params: gst::AllocationParams::default(),
        }
    }
    
    pub fn alloc_buffer(&self, size: usize) -> gst::Buffer {
        let memory = self.allocator.alloc(size, Some(&self.params));
        gst::Buffer::from_memory(memory)
    }
}
```

### 4.2 Inference Batching
```rust
// src/inference/batching.rs
pub struct BatchedInference {
    batch_size: usize,
    timeout: Duration,
    pending_frames: VecDeque<PendingFrame>,
    backend: Box<dyn InferenceBackend>,
}
```

### 4.3 Hardware Acceleration Pipeline
```rust
// src/acceleration/mod.rs
pub enum AccelerationBackend {
    CoreML,
    Metal,
    CUDA,
    OpenVINO,
}

pub struct HardwareAcceleratedPipeline {
    backend: AccelerationBackend,
    memory_pool: Arc<GpuMemoryPool>,
}
```

## Phase 5: Advanced Features (Weeks 9-10)

### 5.1 Model Ensemble Support
```rust
// src/ensemble/mod.rs
pub struct EnsembleInference {
    models: Vec<Box<dyn InferenceBackend>>,
    fusion_strategy: Box<dyn FusionStrategy>,
}

pub trait FusionStrategy {
    fn fuse_predictions(&self, predictions: Vec<Vec<Detection>>) -> Vec<Detection>;
}
```

### 5.2 Real-time Tracking Integration
```rust
// src/tracking/mod.rs
pub struct ObjectTracker {
    tracker_backend: Box<dyn TrackingBackend>,
    track_history: HashMap<TrackId, Track>,
}

pub trait TrackingBackend {
    fn update(&mut self, detections: Vec<Detection>) -> Vec<Track>;
    fn predict(&self, track_id: TrackId) -> Option<Detection>;
}
```

### 5.3 Streaming and Recording
```rust
// src/streaming/mod.rs
pub struct StreamingPipeline {
    rtmp_server: RtmpServer,
    recording_manager: RecordingManager,
    quality_controller: AdaptiveQualityController,
}
```

## Implementation Priorities

### High Priority
1. **Modular refactoring** - Essential for maintainability
2. **Configuration system** - Enables easy deployment variations
3. **ORT backend optimization** - Core inference performance
4. **GStreamer plugin development** - Pipeline integration

### Medium Priority
1. **Model plugin system** - Extensibility for different models
2. **Performance monitoring** - Production readiness
3. **Multi-stream support** - Scalability

### Low Priority
1. **Ensemble inference** - Advanced ML features
2. **Real-time tracking** - Value-added functionality
3. **Streaming infrastructure** - Deployment features

## Migration Strategy

### Week 1: Foundation
- Extract configuration management
- Create basic module structure
- Implement inference traits

### Week 2: Core Refactoring
- Move GStreamer code to pipeline module
- Implement ORT backend
- Create preprocessing pipeline

### Week 3: Integration
- Build unified API
- Create configuration-driven main application
- Implement basic testing framework

### Week 4: GStreamer-RS Integration & Optimization
- Implement proper GStreamer plugin registration following gst-plugins-rs patterns
- Use gstreamer-rs VideoFrame for safe memory management
- Add GStreamer bus message handling and error propagation
- Profile performance bottlenecks using GStreamer tools
- Add metrics collection via GStreamer tracers

## Success Metrics

- **Code maintainability**: Reduced cyclomatic complexity, improved test coverage
- **Performance**: <10ms inference latency, >30 FPS processing
- **Extensibility**: New models addable without core code changes
- **Memory efficiency**: <50% memory usage reduction through zero-copy
- **Developer experience**: <5 minutes to add new model support

## Technologies and Dependencies

### Core GStreamer-RS Ecosystem
- **gstreamer-rs**: Official Rust bindings (v0.23+)
- **gstreamer-video-rs**: Video-specific functionality
- **gstreamer-base-rs**: Base classes for custom elements
- **gstreamer-app-rs**: Application integration
- **gst-plugins-rs**: Reference implementation patterns

### Updated Cargo.toml Dependencies
```toml
[dependencies]
# Official GStreamer ecosystem
gstreamer = { version = "0.23", features = ["v1_14"] }
gstreamer-base = "0.23"
gstreamer-video = "0.23"
gstreamer-app = "0.23"

# ML and inference
ort = { version = "2.0.0-rc.9", features = ["coreml", "load-dynamic"] }
ndarray = "0.15"

# Configuration and serialization  
serde = { version = "1.0", features = ["derive"] }
toml = "0.8"

# Async and parallelism
tokio = { version = "1.0", features = ["full"] }
rayon = "1.7"

# Logging and error handling
tracing = "0.1"
tracing-subscriber = "0.3"
anyhow = "1.0"

# CLI
clap = { version = "4", features = ["derive"] }

# Plugin building
glib = "0.20"
once_cell = "1.19"
```

### Build Integration
```toml
# Following gst-plugins-rs patterns
[lib]
name = "gstpup"
crate-type = ["cdylib", "rlib"]

[[bin]]
name = "pup"
path = "src/bin/main.rs"

[package.metadata.capi]
min_version = "1.14.0"

[package.metadata.capi.header]
name = "gstpup"
subdirectory = "gstreamer-1.0"
targets = ["x86_64-unknown-linux-gnu", "aarch64-apple-darwin"]
```

### Core Technologies
- **GStreamer-RS**: Video pipeline management with safe Rust bindings
- **ONNX Runtime**: ML inference with hardware acceleration
- **Tokio**: Async runtime for I/O operations
- **Rayon**: Data parallelism

### Additional Integrations
- **Hugging Face Hub**: Model distribution
- **serde**: Configuration serialization
- **tracing**: Structured logging and metrics
- **clap**: CLI interface

### Model Sources
- **Hugging Face Model Hub**: Pre-trained models with automatic downloads
- **Ultralytics**: YOLO variants (YOLOv8, YOLOv9, YOLOv10)
- **ONNX Model Zoo**: Standard computer vision models
- **Custom training**: Domain-specific models

This roadmap transforms pup from a monolithic proof-of-concept into a production-ready, extensible video processing framework whilst maintaining the core strengths of Rust performance and GStreamer flexibility.