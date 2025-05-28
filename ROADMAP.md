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
    fn infer(&self, input: &Tensor) -> Result<TaskOutput, Self::Error>;
    fn get_input_shape(&self) -> &[usize];
    fn get_task_type(&self) -> TaskType;
}

pub trait ModelPostProcessor {
    fn process_raw_output(&self, output: RawInferenceOutput) -> TaskOutput;
    fn get_supported_tasks(&self) -> Vec<TaskType>;
}

#[derive(Debug, Clone)]
pub enum TaskType {
    ObjectDetection,
    KeypointDetection,
    PoseEstimation,
    ImageClassification,
    SemanticSegmentation,
    InstanceSegmentation,
    FacialRecognition,
    ActivityRecognition,
}

#[derive(Debug, Clone)]
pub enum TaskOutput {
    Detections(Vec<Detection>),
    Keypoints(Vec<Keypoint>),
    Poses(Vec<Pose>),
    Classifications(Vec<Classification>),
    Segmentation(SegmentationMask),
    Activities(Vec<Activity>),
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

### 2.2 Multi-Task Model Plugin System
```rust
// src/models/plugin_manager.rs
pub struct ModelPluginManager {
    registered_models: HashMap<String, Box<dyn ModelPlugin>>,
    task_registry: HashMap<TaskType, Vec<String>>, // task -> model names
    model_chains: HashMap<String, ModelChain>, // predefined model chains
}

pub trait ModelPlugin {
    fn name(&self) -> &str;
    fn supported_tasks(&self) -> Vec<TaskType>;
    fn create_backend(&self, config: &InferenceConfig) -> Box<dyn InferenceBackend>;
    fn create_postprocessor(&self) -> Box<dyn ModelPostProcessor>;
    fn get_metadata(&self) -> ModelMetadata;
}

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub name: String,
    pub task_type: TaskType,
    pub input_shape: Vec<usize>,
    pub output_format: OutputFormat,
    pub preprocessing_requirements: PreprocessingRequirements,
    pub performance_profile: PerformanceProfile,
}

// Support for model chaining (e.g., detect objects -> extract keypoints)
pub struct ModelChain {
    pub stages: Vec<ModelStage>,
    pub fusion_strategy: Option<FusionStrategy>,
}

pub struct ModelStage {
    pub model_name: String,
    pub depends_on: Option<String>, // previous stage output
    pub roi_extraction: Option<RoiExtractionStrategy>, // for keypoint detection on detected objects
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

### 3.3 Multi-Stream & Multi-Task Support
```rust
// src/pipeline/multi_stream.rs
pub struct MultiStreamPipeline {
    streams: HashMap<StreamId, StreamPipeline>,
    model_pool: Arc<ModelPool>, // shared models across streams
    task_scheduler: Box<dyn TaskScheduler>,
    model_chains: HashMap<String, ModelChain>,
}

pub struct ModelPool {
    models: HashMap<String, Arc<dyn InferenceBackend>>,
    task_router: TaskRouter,
    resource_manager: ResourceManager,
}

// Route frames to appropriate models based on content or configuration
pub struct TaskRouter {
    routing_rules: Vec<RoutingRule>,
    default_tasks: Vec<TaskType>,
}

pub struct RoutingRule {
    pub condition: RoutingCondition,
    pub target_tasks: Vec<TaskType>,
    pub priority: u8,
}

pub enum RoutingCondition {
    SceneType(SceneType),
    ObjectCount(usize),
    FrameRate(f32),
    Custom(Box<dyn Fn(&Frame) -> bool>),
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

## Phase 5: Performance Optimisation & Multi-Model Coordination (Weeks 9-10)

### 5.1 Multi-Model Performance Optimisation
```rust
// src/inference/multi_model.rs
pub struct MultiModelInference {
    models: HashMap<TaskType, Vec<Arc<dyn InferenceBackend>>>,
    scheduler: ModelScheduler,
    resource_pool: SharedResourcePool,
}

pub struct ModelScheduler {
    execution_strategy: ExecutionStrategy,
    load_balancer: LoadBalancer,
    performance_monitor: PerformanceMonitor,
}

pub enum ExecutionStrategy {
    Sequential, // Run models one after another
    Parallel,   // Run compatible models simultaneously
    Adaptive,   // Switch based on system resources
    Conditional, // Run models based on previous results
}

// Example: Detect objects, then run keypoint detection only on detected people
pub struct ConditionalExecution {
    primary_task: TaskType,
    conditional_tasks: HashMap<i32, TaskType>, // class_id -> task
    roi_extractor: RoiExtractor,
}
```

### 5.2 Advanced Model Chaining
```rust
// src/models/chaining.rs
pub struct ModelChainExecutor {
    chain_definition: ModelChain,
    execution_context: ExecutionContext,
    result_aggregator: ResultAggregator,
}

impl ModelChainExecutor {
    // Execute: Object Detection -> Keypoint Detection on detected persons
    pub async fn execute_person_keypoint_chain(&self, frame: &Frame) -> Result<ChainedResult> {
        // Stage 1: Object detection
        let detections = self.execute_stage("object_detection", frame).await?;
        
        // Stage 2: Extract person ROIs and run keypoint detection
        let person_detections = detections.filter_by_class(PERSON_CLASS_ID);
        let mut keypoint_results = Vec::new();
        
        for detection in person_detections {
            let roi = self.extract_roi(frame, &detection);
            let keypoints = self.execute_stage("keypoint_detection", &roi).await?;
            keypoint_results.push((detection, keypoints));
        }
        
        Ok(ChainedResult::PersonKeypoints(keypoint_results))
    }
}

#[derive(Debug, Clone)]
pub enum ChainedResult {
    PersonKeypoints(Vec<(Detection, Vec<Keypoint>)>),
    ObjectsWithPoses(Vec<(Detection, Pose)>),
    ActivityWithContext(Activity, Vec<Detection>),
}
```

## Phase 6: Advanced Multi-Task Computer Vision (Weeks 11-12)

### 6.1 Computer Vision Task Library
```rust
// src/vision/tasks/mod.rs
pub mod object_detection;
pub mod keypoint_detection;
pub mod pose_estimation;
pub mod activity_recognition;
pub mod facial_recognition;
pub mod segmentation;

// src/vision/tasks/keypoint_detection.rs
#[derive(Debug, Clone)]
pub struct Keypoint {
    pub x: f32,
    pub y: f32,
    pub confidence: f32,
    pub keypoint_type: KeypointType,
}

#[derive(Debug, Clone)]
pub enum KeypointType {
    // COCO-style 17 keypoints
    Nose, LeftEye, RightEye, LeftEar, RightEar,
    LeftShoulder, RightShoulder, LeftElbow, RightElbow,
    LeftWrist, RightWrist, LeftHip, RightHip,
    LeftKnee, RightKnee, LeftAnkle, RightAnkle,
    // Face keypoints (68 points)
    FaceLandmark(u8),
    // Hand keypoints (21 points per hand)
    HandLandmark(HandSide, u8),
}

#[derive(Debug, Clone)]
pub struct Pose {
    pub keypoints: Vec<Keypoint>,
    pub skeleton_connections: Vec<(usize, usize)>,
    pub pose_confidence: f32,
    pub bounding_box: Option<BoundingBox>,
}

// src/vision/tasks/pose_estimation.rs
pub struct PoseEstimator {
    backend: Box<dyn InferenceBackend>,
    pose_config: PoseConfig,
}

#[derive(Debug, Clone)]
pub struct PoseConfig {
    pub skeleton_type: SkeletonType,
    pub minimum_keypoint_confidence: f32,
    pub minimum_pose_confidence: f32,
}

#[derive(Debug, Clone)]
pub enum SkeletonType {
    Coco17,      // 17 keypoints for whole body
    Body25,      // OpenPose 25 keypoints
    Face68,      // 68 facial landmarks
    Hand21,      // 21 hand keypoints
    WholeBody,   // Body + Face + Hands
}
```

### 6.2 Real-time Multi-Task Processing
```rust
// src/pipeline/multi_task.rs
pub struct MultiTaskPipeline {
    task_graph: TaskGraph,
    execution_engine: TaskExecutionEngine,
    result_compositor: ResultCompositor,
}

pub struct TaskGraph {
    nodes: HashMap<TaskId, TaskNode>,
    dependencies: HashMap<TaskId, Vec<TaskId>>,
    execution_order: Vec<TaskId>,
}

pub struct TaskNode {
    pub task_type: TaskType,
    pub model_name: String,
    pub input_requirements: InputRequirements,
    pub output_format: OutputFormat,
}

// Example task graph: Webcam -> Object Detection -> Keypoint Detection (on persons) + Activity Recognition
pub fn create_realtime_analysis_pipeline() -> MultiTaskPipeline {
    let mut graph = TaskGraph::new();
    
    // Object detection on full frame
    graph.add_task("object_detection", TaskType::ObjectDetection, "yolov8n");
    
    // Keypoint detection on detected persons
    graph.add_task("keypoint_detection", TaskType::KeypointDetection, "pose_hrnet")
         .depends_on("object_detection")
         .with_filter(|detection| detection.class_id == PERSON_CLASS_ID);
    
    // Activity recognition on full frame
    graph.add_task("activity_recognition", TaskType::ActivityRecognition, "slowfast")
         .with_temporal_window(16); // 16 frames for temporal analysis
    
    MultiTaskPipeline::new(graph)
}
```

### 6.3 Enhanced Visualization System
```rust
// src/visualization/overlay.rs
pub struct MultiTaskOverlay {
    renderers: HashMap<TaskType, Box<dyn TaskRenderer>>,
    composition_strategy: CompositionStrategy,
}

pub trait TaskRenderer {
    fn render(&self, result: &TaskOutput, canvas: &mut Canvas) -> Result<()>;
    fn get_render_priority(&self) -> u8;
}

// src/visualization/renderers/keypoint_renderer.rs
pub struct KeypointRenderer {
    skeleton_config: SkeletonConfig,
    style_config: KeypointStyleConfig,
}

impl TaskRenderer for KeypointRenderer {
    fn render(&self, result: &TaskOutput, canvas: &mut Canvas) -> Result<()> {
        if let TaskOutput::Keypoints(keypoints) = result {
            for keypoint in keypoints {
                self.draw_keypoint(canvas, keypoint)?;
            }
            self.draw_skeleton_connections(canvas, keypoints)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct KeypointStyleConfig {
    pub keypoint_radius: f32,
    pub keypoint_color: Color,
    pub skeleton_line_width: f32,
    pub skeleton_color: Color,
    pub confidence_threshold: f32,
}
```

### 6.4 Model Hub Integration for Multi-Task Models
```rust
// src/models/hub_integration.rs
pub struct MultiTaskModelHub {
    hub_client: HuggingFaceClient,
    local_cache: ModelCache,
    task_model_registry: TaskModelRegistry,
}

impl MultiTaskModelHub {
    pub async fn download_keypoint_model(&self, variant: KeypointModelVariant) -> Result<ModelBundle> {
        let repo_id = match variant {
            KeypointModelVariant::HRNet => "microsoft/hrnet-w32-pose",
            KeypointModelVariant::OpenPose => "openpose/body_25",
            KeypointModelVariant::MediaPipe => "mediapipe/pose_landmark",
        };
        self.download_and_convert_to_onnx(repo_id).await
    }
    
    pub async fn download_activity_model(&self, variant: ActivityModelVariant) -> Result<ModelBundle> {
        let repo_id = match variant {
            ActivityModelVariant::SlowFast => "facebookresearch/slowfast-r50",
            ActivityModelVariant::X3D => "facebookresearch/x3d-m",
            ActivityModelVariant::VideoMAE => "videomae/videomae-base",
        };
        self.download_and_convert_to_onnx(repo_id).await
    }
}
```

### 6.5 Example Multi-Task Configuration
```toml
# config/multi_task_realtime.toml
[pipeline]
video_source = "webcam"
display_enabled = true
framerate = 30

[tasks]
# Primary object detection
[[tasks.models]]
name = "primary_detector"
task_type = "object_detection"
model_path = "models/yolov8n.onnx"
confidence_threshold = 0.5
execution_order = 1

# Keypoint detection on detected persons
[[tasks.models]]
name = "pose_estimator"
task_type = "keypoint_detection"
model_path = "models/hrnet_w32_pose.onnx"
confidence_threshold = 0.3
execution_order = 2
depends_on = "primary_detector"
filter_classes = [0]  # Person class only

# Activity recognition with temporal context
[[tasks.models]]
name = "activity_classifier"
task_type = "activity_recognition"
model_path = "models/slowfast_r50.onnx"
confidence_threshold = 0.7
execution_order = 3
temporal_window = 16

[visualization]
# Overlay configuration for different tasks
[visualization.object_detection]
enabled = true
show_labels = true
show_confidence = true
bbox_color = "#FF0000"

[visualization.keypoint_detection]
enabled = true
skeleton_type = "coco17"
keypoint_radius = 3
skeleton_line_width = 2
keypoint_color = "#00FF00"
skeleton_color = "#0000FF"

[visualization.activity_recognition]
enabled = true
show_top_k = 3
text_position = "top_left"
font_size = 16
```

## Phase 7: Advanced Features & Production Readiness (Weeks 13-14)

### 7.1 Model Ensemble Support
```rust
// src/ensemble/mod.rs
pub struct EnsembleInference {
    models: Vec<Box<dyn InferenceBackend>>,
    fusion_strategy: Box<dyn FusionStrategy>,
    voting_mechanism: VotingMechanism,
}

pub trait FusionStrategy {
    fn fuse_detections(&self, predictions: Vec<Vec<Detection>>) -> Vec<Detection>;
    fn fuse_keypoints(&self, predictions: Vec<Vec<Keypoint>>) -> Vec<Keypoint>;
    fn fuse_activities(&self, predictions: Vec<Vec<Activity>>) -> Vec<Activity>;
}
```

### 7.2 Real-time Tracking Integration
```rust
// src/tracking/mod.rs
pub struct MultiTaskTracker {
    object_tracker: Box<dyn TrackingBackend>,
    pose_tracker: Box<dyn PoseTrackingBackend>,
    activity_tracker: Box<dyn ActivityTrackingBackend>,
    track_history: HashMap<TrackId, MultiModalTrack>,
}

pub struct MultiModalTrack {
    pub track_id: TrackId,
    pub object_track: ObjectTrack,
    pub pose_history: VecDeque<Pose>,
    pub activity_sequence: VecDeque<Activity>,
    pub last_updated: Instant,
}
```

### 7.3 Streaming and Recording
```rust
// src/streaming/mod.rs
pub struct StreamingPipeline {
    rtmp_server: RtmpServer,
    recording_manager: RecordingManager,
    quality_controller: AdaptiveQualityController,
    multi_task_overlay: MultiTaskOverlay,
}
```

## Implementation Priorities

### High Priority
1. **✅ Modular refactoring** - Essential for maintainability (COMPLETED)
2. **Multi-task architecture** - Support for keypoint detection, pose estimation, activity recognition
3. **Configuration system enhancement** - Multi-model configuration support
4. **GStreamer plugin development** - Pipeline integration with multi-task support

### Medium Priority
1. **Model plugin system** - Extensibility for different computer vision tasks
2. **Model chaining and coordination** - Sequential and parallel task execution
3. **Performance monitoring** - Production readiness for multi-model pipelines
4. **Enhanced visualization** - Keypoints, poses, activities overlay

### Low Priority
1. **Model ensemble support** - Advanced ML features for improved accuracy
2. **Real-time tracking** - Multi-modal tracking across tasks
3. **Streaming infrastructure** - Deployment features with multi-task overlays

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

#### Object Detection
- **Ultralytics**: YOLO variants (YOLOv8, YOLOv9, YOLOv10)
- **Microsoft**: DETR, DINO variants
- **Facebook**: DETR, Detectron2 models

#### Keypoint Detection & Pose Estimation  
- **Microsoft**: HRNet variants for pose estimation
- **Google**: MediaPipe Pose, BlazePose
- **CMU**: OpenPose models (Body25, COCO)
- **Facebook**: DensePose for dense pose estimation

#### Activity Recognition
- **Facebook**: SlowFast, X3D models
- **Google**: VideoMAE, ViViT
- **Microsoft**: VideoSwin Transformer

#### Facial Analysis
- **MediaPipe**: Face detection and landmark detection
- **InsightFace**: Face recognition and analysis
- **MTCNN**: Multi-task CNN for face detection

#### General Sources
- **Hugging Face Model Hub**: Pre-trained models with automatic downloads
- **ONNX Model Zoo**: Standard computer vision models
- **Custom training**: Domain-specific models

## Real-time Multi-Task Computer Vision Capabilities

Inspired by projects like [minitflite-cpp-example](https://github.com/tallamjr/minitflite-cpp-example), pup will support comprehensive real-time computer vision tasks:

### Supported Tasks
- **Object Detection**: Real-time detection of multiple object classes
- **Keypoint Detection**: Human pose keypoints (COCO-17, Body25, custom)
- **Pose Estimation**: Full body pose analysis with skeleton rendering
- **Activity Recognition**: Temporal activity classification from video sequences
- **Facial Analysis**: Face detection, landmark detection, recognition
- **Segmentation**: Instance and semantic segmentation

### Multi-Model Coordination
- **Sequential Processing**: Detect objects → Extract keypoints on detected persons
- **Parallel Processing**: Run multiple models simultaneously on same frame
- **Conditional Processing**: Adaptive model selection based on scene content
- **Model Chaining**: Complex pipelines with multiple processing stages

### Real-time Performance
- **Hardware Acceleration**: CoreML, Metal, CUDA support via ONNX Runtime
- **Model Optimization**: TensorRT, OpenVINO integration
- **Memory Management**: Zero-copy processing with GStreamer buffers
- **Batched Inference**: Efficient processing of multiple frames

### Example Use Cases
1. **Sports Analysis**: Detect players → Extract pose keypoints → Analyze movement patterns
2. **Security Monitoring**: Detect persons → Facial recognition → Activity classification
3. **Fitness Applications**: Pose estimation → Form analysis → Rep counting
4. **Retail Analytics**: Person detection → Pose analysis → Behaviour understanding

This roadmap transforms pup from a monolithic proof-of-concept into a production-ready, extensible multi-task computer vision framework whilst maintaining the core strengths of Rust performance and GStreamer flexibility.