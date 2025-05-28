# Roadmap: Unified Pup Video Processing System

This roadmap outlines the development of a unified, production-ready video processing system that combines the best features of the current `pup` and `demo` binaries into a single, mode-driven application.

## Current State Analysis

### ✅ **Completed Achievements**
- ~~**Monolithic design**: All functionality crammed into `main.rs`~~ **FIXED**: Modular architecture implemented
- ~~**Limited extensibility**: Adding new models requires code modification~~ **FIXED**: Configuration-driven model loading
- ~~**Memory inefficiency**: Redundant data copies~~ **IMPROVED**: Using GStreamer buffers and ONNX Runtime efficiently
- ~~**Testing challenges**: Monolithic structure~~ **IMPROVED**: Modular testing structure in place
- ✅ **Basic GStreamer Integration**: Working pipelines for file and webcam input
- ✅ **ONNX Runtime Integration**: Successfully using ORT with "download-binaries" feature
- ✅ **Real-time Video Overlays**: Live mode with bitmap font rendering
- ✅ **Cross-platform Webcam Support**: Using autovideosrc for camera access
- ✅ **Configuration System**: Basic TOML configuration support

### 🚧 **Current Architecture Issues**
- **Dual Binary Confusion**: Users need to choose between `pup` and `demo`
- **CoreML Underutilized**: Not optimally configured for macOS performance
- **Limited Input Sources**: Only basic file/webcam support
- **Inconsistent UX**: Different interfaces for similar functionality

## Phase 1: Unified Binary Architecture (Weeks 1-2)

### 1.1 **Single Entry Point Design**
```rust
// New unified command structure
pup run --config production.toml              # Production mode
pup live --input webcam --model yolov8n.onnx  # Live video mode  
pup detect --input video.mp4 --output results.json  # Detection mode
pup benchmark --model yolov8n.onnx --frames 100    # Performance testing
```

**Migration Strategy**: 
- ~~Separate `pup` and `demo` binaries~~ → **Single `pup` binary with subcommands**
- Preserve all existing functionality from both binaries
- Maintain backward compatibility through configuration

### 1.2 **Enhanced Input Source Management**
```rust
#[derive(Debug, Clone)]
pub enum InputSource {
    Webcam { device_id: Option<u32> },
    File { path: PathBuf },
    Rtsp { url: String },
    Test { pattern: TestPattern },  // For CI/testing
}

pub struct InputManager {
    source_type: InputSource,
    gst_elements: Vec<gst::Element>,
    caps_filter: Option<gst::Caps>,
}
```

**Supported Sources**:
- 📹 **Webcam/USB Camera**: Auto-detection with device enumeration
- 📁 **Video Files**: MP4, AVI, MOV, WebM support
- 🌐 **RTSP Streams**: Network camera support
- 🧪 **Test Patterns**: Synthetic data for testing/CI

### 1.3 **CoreML-Optimized Configuration**
**Critical Issue**: [ORT CoreML Performance](https://github.com/pykeio/ort/issues/341#issuecomment-2913788946) shows 4x performance difference with proper setup.

```rust
// src/inference/coreml_config.rs
pub fn create_optimized_coreml_session(model_path: &Path) -> Result<Session> {
    let coreml_provider = CoreMLExecutionProvider::default()
        .with_compute_units(CoreMLComputeUnits::All)
        .with_static_input_shapes(true)
        .with_model_format(CoreMLModelFormat::MLProgram)
        .with_specialization_strategy(CoreMLSpecializationStrategy::FastPrediction);
    
    Session::builder()?
        .with_execution_providers([coreml_provider])?
        .with_model_from_file(model_path)
}
```

**Testing Priority**: 
- [ ] Benchmark inference latency with/without CoreML optimization
- [ ] Test memory usage patterns on various macOS hardware
- [ ] Validate model compatibility with CoreML provider

## Phase 2: Production Readiness (Weeks 3-4)

### 2.1 **Configuration System Enhancement**
```toml
# config/production.toml
[mode]
type = "production"  # production, live, detection, benchmark

[input]
source = "webcam"  # webcam, file path, rtsp://url
device_id = 0
caps = "video/x-raw,width=1280,height=720,framerate=30/1"

[inference]
backend = "ort"
execution_providers = ["coreml", "cpu"]  # Fallback order
model_path = "models/yolov8n.onnx"
confidence_threshold = 0.5
batch_size = 1

[output]
display_enabled = true
recording_enabled = false
output_format = "mp4"  # mp4, json, rtmp
```

### 2.2 **Comprehensive Error Handling**
```rust
#[derive(thiserror::Error, Debug)]
pub enum PupError {
    #[error("Input source not available: {source}")]
    InputNotAvailable { source: String },
    
    #[error("CoreML initialization failed: {reason}")]
    CoreMLFailure { reason: String },
    
    #[error("GStreamer pipeline error: {gst_error}")]
    PipelineError { gst_error: gst::glib::Error },
    
    #[error("Model loading failed: {path}")]
    ModelLoadError { path: PathBuf },
}
```

### 2.3 **Robust Testing Infrastructure**
```rust
// tests/integration_tests.rs - Core functionality tests
mod integration_tests {
    use super::*;
    
    #[tokio::test]
    async fn test_unified_binary_modes() {
        // Test all command modes work correctly
        unimplemented!("Verify pup run, live, detect, benchmark commands")
    }
    
    #[test]
    fn test_input_source_detection() {
        // Test automatic detection of webcam, files, RTSP
        unimplemented!("Verify InputManager handles all source types")
    }
    
    #[test] 
    fn test_coreml_optimization() {
        // Benchmark CoreML vs CPU performance
        unimplemented!("Verify CoreML configuration improves performance")
    }
    
    #[test]
    fn test_configuration_validation() {
        // Test TOML config parsing and validation
        unimplemented!("Verify config handles invalid/missing values gracefully")
    }
}

// tests/performance_tests.rs - Hardware-specific benchmarks
mod performance_tests {
    #[test]
    fn benchmark_coreml_vs_cpu() {
        unimplemented!("Compare inference times: CoreML vs CPU")
    }
    
    #[test] 
    fn benchmark_memory_usage() {
        unimplemented!("Verify memory usage stays within bounds")
    }
    
    #[test]
    fn benchmark_realtime_performance() {
        unimplemented!("Verify >30fps processing on live video")
    }
}

// tests/platform_tests.rs - Cross-platform compatibility  
mod platform_tests {
    #[cfg(target_os = "macos")]
    #[test]
    fn test_macos_video_display() {
        unimplemented!("Verify video window displays correctly on macOS")
    }
    
    #[test]
    fn test_webcam_enumeration() {
        unimplemented!("Verify webcam detection works across platforms")
    }
}
```

## Phase 3: Advanced Features (Weeks 5-6)

### 3.1 **Multi-Model Support Foundation**
~~**Complex multi-task architecture**~~ → **Simplified, extensible model loading**

```rust
pub struct ModelRegistry {
    models: HashMap<String, ModelInfo>,
    active_model: Option<String>,
}

pub struct ModelInfo {
    pub path: PathBuf,
    pub task_type: TaskType,
    pub metadata: ModelMetadata,
    pub performance_profile: PerformanceProfile,
}

#[derive(Debug, Clone)]
pub enum TaskType {
    ObjectDetection,
    // Future: Keypoint, Pose, Activity (when needed)
}
```

### 3.2 **Enhanced Video Output Options**
```rust
pub enum OutputMode {
    Display { 
        overlay_enabled: bool,
        fullscreen: bool,
    },
    File { 
        path: PathBuf,
        format: VideoFormat,
        quality: Quality,
    },
    Stream { 
        protocol: StreamProtocol,  // RTMP, WebRTC
        target: String,
    },
    Headless {
        json_output: Option<PathBuf>,
        metrics_enabled: bool,
    },
}
```

### 3.3 **Performance Monitoring**
```rust
pub struct PerformanceMonitor {
    metrics: Arc<Metrics>,
    reporter: Box<dyn MetricsReporter>,
}

pub struct Metrics {
    pub fps: AtomicF64,
    pub inference_latency_ms: AtomicF64,
    pub memory_usage_mb: AtomicUsize,
    pub dropped_frames: AtomicUsize,
}
```

## Phase 4: Production Deployment (Weeks 7-8)

### 4.1 **CI/CD Integration**
```yaml
# .github/workflows/test.yml
- name: Test CoreML Performance
  run: cargo test test_coreml_optimization --release
  
- name: Test Cross-Platform Compatibility
  run: cargo test --tests --release
  
- name: Benchmark Performance
  run: cargo test benchmark_ --release -- --nocapture
```

### 4.2 **Documentation and Examples**
```bash
# examples/basic_usage.md
pup live --input webcam                    # Quick start
pup run --config examples/production.toml  # Production deployment
pup detect --input video.mp4 --output results.json  # Batch processing
pup benchmark --model yolov8n.onnx         # Performance testing
```

### 4.3 **Packaging and Distribution**
- **Homebrew Formula**: For easy macOS installation
- **Binary Releases**: Cross-compiled binaries for major platforms
- **Container Images**: Docker/Podman support for deployment

## Implementation Priorities

### 🚨 **Critical Path (Weeks 1-2)**
1. **Unified Binary Design**: Single entry point with subcommands
2. **CoreML Optimization**: Implement proper CoreML configuration
3. **Input Source Abstraction**: Robust handling of webcam/file/RTSP
4. **Configuration Migration**: TOML config for all modes

### 🎯 **High Priority (Weeks 3-4)**  
1. **Production Readiness**: Error handling, logging, monitoring
2. **Testing Infrastructure**: Comprehensive test suite with hardware tests
3. **Performance Validation**: Benchmarking and optimization
4. **Documentation**: User guides and API documentation

### 📈 **Future Enhancements (Weeks 5+)**
1. **Multi-Model Support**: When specific use cases require it
2. **Advanced Output Options**: Streaming, recording, formats
3. **Plugin System**: For custom model integrations
4. **Web Interface**: Optional web-based control panel

## Success Metrics

### **Performance Targets**
- ✅ **Inference Latency**: <50ms for YOLOv8n on CoreML (currently achieved)
- 🎯 **Real-time Processing**: >30 FPS on 720p video (target)
- 🎯 **Memory Efficiency**: <500MB RAM usage during operation (target)
- 🎯 **Startup Time**: <3 seconds to first frame (target)

### **User Experience Goals**
- **Single Command**: `pup live` gets users started immediately
- **Configuration Flexibility**: Production deployments use TOML config
- **Error Clarity**: Clear, actionable error messages
- **Cross-Platform**: Consistent behavior on macOS, Linux, Windows

### **Development Metrics**
- **Test Coverage**: >80% code coverage with integration tests
- **Build Time**: <2 minutes for full build including tests
- **Documentation**: Every public API documented with examples
- **Backwards Compatibility**: Existing configurations continue to work

## Technology Stack

### **Core Dependencies**
```toml
[dependencies]
# GStreamer ecosystem
gstreamer = { version = "0.23", features = ["v1_14"] }
gstreamer-base = "0.23"
gstreamer-video = "0.23"
gstreamer-app = "0.23"

# ML inference with CoreML optimization
ort = { version = "2.0.0-rc.9", features = ["coreml", "download-binaries"] }

# Configuration and CLI
clap = { version = "4", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
toml = "0.8"

# Error handling and logging
anyhow = "1.0"
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = "0.3"

# Async runtime
tokio = { version = "1.0", features = ["full"] }
```

### **Development Tools**
- **Performance Profiling**: `cargo flamegraph`, Instruments.app on macOS
- **Memory Analysis**: `valgrind` on Linux, `leaks` on macOS  
- **CI/CD**: GitHub Actions with cross-platform testing
- **Documentation**: `cargo doc` with examples

## Migration Strategy

### **Week 1: Foundation**
- [x] ✅ Create unified `main.rs` with subcommand structure
- [x] ✅ Migrate existing `demo` modes into subcommands
- [x] ✅ Implement CoreML-optimized ONNX Runtime configuration
- [x] ✅ Add comprehensive input source management

### **Week 2: Integration** 
- [ ] 🎯 Implement production-grade error handling
- [ ] 🎯 Add comprehensive configuration validation
- [ ] 🎯 Create performance monitoring infrastructure
- [ ] 🎯 Build automated test suite with CI/CD

### **Week 3: Polish**
- [ ] 📋 Add documentation and examples
- [ ] 📋 Implement packaging and distribution
- [ ] 📋 Performance optimization and benchmarking
- [ ] 📋 Cross-platform compatibility testing

### **Week 4: Release**
- [ ] 🚀 Final testing and validation
- [ ] 🚀 Release preparation and documentation
- [ ] 🚀 Community feedback and iteration
- [ ] 🚀 Future roadmap planning

## Architectural Decisions Made

### **✅ Decisions Finalized**
1. **Single Binary**: Eliminates user confusion between `pup` and `demo`
2. **Mode-Based Operation**: Clean separation of concerns via subcommands
3. **CoreML First**: Optimize for macOS development environment first
4. **GStreamer Foundation**: Proven, robust video processing pipeline
5. **Configuration Hierarchy**: CLI → TOML → Defaults (in priority order)

### **🤔 Decisions Pending**
1. **Plugin Architecture**: How extensible should the model system be?
2. **Web Interface**: Whether to add optional web-based control
3. **Multi-Model Coordination**: When and how to implement model chaining
4. **Cloud Integration**: Support for cloud-based inference services

### **❌ Decisions Rejected**
- ~~Complex multi-task vision pipeline~~ → **Too complex for current needs**
- ~~Separate plugin binaries~~ → **Increases deployment complexity**  
- ~~Custom video codecs~~ → **GStreamer handles this well**
- ~~Multiple configuration formats~~ → **TOML is sufficient**

## Risk Mitigation

### **Technical Risks**
1. **CoreML Compatibility**: Some models may not work with CoreML provider
   - *Mitigation*: Always include CPU fallback, test model compatibility
2. **GStreamer Dependencies**: Platform-specific installation complexity  
   - *Mitigation*: Clear installation docs, consider static linking
3. **Performance Regression**: Changes might slow down inference
   - *Mitigation*: Continuous benchmarking in CI/CD

### **Project Risks**  
1. **Scope Creep**: Feature requests for complex multi-model support
   - *Mitigation*: Clear roadmap priorities, defer non-essential features
2. **Platform Fragmentation**: Behavior differences across macOS/Linux/Windows
   - *Mitigation*: Automated cross-platform testing, platform-specific docs

This roadmap provides a practical, achievable path to a production-ready, unified video processing system while maintaining the flexibility to expand into more complex computer vision tasks when specific use cases demand it.