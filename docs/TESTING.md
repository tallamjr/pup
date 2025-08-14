# Testing Documentation for Pup

This document explains the comprehensive testing framework created for the pup video processing application, designed to validate both current functionality and future roadmap implementations.

## Test Suite Overview

The testing framework consists of four main categories:

### 1. Unit Tests (`tests/unit_tests.rs`)
**Purpose**: Test individual components and functions in isolation

**Test Categories**:
- **Letterbox Tests**: Validate image resizing with aspect ratio preservation
  - Square image handling (640x640)
  - Wide image handling (1920x1080 → 640x640)
  - Tall image handling (480x800 → 640x640)
  - Small image upscaling

- **Detection Tests**: Validate detection data structures
  - Detection creation and properties
  - Bounding box calculations (area, center, dimensions)

- **Modular Architecture Tests**: Mock implementations for future roadmap
  - Inference backend trait testing
  - Model post-processor trait testing
  - Configuration structure validation

- **ONNX Model Tests**: Model file validation
  - File structure and signature verification
  - Model loading performance

- **Preprocessing Tests**: Image processing pipeline validation
  - RGB normalization (0-255 → 0.0-1.0)
  - Tensor shape calculations
  - Colour channel ordering (RGB vs BGR)

- **GStreamer Plugin Tests**: Mock plugin architecture
  - Property management system
  - Pipeline construction validation
  - Caps negotiation patterns

**Run Command**:
```bash
cargo test --test unit_tests
```

### 2. Integration Tests (`tests/integration_tests.rs`)
**Purpose**: Test complete system functionality and end-to-end workflows

**Test Categories**:
- **Build Verification**: Ensure application compiles successfully
- **CLI Interface**: Test command-line argument handling and help output
- **Error Handling**: Validate graceful failure with missing files
- **GStreamer Initialization**: Verify GStreamer can be initialised
- **Asset Verification**: Check required test files exist
- **Pipeline Processing**: End-to-end video processing (if model available)
- **Performance Baseline**: Basic processing speed measurements
- **Memory Usage**: Monitor memory consumption patterns (macOS only)

**Run Command**:
```bash
cargo test --test integration_tests
```

### 3. Roadmap Validation Tests (`tests/roadmap_validation_tests.rs`)
**Purpose**: Validate each phase of the roadmap implementation meets requirements

**Test Phases**:

#### Phase 1: Modular Refactoring
- Module structure verification
- Configuration-driven design validation
- Trait-based inference system testing
- Monolithic-to-modular migration tracking

#### Phase 2: GStreamer-RS Plugin Architecture
- Plugin registration structure validation
- Element property system testing
- GStreamer caps definition verification
- Pipeline integration validation

#### Phase 3: Advanced Pipeline Management
- Declarative configuration parsing (YAML)
- Performance metrics structure
- Multi-stream architecture testing

#### Phase 4: Performance Optimisation
- Zero-copy buffer design validation
- Inference batching architecture
- Hardware acceleration configuration
- Memory pool design testing

#### Phase 5: Advanced Features
- Ensemble inference structure
- Object tracking system design
- Streaming pipeline architecture

#### Success Metrics Validation
- <10ms inference latency requirement
- >30 FPS processing requirement
- <50% memory usage reduction target
- Extensibility requirements (new model support)
- <5 minutes developer experience target

**Run Command**:
```bash
cargo test --test roadmap_validation_tests
```

### 4. Benchmark Tests (`tests/benchmark_tests.rs`)
**Purpose**: Establish performance baselines and validate optimisations

**Benchmark Categories**:

#### Memory Benchmarks
- Baseline memory usage measurement
- Zero-copy memory target validation
- Memory scaling with load

#### Performance Benchmarks
- Inference latency baseline vs optimised targets
- Preprocessing performance measurement
- Frame processing rate (FPS) validation
- Sustained throughput over time

#### Scalability Benchmarks
- Single stream performance baseline
- Multi-stream processing scalability
- Batch processing efficiency
- Memory usage scaling

#### Hardware Acceleration Benchmarks
- CPU vs CoreML performance comparison
- Metal compute performance for preprocessing
- Device memory transfer overhead measurement

#### Integration Performance
- End-to-end pipeline latency
- Pipeline stability over time

**Run Command**:
```bash
cargo test --test benchmark_tests
```

## Criterion Benchmarks (`benches/performance_benchmarks.rs`)
**Purpose**: Precise performance measurements using the Criterion framework

**Benchmark Categories**:
- **Letterbox Resize**: Different resolution performance (VGA, HD, FullHD, 4K)
- **RGB Normalisation**: Sequential vs parallel processing
- **Tensor Conversion**: HWC to CHW format conversion
- **Memory Allocation**: Standard vs buffer reuse vs memory pool
- **Detection Post-processing**: Confidence thresholding and NMS
- **GStreamer Buffers**: Copy-based vs in-place vs memory-mapped processing

**Run Command**:
```bash
cargo bench
```

**HTML Reports**: Generated in `target/criterion/` directory

## Running All Tests

### Quick Test Run
```bash
# Run all unit and integration tests
cargo test

# Run only unit tests
cargo test --test unit_tests

# Run only integration tests
cargo test --test integration_tests

# Run specific test
cargo test --test unit_tests test_letterbox_square_image
```

### Complete Validation
```bash
# Run all tests including roadmap validation
cargo test --test unit_tests
cargo test --test integration_tests
cargo test --test roadmap_validation_tests
cargo test --test benchmark_tests

# Run performance benchmarks
cargo bench
```

### Continuous Integration
```bash
# Format and lint
cargo fmt
cargo check

# Run tests with output
cargo test -- --nocapture

# Generate test coverage (if tools installed)
cargo tarpaulin --out Html
```

## Test Data Requirements

### Required Assets
- `assets/sample.mp4` - Video file for processing tests
- `assets/bike.jpeg` - Image file for preprocessing tests
- `assets/seal.jpeg` - Additional image for validation
- `models/yolov8n.onnx` - ONNX model file (optional, tests skip if missing)

### Test Environment
- **macOS**: Full test suite including memory profiling
- **Linux/Windows**: Core functionality tests (some macOS-specific tests skipped)
- **CI/CD**: All tests should pass without requiring model files

## Interpreting Test Results

### Success Criteria
- **All unit tests pass**: Core functionality is correct
- **Integration tests pass**: System works end-to-end
- **Performance benchmarks within targets**:
  - Inference <10ms
  - Processing >30 FPS
  - Memory usage reasonable
- **Roadmap validation passes**: Implementation meets design requirements

### Common Issues
- **Missing model file**: Many tests skip gracefully, not a failure
- **GStreamer initialization fails**: Check GStreamer installation
- **OpenCV errors**: Verify OpenCV development libraries installed
- **Performance regression**: Compare benchmark results over time

## Future Test Expansion

As roadmap phases are implemented:

1. **Phase 1**: Add module integration tests
2. **Phase 2**: Add real GStreamer plugin tests
3. **Phase 3**: Add multi-stream performance tests
4. **Phase 4**: Add hardware acceleration validation
5. **Phase 5**: Add ensemble and tracking validation

## Maintenance

- **Update baselines**: When optimisations are implemented
- **Add new test cases**: For each new feature or bug fix
- **Monitor performance**: Regular benchmark runs to catch regressions
- **Review test coverage**: Ensure new code is adequately tested

This testing framework provides comprehensive validation for both current functionality and future development, ensuring the pup video processing system remains robust and performant throughout its evolution.
