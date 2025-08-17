# Configuration Files

This directory contains pre-configured TOML files for different use cases of the Pup video processing application.

## Available Configurations

### 1. **Webcam Live View with Overlays** (`webcam_live.toml`)
- **Purpose**: Real-time webcam processing with bounding box overlays
- **Features**: Live video window, real-time object detection, overlay rendering
- **Usage**: `cargo run --release -- --config configs/webcam_live.toml`

### 2. **Sample Video Live View** (`sample_video_live.toml`)
- **Purpose**: Process sample video file with live overlays
- **Features**: Video playback with bounding box overlays and labels
- **Usage**: `cargo run --release -- --config configs/sample_video_live.toml`

### 3. **Detection Only Mode** (`sample_video_detection.toml`)
- **Purpose**: Terminal-only detection output, no video display
- **Features**: Fast processing, text output, no GUI
- **Usage**: `cargo run --release -- --config configs/sample_video_detection.toml`

### 4. **Production Mode** (`sample_video_production.toml`)
- **Purpose**: Original application behaviour (backwards compatibility)
- **Features**: Video display + terminal detection output
- **Usage**: `cargo run --release -- --config configs/sample_video_production.toml`

### 5. **High Confidence Live** (`high_confidence_live.toml`)
- **Purpose**: Live view with higher confidence threshold (fewer, more accurate detections)
- **Features**: Confidence threshold of 0.8, cleaner overlay display
- **Usage**: `cargo run --release -- --config configs/high_confidence_live.toml`

## Quick Test Commands

```bash
# Test webcam with overlays (recommended first test)
cargo run --release -- --config configs/webcam_live.toml

# Test sample video with overlays
cargo run --release -- --config configs/sample_video_live.toml

# Test detection only (fastest)
cargo run --release -- --config configs/sample_video_detection.toml

# Test original behaviour
cargo run --release -- --config configs/sample_video_production.toml

# Test high confidence mode
cargo run --release -- --config configs/high_confidence_live.toml
```

## Alternative Command Line Usage

You can also use the new command line arguments directly:

```bash
# Live mode with overlays
cargo run --release -- --mode live --model models/yolov8n.onnx --video assets/sample.mp4

# Webcam live mode
cargo run --release -- --mode live --model models/yolov8n.onnx --video webcam

# Detection only mode
cargo run --release -- --mode detection --model models/yolov8n.onnx --video assets/sample.mp4

# Customize overlay display
cargo run --release -- --mode live --model models/yolov8n.onnx --video assets/sample.mp4 --confidence 0.8 --show-labels --show-confidence
```

## Troubleshooting

### **ONNX Runtime Issues (dyld: missing symbol called / Abort trap: 6)**

If you get the error `dyld[xxx]: missing symbol called` followed by `Abort trap: 6`, this means the ONNX Runtime library cannot be found. **Solution:**

1. **Use the setup script** (recommended):
   ```bash
   ./setup_onnx.sh
   ./run_with_overlays.sh sample
   ```

2. **Manual setup**:
   ```bash
   export DYLD_LIBRARY_PATH="/Users/$USER/.ort/lib:$DYLD_LIBRARY_PATH"
   cargo run --release -- --config configs/sample_video_live.toml
   ```

3. **Quick test**:
   ```bash
   # Test with sample video
   ./run_with_overlays.sh sample

   # Test with webcam
   ./run_with_overlays.sh webcam
   ```

### **Other Common Issues**

- **No video window**: Check that `display_enabled = true` in the config
- **No overlays**: Ensure mode is set to `"live"`
- **No detections**: Lower the `confidence_threshold` value
- **Webcam not working**: Check `device_id` or try `video = "webcam"` in command line
- **Model not found**: Ensure `models/yolov8n.onnx` exists in your project directory

### **Environment Variables**

For best results, set these environment variables:
```bash
export DYLD_LIBRARY_PATH="/Users/$USER/.ort/lib:$DYLD_LIBRARY_PATH"
export ORT_DYLIB_PATH="/Users/$USER/.ort/lib/libonnxruntime.dylib"
```
