#!/bin/bash

echo "Testing basic application without inference..."

# Set environment variables for ONNX runtime
export DYLD_LIBRARY_PATH="$HOME/.cargo/registry/src/index.crates.io-6f17d22bba15001f/ort-sys-2.0.0-rc.9/dist/onnxruntime-osx-arm64-1.20.1/lib:$DYLD_LIBRARY_PATH"

# Try to find the ONNX runtime library
find ~/.cargo -name "libonnxruntime*.dylib" 2>/dev/null

echo "Running cargo run with production mode..."
timeout 5s cargo run --release -- --config configs/sample_video_production.toml || echo "Test completed"