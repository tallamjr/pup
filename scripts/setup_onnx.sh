#!/bin/bash

echo "Setting up ONNX Runtime for macOS ARM64..."

# Create directory for ONNX runtime
mkdir -p ~/.ort/lib

# Download ONNX Runtime if not exists
ONNX_VERSION="1.20.1"
ONNX_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/onnxruntime-osx-arm64-${ONNX_VERSION}.tgz"
ONNX_DIR="$HOME/.ort"

if [ ! -f "$ONNX_DIR/lib/libonnxruntime.dylib" ]; then
    echo "Downloading ONNX Runtime..."
    cd /tmp
    curl -L -O "$ONNX_URL"
    tar -xzf "onnxruntime-osx-arm64-${ONNX_VERSION}.tgz"

    # Copy library to our directory
    cp "onnxruntime-osx-arm64-${ONNX_VERSION}/lib/libonnxruntime.${ONNX_VERSION}.dylib" "$ONNX_DIR/lib/"

    # Create symlinks
    cd "$ONNX_DIR/lib"
    ln -sf "libonnxruntime.${ONNX_VERSION}.dylib" "libonnxruntime.dylib"
    ln -sf "libonnxruntime.${ONNX_VERSION}.dylib" "libonnxruntime.1.dylib"

    echo "ONNX Runtime installed to $ONNX_DIR/lib"
else
    echo "ONNX Runtime already installed"
fi

# Set environment variables
export DYLD_LIBRARY_PATH="$ONNX_DIR/lib:$DYLD_LIBRARY_PATH"
export ORT_DYLIB_PATH="$ONNX_DIR/lib/libonnxruntime.dylib"

echo "Environment set. You can now run:"
echo "export DYLD_LIBRARY_PATH=\"$ONNX_DIR/lib:\$DYLD_LIBRARY_PATH\""
echo "cargo run --release -- --config configs/sample_video_live.toml"
