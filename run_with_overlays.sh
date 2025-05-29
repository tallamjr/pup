#!/bin/bash

# Pup Video Processing - Run with Overlays
# This script sets up the environment and runs the application with overlay functionality

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🎥 Pup Video Processing - Live Overlay Mode${NC}"
echo ""

# Check if ONNX Runtime is installed
ONNX_DIR="$HOME/.ort"
if [ ! -f "$ONNX_DIR/lib/libonnxruntime.dylib" ]; then
    echo -e "${YELLOW}⚠️  ONNX Runtime not found. Setting up...${NC}"
    ./setup_onnx.sh
    echo ""
fi

# Set environment variables
export DYLD_LIBRARY_PATH="$ONNX_DIR/lib:$DYLD_LIBRARY_PATH"
export ORT_DYLIB_PATH="$ONNX_DIR/lib/libonnxruntime.dylib"

echo -e "${GREEN}✅ Environment configured${NC}"
echo -e "${BLUE}📚 Available configurations:${NC}"
echo "  1. 📹 Webcam Live:        ./run_with_overlays.sh webcam"
echo "  2. 🎬 Sample Video Live:  ./run_with_overlays.sh sample"
echo "  3. 📊 Detection Only:     ./run_with_overlays.sh detection"
echo "  4. 🏭 Production Mode:    ./run_with_overlays.sh production"
echo "  5. 🔍 High Confidence:    ./run_with_overlays.sh high_confidence"
echo ""

# Determine which config to use
case "${1:-sample}" in
    "webcam")
        CONFIG="configs/webcam_live.toml"
        echo -e "${GREEN}🚀 Starting webcam live processing...${NC}"
        ;;
    "sample")
        CONFIG="configs/sample_video_live.toml"
        echo -e "${GREEN}🚀 Starting sample video live processing...${NC}"
        ;;
    "detection")
        CONFIG="configs/sample_video_detection.toml"
        echo -e "${GREEN}🚀 Starting detection-only mode...${NC}"
        ;;
    "production")
        CONFIG="configs/sample_video_production.toml"
        echo -e "${GREEN}🚀 Starting production mode...${NC}"
        ;;
    "high_confidence")
        CONFIG="configs/high_confidence_live.toml"
        echo -e "${GREEN}🚀 Starting high confidence mode...${NC}"
        ;;
    *)
        echo -e "${RED}❌ Unknown mode: $1${NC}"
        echo "Usage: ./run_with_overlays.sh [webcam|sample|detection|production|high_confidence]"
        exit 1
        ;;
esac

echo -e "${BLUE}📁 Using config: $CONFIG${NC}"
echo -e "${BLUE}⚡ Press Ctrl+C to stop${NC}"
echo ""

# Run the application
cargo run --release -- --config "$CONFIG"