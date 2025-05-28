#!/bin/bash

# PUP Codebase Verification Script
# Verifies audit claims and codebase improvements

set -e  # Exit on any error

echo "========================================="
echo "🔍 PUP Codebase Verification Script"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Verification 1: Test Suite
echo "1. 🧪 Testing all unit tests..."
echo "   Running: cargo test --lib"
if cargo test --lib --quiet > /tmp/test_output.log 2>&1; then
    TEST_RESULT=$(grep "test result:" /tmp/test_output.log)
    print_success "All tests pass: $TEST_RESULT"
else
    print_error "Some tests failed"
    echo "   Check /tmp/test_output.log for details"
fi
echo ""

# Verification 2: Specific YOLO test
echo "2. 🎯 Testing previously broken YOLO test..."
echo "   Running: cargo test test_yolo_output_processing"
if cargo test test_yolo_output_processing --quiet > /tmp/yolo_test.log 2>&1; then
    print_success "YOLO output processing test passes"
else
    print_error "YOLO test still failing"
    cat /tmp/yolo_test.log
fi
echo ""

# Verification 3: Unimplemented code check
echo "3. 🔧 Checking for unimplemented code..."
UNIMPL_COUNT=$(grep -r "unimplemented!" src/ 2>/dev/null | wc -l | tr -d ' ')
if [ "$UNIMPL_COUNT" -eq 0 ]; then
    print_success "No unimplemented! macros found in src/"
else
    print_error "Found $UNIMPL_COUNT unimplemented! macros:"
    grep -r "unimplemented!" src/ | head -5
fi
echo ""

# Verification 4: TODO analysis
echo "4. 📝 Checking remaining TODO items..."
TODO_COUNT=$(grep -r "TODO" src/ 2>/dev/null | wc -l | tr -d ' ')
if [ "$TODO_COUNT" -eq 0 ]; then
    print_success "No TODO items found"
else
    print_info "Found $TODO_COUNT TODO items (expected for future enhancements)"
    echo "   First few TODOs:"
    grep -r "TODO" src/ | head -3 | sed 's/^/   /'
fi
echo ""

# Verification 5: Compilation check
echo "5. 🛠️  Testing compilation..."
echo "   Running: cargo check"
if cargo check --quiet > /tmp/compile_output.log 2>&1; then
    print_success "Code compiles cleanly"
else
    print_error "Compilation issues found"
    echo "   Last 10 lines of compiler output:"
    tail -10 /tmp/compile_output.log | sed 's/^/   /'
fi
echo ""

# Verification 6: Binary help commands
echo "6. 📖 Testing help commands..."
echo "   Testing: cargo run --bin pup -- --help"
if timeout 10 cargo run --bin pup -- --help > /tmp/pup_help.log 2>&1; then
    print_success "Main binary (pup) help works"
else
    print_error "Main binary help failed or timed out"
fi

echo "   Testing: cargo run --bin demo -- --help"
if timeout 10 cargo run --bin demo -- --help > /tmp/demo_help.log 2>&1; then
    print_success "Demo binary help works"
else
    print_error "Demo binary help failed or timed out"
fi
echo ""

# Verification 7: Required files check
echo "7. 📁 Checking required files..."
if [ -f "models/yolov8n.onnx" ]; then
    MODEL_SIZE=$(ls -lh models/yolov8n.onnx | awk '{print $5}')
    print_success "ONNX model found (size: $MODEL_SIZE)"
else
    print_warning "ONNX model not found at models/yolov8n.onnx"
    print_info "Download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx"
fi

if [ -f "assets/sample.mp4" ]; then
    VIDEO_SIZE=$(ls -lh assets/sample.mp4 | awk '{print $5}')
    print_success "Sample video found (size: $VIDEO_SIZE)"
else
    print_warning "Sample video not found at assets/sample.mp4"
fi

if [ -d "assets" ]; then
    ASSET_COUNT=$(ls assets/ | wc -l | tr -d ' ')
    print_info "Found $ASSET_COUNT files in assets directory"
fi
echo ""

# Verification 8: Documentation accuracy
echo "8. 📚 Testing README command examples..."
echo "   Verifying command structure (without execution):"

# Check if the README examples are syntactically correct
if cargo run --bin pup -- --help 2>&1 | grep -q "model.*MODEL"; then
    print_success "Main binary requires --model parameter (as documented)"
else
    print_warning "Main binary help doesn't show required --model parameter"
fi

if cargo run --bin demo -- --help 2>&1 | grep -q "mode.*MODE"; then
    print_success "Demo binary has --mode parameter (as documented)"
else
    print_warning "Demo binary help doesn't show --mode parameter"
fi
echo ""

# Verification 9: GStreamer functionality (if possible)
echo "9. 🎬 Testing basic GStreamer functionality..."
if command -v gst-inspect-1.0 >/dev/null 2>&1; then
    print_success "GStreamer tools available"
    if timeout 5 cargo build --quiet > /tmp/build.log 2>&1; then
        print_success "Project builds successfully"
    else
        print_error "Build failed"
    fi
else
    print_warning "GStreamer tools not found - install gstreamer development packages"
fi
echo ""

# Verification 10: Code quality metrics
echo "10. 📊 Code quality summary..."
RUST_FILES=$(find src/ -name "*.rs" | wc -l | tr -d ' ')
TOTAL_LINES=$(find src/ -name "*.rs" -exec cat {} \; | wc -l | tr -d ' ')
print_info "Rust source files: $RUST_FILES"
print_info "Total lines of code: $TOTAL_LINES"

# Count warnings in a clean build
echo "    Checking for compiler warnings..."
if cargo check 2>&1 | grep -q "warning:"; then
    WARNING_COUNT=$(cargo check 2>&1 | grep "warning:" | wc -l | tr -d ' ')
    print_warning "Found $WARNING_COUNT compiler warnings"
else
    print_success "No compiler warnings found"
fi
echo ""

# Summary
echo "========================================="
echo "📋 VERIFICATION SUMMARY"
echo "========================================="

# Check if key files exist for functional testing
CAN_TEST_FULL=true
if [ ! -f "models/yolov8n.onnx" ]; then
    CAN_TEST_FULL=false
fi

if [ ! -f "assets/sample.mp4" ]; then
    CAN_TEST_FULL=false
fi

if [ "$CAN_TEST_FULL" = true ]; then
    print_success "All required files present - you can test full functionality"
    echo ""
    echo "🚀 Try these commands to test the improvements:"
    echo "   cargo run --bin demo -- --mode detection --input assets/sample.mp4"
    echo "   cargo run --bin pup -- --model models/yolov8n.onnx --video assets/sample.mp4"
else
    print_warning "Missing model or video files - functional testing limited"
    echo ""
    echo "📥 To enable full testing:"
    echo "   1. Download yolov8n.onnx to models/ directory"
    echo "   2. Ensure sample video exists in assets/ directory"
fi

echo ""
echo "🔍 Audit verification complete!"
echo "   Check output above for any ❌ errors that need attention"
echo ""

# Cleanup temporary files
rm -f /tmp/test_output.log /tmp/yolo_test.log /tmp/compile_output.log 
rm -f /tmp/pup_help.log /tmp/demo_help.log /tmp/build.log

exit 0