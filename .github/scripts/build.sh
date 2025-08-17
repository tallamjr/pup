#!/bin/bash
set -e

echo "Building pup project..."

# Check if we're building in release mode
BUILD_TYPE="--release"
if [ "$1" = "--debug" ]; then
    BUILD_TYPE=""
    echo "Building in debug mode"
else
    echo "Building in release mode"
fi

# Build the project
cargo build $BUILD_TYPE

# Build all binaries
cargo build $BUILD_TYPE --bins

echo "Build completed successfully!"

# Show the built binaries
if [ -n "$BUILD_TYPE" ]; then
    echo "Built binaries are located in: $(pwd)/target/release/"
    ls -la target/release/pup target/release/demo target/release/benchmark 2>/dev/null || true
else
    echo "Built binaries are located in: $(pwd)/target/debug/"
    ls -la target/debug/pup target/debug/demo target/debug/benchmark 2>/dev/null || true
fi
