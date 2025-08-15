fn main() {
    // Platform-specific configuration
    #[cfg(target_os = "macos")]
    {
        // Link required macOS frameworks for CoreML support
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
        println!("cargo:rustc-link-lib=framework=Foundation");

        // Add Homebrew library paths for potential dependencies like libomp
        println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
        println!("cargo:rustc-link-search=native=/usr/local/lib");

        // Link libomp if available (required for ONNX Runtime on macOS)
        // This is a weak link - it won't fail if not found
        println!("cargo:rustc-link-lib=dylib=omp");
    }

    // Remove the Python virtual environment path as we're using download-binaries feature
    // The ort crate with download-binaries should handle ONNX Runtime linking automatically
    // println!("cargo:rustc-link-search=native=./.venv/lib/python3.11/site-packages/onnxruntime/capi");
    // println!("cargo:rustc-link-lib=dylib=onnxruntime");
}
