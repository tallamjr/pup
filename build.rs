fn main() {
    // println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
    println!(
        "cargo:rustc-link-search=native=./.venv/lib/python3.11/site-packages/onnxruntime/capi"
    );
    println!("cargo:rustc-link-lib=dylib=onnxruntime");
}
