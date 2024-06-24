use std::env;
use std::process::Command;

fn main() {
    // Run the sed command
    let output = Command::new("sh")
        .arg("-c")
        .arg("rust -vV | sed -n 's/^host: \\(.*\\)$/\\1/p'")
        .output()
        .expect("Failed to execute command");

    // Check if the command was successful
    if output.status.success() {
        let result = std::str::from_utf8(&output.stdout).expect("Failed to parse output");
        println!("Extracted host: {}", result.trim());
    } else {
        let error = std::str::from_utf8(&output.stderr).expect("Failed to parse error");
        eprintln!("Error: {}", error);
    }

    // Set an environment variable based on the extracted host
    println!("cargo:rustc-env=HOST_ARCH={}", result.trim());

    // Retrieve the target triple
    let target = env::var("HOST_ARCH").unwrap();

    let xcode_path = Command::new("sh")
        .arg("-c")
        .arg("xcode-select -p")
        .output()
        .expect("Failed to execute command");

    if output.status.success() {
        let path = std::str::from_utf8(&output.stdout).expect("Failed to parse output");
        println!("Extracted host: {}", result.trim());
    } else {
        let error = std::str::from_utf8(&output.stderr).expect("Failed to parse error");
        eprintln!("Error: {}", error);
    }

    // Check if the target is aarch64-apple-darwin
    if target == "aarch64-apple-darwin" {
        // Set custom RUSTFLAGS for the target
        let rustflags = format!("-C link-arg=-fuse-ld={}/usr/bin/ld", path.trim());

        // Export the RUSTFLAGS environment variable
        println!("cargo:rustc-env=RUSTFLAGS={}", rustflags);
    }

    // Optionally, run any other build script actions here
    println!("cargo:rerun-if-changed=build.rs");
}
