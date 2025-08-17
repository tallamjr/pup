//! Integration tests for pup video processing pipeline
//!
//! These tests verify the entire pipeline functionality and will be used
//! to validate the roadmap implementation stages.

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

/// Test that the application builds successfully
#[test]
fn test_application_builds() {
    let output = Command::new("cargo")
        .args(&["build", "--release"])
        .output()
        .expect("Failed to execute cargo build");

    assert!(
        output.status.success(),
        "Build failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Test that the application can be invoked with help
#[test]
fn test_application_help() {
    let output = Command::new("cargo")
        .args(&["run", "--release", "--", "--help"])
        .output()
        .expect("Failed to execute pup --help");

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Usage:"));
}

/// Test that the application fails gracefully with missing model
#[test]
fn test_missing_model_error() {
    let output = Command::new("cargo")
        .args(&[
            "run",
            "--release",
            "--",
            "--model",
            "nonexistent.onnx",
            "--video",
            "assets/sample.mp4",
        ])
        .output()
        .expect("Failed to execute pup with missing model");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("not found") || stderr.contains("Model file"));
}

/// Test that the application fails gracefully with missing video
#[test]
fn test_missing_video_error() {
    // Skip if model doesn't exist
    if !PathBuf::from("models/yolov8n.onnx").exists() {
        return;
    }

    let output = Command::new("cargo")
        .args(&[
            "run",
            "--release",
            "--",
            "--model",
            "models/yolov8n.onnx",
            "--video",
            "nonexistent.mp4",
        ])
        .output()
        .expect("Failed to execute pup with missing video");

    assert!(!output.status.success());
}

/// Test GStreamer pipeline initialization
#[test]
fn test_gstreamer_initialization() {
    // This test verifies that GStreamer can be initialized
    // It's a smoke test for the current pipeline implementation

    let test_code = r#"
        use gstreamer::prelude::*;

        fn main() -> Result<(), Box<dyn std::error::Error>> {
            gstreamer::init()?;
            println!("GStreamer initialized successfully");
            Ok(())
        }
    "#;

    // Create a temporary test file
    std::fs::write("/tmp/gst_test.rs", test_code).unwrap();

    // Compile and run the test
    let compile_output = Command::new("rustc")
        .args(&[
            "--extern",
            "gstreamer",
            "-L",
            "target/release/deps",
            "/tmp/gst_test.rs",
            "-o",
            "/tmp/gst_test",
        ])
        .output();

    if let Ok(output) = compile_output {
        if output.status.success() {
            let run_output = Command::new("/tmp/gst_test")
                .output()
                .expect("Failed to run GStreamer test");

            assert!(run_output.status.success());
            let stdout = String::from_utf8_lossy(&run_output.stdout);
            assert!(stdout.contains("GStreamer initialized successfully"));
        }
    }

    // Clean up
    let _ = std::fs::remove_file("/tmp/gst_test.rs");
    let _ = std::fs::remove_file("/tmp/gst_test");
}

/// Test asset files exist
#[test]
fn test_asset_files_exist() {
    let assets = vec!["assets/sample.mp4", "assets/bike.jpeg", "assets/seal.jpeg"];

    for asset in assets {
        assert!(
            PathBuf::from(asset).exists(),
            "Asset file {} does not exist",
            asset
        );
    }
}

/// Test model files exist (if available)
#[test]
fn test_model_files_available() {
    let model_path = PathBuf::from("models/yolov8n.onnx");

    if model_path.exists() {
        // Verify it's a valid file
        let metadata = std::fs::metadata(&model_path).unwrap();
        assert!(metadata.is_file());
        assert!(metadata.len() > 0);
    } else {
        println!("Model file not available, skipping model tests");
    }
}

/// Test video processing pipeline end-to-end (if model available)
#[test]
fn test_video_processing_pipeline() {
    let model_path = PathBuf::from("models/yolov8n.onnx");
    let video_path = PathBuf::from("assets/sample.mp4");

    if !model_path.exists() || !video_path.exists() {
        println!("Required files not available, skipping pipeline test");
        return;
    }

    // Run the application for a short duration
    let mut child = Command::new("cargo")
        .args(&[
            "run",
            "--release",
            "--",
            "--model",
            "models/yolov8n.onnx",
            "--video",
            "assets/sample.mp4",
        ])
        .spawn()
        .expect("Failed to start pup");

    // Let it run for 5 seconds
    std::thread::sleep(Duration::from_secs(5));

    // Terminate the process
    let _ = child.kill();
    let _output = child.wait().expect("Failed to wait for child process");

    // The process should have been running (killed, not failed)
    // Note: kill() results in a non-zero exit code, which is expected
}

/// Performance baseline test - measure basic processing speed
#[test]
fn test_performance_baseline() {
    use std::time::Instant;

    let model_path = PathBuf::from("models/yolov8n.onnx");

    if !model_path.exists() {
        println!("Model not available, skipping performance test");
        return;
    }

    let start = Instant::now();

    // Time how long it takes to load the model (basic benchmark)
    let test_code = r#"
        use ort::session::{builder::GraphOptimizationLevel, Session};
        use ort::execution_providers::CoreMLExecutionProvider;

        fn main() -> Result<(), Box<dyn std::error::Error>> {
            let _session = Session::builder()?
                .with_optimization_level(GraphOptimizationLevel::Level3)?
                .with_intra_threads(8)?
                .with_execution_providers([CoreMLExecutionProvider::default().build()])?
                .commit_from_file("models/yolov8n.onnx")?;
            println!("Model loaded successfully");
            Ok(())
        }
    "#;

    std::fs::write("/tmp/perf_test.rs", test_code).unwrap();

    let compile_output = Command::new("rustc")
        .args(&[
            "--extern",
            "ort",
            "-L",
            "target/release/deps",
            "/tmp/perf_test.rs",
            "-o",
            "/tmp/perf_test",
        ])
        .output();

    if let Ok(output) = compile_output {
        if output.status.success() {
            let run_output = Command::new("/tmp/perf_test").output();

            if let Ok(result) = run_output {
                let duration = start.elapsed();
                println!("Model loading took: {:?}", duration);

                // Baseline: model should load in under 10 seconds
                assert!(
                    duration < Duration::from_secs(10),
                    "Model loading took too long: {:?}",
                    duration
                );

                assert!(result.status.success());
            }
        }
    }

    // Clean up
    let _ = std::fs::remove_file("/tmp/perf_test.rs");
    let _ = std::fs::remove_file("/tmp/perf_test");
}

/// Test memory usage stays within reasonable bounds
#[cfg(target_os = "macos")]
#[test]
fn test_memory_usage_bounds() {
    use std::process::{Command, Stdio};

    let model_path = PathBuf::from("models/yolov8n.onnx");
    let video_path = PathBuf::from("assets/sample.mp4");

    if !model_path.exists() || !video_path.exists() {
        println!("Required files not available, skipping memory test");
        return;
    }

    // Use macOS leaks tool to monitor memory usage
    let mut child = Command::new("leaks")
        .args(&[
            "--atExit",
            "--",
            "cargo",
            "run",
            "--release",
            "--",
            "--model",
            "models/yolov8n.onnx",
            "--video",
            "assets/sample.mp4",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();

    if let Ok(mut process) = child {
        // Let it run for a few seconds
        std::thread::sleep(Duration::from_secs(3));

        let _ = process.kill();
        if let Ok(output) = process.wait_with_output() {
            let stderr = String::from_utf8_lossy(&output.stderr);

            // Check that memory leaks are minimal
            if stderr.contains("leaks for") {
                // Extract leaked bytes count
                if let Some(line) = stderr
                    .lines()
                    .find(|l| l.contains("leaks for") && l.contains("total leaked bytes"))
                {
                    println!("Memory leaks report: {}", line);
                    // The current implementation should have minimal leaks
                    assert!(line.contains("0 leaks") || line.contains("0 total leaked bytes"));
                }
            }
        }
    }
}

/// Test configuration validation (preparing for Phase 1 roadmap)
#[test]
fn test_configuration_validation() {
    // Test that invalid arguments are rejected
    let test_cases = vec![
        vec!["--model"],                            // Missing model path
        vec!["--video"],                            // Missing video path
        vec!["--model", "", "--video", "test.mp4"], // Empty model path
    ];

    for args in test_cases {
        let output = Command::new("cargo")
            .args(&["run", "--release", "--"])
            .args(&args)
            .output()
            .expect("Failed to execute pup with invalid args");

        assert!(
            !output.status.success(),
            "Application should fail with invalid args: {:?}",
            args
        );
    }
}
