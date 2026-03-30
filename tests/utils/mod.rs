//! Comprehensive test utilities and fixtures for the Pup testing framework
//!
//! This module provides shared testing infrastructure including mock objects,
//! test fixtures, custom assertions, and utilities for comprehensive edge case testing.

use gstpup::config::{AppConfig, InferenceConfig, InputConfig, OutputConfig, PreprocessingConfig};
use gstpup::error::{PupError, PupResult};
use std::fs;
use std::io::Write;
use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;
use std::sync::Arc;
use tempfile::{NamedTempFile, TempDir};

pub mod assertions;
pub mod fixtures;
pub mod generators;
pub mod mocks;
pub mod stress;

// Re-export key types for convenience
pub use fixtures::ConfigFixtures;
pub use generators::ConfigGenerator;

/// Test configuration builder for creating various config scenarios
pub struct TestConfigBuilder {
    config: AppConfig,
}

impl Default for TestConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl TestConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: AppConfig::default(),
        }
    }

    /// Create a minimal valid configuration
    pub fn minimal_valid() -> Self {
        let mut builder = Self::new();
        builder.config.inference.model_path = PathBuf::from("models/test_model.onnx");
        builder.config.input.source = "webcam".to_string();
        builder
    }

    /// Create configuration with invalid mode
    pub fn with_invalid_mode(mut self, mode: &str) -> Self {
        self.config.mode.mode_type = mode.to_string();
        self
    }

    /// Create configuration with invalid confidence threshold
    pub fn with_invalid_confidence(mut self, threshold: f32) -> Self {
        self.config.inference.confidence_threshold = threshold;
        self
    }

    /// Create configuration with invalid batch size
    pub fn with_invalid_batch_size(mut self, batch_size: usize) -> Self {
        self.config.inference.batch_size = batch_size;
        self
    }

    /// Create configuration with invalid execution provider
    pub fn with_invalid_execution_provider(mut self, provider: &str) -> Self {
        self.config.inference.execution_providers = vec![provider.to_string()];
        self
    }

    /// Create configuration with invalid device ID
    pub fn with_invalid_device_id(mut self, device_id: u32) -> Self {
        self.config.input.device_id = Some(device_id);
        self
    }

    /// Create configuration with invalid target size
    pub fn with_invalid_target_size(mut self, width: u32, height: u32) -> Self {
        if let Some(ref mut preprocessing) = self.config.preprocessing {
            preprocessing.target_size = [width, height];
        } else {
            self.config.preprocessing = Some(PreprocessingConfig {
                target_size: [width, height],
                letterbox: true,
                normalize: true,
            });
        }
        self
    }

    /// Create configuration with non-existent model file
    pub fn with_nonexistent_model(mut self) -> Self {
        self.config.inference.model_path = PathBuf::from("models/nonexistent_model.onnx");
        self
    }

    /// Create configuration with invalid RTSP URL
    pub fn with_invalid_rtsp_url(mut self, url: &str) -> Self {
        self.config.input.source = url.to_string();
        self
    }

    /// Create configuration with invalid output format
    pub fn with_invalid_output_format(mut self, format: &str) -> Self {
        self.config.output.output_format = format.to_string();
        self
    }

    /// Create configuration with invalid caps
    pub fn with_invalid_caps(mut self, caps: &str) -> Self {
        self.config.input.caps = Some(caps.to_string());
        self
    }

    /// Build the configuration
    pub fn build(self) -> AppConfig {
        self.config
    }
}

/// Test file generator for creating temporary test files
pub struct TestFileGenerator {
    temp_dir: TempDir,
}

impl TestFileGenerator {
    pub fn new() -> std::result::Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            temp_dir: TempDir::new()?,
        })
    }

    /// Create a temporary TOML config file with given content
    pub fn create_toml_file(
        &self,
        content: &str,
    ) -> std::result::Result<PathBuf, Box<dyn std::error::Error>> {
        let file_path = self.temp_dir.path().join("test_config.toml");
        fs::write(&file_path, content)?;
        Ok(file_path)
    }

    /// Create an invalid TOML file
    pub fn create_invalid_toml(&self) -> std::result::Result<PathBuf, Box<dyn std::error::Error>> {
        let content = r#"
            [mode
            type = "production"  # Missing closing bracket

            [input]
            source = "webcam"
            device_id = "invalid_number"  # Should be number, not string
        "#;
        self.create_toml_file(content)
    }

    /// Create a TOML file with missing required fields
    pub fn create_incomplete_toml(
        &self,
    ) -> std::result::Result<PathBuf, Box<dyn std::error::Error>> {
        let content = r#"
            [mode]
            type = "production"

            # Missing inference section entirely
            [input]
            source = "webcam"
        "#;
        self.create_toml_file(content)
    }

    /// Create a temporary mock ONNX model file
    pub fn create_mock_onnx_file(
        &self,
    ) -> std::result::Result<PathBuf, Box<dyn std::error::Error>> {
        let file_path = self.temp_dir.path().join("mock_model.onnx");
        fs::write(&file_path, b"mock onnx content")?;
        Ok(file_path)
    }

    /// Create a temporary video file
    pub fn create_mock_video_file(
        &self,
    ) -> std::result::Result<PathBuf, Box<dyn std::error::Error>> {
        let file_path = self.temp_dir.path().join("mock_video.mp4");
        fs::write(&file_path, b"mock video content")?;
        Ok(file_path)
    }

    /// Create a file with invalid extension
    pub fn create_invalid_model_file(
        &self,
    ) -> std::result::Result<PathBuf, Box<dyn std::error::Error>> {
        let file_path = self.temp_dir.path().join("invalid_model.txt");
        fs::write(&file_path, b"not an onnx file")?;
        Ok(file_path)
    }

    /// Get the temp directory path
    pub fn temp_dir(&self) -> &std::path::Path {
        self.temp_dir.path()
    }
}

impl Default for TestFileGenerator {
    fn default() -> Self {
        Self::new().expect("Failed to create temp directory")
    }
}

/// Resource constraint simulator for testing edge cases
pub struct ResourceConstraints {
    temp_dir: TempDir,
}

impl ResourceConstraints {
    pub fn new() -> std::result::Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            temp_dir: TempDir::new()?,
        })
    }

    /// Create a directory without write permissions (platform-specific)
    pub fn create_readonly_directory(
        &self,
    ) -> std::result::Result<PathBuf, Box<dyn std::error::Error>> {
        let dir_path = self.temp_dir.path().join("readonly");
        fs::create_dir(&dir_path)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&dir_path)?.permissions();
            perms.set_mode(0o444); // Read-only
            fs::set_permissions(&dir_path, perms)?;
        }

        Ok(dir_path)
    }

    /// Create a file that simulates insufficient disk space scenario
    pub fn create_large_file(
        &self,
        size_mb: usize,
    ) -> std::result::Result<PathBuf, Box<dyn std::error::Error>> {
        let file_path = self.temp_dir.path().join("large_file.dat");
        let mut file = fs::File::create(&file_path)?;

        // Write in chunks to avoid memory issues
        let chunk = vec![0u8; 1024 * 1024]; // 1MB chunk
        for _ in 0..size_mb {
            file.write_all(&chunk)?;
        }

        Ok(file_path)
    }
}

/// Memory leak detector for integration tests
pub struct MemoryLeakDetector {
    initial_memory: usize,
}

impl MemoryLeakDetector {
    pub fn new() -> Self {
        Self {
            initial_memory: Self::get_current_memory(),
        }
    }

    /// Check for memory leaks (threshold in MB)
    pub fn check_for_leaks(&self, threshold_mb: usize) -> bool {
        let current_memory = Self::get_current_memory();
        let memory_increase = current_memory.saturating_sub(self.initial_memory);
        memory_increase > threshold_mb
    }

    #[cfg(target_os = "macos")]
    fn get_current_memory() -> usize {
        use std::process::Command;

        let output = Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .unwrap_or_else(|_| std::process::Output {
                status: std::process::ExitStatus::from_raw(1),
                stdout: b"0".to_vec(),
                stderr: Vec::new(),
            });

        String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<usize>()
            .unwrap_or(0)
            / 1024 // Convert KB to MB
    }

    #[cfg(target_os = "linux")]
    fn get_current_memory() -> usize {
        let status_file = format!("/proc/{}/status", std::process::id());
        let content = std::fs::read_to_string(status_file).unwrap_or_default();

        for line in content.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    return parts[1].parse::<usize>().unwrap_or(0) / 1024; // Convert KB to MB
                }
            }
        }
        0
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    fn get_current_memory() -> usize {
        0 // Fallback for unsupported platforms
    }
}

/// Concurrent test executor for testing race conditions
pub struct ConcurrentTestExecutor;

impl ConcurrentTestExecutor {
    /// Execute multiple operations concurrently and collect results
    pub async fn execute_concurrent<F, T>(operations: Vec<F>) -> Vec<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        use tokio::task;

        let handles: Vec<_> = operations
            .into_iter()
            .map(|op| task::spawn_blocking(op))
            .collect();

        let mut results = Vec::new();
        for handle in handles {
            if let Ok(result) = handle.await {
                results.push(result);
            }
        }
        results
    }

}

/// Property-based test generator
pub struct PropertyTestGenerator;

impl PropertyTestGenerator {
    /// Generate random valid confidence thresholds
    pub fn valid_confidence_thresholds(count: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..count).map(|_| rng.gen_range(0.0..=1.0)).collect()
    }

    /// Generate invalid confidence thresholds
    pub fn invalid_confidence_thresholds(count: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut invalid = Vec::new();

        for _ in 0..count {
            let threshold = if rng.gen::<bool>() {
                rng.gen_range(-10.0..0.0) // Negative values
            } else {
                rng.gen_range(1.1..10.0) // Values > 1.0
            };
            invalid.push(threshold);
        }
        invalid
    }

    /// Generate random valid batch sizes
    pub fn valid_batch_sizes(count: usize) -> Vec<usize> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        (0..count).map(|_| rng.gen_range(1..=32)).collect()
    }

    /// Generate invalid batch sizes
    pub fn invalid_batch_sizes(count: usize) -> Vec<usize> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut invalid = Vec::new();

        for i in 0..count {
            let batch_size = if i % 2 == 0 {
                0 // Zero batch size
            } else {
                rng.gen_range(33..1000) // Too large batch size
            };
            invalid.push(batch_size);
        }
        invalid
    }

    /// Generate random valid target sizes
    pub fn valid_target_sizes(count: usize) -> Vec<[u32; 2]> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let valid_sizes = [
            [224, 224],
            [416, 416],
            [640, 640],
            [800, 600],
            [1024, 768],
            [1280, 720],
            [1920, 1080],
        ];

        (0..count)
            .map(|_| valid_sizes[rng.gen_range(0..valid_sizes.len())])
            .collect()
    }

    /// Generate invalid target sizes
    pub fn invalid_target_sizes(count: usize) -> Vec<[u32; 2]> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut invalid = Vec::new();

        for i in 0..count {
            let size = match i % 4 {
                0 => [0, 640],                                                 // Zero width
                1 => [640, 0],                                                 // Zero height
                2 => [0, 0],                                                   // Both zero
                _ => [rng.gen_range(5000..10000), rng.gen_range(5000..10000)], // Too large
            };
            invalid.push(size);
        }
        invalid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builder() {
        let config = TestConfigBuilder::new()
            .with_invalid_confidence(-0.5)
            .build();

        assert_eq!(config.inference.confidence_threshold, -0.5);
    }

    #[test]
    fn test_file_generator() {
        let generator = TestFileGenerator::new().unwrap();

        let toml_path = generator
            .create_toml_file("[mode]\ntype = \"test\"")
            .unwrap();
        assert!(toml_path.exists());

        let content = fs::read_to_string(toml_path).unwrap();
        assert!(content.contains("test"));
    }

    #[test]
    fn test_property_generator() {
        let valid_thresholds = PropertyTestGenerator::valid_confidence_thresholds(10);
        assert_eq!(valid_thresholds.len(), 10);
        assert!(valid_thresholds.iter().all(|&t| t >= 0.0 && t <= 1.0));

        let invalid_thresholds = PropertyTestGenerator::invalid_confidence_thresholds(10);
        assert_eq!(invalid_thresholds.len(), 10);
        assert!(invalid_thresholds.iter().all(|&t| t < 0.0 || t > 1.0));
    }

    #[test]
    fn test_memory_leak_detector() {
        let detector = MemoryLeakDetector::new();
        // Should not detect leaks immediately
        assert!(!detector.check_for_leaks(1000));
    }
}
