//! Test utilities and fixtures for the Pup testing framework
//!
//! This module provides shared testing infrastructure including
//! test fixtures, custom assertions, and property-based test generators.

use gstpup::config::{AppConfig, PreprocessingConfig};
use std::path::PathBuf;

pub mod assertions;
pub mod fixtures;
pub mod generators;

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
    fn test_property_generator() {
        let valid_thresholds = PropertyTestGenerator::valid_confidence_thresholds(10);
        assert_eq!(valid_thresholds.len(), 10);
        assert!(valid_thresholds.iter().all(|&t| t >= 0.0 && t <= 1.0));

        let invalid_thresholds = PropertyTestGenerator::invalid_confidence_thresholds(10);
        assert_eq!(invalid_thresholds.len(), 10);
        assert!(invalid_thresholds.iter().all(|&t| t < 0.0 || t > 1.0));
    }
}
