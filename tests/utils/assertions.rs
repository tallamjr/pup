//! Custom assertions for domain-specific testing in the Pup video processing system

use gstpup::config::AppConfig;
use gstpup::error::PupError;
use gstpup::metrics::Metrics;
use std::sync::Arc;

/// Custom assertion for configuration validation errors
pub fn assert_config_validation_error(result: Result<(), PupError>, expected_field: &str) {
    match result {
        Err(PupError::InvalidConfigValue { field, .. }) => {
            assert_eq!(
                field, expected_field,
                "Expected validation error for field '{}', got '{}'",
                expected_field, field
            );
        }
        Err(other_error) => {
            panic!(
                "Expected InvalidConfigValue error for field '{}', got: {:?}",
                expected_field, other_error
            );
        }
        Ok(()) => {
            panic!(
                "Expected validation error for field '{}', but validation passed",
                expected_field
            );
        }
    }
}

/// Assert that a configuration parsing error occurs
pub fn assert_config_parse_error(result: Result<AppConfig, PupError>) {
    match result {
        Err(PupError::ConfigParseError(_)) => { /* Expected */ }
        Err(other_error) => {
            panic!("Expected ConfigParseError, got: {:?}", other_error);
        }
        Ok(_) => {
            panic!("Expected configuration parsing to fail, but it succeeded");
        }
    }
}

/// Assert that a file not found error occurs
pub fn assert_file_not_found_error(result: Result<AppConfig, PupError>, expected_path: &str) {
    match result {
        Err(PupError::ConfigNotFound(path)) => {
            assert!(
                path.to_string_lossy().contains(expected_path),
                "Expected path '{}' in error, got '{}'",
                expected_path,
                path.display()
            );
        }
        Err(other_error) => {
            panic!("Expected ConfigNotFound error, got: {:?}", other_error);
        }
        Ok(_) => {
            panic!("Expected file not found error, but operation succeeded");
        }
    }
}

/// Assert that metrics are within expected ranges
pub fn assert_metrics_in_range(metrics: &Metrics, min_fps: f64, max_fps: f64, max_latency_ms: f64) {
    let fps = metrics.get_fps();
    let latency = metrics.get_inference_latency_ms();

    assert!(
        fps >= min_fps && fps <= max_fps,
        "FPS {} is not within range [{}, {}]",
        fps,
        min_fps,
        max_fps
    );

    assert!(
        latency <= max_latency_ms,
        "Latency {}ms exceeds maximum allowed {}ms",
        latency,
        max_latency_ms
    );
}

/// Assert that memory usage is reasonable
pub fn assert_memory_usage_reasonable(metrics: &Metrics, max_memory_mb: usize) {
    let memory = metrics.get_memory_usage_mb();
    let peak_memory = metrics.get_peak_memory_mb();

    assert!(
        memory <= max_memory_mb,
        "Memory usage {}MB exceeds maximum allowed {}MB",
        memory,
        max_memory_mb
    );

    assert!(
        peak_memory >= memory,
        "Peak memory {}MB should be >= current memory {}MB",
        peak_memory,
        memory
    );
}

/// Assert that frame drop rate is acceptable
pub fn assert_frame_drop_rate_acceptable(metrics: &Metrics, max_drop_rate_percent: f64) {
    let drop_rate = metrics.get_frame_drop_rate();

    assert!(
        drop_rate <= max_drop_rate_percent,
        "Frame drop rate {:.2}% exceeds maximum allowed {:.2}%",
        drop_rate,
        max_drop_rate_percent
    );
}

/// Assert that performance targets are met
pub fn assert_performance_targets_met(metrics: &Metrics, target_fps: f64, max_latency_ms: f64) {
    let result = metrics.check_performance_targets(target_fps, max_latency_ms);
    assert!(result.is_ok(), "Performance targets not met: {:?}", result);
}

/// Assert that an error contains specific context
pub fn assert_error_contains_context(error: &PupError, expected_context: &str) {
    let error_message = error.to_string();
    assert!(
        error_message.contains(expected_context),
        "Error message '{}' does not contain expected context '{}'",
        error_message,
        expected_context
    );
}

/// Assert that a model loading error occurs
pub fn assert_model_loading_error(result: Result<AppConfig, PupError>) {
    match result {
        Err(PupError::ModelLoadError(_)) => { /* Expected */ }
        Err(PupError::ModelFormatError(_)) => { /* Also acceptable */ }
        Err(other_error) => {
            panic!(
                "Expected ModelLoadError or ModelFormatError, got: {:?}",
                other_error
            );
        }
        Ok(_) => {
            panic!("Expected model loading error, but validation succeeded");
        }
    }
}

/// Assert that concurrent operations complete without data corruption
pub fn assert_concurrent_operations_safe(metrics: Arc<Metrics>, expected_total_operations: usize) {
    let total_frames = metrics.get_total_frames();
    let dropped_frames = metrics.get_dropped_frames();

    assert!(
        total_frames <= expected_total_operations,
        "Total frames {} exceeds expected operations {}",
        total_frames,
        expected_total_operations
    );

    assert!(
        dropped_frames <= total_frames,
        "Dropped frames {} cannot exceed total frames {}",
        dropped_frames,
        total_frames
    );
}

/// Assert that a configuration has valid defaults
pub fn assert_valid_default_config(config: &AppConfig) {
    // Mode should be valid
    assert!(
        ["production", "live", "detection", "benchmark"].contains(&config.mode.mode_type.as_str()),
        "Invalid default mode: {}",
        config.mode.mode_type
    );

    // Inference config should be valid
    assert!(
        config.inference.confidence_threshold >= 0.0
            && config.inference.confidence_threshold <= 1.0,
        "Invalid default confidence threshold: {}",
        config.inference.confidence_threshold
    );

    assert!(
        config.inference.batch_size > 0 && config.inference.batch_size <= 32,
        "Invalid default batch size: {}",
        config.inference.batch_size
    );

    // Input config should be valid
    if let Some(device_id) = config.input.device_id {
        assert!(device_id <= 99, "Invalid default device ID: {}", device_id);
    }

    // Output format should be valid
    assert!(
        ["mp4", "json", "rtmp", "avi", "mov"].contains(&config.output.output_format.as_str()),
        "Invalid default output format: {}",
        config.output.output_format
    );
}

/// Assert that boundary values are handled correctly
pub fn assert_boundary_value_handling<T, F>(
    test_fn: F,
    boundary_values: Vec<T>,
    expected_failures: usize,
) where
    F: Fn(T) -> Result<(), PupError>,
    T: std::fmt::Debug + Clone,
{
    let mut failure_count = 0;

    for value in boundary_values {
        match test_fn(value.clone()) {
            Ok(()) => { /* Success case */ }
            Err(_) => failure_count += 1,
        }
    }

    assert_eq!(
        failure_count, expected_failures,
        "Expected {} boundary value failures, got {}",
        expected_failures, failure_count
    );
}

/// Assert that resource exhaustion is handled gracefully
pub fn assert_resource_exhaustion_handled<F>(operation: F)
where
    F: FnOnce() -> Result<(), PupError>,
{
    match operation() {
        Err(PupError::InsufficientMemory { .. }) => { /* Expected */ }
        Err(PupError::InsufficientDiskSpace(_)) => { /* Expected */ }
        Err(PupError::PermissionDenied(_)) => { /* Expected */ }
        Err(other_error) => {
            panic!("Expected resource exhaustion error, got: {:?}", other_error);
        }
        Ok(()) => {
            panic!("Expected resource exhaustion error, but operation succeeded");
        }
    }
}

/// Assert that TOML serialization roundtrip preserves data
pub fn assert_toml_roundtrip_preserves_data(original: &AppConfig) {
    let serialized = toml::to_string_pretty(original).expect("Serialization failed");
    let deserialized: AppConfig = toml::from_str(&serialized).expect("Deserialization failed");

    // Compare key fields
    assert_eq!(original.mode.mode_type, deserialized.mode.mode_type);
    assert_eq!(original.input.source, deserialized.input.source);
    assert_eq!(
        original.inference.confidence_threshold,
        deserialized.inference.confidence_threshold
    );
    assert_eq!(
        original.inference.batch_size,
        deserialized.inference.batch_size
    );
    assert_eq!(
        original.output.output_format,
        deserialized.output.output_format
    );
}

/// Assert that error recovery works correctly
pub fn assert_error_recovery<F, R>(operation: F, recovery: R)
where
    F: FnOnce() -> Result<(), PupError>,
    R: FnOnce(PupError) -> Result<(), PupError>,
{
    match operation() {
        Ok(()) => { /* No error to recover from */ }
        Err(error) => {
            match recovery(error) {
                Ok(()) => { /* Recovery successful */ }
                Err(recovery_error) => {
                    panic!("Error recovery failed: {:?}", recovery_error);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{PropertyTestGenerator, TestConfigBuilder};

    #[test]
    fn test_config_validation_assertion() {
        let config = TestConfigBuilder::new()
            .with_invalid_confidence(1.5)
            .build();

        let result = config.validate();
        assert_config_validation_error(result, "inference.confidence_threshold");
    }

    #[test]
    fn test_metrics_assertions() {
        let metrics = Metrics::new();
        metrics.update_fps(30.0);
        metrics.update_inference_latency(50.0);
        metrics.update_memory_usage(256);

        assert_metrics_in_range(&metrics, 20.0, 40.0, 100.0);
        assert_memory_usage_reasonable(&metrics, 512);
    }

    #[test]
    fn test_boundary_value_assertion() {
        let test_confidence = |threshold: f32| -> Result<(), PupError> {
            if threshold < 0.0 || threshold > 1.0 {
                Err(PupError::InvalidConfigValue {
                    field: "confidence".to_string(),
                    value: threshold.to_string(),
                })
            } else {
                Ok(())
            }
        };

        let boundary_values = vec![-0.1, 0.0, 0.5, 1.0, 1.1];
        assert_boundary_value_handling(test_confidence, boundary_values, 2); // -0.1 and 1.1 should fail
    }

    #[test]
    fn test_default_config_assertion() {
        let config = AppConfig::default();
        assert_valid_default_config(&config);
    }
}
