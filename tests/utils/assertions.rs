//! Custom assertions for domain-specific testing in the Pup video processing system

use gstpup::config::AppConfig;
use gstpup::error::PupError;

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
        Err(PupError::InvalidConfigValue { field, .. }) if field.contains("model_path") => {
            /* Also acceptable for format errors */
        }
        Err(other_error) => {
            panic!(
                "Expected ModelLoadError or model-related InvalidConfigValue, got: {:?}",
                other_error
            );
        }
        Ok(_) => {
            panic!("Expected model loading error, but validation succeeded");
        }
    }
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
