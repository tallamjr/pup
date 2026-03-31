//! Test fixtures for consistent test data across the test suite

use gstpup::config::*;
use gstpup::error::PupError;
use std::fs;
use std::path::PathBuf;
use tempfile::{NamedTempFile, TempDir};

/// Collection of standard test configurations
pub struct ConfigFixtures;

impl ConfigFixtures {
    /// Minimal valid configuration for testing
    pub fn minimal_valid() -> AppConfig {
        AppConfig {
            mode: ModeConfig {
                mode_type: "production".to_string(),
            },
            input: InputConfig {
                source: "webcam".to_string(),
                device_id: Some(0),
                caps: None,
            },
            inference: InferenceConfig {
                backend: "ort".to_string(),
                execution_providers: vec!["cpu".to_string()],
                model_path: PathBuf::from("models/yolov8n.onnx"),
                confidence_threshold: 0.5,
                batch_size: 1,
                device: None,
            },
            output: OutputConfig {
                display_enabled: true,
                recording_enabled: false,
                output_format: "mp4".to_string(),
            },
            preprocessing: Some(PreprocessingConfig {
                target_size: [640, 640],
                letterbox: true,
                normalize: true,
            }),
            pipeline: None,
        }
    }

    /// High-performance configuration
    pub fn high_performance() -> AppConfig {
        AppConfig {
            mode: ModeConfig {
                mode_type: "production".to_string(),
            },
            input: InputConfig {
                source: "webcam".to_string(),
                device_id: Some(0),
                caps: Some("video/x-raw,width=1920,height=1080,framerate=60/1".to_string()),
            },
            inference: InferenceConfig {
                backend: "ort".to_string(),
                execution_providers: vec!["coreml".to_string(), "cpu".to_string()],
                model_path: PathBuf::from("models/yolov8n.onnx"),
                confidence_threshold: 0.3,
                batch_size: 4,
                device: None,
            },
            output: OutputConfig {
                display_enabled: true,
                recording_enabled: true,
                output_format: "mp4".to_string(),
            },
            preprocessing: Some(PreprocessingConfig {
                target_size: [640, 640],
                letterbox: true,
                normalize: true,
            }),
            pipeline: None,
        }
    }

    /// Configuration for file input
    pub fn file_input() -> AppConfig {
        let mut config = Self::minimal_valid();
        config.input.source = "assets/sample.mp4".to_string();
        config.input.device_id = None;
        config
    }

    /// Configuration for RTSP input
    pub fn rtsp_input() -> AppConfig {
        let mut config = Self::minimal_valid();
        config.input.source = "rtsp://example.com/stream".to_string();
        config.input.device_id = None;
        config
    }

    /// Configuration with invalid mode
    pub fn invalid_mode() -> AppConfig {
        let mut config = Self::minimal_valid();
        config.mode.mode_type = "invalid_mode".to_string();
        config
    }

    /// Configuration with invalid confidence threshold
    pub fn invalid_confidence() -> AppConfig {
        let mut config = Self::minimal_valid();
        config.inference.confidence_threshold = 1.5;
        config
    }

    /// Configuration with invalid batch size
    pub fn invalid_batch_size() -> AppConfig {
        let mut config = Self::minimal_valid();
        config.inference.batch_size = 0;
        config
    }

    /// Configuration with invalid execution provider
    pub fn invalid_execution_provider() -> AppConfig {
        let mut config = Self::minimal_valid();
        config.inference.execution_providers = vec!["invalid_provider".to_string()];
        config
    }

    /// Configuration with invalid device ID
    pub fn invalid_device_id() -> AppConfig {
        let mut config = Self::minimal_valid();
        config.input.device_id = Some(999);
        config
    }

    /// Configuration with invalid target size
    pub fn invalid_target_size() -> AppConfig {
        let mut config = Self::minimal_valid();
        config.preprocessing = Some(PreprocessingConfig {
            target_size: [0, 640],
            letterbox: true,
            normalize: true,
        });
        config
    }

    /// Configuration with non-existent model
    pub fn nonexistent_model() -> AppConfig {
        let mut config = Self::minimal_valid();
        config.inference.model_path = PathBuf::from("models/nonexistent.onnx");
        config
    }

    /// Configuration with invalid model format
    pub fn invalid_model_format() -> AppConfig {
        let mut config = Self::minimal_valid();
        config.inference.model_path = PathBuf::from("models/invalid.txt");
        config
    }

    /// Configuration with invalid output format
    pub fn invalid_output_format() -> AppConfig {
        let mut config = Self::minimal_valid();
        config.output.output_format = "invalid_format".to_string();
        config
    }

    /// Configuration with invalid caps
    pub fn invalid_caps() -> AppConfig {
        let mut config = Self::minimal_valid();
        config.input.caps = Some("invalid/caps".to_string());
        config
    }

    /// Configuration with extreme values (boundary testing)
    pub fn extreme_values() -> AppConfig {
        AppConfig {
            mode: ModeConfig {
                mode_type: "benchmark".to_string(),
            },
            input: InputConfig {
                source: "webcam".to_string(),
                device_id: Some(99), // Maximum valid device ID
                caps: Some("video/x-raw,width=4096,height=4096,framerate=120/1".to_string()),
            },
            inference: InferenceConfig {
                backend: "ort".to_string(),
                execution_providers: vec![
                    "coreml".to_string(),
                    "cuda".to_string(),
                    "cpu".to_string(),
                ],
                model_path: PathBuf::from("models/test.onnx"),
                confidence_threshold: 1.0, // Maximum valid confidence
                batch_size: 32,            // Maximum valid batch size
                device: None,
            },
            output: OutputConfig {
                display_enabled: true,
                recording_enabled: true,
                output_format: "mov".to_string(),
            },
            preprocessing: Some(PreprocessingConfig {
                target_size: [4096, 4096], // Maximum reasonable size
                letterbox: false,
                normalize: false,
            }),
            pipeline: None,
        }
    }

    /// Minimal configuration for memory-constrained environments
    pub fn memory_constrained() -> AppConfig {
        AppConfig {
            mode: ModeConfig {
                mode_type: "live".to_string(),
            },
            input: InputConfig {
                source: "webcam".to_string(),
                device_id: Some(0),
                caps: Some("video/x-raw,width=320,height=240,framerate=15/1".to_string()),
            },
            inference: InferenceConfig {
                backend: "ort".to_string(),
                execution_providers: vec!["cpu".to_string()],
                model_path: PathBuf::from("models/test.onnx"),
                confidence_threshold: 0.7,
                batch_size: 1,
                device: None,
            },
            output: OutputConfig {
                display_enabled: false,
                recording_enabled: false,
                output_format: "json".to_string(),
            },
            preprocessing: Some(PreprocessingConfig {
                target_size: [224, 224], // Smaller target size
                letterbox: true,
                normalize: true,
            }),
            pipeline: None,
        }
    }
}

/// Collection of TOML configuration strings for testing
pub struct TomlFixtures;

impl TomlFixtures {
    /// Valid minimal TOML configuration
    pub fn minimal_valid() -> &'static str {
        r#"
[mode]
type = "production"

[input]
source = "webcam"
device_id = 0

[inference]
backend = "ort"
execution_providers = ["cpu"]
model_path = "models/test.onnx"
confidence_threshold = 0.5
batch_size = 1

[output]
display_enabled = true
recording_enabled = false
output_format = "mp4"

[preprocessing]
target_size = [640, 640]
letterbox = true
normalize = true
"#
    }

    /// Invalid TOML syntax
    pub fn invalid_syntax() -> &'static str {
        r#"
[mode
type = "production"  # Missing closing bracket

[input]
source = "webcam"
device_id = "invalid"  # Should be number
"#
    }

    /// Missing required sections
    pub fn missing_sections() -> &'static str {
        r#"
[mode]
type = "production"

[input]
source = "webcam"

# Missing inference section
[output]
display_enabled = true
"#
    }

    /// Invalid field values
    pub fn invalid_values() -> &'static str {
        r#"
[mode]
type = "invalid_mode"

[input]
source = "webcam"
device_id = 999

[inference]
backend = "invalid_backend"
execution_providers = ["invalid_provider"]
model_path = "models/test.onnx"
confidence_threshold = 1.5
batch_size = 0

[output]
display_enabled = true
output_format = "invalid_format"

[preprocessing]
target_size = [0, 640]
"#
    }

    /// TOML with extra fields (should be ignored)
    pub fn extra_fields() -> &'static str {
        r#"
[mode]
type = "production"
extra_field = "should_be_ignored"

[input]
source = "webcam"
device_id = 0
unknown_field = 123

[inference]
backend = "ort"
execution_providers = ["cpu"]
model_path = "models/test.onnx"
confidence_threshold = 0.5
batch_size = 1
deprecated_field = "old_value"

[output]
display_enabled = true
recording_enabled = false
output_format = "mp4"

[preprocessing]
target_size = [640, 640]
letterbox = true
normalize = true

[unknown_section]
field1 = "value1"
field2 = 42
"#
    }

    /// Complex configuration with all fields
    pub fn complex_complete() -> &'static str {
        r#"
[mode]
type = "benchmark"

[input]
source = "rtsp://example.com/stream"
device_id = 1
caps = "video/x-raw,width=1920,height=1080,framerate=30/1"

[inference]
backend = "ort"
execution_providers = ["coreml", "cuda", "cpu"]
model_path = "models/yolov8l.onnx"
confidence_threshold = 0.25
batch_size = 8
device = "auto"

[output]
display_enabled = true
recording_enabled = true
output_format = "avi"

[preprocessing]
target_size = [1024, 1024]
letterbox = false
normalize = false

[pipeline]
video_source = "rtsp://example.com/stream"
display_enabled = true
framerate = 60
"#
    }

    /// Edge case values
    pub fn edge_case_values() -> &'static str {
        r#"
[mode]
type = "detection"

[input]
source = "/dev/video99"
device_id = 99
caps = "video/x-raw,width=4096,height=4096,framerate=120/1"

[inference]
backend = "ort"
execution_providers = ["tensorrt", "openvino", "cpu"]
model_path = "models/edge_case.onnx"
confidence_threshold = 0.001
batch_size = 32

[output]
display_enabled = false
recording_enabled = true
output_format = "rtmp"

[preprocessing]
target_size = [4096, 4096]
letterbox = true
normalize = true
"#
    }
}

/// Collection of error scenarios for testing
pub struct ErrorFixtures;

impl ErrorFixtures {
    /// Create various PupError instances for testing
    pub fn input_not_available() -> PupError {
        PupError::InputNotAvailable("webcam not found".to_string())
    }

    pub fn video_file_error() -> PupError {
        PupError::VideoFileError(PathBuf::from("missing_video.mp4"))
    }

    pub fn model_load_error() -> PupError {
        PupError::ModelLoadError(PathBuf::from("invalid_model.onnx"))
    }

    pub fn inference_error() -> PupError {
        PupError::InferenceError("model execution failed".to_string())
    }

    pub fn pipeline_error() -> PupError {
        PupError::PipelineError("element creation failed".to_string())
    }

    pub fn config_not_found() -> PupError {
        PupError::ConfigNotFound(PathBuf::from("missing_config.toml"))
    }

    pub fn config_parse_error() -> PupError {
        PupError::ConfigParseError("invalid TOML syntax".to_string())
    }

    pub fn invalid_config_value() -> PupError {
        PupError::InvalidConfigValue {
            field: "confidence_threshold".to_string(),
            value: "1.5".to_string(),
        }
    }

    pub fn unexpected() -> PupError {
        PupError::Unexpected("something went wrong".to_string())
    }
}

/// File fixtures for testing file operations
pub struct FileFixtures {
    temp_dir: TempDir,
}

impl FileFixtures {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            temp_dir: TempDir::new()?,
        })
    }

    /// Create a valid TOML config file
    pub fn create_valid_config_file(&self) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let file_path = self.temp_dir.path().join("valid_config.toml");
        fs::write(&file_path, TomlFixtures::minimal_valid())?;
        Ok(file_path)
    }

    /// Create an invalid TOML config file
    pub fn create_invalid_config_file(&self) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let file_path = self.temp_dir.path().join("invalid_config.toml");
        fs::write(&file_path, TomlFixtures::invalid_syntax())?;
        Ok(file_path)
    }

    /// Create a mock ONNX model file
    pub fn create_mock_model_file(&self) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let file_path = self.temp_dir.path().join("mock_model.onnx");
        fs::write(&file_path, b"mock onnx model data")?;
        Ok(file_path)
    }

    /// Create a file with wrong extension
    pub fn create_wrong_extension_file(&self) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let file_path = self.temp_dir.path().join("model.txt");
        fs::write(&file_path, b"not an onnx file")?;
        Ok(file_path)
    }

    /// Create a readonly directory
    pub fn create_readonly_directory(&self) -> Result<PathBuf, Box<dyn std::error::Error>> {
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

    /// Get temp directory path
    pub fn temp_dir(&self) -> &std::path::Path {
        self.temp_dir.path()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_fixtures() {
        let config = ConfigFixtures::minimal_valid();
        assert_eq!(config.mode.mode_type, "production");
        assert!(config.validate().is_ok());

        let invalid_config = ConfigFixtures::invalid_confidence();
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_toml_fixtures() {
        let toml_content = TomlFixtures::minimal_valid();
        let config: Result<AppConfig, _> = toml::from_str(toml_content);
        assert!(config.is_ok());

        let invalid_toml = TomlFixtures::invalid_syntax();
        let invalid_config: Result<AppConfig, _> = toml::from_str(invalid_toml);
        assert!(invalid_config.is_err());
    }

    #[test]
    fn test_error_fixtures() {
        let error = ErrorFixtures::input_not_available();
        assert!(error.to_string().contains("webcam not found"));

        let error = ErrorFixtures::model_load_error();
        assert!(error.to_string().contains("invalid_model.onnx"));
    }

    #[test]
    fn test_file_fixtures() {
        let fixtures = FileFixtures::new().unwrap();

        let config_path = fixtures.create_valid_config_file().unwrap();
        assert!(config_path.exists());

        let model_path = fixtures.create_mock_model_file().unwrap();
        assert!(model_path.exists());
        assert!(model_path.extension().unwrap() == "onnx");
    }
}
