//! Data generators for property-based and fuzz testing

use gstpup::config::*;
use rand::Rng;
use std::path::PathBuf;

/// Configuration generator for property-based testing
pub struct ConfigGenerator;

impl ConfigGenerator {
    /// Generate a random valid configuration
    pub fn random_valid() -> AppConfig {
        let mut rng = rand::thread_rng();

        AppConfig {
            mode: ModeConfig {
                mode_type: Self::random_mode(),
            },
            input: InputConfig {
                source: Self::random_input_source(),
                device_id: Some(rng.gen_range(0..=3)),
                caps: Some(Self::random_caps()),
            },
            inference: InferenceConfig {
                backend: "ort".to_string(),
                execution_providers: Self::random_execution_providers(),
                model_path: PathBuf::from("models/test.onnx"),
                confidence_threshold: rng.gen_range(0.1..=0.9),
                batch_size: rng.gen_range(1..=16),
                device: None,
            },
            output: OutputConfig {
                display_enabled: rng.gen_bool(0.8),
                recording_enabled: rng.gen_bool(0.3),
                output_format: Self::random_output_format(),
            },
            preprocessing: Some(PreprocessingConfig {
                target_size: Self::random_target_size(),
                letterbox: rng.gen_bool(0.7),
                normalize: rng.gen_bool(0.9),
            }),
            pipeline: None,
        }
    }

    /// Generate configurations with specific invalid fields
    pub fn with_invalid_field(field: &str) -> AppConfig {
        let mut config = Self::random_valid();
        let mut rng = rand::thread_rng();

        match field {
            "mode" => config.mode.mode_type = format!("invalid_{}", rng.gen::<u32>()),
            "confidence" => {
                config.inference.confidence_threshold =
                    f32::max(rng.gen_range(-1.0..0.0) as f32, 1.1_f32)
            }
            "batch_size" => {
                config.inference.batch_size = if rng.gen_bool(0.5) {
                    0
                } else {
                    rng.gen_range(33..100)
                }
            }
            "device_id" => config.input.device_id = Some(rng.gen_range(100..999)),
            "execution_provider" => {
                config.inference.execution_providers = vec![format!("invalid_{}", rng.gen::<u32>())]
            }
            "output_format" => {
                config.output.output_format = format!("invalid_{}", rng.gen::<u32>())
            }
            "target_size" => {
                if let Some(ref mut preprocessing) = config.preprocessing {
                    preprocessing.target_size = if rng.gen_bool(0.5) {
                        [0, 640]
                    } else {
                        [rng.gen_range(5000..10000), rng.gen_range(5000..10000)]
                    };
                }
            }
            _ => {}
        }

        config
    }

    fn random_mode() -> String {
        let modes = ["production", "live", "detection", "benchmark"];
        let mut rng = rand::thread_rng();
        modes[rng.gen_range(0..modes.len())].to_string()
    }

    fn random_input_source() -> String {
        let sources = [
            "webcam",
            "assets/sample.mp4",
            "rtsp://example.com/stream",
            "/dev/video0",
        ];
        let mut rng = rand::thread_rng();
        sources[rng.gen_range(0..sources.len())].to_string()
    }

    fn random_caps() -> String {
        let resolutions = [(640, 480), (1280, 720), (1920, 1080), (320, 240)];
        let framerates = [15, 30, 60];
        let mut rng = rand::thread_rng();

        let (width, height) = resolutions[rng.gen_range(0..resolutions.len())];
        let framerate = framerates[rng.gen_range(0..framerates.len())];

        format!(
            "video/x-raw,width={},height={},framerate={}/1",
            width, height, framerate
        )
    }

    fn random_execution_providers() -> Vec<String> {
        let providers = ["coreml", "cpu", "cuda", "tensorrt", "openvino"];
        let mut rng = rand::thread_rng();
        let count = rng.gen_range(1..=3);

        let mut selected = Vec::new();
        for _ in 0..count {
            let provider = providers[rng.gen_range(0..providers.len())];
            if !selected.contains(&provider.to_string()) {
                selected.push(provider.to_string());
            }
        }

        if selected.is_empty() {
            vec!["cpu".to_string()]
        } else {
            selected
        }
    }

    fn random_output_format() -> String {
        let formats = ["mp4", "json", "rtmp", "avi", "mov"];
        let mut rng = rand::thread_rng();
        formats[rng.gen_range(0..formats.len())].to_string()
    }

    fn random_target_size() -> [u32; 2] {
        let sizes = [
            [224, 224],
            [416, 416],
            [640, 640],
            [800, 600],
            [1024, 768],
            [1280, 720],
        ];
        let mut rng = rand::thread_rng();
        sizes[rng.gen_range(0..sizes.len())]
    }

    /// Generate boundary value configurations
    pub fn boundary_values() -> Vec<AppConfig> {
        vec![
            Self::min_values(),
            Self::max_values(),
            Self::zero_values(),
            Self::negative_values(),
            Self::extreme_values(),
        ]
    }

    fn min_values() -> AppConfig {
        AppConfig {
            mode: ModeConfig {
                mode_type: "production".to_string(),
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
                confidence_threshold: 0.0,
                batch_size: 1,
                device: None,
            },
            output: OutputConfig {
                display_enabled: false,
                recording_enabled: false,
                output_format: "json".to_string(),
            },
            preprocessing: Some(PreprocessingConfig {
                target_size: [224, 224],
                letterbox: false,
                normalize: false,
            }),
            pipeline: None,
        }
    }

    fn max_values() -> AppConfig {
        AppConfig {
            mode: ModeConfig {
                mode_type: "benchmark".to_string(),
            },
            input: InputConfig {
                source: "webcam".to_string(),
                device_id: Some(99),
                caps: Some("video/x-raw,width=4096,height=4096,framerate=120/1".to_string()),
            },
            inference: InferenceConfig {
                backend: "ort".to_string(),
                execution_providers: vec![
                    "coreml".to_string(),
                    "cuda".to_string(),
                    "tensorrt".to_string(),
                    "openvino".to_string(),
                    "cpu".to_string(),
                ],
                model_path: PathBuf::from("models/test.onnx"),
                confidence_threshold: 1.0,
                batch_size: 32,
                device: None,
            },
            output: OutputConfig {
                display_enabled: true,
                recording_enabled: true,
                output_format: "mov".to_string(),
            },
            preprocessing: Some(PreprocessingConfig {
                target_size: [4096, 4096],
                letterbox: true,
                normalize: true,
            }),
            pipeline: None,
        }
    }

    fn zero_values() -> AppConfig {
        let mut config = Self::min_values();
        config.inference.confidence_threshold = 0.0;
        if let Some(ref mut preprocessing) = config.preprocessing {
            preprocessing.target_size = [0, 640]; // Invalid - will fail validation
        }
        config
    }

    fn negative_values() -> AppConfig {
        let mut config = Self::min_values();
        config.inference.confidence_threshold = -0.5; // Invalid
        config.inference.batch_size = 0; // Invalid
        config
    }

    fn extreme_values() -> AppConfig {
        let mut config = Self::max_values();
        config.inference.confidence_threshold = 2.0; // Invalid
        config.inference.batch_size = 1000; // Invalid
        config.input.device_id = Some(999); // Invalid
        if let Some(ref mut preprocessing) = config.preprocessing {
            preprocessing.target_size = [10000, 10000]; // Invalid - too large
        }
        config
    }
}

/// TOML string generator for testing malformed configurations
pub struct TomlGenerator;

impl TomlGenerator {
    /// Generate syntactically invalid TOML
    pub fn invalid_syntax() -> Vec<String> {
        vec![
            // Missing closing bracket
            r#"
[mode
type = "production"
"#
            .to_string(),
            // Invalid value types
            r#"
[mode]
type = "production"

[input]
device_id = "not_a_number"
"#
            .to_string(),
            // Unclosed strings
            r#"
[mode]
type = "production

[input]
source = "webcam"
"#
            .to_string(),
            // Invalid keys
            r#"
[mode]
123invalid = "production"
"#
            .to_string(),
            // Nested section errors
            r#"
[mode]
type = "production"

[input
source = "webcam"
[input.nested]
"#
            .to_string(),
            // Array syntax errors
            r#"
[inference]
execution_providers = ["cpu", "coreml"  # Missing closing bracket
"#
            .to_string(),
        ]
    }

    /// Generate TOML with missing required sections
    pub fn missing_sections() -> Vec<String> {
        vec![
            // Missing inference section
            r#"
[mode]
type = "production"

[input]
source = "webcam"
"#
            .to_string(),
            // Missing mode section
            r#"
[input]
source = "webcam"

[inference]
backend = "ort"
model_path = "models/test.onnx"
"#
            .to_string(),
            // Completely empty
            "".to_string(),
            // Only comments
            r#"
# This is a comment
# Another comment
"#
            .to_string(),
        ]
    }

    /// Generate TOML with invalid field values
    pub fn invalid_values() -> Vec<String> {
        vec![
            // Invalid mode
            r#"
[mode]
type = "invalid_mode"

[input]
source = "webcam"

[inference]
backend = "ort"
model_path = "models/test.onnx"
confidence_threshold = 0.5
batch_size = 1
execution_providers = ["cpu"]
"#
            .to_string(),
            // Invalid confidence threshold
            r#"
[mode]
type = "production"

[input]
source = "webcam"

[inference]
backend = "ort"
model_path = "models/test.onnx"
confidence_threshold = 1.5
batch_size = 1
execution_providers = ["cpu"]
"#
            .to_string(),
            // Invalid batch size
            r#"
[mode]
type = "production"

[input]
source = "webcam"

[inference]
backend = "ort"
model_path = "models/test.onnx"
confidence_threshold = 0.5
batch_size = 0
execution_providers = ["cpu"]
"#
            .to_string(),
        ]
    }

    /// Generate TOML with edge case values
    pub fn edge_cases() -> Vec<String> {
        vec![
            // Very long strings
            format!(
                r#"
[mode]
type = "production"

[input]
source = "{}"
"#,
                "a".repeat(10000)
            ),
            // Unicode characters
            r#"
[mode]
type = "production"

[input]
source = "测试摄像头🎥"

[inference]
backend = "ort"
model_path = "models/模型.onnx"
confidence_threshold = 0.5
batch_size = 1
execution_providers = ["cpu"]
"#
            .to_string(),
            // Special characters in paths
            r#"
[mode]
type = "production"

[input]
source = "path with spaces/video file.mp4"

[inference]
backend = "ort"
model_path = "models/model with-special_chars.onnx"
confidence_threshold = 0.5
batch_size = 1
execution_providers = ["cpu"]
"#
            .to_string(),
        ]
    }
}

/// Error scenario generator for testing error handling
pub struct ErrorScenarioGenerator;

impl ErrorScenarioGenerator {
    /// Generate input source error scenarios
    pub fn input_errors() -> Vec<(&'static str, String)> {
        vec![
            ("webcam_not_found", "webcam_99".to_string()),
            (
                "invalid_rtsp",
                "rtsp://invalid.url.that.does.not.exist/stream".to_string(),
            ),
            ("missing_file", "assets/nonexistent_video.mp4".to_string()),
            ("invalid_device", "/dev/video999".to_string()),
            ("empty_source", "".to_string()),
        ]
    }

    /// Generate model loading error scenarios
    pub fn model_errors() -> Vec<(&'static str, PathBuf)> {
        vec![
            ("missing_model", PathBuf::from("models/missing.onnx")),
            ("invalid_extension", PathBuf::from("models/model.txt")),
            ("empty_path", PathBuf::from("")),
            ("directory_instead_of_file", PathBuf::from("models/")),
            ("permission_denied", PathBuf::from("/root/model.onnx")),
        ]
    }

    /// Generate resource constraint scenarios
    pub fn resource_constraints() -> Vec<(&'static str, usize)> {
        vec![
            ("low_memory", 64),               // 64MB
            ("very_low_memory", 16),          // 16MB
            ("no_memory", 0),                 // 0MB
            ("excessive_memory", usize::MAX), // Maximum value
        ]
    }

    /// Generate network error scenarios
    pub fn network_errors() -> Vec<(&'static str, String)> {
        vec![
            ("timeout", "rtsp://slow.server.com/stream".to_string()),
            (
                "connection_refused",
                "rtsp://127.0.0.1:12345/stream".to_string(),
            ),
            (
                "dns_failure",
                "rtsp://non.existent.domain/stream".to_string(),
            ),
            ("malformed_url", "invalid://not.a.url".to_string()),
            ("missing_protocol", "example.com/stream".to_string()),
        ]
    }

    /// Generate concurrent access scenarios
    pub fn concurrent_scenarios() -> Vec<(&'static str, usize)> {
        vec![
            ("light_concurrent", 2),
            ("moderate_concurrent", 8),
            ("heavy_concurrent", 32),
            ("extreme_concurrent", 100),
        ]
    }
}

/// Performance test scenario generator
pub struct PerformanceGenerator;

impl PerformanceGenerator {
    /// Generate different load scenarios
    pub fn load_scenarios() -> Vec<(&'static str, LoadScenario)> {
        vec![
            (
                "light_load",
                LoadScenario {
                    fps: 15.0,
                    resolution: (640, 480),
                    batch_size: 1,
                    confidence_threshold: 0.5,
                    duration_seconds: 10,
                },
            ),
            (
                "moderate_load",
                LoadScenario {
                    fps: 30.0,
                    resolution: (1280, 720),
                    batch_size: 2,
                    confidence_threshold: 0.3,
                    duration_seconds: 30,
                },
            ),
            (
                "heavy_load",
                LoadScenario {
                    fps: 60.0,
                    resolution: (1920, 1080),
                    batch_size: 4,
                    confidence_threshold: 0.1,
                    duration_seconds: 60,
                },
            ),
            (
                "stress_test",
                LoadScenario {
                    fps: 120.0,
                    resolution: (4096, 2160),
                    batch_size: 8,
                    confidence_threshold: 0.05,
                    duration_seconds: 120,
                },
            ),
        ]
    }

    /// Generate memory pressure scenarios
    pub fn memory_scenarios() -> Vec<(&'static str, MemoryScenario)> {
        vec![
            (
                "normal_memory",
                MemoryScenario {
                    initial_allocation_mb: 100,
                    growth_rate_mb_per_sec: 0,
                    max_memory_mb: 512,
                },
            ),
            (
                "memory_leak",
                MemoryScenario {
                    initial_allocation_mb: 100,
                    growth_rate_mb_per_sec: 10,
                    max_memory_mb: 1024,
                },
            ),
            (
                "memory_pressure",
                MemoryScenario {
                    initial_allocation_mb: 400,
                    growth_rate_mb_per_sec: 5,
                    max_memory_mb: 512,
                },
            ),
            (
                "memory_exhaustion",
                MemoryScenario {
                    initial_allocation_mb: 800,
                    growth_rate_mb_per_sec: 50,
                    max_memory_mb: 1024,
                },
            ),
        ]
    }
}

#[derive(Debug, Clone)]
pub struct LoadScenario {
    pub fps: f64,
    pub resolution: (u32, u32),
    pub batch_size: usize,
    pub confidence_threshold: f32,
    pub duration_seconds: u64,
}

#[derive(Debug, Clone)]
pub struct MemoryScenario {
    pub initial_allocation_mb: usize,
    pub growth_rate_mb_per_sec: usize,
    pub max_memory_mb: usize,
}

/// Random data generator for fuzz testing
pub struct FuzzDataGenerator;

impl FuzzDataGenerator {
    /// Generate random byte sequences
    pub fn random_bytes(size: usize) -> Vec<u8> {
        let mut rng = rand::thread_rng();
        (0..size).map(|_| rng.gen()).collect()
    }

    /// Generate random strings with various encodings
    pub fn random_strings(count: usize) -> Vec<String> {
        let mut rng = rand::thread_rng();
        let mut strings = Vec::new();

        for _ in 0..count {
            let length = rng.gen_range(0..1000);
            let string = match rng.gen_range(0..4) {
                0 => Self::random_ascii_string(length),
                1 => Self::random_unicode_string(length),
                2 => Self::random_control_chars(length),
                _ => Self::random_mixed_string(length),
            };
            strings.push(string);
        }

        strings
    }

    fn random_ascii_string(length: usize) -> String {
        let mut rng = rand::thread_rng();
        (0..length)
            .map(|_| rng.gen_range(32..127) as u8 as char)
            .collect()
    }

    fn random_unicode_string(length: usize) -> String {
        let mut rng = rand::thread_rng();
        let unicode_ranges = [
            (0x0020, 0x007F),   // Basic Latin
            (0x00A0, 0x00FF),   // Latin-1 Supplement
            (0x0100, 0x017F),   // Latin Extended-A
            (0x4E00, 0x9FFF),   // CJK Unified Ideographs
            (0x1F600, 0x1F64F), // Emoticons
        ];

        (0..length)
            .map(|_| {
                let (start, end) = unicode_ranges[rng.gen_range(0..unicode_ranges.len())];
                std::char::from_u32(rng.gen_range(start..=end)).unwrap_or('?')
            })
            .collect()
    }

    fn random_control_chars(length: usize) -> String {
        let mut rng = rand::thread_rng();
        (0..length)
            .map(|_| rng.gen_range(0..32) as u8 as char)
            .collect()
    }

    fn random_mixed_string(length: usize) -> String {
        let mut rng = rand::thread_rng();
        (0..length)
            .map(|_| rng.gen_range(0..256) as u8 as char)
            .collect()
    }

    /// Generate random numerical values including edge cases
    pub fn random_numbers<T>(count: usize) -> Vec<T>
    where
        T: rand::distributions::uniform::SampleUniform + Copy + Default + std::fmt::Debug,
        rand::distributions::Standard: rand::distributions::Distribution<T>,
    {
        let mut rng = rand::thread_rng();
        (0..count).map(|_| rng.gen()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_generator() {
        let config = ConfigGenerator::random_valid();
        // Should be valid (though we can't guarantee model file exists)
        assert!(["production", "live", "detection", "benchmark"]
            .contains(&config.mode.mode_type.as_str()));
        assert!(
            config.inference.confidence_threshold >= 0.0
                && config.inference.confidence_threshold <= 1.0
        );
        assert!(config.inference.batch_size > 0 && config.inference.batch_size <= 32);
    }

    #[test]
    fn test_invalid_config_generation() {
        let config = ConfigGenerator::with_invalid_field("confidence");
        assert!(
            config.inference.confidence_threshold < 0.0
                || config.inference.confidence_threshold > 1.0
        );
    }

    #[test]
    fn test_toml_generator() {
        let invalid_tomls = TomlGenerator::invalid_syntax();
        assert!(!invalid_tomls.is_empty());

        for toml_str in invalid_tomls {
            let result: Result<AppConfig, _> = toml::from_str(&toml_str);
            assert!(
                result.is_err(),
                "Expected TOML parsing to fail for: {}",
                toml_str
            );
        }
    }

    #[test]
    fn test_error_scenarios() {
        let input_errors = ErrorScenarioGenerator::input_errors();
        assert!(!input_errors.is_empty());

        let model_errors = ErrorScenarioGenerator::model_errors();
        assert!(!model_errors.is_empty());

        for (name, _path) in model_errors {
            assert!(!name.is_empty());
        }
    }

    #[test]
    fn test_performance_scenarios() {
        let load_scenarios = PerformanceGenerator::load_scenarios();
        assert!(!load_scenarios.is_empty());

        for (name, scenario) in load_scenarios {
            assert!(!name.is_empty());
            assert!(scenario.fps > 0.0);
            assert!(scenario.resolution.0 > 0);
            assert!(scenario.resolution.1 > 0);
        }
    }

    #[test]
    fn test_fuzz_data_generator() {
        let bytes = FuzzDataGenerator::random_bytes(100);
        assert_eq!(bytes.len(), 100);

        let strings = FuzzDataGenerator::random_strings(5);
        assert_eq!(strings.len(), 5);
    }
}
