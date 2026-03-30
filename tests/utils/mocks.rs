//! Mock objects for testing complex scenarios

use gstpup::error::{PupError, PupResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Mock inference backend for testing inference operations
pub struct MockInferenceBackend {
    model_loaded: bool,
    inference_latency_ms: Duration,
    should_fail: bool,
    failure_rate: f64,
    inference_count: Arc<Mutex<usize>>,
}

impl MockInferenceBackend {
    pub fn new() -> Self {
        Self {
            model_loaded: false,
            inference_latency_ms: Duration::from_millis(50),
            should_fail: false,
            failure_rate: 0.0,
            inference_count: Arc::new(Mutex::new(0)),
        }
    }

    pub fn with_model_loaded(mut self) -> Self {
        self.model_loaded = true;
        self
    }

    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.inference_latency_ms = latency;
        self
    }

    pub fn with_failure(mut self) -> Self {
        self.should_fail = true;
        self
    }

    pub fn with_failure_rate(mut self, rate: f64) -> Self {
        self.failure_rate = rate.clamp(0.0, 1.0);
        self
    }

    pub fn load_model(&mut self, _model_path: &std::path::Path) -> PupResult<()> {
        if self.should_fail {
            return Err(PupError::ModelLoadError(_model_path.to_path_buf()));
        }
        self.model_loaded = true;
        Ok(())
    }

    pub fn run_inference(&self, _input_data: &[f32]) -> PupResult<Vec<f32>> {
        *self.inference_count.lock().unwrap() += 1;

        if self.should_fail {
            return Err(PupError::InferenceError(
                "Mock inference failure".to_string(),
            ));
        }

        // Simulate random failures based on failure rate
        if self.failure_rate > 0.0 {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            if rng.gen::<f64>() < self.failure_rate {
                return Err(PupError::InferenceError("Random failure".to_string()));
            }
        }

        if !self.model_loaded {
            return Err(PupError::InferenceError("Model not loaded".to_string()));
        }

        // Simulate inference latency
        std::thread::sleep(self.inference_latency_ms);

        // Return mock detection results
        Ok(vec![0.1, 0.2, 0.8, 0.9, 0.95, 1.0]) // Mock bounding box + confidence + class
    }

    pub fn get_inference_count(&self) -> usize {
        *self.inference_count.lock().unwrap()
    }
}

impl Default for MockInferenceBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock video source for testing input scenarios
pub struct MockVideoSource {
    frame_count: Arc<Mutex<usize>>,
    should_fail: bool,
    failure_after_frames: Option<usize>,
    frame_rate: Duration,
}

impl MockVideoSource {
    pub fn new() -> Self {
        Self {
            frame_count: Arc::new(Mutex::new(0)),
            should_fail: false,
            failure_after_frames: None,
            frame_rate: Duration::from_millis(33), // ~30 FPS
        }
    }

    pub fn with_failure(mut self) -> Self {
        self.should_fail = true;
        self
    }

    pub fn with_failure_after_frames(mut self, frames: usize) -> Self {
        self.failure_after_frames = Some(frames);
        self
    }

    pub fn with_frame_rate(mut self, fps: u32) -> Self {
        self.frame_rate = Duration::from_millis(1000 / fps as u64);
        self
    }

    pub fn get_next_frame(&self) -> PupResult<MockFrame> {
        let mut count = self.frame_count.lock().unwrap();
        *count += 1;

        if self.should_fail {
            return Err(PupError::InputNotAvailable(
                "Mock source failure".to_string(),
            ));
        }

        if let Some(failure_point) = self.failure_after_frames {
            if *count > failure_point {
                return Err(PupError::InputNotAvailable(
                    "Source failed after frames".to_string(),
                ));
            }
        }

        // Simulate frame timing
        std::thread::sleep(self.frame_rate);

        Ok(MockFrame {
            id: *count,
            timestamp: Instant::now(),
            data: vec![128u8; 640 * 480 * 3], // Mock RGB data
        })
    }

    pub fn get_frame_count(&self) -> usize {
        *self.frame_count.lock().unwrap()
    }
}

impl Default for MockVideoSource {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct MockFrame {
    pub id: usize,
    pub timestamp: Instant,
    pub data: Vec<u8>,
}

/// Mock file system for testing I/O operations
pub struct MockFileSystem {
    files: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    readonly_paths: Arc<Mutex<Vec<String>>>,
    should_simulate_disk_full: bool,
}

impl MockFileSystem {
    pub fn new() -> Self {
        Self {
            files: Arc::new(Mutex::new(HashMap::new())),
            readonly_paths: Arc::new(Mutex::new(Vec::new())),
            should_simulate_disk_full: false,
        }
    }

    pub fn with_disk_full(mut self) -> Self {
        self.should_simulate_disk_full = true;
        self
    }

    pub fn add_file(&self, path: &str, content: Vec<u8>) {
        self.files.lock().unwrap().insert(path.to_string(), content);
    }

    pub fn add_readonly_path(&self, path: &str) {
        self.readonly_paths.lock().unwrap().push(path.to_string());
    }

    pub fn read_file(&self, path: &str) -> PupResult<Vec<u8>> {
        let files = self.files.lock().unwrap();
        files
            .get(path)
            .cloned()
            .ok_or_else(|| PupError::VideoFileError(std::path::PathBuf::from(path)))
    }

    pub fn write_file(&self, path: &str, content: Vec<u8>) -> PupResult<()> {
        if self.should_simulate_disk_full {
            return Err(PupError::Unexpected(format!(
                "Disk full: cannot write {} bytes",
                content.len()
            )));
        }

        let readonly_paths = self.readonly_paths.lock().unwrap();
        if readonly_paths.contains(&path.to_string()) {
            return Err(PupError::Unexpected(format!(
                "Permission denied: {}",
                path
            )));
        }

        self.files.lock().unwrap().insert(path.to_string(), content);
        Ok(())
    }

    pub fn file_exists(&self, path: &str) -> bool {
        self.files.lock().unwrap().contains_key(path)
    }

    pub fn get_file_count(&self) -> usize {
        self.files.lock().unwrap().len()
    }
}

impl Default for MockFileSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Mock network client for testing streaming scenarios
pub struct MockNetworkClient {
    connection_successful: bool,
    latency: Duration,
    should_timeout: bool,
    packet_loss_rate: f64,
    sent_packets: Arc<Mutex<usize>>,
    received_packets: Arc<Mutex<usize>>,
}

impl MockNetworkClient {
    pub fn new() -> Self {
        Self {
            connection_successful: true,
            latency: Duration::from_millis(50),
            should_timeout: false,
            packet_loss_rate: 0.0,
            sent_packets: Arc::new(Mutex::new(0)),
            received_packets: Arc::new(Mutex::new(0)),
        }
    }

    pub fn with_connection_failure(mut self) -> Self {
        self.connection_successful = false;
        self
    }

    pub fn with_timeout(mut self) -> Self {
        self.should_timeout = true;
        self
    }

    pub fn with_packet_loss(mut self, loss_rate: f64) -> Self {
        self.packet_loss_rate = loss_rate.clamp(0.0, 1.0);
        self
    }

    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.latency = latency;
        self
    }

    pub fn connect(&self, _url: &str) -> PupResult<()> {
        if !self.connection_successful {
            return Err(PupError::Unexpected(format!(
                "Connection failed: {}",
                _url
            )));
        }

        if self.should_timeout {
            std::thread::sleep(Duration::from_secs(5));
            return Err(PupError::Unexpected("Connection timeout".to_string()));
        }

        // Simulate connection latency
        std::thread::sleep(self.latency);
        Ok(())
    }

    pub fn send_packet(&self, _data: &[u8]) -> PupResult<()> {
        *self.sent_packets.lock().unwrap() += 1;

        // Simulate packet loss
        if self.packet_loss_rate > 0.0 {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            if rng.gen::<f64>() < self.packet_loss_rate {
                return Ok(()); // Packet lost, but not an error from sender perspective
            }
        }

        *self.received_packets.lock().unwrap() += 1;
        Ok(())
    }

    pub fn get_sent_packets(&self) -> usize {
        *self.sent_packets.lock().unwrap()
    }

    pub fn get_received_packets(&self) -> usize {
        *self.received_packets.lock().unwrap()
    }

    pub fn get_packet_loss_rate(&self) -> f64 {
        let sent = self.get_sent_packets();
        let received = self.get_received_packets();

        if sent == 0 {
            0.0
        } else {
            1.0 - (received as f64 / sent as f64)
        }
    }
}

impl Default for MockNetworkClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_inference_backend() {
        let mut backend = MockInferenceBackend::new();
        let model_path = std::path::Path::new("test.onnx");

        assert!(backend.load_model(model_path).is_ok());

        let input = vec![1.0, 2.0, 3.0];
        let result = backend.run_inference(&input);
        assert!(result.is_ok());
        assert_eq!(backend.get_inference_count(), 1);
    }

    #[test]
    fn test_mock_video_source() {
        let source = MockVideoSource::new();

        let frame = source.get_next_frame();
        assert!(frame.is_ok());
        assert_eq!(source.get_frame_count(), 1);

        let frame_data = frame.unwrap();
        assert_eq!(frame_data.id, 1);
        assert!(!frame_data.data.is_empty());
    }

    #[test]
    fn test_mock_file_system() {
        let fs = MockFileSystem::new();
        let path = "test.txt";
        let content = b"test content".to_vec();

        assert!(fs.write_file(path, content.clone()).is_ok());
        assert!(fs.file_exists(path));

        let read_content = fs.read_file(path);
        assert!(read_content.is_ok());
        assert_eq!(read_content.unwrap(), content);
    }

    #[test]
    fn test_mock_network_client() {
        let client = MockNetworkClient::new();

        assert!(client.connect("rtsp://test.com/stream").is_ok());
        assert!(client.send_packet(b"test data").is_ok());

        assert_eq!(client.get_sent_packets(), 1);
        assert_eq!(client.get_received_packets(), 1);
    }
}
