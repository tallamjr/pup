//! Stress testing utilities for comprehensive system testing under load

use gstpup::config::AppConfig;
use gstpup::error::{PupError, PupResult};
use gstpup::metrics::{Metrics, PerformanceMonitor};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};

/// Stress test configuration
#[derive(Debug, Clone)]
pub struct StressTestConfig {
    /// Number of concurrent threads
    pub thread_count: usize,
    /// Test duration in seconds
    pub duration_seconds: u64,
    /// Operations per second target
    pub ops_per_second: u64,
    /// Memory limit in MB
    pub memory_limit_mb: usize,
    /// CPU limit as percentage (0-100)
    pub cpu_limit_percent: f64,
    /// Whether to inject random failures
    pub inject_failures: bool,
    /// Failure injection rate (0.0-1.0)
    pub failure_rate: f64,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            thread_count: 4,
            duration_seconds: 30,
            ops_per_second: 100,
            memory_limit_mb: 1024,
            cpu_limit_percent: 80.0,
            inject_failures: false,
            failure_rate: 0.0,
        }
    }
}

/// Results of a stress test
#[derive(Debug, Clone)]
pub struct StressTestResults {
    /// Total operations completed
    pub total_operations: usize,
    /// Operations that succeeded
    pub successful_operations: usize,
    /// Operations that failed
    pub failed_operations: usize,
    /// Test duration
    pub actual_duration: Duration,
    /// Peak memory usage in MB
    pub peak_memory_mb: usize,
    /// Average CPU usage percentage
    pub avg_cpu_percent: f64,
    /// Average operations per second
    pub avg_ops_per_second: f64,
    /// Whether memory limit was exceeded
    pub memory_limit_exceeded: bool,
    /// Whether CPU limit was exceeded
    pub cpu_limit_exceeded: bool,
    /// Any errors encountered
    pub errors: Vec<String>,
}

impl StressTestResults {
    /// Check if the stress test passed all criteria
    pub fn passed(&self) -> bool {
        self.failed_operations == 0
            && !self.memory_limit_exceeded
            && !self.cpu_limit_exceeded
            && self.errors.is_empty()
    }

    /// Get success rate as percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_operations == 0 {
            0.0
        } else {
            (self.successful_operations as f64 / self.total_operations as f64) * 100.0
        }
    }
}

/// Stress test executor
pub struct StressTestExecutor {
    config: StressTestConfig,
    metrics: Arc<Metrics>,
    stop_flag: Arc<AtomicBool>,
    operation_count: Arc<AtomicUsize>,
    success_count: Arc<AtomicUsize>,
    failure_count: Arc<AtomicUsize>,
}

impl StressTestExecutor {
    pub fn new(config: StressTestConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(Metrics::new()),
            stop_flag: Arc::new(AtomicBool::new(false)),
            operation_count: Arc::new(AtomicUsize::new(0)),
            success_count: Arc::new(AtomicUsize::new(0)),
            failure_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Run configuration validation stress test
    pub fn run_config_validation_stress<F>(&self, test_fn: F) -> StressTestResults
    where
        F: Fn() -> PupResult<AppConfig> + Send + Sync + 'static,
    {
        let test_fn = Arc::new(test_fn);
        self.run_generic_stress_test(move || {
            let config_result = test_fn();
            match config_result {
                Ok(config) => {
                    // Validate the configuration
                    config.validate().map(|_| ())
                }
                Err(e) => Err(e),
            }
        })
    }

    /// Run metrics update stress test
    pub fn run_metrics_stress_test(&self) -> StressTestResults {
        let metrics = self.metrics.clone();
        self.run_generic_stress_test(move || {
            use rand::Rng;
            let mut rng = rand::thread_rng();

            // Perform various metrics operations
            metrics.update_fps(rng.gen_range(10.0..120.0));
            metrics.update_inference_latency(rng.gen_range(1.0..100.0));
            metrics.update_memory_usage(rng.gen_range(100..2048));
            metrics.increment_total_frames();

            if rng.gen_bool(0.1) {
                metrics.increment_dropped_frames();
            }

            Ok(())
        })
    }

    /// Run concurrent configuration access stress test
    pub fn run_concurrent_config_access(&self, configs: Vec<AppConfig>) -> StressTestResults {
        let configs = Arc::new(configs);
        self.run_generic_stress_test(move || {
            use rand::Rng;
            let mut rng = rand::thread_rng();

            // Randomly select and validate a configuration
            let config = &configs[rng.gen_range(0..configs.len())];

            // Perform various read operations
            let _ = config.model_path();
            let _ = config.input_source();
            let _ = config.video_source();
            let _ = config.model_exists();
            let _ = config.video_exists();

            // Validate configuration
            config.validate()
        })
    }

    /// Run memory allocation stress test
    pub fn run_memory_stress_test(&self) -> StressTestResults {
        self.run_generic_stress_test(|| {
            // Allocate and deallocate memory to stress the system
            let size = rand::random::<usize>() % (1024 * 1024); // Up to 1MB
            let mut data = Vec::with_capacity(size);

            // Fill with random data
            for _ in 0..size {
                data.push(rand::random::<u8>());
            }

            // Force deallocation
            drop(data);

            Ok(())
        })
    }

    /// Run file I/O stress test
    pub fn run_file_io_stress_test(&self, temp_dir: &std::path::Path) -> StressTestResults {
        let temp_dir = temp_dir.to_path_buf();
        self.run_generic_stress_test(move || {
            use rand::Rng;
            use std::fs;

            let mut rng = rand::thread_rng();
            let file_name = format!("stress_test_{}.tmp", rng.gen::<u64>());
            let file_path = temp_dir.join(file_name);

            // Generate random data
            let data_size = rng.gen_range(1..10240); // 1-10KB
            let data: Vec<u8> = (0..data_size).map(|_| rng.gen()).collect();

            // Write file
            fs::write(&file_path, &data)
                .map_err(|e| PupError::Unexpected(format!("Write failed: {}", e)))?;

            // Read file back
            let read_data = fs::read(&file_path)
                .map_err(|e| PupError::Unexpected(format!("Read failed: {}", e)))?;

            // Verify data integrity
            if data != read_data {
                return Err(PupError::Unexpected("Data corruption detected".to_string()));
            }

            // Clean up
            let _ = fs::remove_file(&file_path);

            Ok(())
        })
    }

    /// Generic stress test runner
    fn run_generic_stress_test<F>(&self, operation: F) -> StressTestResults
    where
        F: Fn() -> PupResult<()> + Send + Sync + 'static,
    {
        let start_time = Instant::now();
        let operation = Arc::new(operation);

        // Reset counters
        self.stop_flag.store(false, Ordering::Relaxed);
        self.operation_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        self.failure_count.store(0, Ordering::Relaxed);

        // Create barrier for synchronized start
        let barrier = Arc::new(Barrier::new(self.config.thread_count + 1));
        let mut handles = Vec::new();

        // Start monitoring thread
        let monitor_handle = self.start_monitoring_thread();

        // Start worker threads
        for thread_id in 0..self.config.thread_count {
            let operation = operation.clone();
            let stop_flag = self.stop_flag.clone();
            let operation_count = self.operation_count.clone();
            let success_count = self.success_count.clone();
            let failure_count = self.failure_count.clone();
            let barrier = barrier.clone();
            let config = self.config.clone();

            let handle = thread::spawn(move || {
                barrier.wait(); // Wait for synchronized start

                let thread_ops_per_second =
                    config.ops_per_second / config.thread_count.max(1) as u64;
                let operation_interval = Duration::from_millis(1000 / thread_ops_per_second.max(1));

                while !stop_flag.load(Ordering::Relaxed) {
                    let op_start = Instant::now();

                    // Execute operation
                    match operation() {
                        Ok(()) => {
                            success_count.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(_) => {
                            failure_count.fetch_add(1, Ordering::Relaxed);
                        }
                    }

                    operation_count.fetch_add(1, Ordering::Relaxed);

                    // Rate limiting
                    let elapsed = op_start.elapsed();
                    if elapsed < operation_interval {
                        thread::sleep(operation_interval - elapsed);
                    }
                }
            });

            handles.push(handle);
        }

        // Start the test
        barrier.wait();

        // Run for specified duration
        thread::sleep(Duration::from_secs(self.config.duration_seconds));

        // Stop all threads
        self.stop_flag.store(true, Ordering::Relaxed);

        // Wait for all threads to complete
        for handle in handles {
            let _ = handle.join();
        }

        // Stop monitoring
        let _ = monitor_handle.join();

        let actual_duration = start_time.elapsed();

        // Collect results
        let total_ops = self.operation_count.load(Ordering::Relaxed);
        let successful_ops = self.success_count.load(Ordering::Relaxed);
        let failed_ops = self.failure_count.load(Ordering::Relaxed);

        StressTestResults {
            total_operations: total_ops,
            successful_operations: successful_ops,
            failed_operations: failed_ops,
            actual_duration,
            peak_memory_mb: self.metrics.get_peak_memory_mb(),
            avg_cpu_percent: self.metrics.get_cpu_usage_percent(),
            avg_ops_per_second: total_ops as f64 / actual_duration.as_secs_f64(),
            memory_limit_exceeded: self.metrics.get_peak_memory_mb() > self.config.memory_limit_mb,
            cpu_limit_exceeded: self.metrics.get_cpu_usage_percent()
                > self.config.cpu_limit_percent,
            errors: Vec::new(), // TODO: Collect actual errors
        }
    }

    /// Start monitoring thread to track resource usage
    fn start_monitoring_thread(&self) -> thread::JoinHandle<()> {
        let metrics = self.metrics.clone();
        let stop_flag = self.stop_flag.clone();

        thread::spawn(move || {
            let mut monitor = crate::utils::MemoryLeakDetector::new();

            while !stop_flag.load(Ordering::Relaxed) {
                // Update memory metrics
                if let Ok(memory_mb) = Self::get_current_memory_usage() {
                    metrics.update_memory_usage(memory_mb);
                }

                // Update CPU metrics (mock implementation)
                let cpu_usage = Self::get_current_cpu_usage();
                metrics.update_cpu_usage(cpu_usage);

                thread::sleep(Duration::from_millis(100)); // Monitor every 100ms
            }
        })
    }

    #[cfg(target_os = "macos")]
    fn get_current_memory_usage() -> PupResult<usize> {
        use std::process::Command;

        let output = Command::new("ps")
            .args(["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .map_err(|e| PupError::Unexpected(format!("Failed to get memory usage: {}", e)))?;

        let rss_kb = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<usize>()
            .unwrap_or(0);

        Ok(rss_kb / 1024) // Convert KB to MB
    }

    #[cfg(target_os = "linux")]
    fn get_current_memory_usage() -> PupResult<usize> {
        let status_file = format!("/proc/{}/status", std::process::id());
        let content = std::fs::read_to_string(status_file)
            .map_err(|e| PupError::Unexpected(format!("Failed to read proc status: {}", e)))?;

        for line in content.lines() {
            if line.starts_with("VmRSS:") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let rss_kb = parts[1].parse::<usize>().unwrap_or(0);
                    return Ok(rss_kb / 1024); // Convert KB to MB
                }
            }
        }

        Ok(0)
    }

    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    fn get_current_memory_usage() -> PupResult<usize> {
        Ok(0) // Fallback for unsupported platforms
    }

    fn get_current_cpu_usage() -> f64 {
        // Mock CPU usage - in a real implementation, this would use platform-specific APIs
        use rand::Rng;
        let mut rng = rand::thread_rng();
        rng.gen_range(5.0..25.0) // Mock CPU usage between 5-25%
    }
}

/// Load testing framework for sustained performance testing
pub struct LoadTester {
    base_config: AppConfig,
    metrics_history: Arc<std::sync::Mutex<Vec<MetricsSnapshot>>>,
}

#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub timestamp: Instant,
    pub fps: f64,
    pub memory_mb: usize,
    pub cpu_percent: f64,
    pub operation_count: usize,
}

impl LoadTester {
    pub fn new(base_config: AppConfig) -> Self {
        Self {
            base_config,
            metrics_history: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    /// Run sustained load test
    pub fn run_sustained_load_test(&self, duration_minutes: u64) -> LoadTestResults {
        let start_time = Instant::now();
        let duration = Duration::from_secs(duration_minutes * 60);
        let metrics = Arc::new(Metrics::new());
        let operation_count = Arc::new(AtomicUsize::new(0));

        // Sample metrics every second
        let metrics_clone = metrics.clone();
        let history_clone = self.metrics_history.clone();
        let operation_count_clone = operation_count.clone();

        let sampling_handle = thread::spawn(move || {
            while start_time.elapsed() < duration {
                let snapshot = MetricsSnapshot {
                    timestamp: Instant::now(),
                    fps: metrics_clone.get_fps(),
                    memory_mb: metrics_clone.get_memory_usage_mb(),
                    cpu_percent: metrics_clone.get_cpu_usage_percent(),
                    operation_count: operation_count_clone.load(Ordering::Relaxed),
                };

                history_clone.lock().unwrap().push(snapshot);
                thread::sleep(Duration::from_secs(1));
            }
        });

        // Simulate load
        let mut handles = Vec::new();
        for _ in 0..4 {
            let metrics = metrics.clone();
            let operation_count = operation_count.clone();
            let start_time = start_time;

            let handle = thread::spawn(move || {
                while start_time.elapsed() < duration {
                    // Simulate operations
                    metrics.increment_total_frames();
                    metrics.update_fps(30.0);
                    metrics.update_inference_latency(50.0);
                    operation_count.fetch_add(1, Ordering::Relaxed);

                    thread::sleep(Duration::from_millis(33)); // ~30 FPS
                }
            });
            handles.push(handle);
        }

        // Wait for completion
        for handle in handles {
            let _ = handle.join();
        }
        let _ = sampling_handle.join();

        self.analyze_load_test_results(start_time.elapsed())
    }

    fn analyze_load_test_results(&self, actual_duration: Duration) -> LoadTestResults {
        let history = self.metrics_history.lock().unwrap();

        if history.is_empty() {
            return LoadTestResults {
                duration: actual_duration,
                avg_fps: 0.0,
                min_fps: 0.0,
                max_fps: 0.0,
                avg_memory_mb: 0,
                peak_memory_mb: 0,
                avg_cpu_percent: 0.0,
                peak_cpu_percent: 0.0,
                total_operations: 0,
                performance_degradation_detected: false,
                memory_leak_detected: false,
                stable_performance: false,
            };
        }

        let mut fps_values: Vec<f64> = history.iter().map(|s| s.fps).collect();
        let mut memory_values: Vec<usize> = history.iter().map(|s| s.memory_mb).collect();
        let mut cpu_values: Vec<f64> = history.iter().map(|s| s.cpu_percent).collect();

        fps_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        memory_values.sort();
        cpu_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let avg_fps = fps_values.iter().sum::<f64>() / fps_values.len() as f64;
        let avg_memory = memory_values.iter().sum::<usize>() / memory_values.len();
        let avg_cpu = cpu_values.iter().sum::<f64>() / cpu_values.len() as f64;

        // Detect performance degradation (fps drops over time)
        let first_quarter = &fps_values[0..fps_values.len() / 4];
        let last_quarter = &fps_values[fps_values.len() * 3 / 4..];
        let first_avg = first_quarter.iter().sum::<f64>() / first_quarter.len() as f64;
        let last_avg = last_quarter.iter().sum::<f64>() / last_quarter.len() as f64;
        let degradation = (first_avg - last_avg) / first_avg > 0.1; // 10% degradation

        // Detect memory leak (consistent increase over time)
        let first_mem = memory_values[0];
        let last_mem = *memory_values.last().unwrap();
        let memory_increase = last_mem > first_mem && (last_mem - first_mem) > first_mem / 2;

        LoadTestResults {
            duration: actual_duration,
            avg_fps,
            min_fps: fps_values[0],
            max_fps: *fps_values.last().unwrap(),
            avg_memory_mb: avg_memory,
            peak_memory_mb: *memory_values.last().unwrap(),
            avg_cpu_percent: avg_cpu,
            peak_cpu_percent: *cpu_values.last().unwrap(),
            total_operations: history.last().map(|s| s.operation_count).unwrap_or(0),
            performance_degradation_detected: degradation,
            memory_leak_detected: memory_increase,
            stable_performance: !degradation && !memory_increase,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoadTestResults {
    pub duration: Duration,
    pub avg_fps: f64,
    pub min_fps: f64,
    pub max_fps: f64,
    pub avg_memory_mb: usize,
    pub peak_memory_mb: usize,
    pub avg_cpu_percent: f64,
    pub peak_cpu_percent: f64,
    pub total_operations: usize,
    pub performance_degradation_detected: bool,
    pub memory_leak_detected: bool,
    pub stable_performance: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::{ConfigFixtures, TestFileGenerator};

    #[test]
    fn test_stress_test_config() {
        let config = StressTestConfig::default();
        assert_eq!(config.thread_count, 4);
        assert_eq!(config.duration_seconds, 30);
        assert_eq!(config.ops_per_second, 100);
    }

    #[test]
    fn test_metrics_stress_test() {
        let config = StressTestConfig {
            thread_count: 2,
            duration_seconds: 1, // Short test
            ops_per_second: 10,
            ..Default::default()
        };

        let executor = StressTestExecutor::new(config);
        let results = executor.run_metrics_stress_test();

        assert!(results.total_operations > 0);
        assert!(results.successful_operations <= results.total_operations);
    }

    #[test]
    fn test_config_validation_stress() {
        let config = StressTestConfig {
            thread_count: 2,
            duration_seconds: 1,
            ops_per_second: 5,
            ..Default::default()
        };

        let executor = StressTestExecutor::new(config);
        let results = executor.run_config_validation_stress(|| Ok(ConfigFixtures::minimal_valid()));

        assert!(results.successful_operations > 0);
        assert_eq!(results.failed_operations, 0);
    }

    #[test]
    fn test_file_io_stress() {
        let temp_dir = tempfile::TempDir::new().unwrap();
        let config = StressTestConfig {
            thread_count: 2,
            duration_seconds: 1,
            ops_per_second: 10,
            ..Default::default()
        };

        let executor = StressTestExecutor::new(config);
        let results = executor.run_file_io_stress_test(temp_dir.path());

        // Should have some successful operations
        assert!(results.total_operations > 0);
    }

    #[test]
    fn test_load_tester() {
        let config = ConfigFixtures::minimal_valid();
        let tester = LoadTester::new(config);

        // Very short load test
        let results = tester.run_sustained_load_test(1); // 1 minute

        assert!(results.duration.as_secs() >= 60);
        assert!(results.total_operations > 0);
    }

    #[test]
    fn test_stress_test_results() {
        let results = StressTestResults {
            total_operations: 100,
            successful_operations: 95,
            failed_operations: 5,
            actual_duration: Duration::from_secs(30),
            peak_memory_mb: 256,
            avg_cpu_percent: 45.0,
            avg_ops_per_second: 3.3,
            memory_limit_exceeded: false,
            cpu_limit_exceeded: false,
            errors: Vec::new(),
        };

        assert!(!results.passed()); // Has failed operations
        assert_eq!(results.success_rate(), 95.0);
    }
}
