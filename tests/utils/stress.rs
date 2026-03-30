//! Stress testing utilities for comprehensive system testing under load

use gstpup::config::AppConfig;
use gstpup::error::{PupError, PupResult};
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
    stop_flag: Arc<AtomicBool>,
    operation_count: Arc<AtomicUsize>,
    success_count: Arc<AtomicUsize>,
    failure_count: Arc<AtomicUsize>,
}

impl StressTestExecutor {
    pub fn new(config: StressTestConfig) -> Self {
        Self {
            config,
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

        // Start worker threads
        for _thread_id in 0..self.config.thread_count {
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
            peak_memory_mb: 0,
            avg_cpu_percent: 0.0,
            avg_ops_per_second: total_ops as f64 / actual_duration.as_secs_f64(),
            memory_limit_exceeded: false,
            cpu_limit_exceeded: false,
            errors: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::ConfigFixtures;

    #[test]
    fn test_stress_test_config() {
        let config = StressTestConfig::default();
        assert_eq!(config.thread_count, 4);
        assert_eq!(config.duration_seconds, 30);
        assert_eq!(config.ops_per_second, 100);
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
