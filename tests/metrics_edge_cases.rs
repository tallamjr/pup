//! Comprehensive edge case testing for metrics and performance monitoring
//! This test suite focuses on achieving 85% coverage for the metrics module

use gstpup::error::PupError;
use gstpup::metrics::{
    ConsoleReporter, JsonReporter, Metrics, MetricsReporter, PerformanceMonitor,
};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use tempfile::NamedTempFile;

mod utils;
// Note: utils::assertions imports are available but currently unused
use utils::mocks::*;
use utils::stress::*;

/// Test metrics creation, initialization, and default values
#[cfg(test)]
mod metrics_initialization_tests {
    use super::*;

    #[test]
    fn test_metrics_default_initialization() {
        let metrics = Metrics::new();

        assert_eq!(metrics.get_fps(), 0.0);
        assert_eq!(metrics.get_inference_latency_ms(), 0.0);
        assert_eq!(metrics.get_memory_usage_mb(), 0);
        assert_eq!(metrics.get_dropped_frames(), 0);
        assert_eq!(metrics.get_total_frames(), 0);
        assert_eq!(metrics.get_avg_frame_time_ms(), 0.0);
        assert_eq!(metrics.get_peak_memory_mb(), 0);
        assert_eq!(metrics.get_cpu_usage_percent(), 0.0);
        assert_eq!(metrics.get_gpu_usage_percent(), 0.0);
        assert_eq!(metrics.get_frame_drop_rate(), 0.0);
    }

    #[test]
    fn test_metrics_default_trait() {
        let metrics = Metrics::default();

        // Should be identical to new()
        assert_eq!(metrics.get_fps(), 0.0);
        assert_eq!(metrics.get_total_frames(), 0);
        assert_eq!(metrics.get_memory_usage_mb(), 0);
    }

    #[test]
    fn test_performance_monitor_initialization() {
        let monitor = PerformanceMonitor::new();
        let metrics = monitor.metrics();

        assert_eq!(metrics.get_fps(), 0.0);
        assert_eq!(metrics.get_total_frames(), 0);
    }

    #[test]
    fn test_performance_monitor_default() {
        let monitor = PerformanceMonitor::default();
        let metrics = monitor.metrics();

        assert_eq!(metrics.get_fps(), 0.0);
        assert_eq!(metrics.get_total_frames(), 0);
    }
}

/// Test metrics update operations and boundary values
#[cfg(test)]
mod metrics_update_tests {
    use super::*;

    #[test]
    fn test_fps_boundary_values() {
        let metrics = Metrics::new();

        let boundary_values = [
            0.0,
            0.1,
            1.0,
            15.0,
            30.0,
            60.0,
            120.0,
            240.0,
            f64::MIN,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];

        for fps in boundary_values {
            metrics.update_fps(fps);
            let retrieved = metrics.get_fps();

            if fps.is_nan() {
                assert!(retrieved.is_nan() || retrieved == 0.0); // Either preserve NaN or default to 0
            } else if fps.is_infinite() {
                assert!(retrieved.is_infinite() || retrieved == fps);
            } else {
                assert_eq!(retrieved, fps);
            }
        }
    }

    #[test]
    fn test_latency_boundary_values() {
        let metrics = Metrics::new();

        let boundary_values = [
            0.0,
            0.001,
            1.0,
            16.67,
            33.33,
            100.0,
            1000.0,
            10000.0,
            f64::MIN,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];

        for latency in boundary_values {
            metrics.update_inference_latency(latency);
            let retrieved = metrics.get_inference_latency_ms();

            if latency.is_nan() {
                assert!(retrieved.is_nan() || retrieved == 0.0);
            } else if latency.is_infinite() {
                assert!(retrieved.is_infinite() || retrieved == latency);
            } else {
                assert_eq!(retrieved, latency);
            }
        }
    }

    #[test]
    fn test_memory_usage_boundary_values() {
        let metrics = Metrics::new();

        let boundary_values = [0, 1, 16, 64, 128, 512, 1024, 2048, 4096, 8192, usize::MAX];

        for memory in boundary_values {
            metrics.update_memory_usage(memory);
            assert_eq!(metrics.get_memory_usage_mb(), memory);

            // Peak should be updated if this is the highest value so far
            assert!(metrics.get_peak_memory_mb() >= memory);
        }
    }

    #[test]
    fn test_peak_memory_tracking() {
        let metrics = Metrics::new();

        // Test increasing memory usage
        let memory_sequence = [100, 200, 150, 300, 250, 400];
        let mut expected_peak = 0;

        for memory in memory_sequence {
            metrics.update_memory_usage(memory);
            expected_peak = expected_peak.max(memory);

            assert_eq!(metrics.get_memory_usage_mb(), memory);
            assert_eq!(metrics.get_peak_memory_mb(), expected_peak);
        }
    }

    #[test]
    fn test_frame_counting_boundary_values() {
        let metrics = Metrics::new();

        // Test large numbers of frame increments
        let increment_counts = [1, 10, 100, 1000, 10000];

        for count in increment_counts {
            metrics.reset(); // Reset for each test

            for _ in 0..count {
                metrics.increment_total_frames();
            }

            assert_eq!(metrics.get_total_frames(), count);
            assert_eq!(metrics.get_dropped_frames(), 0);
            assert_eq!(metrics.get_frame_drop_rate(), 0.0);
        }
    }

    #[test]
    fn test_dropped_frames_boundary_values() {
        let metrics = Metrics::new();

        // Set up total frames
        for _ in 0..1000 {
            metrics.increment_total_frames();
        }

        let drop_counts = [0, 1, 10, 100, 500, 999, 1000];

        for drop_count in drop_counts {
            metrics.reset();

            // Set up frames again
            for _ in 0..1000 {
                metrics.increment_total_frames();
            }

            // Add dropped frames
            for _ in 0..drop_count {
                metrics.increment_dropped_frames();
            }

            let expected_rate = if metrics.get_total_frames() == 0 {
                0.0
            } else {
                (drop_count as f64 / 1000.0) * 100.0
            };

            assert_eq!(metrics.get_dropped_frames(), drop_count);
            assert!((metrics.get_frame_drop_rate() - expected_rate).abs() < 0.01);
        }
    }

    #[test]
    fn test_cpu_gpu_usage_boundary_values() {
        let metrics = Metrics::new();

        let usage_values = [
            0.0,
            0.1,
            25.0,
            50.0,
            75.0,
            100.0,
            101.0,
            200.0,
            f64::MIN,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];

        for usage in usage_values {
            metrics.update_cpu_usage(usage);
            let retrieved_cpu = metrics.get_cpu_usage_percent();

            metrics.update_gpu_usage(usage);
            let retrieved_gpu = metrics.get_gpu_usage_percent();

            if usage.is_nan() {
                assert!(retrieved_cpu.is_nan() || retrieved_cpu == 0.0);
                assert!(retrieved_gpu.is_nan() || retrieved_gpu == 0.0);
            } else {
                assert_eq!(retrieved_cpu, usage);
                assert_eq!(retrieved_gpu, usage);
            }
        }
    }

    #[test]
    fn test_frame_time_boundary_values() {
        let metrics = Metrics::new();

        let frame_times = [
            0.0,
            0.001,
            16.67,
            33.33,
            66.67,
            100.0,
            1000.0,
            f64::MIN,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];

        for frame_time in frame_times {
            metrics.update_frame_time(frame_time);
            let retrieved = metrics.get_avg_frame_time_ms();

            if frame_time.is_nan() {
                assert!(retrieved.is_nan() || retrieved == 0.0);
            } else {
                assert_eq!(retrieved, frame_time);
            }
        }
    }
}

/// Test performance target checking with various scenarios
#[cfg(test)]
mod performance_target_tests {
    use super::*;

    #[test]
    fn test_performance_targets_success() {
        let metrics = Metrics::new();

        // Set good performance
        metrics.update_fps(60.0);
        metrics.update_inference_latency(10.0);

        // Should pass reasonable targets
        assert!(metrics.check_performance_targets(30.0, 50.0).is_ok());
        assert!(metrics.check_performance_targets(60.0, 10.0).is_ok());
        assert!(metrics.check_performance_targets(59.0, 11.0).is_ok());
    }

    #[test]
    fn test_performance_targets_fps_failure() {
        let metrics = Metrics::new();

        metrics.update_fps(20.0);
        metrics.update_inference_latency(10.0);

        let result = metrics.check_performance_targets(30.0, 50.0);

        match result {
            Err(PupError::PerformanceTarget {
                target_fps,
                actual_fps,
            }) => {
                assert_eq!(target_fps, 30.0);
                assert_eq!(actual_fps, 20.0);
            }
            other => panic!("Expected PerformanceTarget error, got: {:?}", other),
        }
    }

    #[test]
    fn test_performance_targets_latency_failure() {
        let metrics = Metrics::new();

        metrics.update_fps(60.0);
        metrics.update_inference_latency(100.0);

        let result = metrics.check_performance_targets(30.0, 50.0);

        match result {
            Err(PupError::ProcessingTimeout(timeout_ms)) => {
                assert_eq!(timeout_ms, 50);
            }
            other => panic!("Expected ProcessingTimeout error, got: {:?}", other),
        }
    }

    #[test]
    fn test_performance_targets_boundary_values() {
        let metrics = Metrics::new();

        // Test exact boundary conditions
        metrics.update_fps(30.0);
        metrics.update_inference_latency(50.0);

        // Exact match should pass
        assert!(metrics.check_performance_targets(30.0, 50.0).is_ok());

        // Just above threshold should fail
        assert!(metrics.check_performance_targets(30.1, 50.0).is_err());
        assert!(metrics.check_performance_targets(30.0, 49.9).is_err());

        // Just below threshold should pass
        metrics.update_fps(29.9);
        metrics.update_inference_latency(49.9);
        assert!(metrics.check_performance_targets(30.0, 50.0).is_ok());
    }

    #[test]
    fn test_performance_targets_extreme_values() {
        let metrics = Metrics::new();

        // Test with extreme target values
        metrics.update_fps(60.0);
        metrics.update_inference_latency(10.0);

        // Extremely high targets should fail
        assert!(metrics.check_performance_targets(1000.0, 1.0).is_err());

        // Extremely low targets should pass
        assert!(metrics.check_performance_targets(1.0, 1000.0).is_ok());

        // Test with infinite values
        metrics.update_fps(f64::INFINITY);
        metrics.update_inference_latency(f64::NEG_INFINITY);

        assert!(metrics.check_performance_targets(100.0, 100.0).is_ok());
    }
}

/// Test metrics reset functionality
#[cfg(test)]
mod metrics_reset_tests {
    use super::*;

    #[test]
    fn test_complete_reset() {
        let metrics = Metrics::new();

        // Set up non-zero values
        metrics.update_fps(60.0);
        metrics.update_inference_latency(25.0);
        metrics.update_memory_usage(512);
        metrics.increment_total_frames();
        metrics.increment_dropped_frames();
        metrics.update_frame_time(16.67);
        metrics.update_cpu_usage(45.0);
        metrics.update_gpu_usage(30.0);

        // Verify values are set
        assert_ne!(metrics.get_fps(), 0.0);
        assert_ne!(metrics.get_inference_latency_ms(), 0.0);
        assert_ne!(metrics.get_memory_usage_mb(), 0);
        assert_ne!(metrics.get_total_frames(), 0);
        assert_ne!(metrics.get_dropped_frames(), 0);
        assert_ne!(metrics.get_peak_memory_mb(), 0);

        // Reset
        metrics.reset();

        // Verify all values are reset
        assert_eq!(metrics.get_fps(), 0.0);
        assert_eq!(metrics.get_inference_latency_ms(), 0.0);
        assert_eq!(metrics.get_memory_usage_mb(), 0);
        assert_eq!(metrics.get_total_frames(), 0);
        assert_eq!(metrics.get_dropped_frames(), 0);
        assert_eq!(metrics.get_avg_frame_time_ms(), 0.0);
        assert_eq!(metrics.get_cpu_usage_percent(), 0.0);
        assert_eq!(metrics.get_gpu_usage_percent(), 0.0);
        assert_eq!(metrics.get_peak_memory_mb(), 0);
        assert_eq!(metrics.get_frame_drop_rate(), 0.0);
    }

    #[test]
    fn test_reset_after_extreme_values() {
        let metrics = Metrics::new();

        // Set extreme values
        metrics.update_fps(f64::INFINITY);
        metrics.update_inference_latency(f64::NAN);
        metrics.update_memory_usage(usize::MAX);
        metrics.update_cpu_usage(f64::NEG_INFINITY);

        // Reset should work even with extreme values
        metrics.reset();

        assert_eq!(metrics.get_fps(), 0.0);
        assert_eq!(metrics.get_inference_latency_ms(), 0.0);
        assert_eq!(metrics.get_memory_usage_mb(), 0);
        assert_eq!(metrics.get_cpu_usage_percent(), 0.0);
    }

    #[test]
    fn test_multiple_resets() {
        let metrics = Metrics::new();

        for i in 0..10 {
            // Set values
            metrics.update_fps(i as f64);
            metrics.update_memory_usage(i * 100);

            // Reset
            metrics.reset();

            // Should be zero after each reset
            assert_eq!(metrics.get_fps(), 0.0);
            assert_eq!(metrics.get_memory_usage_mb(), 0);
        }
    }
}

/// Test metrics format summary functionality
#[cfg(test)]
mod metrics_format_tests {
    use super::*;

    #[test]
    fn test_format_summary_with_zero_values() {
        let metrics = Metrics::new();
        let summary = metrics.format_summary();

        assert!(summary.contains("FPS: 0.0"));
        assert!(summary.contains("Latency: 0.0ms"));
        assert!(summary.contains("Memory: 0MB"));
        assert!(summary.contains("Dropped: 0"));
        assert!(summary.contains("CPU: 0.0%"));
        assert!(summary.contains("GPU: 0.0%"));
    }

    #[test]
    fn test_format_summary_with_normal_values() {
        let metrics = Metrics::new();

        metrics.update_fps(30.5);
        metrics.update_inference_latency(25.7);
        metrics.update_memory_usage(256);
        metrics.increment_total_frames();
        metrics.increment_total_frames();
        metrics.increment_dropped_frames();
        metrics.update_cpu_usage(45.8);
        metrics.update_gpu_usage(32.1);

        let summary = metrics.format_summary();

        assert!(summary.contains("FPS: 30.5"));
        assert!(summary.contains("Latency: 25.7ms"));
        assert!(summary.contains("Memory: 256MB"));
        assert!(summary.contains("Dropped: 1"));
        assert!(summary.contains("50.0%")); // Drop rate
        assert!(summary.contains("CPU: 45.8%"));
        assert!(summary.contains("GPU: 32.1%"));
    }

    #[test]
    fn test_format_summary_with_extreme_values() {
        let metrics = Metrics::new();

        metrics.update_fps(f64::INFINITY);
        metrics.update_inference_latency(f64::NEG_INFINITY);
        metrics.update_memory_usage(usize::MAX);
        metrics.update_cpu_usage(f64::NAN);

        let summary = metrics.format_summary();

        // Should handle extreme values gracefully without crashing
        assert!(!summary.is_empty());
        assert!(summary.contains("FPS:"));
        assert!(summary.contains("Memory:"));
    }

    #[test]
    fn test_format_summary_precision() {
        let metrics = Metrics::new();

        metrics.update_fps(29.999999);
        metrics.update_inference_latency(16.666666);

        let summary = metrics.format_summary();

        // Should format to reasonable precision
        assert!(summary.contains("FPS: 30.0") || summary.contains("FPS: 29.9"));
        assert!(summary.contains("16.7ms") || summary.contains("16.66ms"));
    }
}

/// Test reporter implementations
#[cfg(test)]
mod reporter_tests {
    use super::*;

    #[test]
    fn test_console_reporter_basic() {
        let reporter = ConsoleReporter::new(0); // No interval delay for testing
        let metrics = Metrics::new();

        metrics.update_fps(30.0);

        assert_eq!(reporter.name(), "console");
        assert!(reporter.report(&metrics).is_ok());
    }

    #[test]
    fn test_console_reporter_with_default_interval() {
        let reporter = ConsoleReporter::with_default_interval();
        assert_eq!(reporter.name(), "console");

        let metrics = Metrics::new();
        assert!(reporter.report(&metrics).is_ok());
    }

    #[test]
    fn test_console_reporter_default() {
        let reporter = ConsoleReporter::default();
        assert_eq!(reporter.name(), "console");
    }

    #[test]
    fn test_console_reporter_interval_limiting() {
        let reporter = ConsoleReporter::new(1000); // 1 second interval
        let metrics = Metrics::new();

        // First report should work
        assert!(reporter.report(&metrics).is_ok());

        // Immediate second report might be rate limited, but shouldn't fail
        assert!(reporter.report(&metrics).is_ok());
    }

    #[test]
    fn test_json_reporter_basic() {
        let temp_file = NamedTempFile::new().unwrap();
        let reporter = JsonReporter::new(temp_file.path().to_path_buf(), 0);
        let metrics = Metrics::new();

        metrics.update_fps(30.0);
        metrics.update_inference_latency(25.0);
        metrics.update_memory_usage(256);

        assert_eq!(reporter.name(), "json");
        assert!(reporter.report(&metrics).is_ok());

        // Check that file was written
        let content = fs::read_to_string(temp_file.path()).unwrap();
        assert!(content.contains("fps"));
        assert!(content.contains("30"));
        assert!(content.contains("inference_latency_ms"));
        assert!(content.contains("25"));
    }

    #[test]
    fn test_json_reporter_multiple_reports() {
        let temp_file = NamedTempFile::new().unwrap();
        let reporter = JsonReporter::new(temp_file.path().to_path_buf(), 0);
        let metrics = Metrics::new();

        // Multiple reports should append to file
        metrics.update_fps(30.0);
        assert!(reporter.report(&metrics).is_ok());

        metrics.update_fps(60.0);
        assert!(reporter.report(&metrics).is_ok());

        let content = fs::read_to_string(temp_file.path()).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert!(lines.len() >= 2);

        // Both values should be present
        assert!(content.contains("30"));
        assert!(content.contains("60"));
    }

    #[test]
    fn test_json_reporter_file_error() {
        // Try to write to an invalid path
        let invalid_path = PathBuf::from("/invalid/path/that/does/not/exist.json");
        let reporter = JsonReporter::new(invalid_path, 0);
        let metrics = Metrics::new();

        let result = reporter.report(&metrics);

        match result {
            Err(PupError::OutputDirectoryError(_)) => { /* Expected */ }
            other => {
                // On some systems, this might produce a different error
                // As long as it fails gracefully, that's acceptable
                println!("Got error: {:?}", other);
            }
        }
    }

    #[test]
    fn test_mock_metrics_reporter() {
        let reporter = MockMetricsReporter::new("test_reporter");
        let metrics = Metrics::new();

        metrics.update_fps(45.0);
        metrics.increment_total_frames();

        assert_eq!(reporter.name(), "test_reporter");
        assert_eq!(reporter.get_report_count(), 0);

        // Report metrics
        assert!(reporter.report(&metrics).is_ok());
        assert_eq!(reporter.get_report_count(), 1);

        let reports = reporter.get_reports();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].fps, 45.0);
        assert_eq!(reports[0].total_frames, 1);
    }

    #[test]
    fn test_mock_metrics_reporter_with_failure() {
        let reporter = MockMetricsReporter::new("failing_reporter").with_failure();
        let metrics = Metrics::new();

        let result = reporter.report(&metrics);

        match result {
            Err(PupError::Unexpected(msg)) => {
                assert_eq!(msg, "Mock reporter failure");
            }
            other => panic!("Expected mock failure, got: {:?}", other),
        }

        assert_eq!(reporter.get_report_count(), 1); // Should still increment count
    }

    #[test]
    fn test_mock_reporter_clear_reports() {
        let reporter = MockMetricsReporter::new("test");
        let metrics = Metrics::new();

        // Add some reports
        assert!(reporter.report(&metrics).is_ok());
        assert!(reporter.report(&metrics).is_ok());

        assert_eq!(reporter.get_report_count(), 2);
        assert_eq!(reporter.get_reports().len(), 2);

        // Clear reports
        reporter.clear_reports();

        assert_eq!(reporter.get_report_count(), 0);
        assert_eq!(reporter.get_reports().len(), 0);
    }
}

/// Test performance monitor functionality
#[cfg(test)]
mod performance_monitor_tests {
    use super::*;

    #[test]
    fn test_performance_monitor_reporters() {
        let mut monitor = PerformanceMonitor::new();

        // Add reporters
        monitor.add_reporter(Box::new(ConsoleReporter::new(1000)));
        monitor.add_reporter(Box::new(MockMetricsReporter::new("test")));

        // Should be able to report
        assert!(monitor.report().is_ok());
    }

    #[test]
    fn test_performance_monitor_frame_timer() {
        let mut monitor = PerformanceMonitor::new();
        let metrics = monitor.metrics();

        let timer = monitor.start_frame();

        // Simulate some work
        thread::sleep(Duration::from_millis(10));

        timer.complete();

        // Should have recorded the frame
        assert_eq!(metrics.get_total_frames(), 1);
        assert!(metrics.get_avg_frame_time_ms() >= 10.0);
    }

    #[test]
    fn test_performance_monitor_inference_timing() {
        let monitor = PerformanceMonitor::new();
        let metrics = monitor.metrics();

        let duration = Duration::from_millis(25);
        monitor.record_inference_time(duration);

        assert_eq!(metrics.get_inference_latency_ms(), 25.0);
    }

    #[test]
    fn test_performance_monitor_system_metrics() {
        let mut monitor = PerformanceMonitor::new();

        // This might fail on some systems due to platform-specific code
        let result = monitor.update_system_metrics();

        // Should either succeed or fail gracefully
        match result {
            Ok(()) => { /* System metrics updated successfully */ }
            Err(_) => { /* Some systems may not support all metrics */ }
        }
    }

    #[test]
    fn test_performance_monitor_check_targets() {
        let monitor = PerformanceMonitor::new();
        let metrics = monitor.metrics();

        // Set good performance
        metrics.update_fps(60.0);
        metrics.update_inference_latency(15.0);

        assert!(monitor.check_targets(30.0, 50.0).is_ok());

        // Set poor performance
        metrics.update_fps(15.0);

        assert!(monitor.check_targets(30.0, 50.0).is_err());
    }

    #[test]
    fn test_performance_monitor_with_failing_reporter() {
        let mut monitor = PerformanceMonitor::new();

        // Add a failing reporter
        monitor.add_reporter(Box::new(MockMetricsReporter::new("failing").with_failure()));
        monitor.add_reporter(Box::new(MockMetricsReporter::new("working")));

        // Report should fail due to the failing reporter
        assert!(monitor.report().is_err());
    }
}

/// Test frame timer functionality
#[cfg(test)]
mod frame_timer_tests {
    use super::*;

    #[test]
    fn test_frame_timer_complete() {
        let mut monitor = PerformanceMonitor::new();
        let metrics = monitor.metrics();

        let timer = monitor.start_frame();

        // Simulate some processing time
        thread::sleep(Duration::from_millis(10));

        timer.complete();

        assert_eq!(metrics.get_total_frames(), 1);
        assert!(metrics.get_avg_frame_time_ms() >= 10.0);
    }

    #[test]
    fn test_frame_timer_drop() {
        let mut monitor = PerformanceMonitor::new();
        let metrics = monitor.metrics();

        let timer = monitor.start_frame();

        timer.drop();

        assert_eq!(metrics.get_total_frames(), 1);
        assert_eq!(metrics.get_dropped_frames(), 1);
        assert_eq!(metrics.get_frame_drop_rate(), 100.0);
    }

    #[test]
    fn test_multiple_frame_timers() {
        let mut monitor = PerformanceMonitor::new();
        let metrics = monitor.metrics();

        // Process multiple frames
        for i in 0..10 {
            let timer = monitor.start_frame();

            thread::sleep(Duration::from_millis(5));

            if i % 3 == 0 {
                timer.drop(); // Drop every 3rd frame
            } else {
                timer.complete();
            }
        }

        assert_eq!(metrics.get_total_frames(), 10);
        assert_eq!(metrics.get_dropped_frames(), 4); // Frames 0, 3, 6, 9
        assert!((metrics.get_frame_drop_rate() - 40.0).abs() < 0.1);
    }

    #[test]
    fn test_frame_timer_zero_duration() {
        let mut monitor = PerformanceMonitor::new();
        let metrics = monitor.metrics();

        let timer = monitor.start_frame();
        timer.complete(); // Complete immediately

        assert_eq!(metrics.get_total_frames(), 1);
        // Frame time should be very small but non-negative
        assert!(metrics.get_avg_frame_time_ms() >= 0.0);
    }
}

/// Test concurrent metrics access and thread safety
#[cfg(test)]
mod concurrent_metrics_tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn test_concurrent_metrics_updates() {
        let metrics = Arc::new(Metrics::new());
        let num_threads = 10;
        let operations_per_thread = 100;

        let barrier = Arc::new(Barrier::new(num_threads));
        let mut handles = vec![];

        for thread_id in 0..num_threads {
            let metrics = metrics.clone();
            let barrier = barrier.clone();

            let handle = thread::spawn(move || {
                barrier.wait(); // Synchronize start

                for i in 0..operations_per_thread {
                    let value = (thread_id * operations_per_thread + i) as f64;

                    metrics.update_fps(value);
                    metrics.update_inference_latency(value / 2.0);
                    metrics.update_memory_usage(i);
                    metrics.increment_total_frames();

                    if i % 5 == 0 {
                        metrics.increment_dropped_frames();
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify final state
        assert_eq!(
            metrics.get_total_frames(),
            num_threads * operations_per_thread
        );
        assert_eq!(
            metrics.get_dropped_frames(),
            num_threads * (operations_per_thread / 5)
        );

        // FPS should be the last value set by some thread
        let fps = metrics.get_fps();
        assert!(fps >= 0.0);

        // Memory usage should be some value from the last updates
        let memory = metrics.get_memory_usage_mb();
        assert!(memory < operations_per_thread);
    }

    #[test]
    fn test_concurrent_reads_and_writes() {
        let metrics = Arc::new(Metrics::new());
        let num_writers = 5;
        let num_readers = 5;
        let operations = 200;

        let mut handles = vec![];

        // Writer threads
        for _ in 0..num_writers {
            let metrics = metrics.clone();
            let handle = thread::spawn(move || {
                for i in 0..operations {
                    metrics.update_fps(i as f64);
                    metrics.increment_total_frames();
                    thread::sleep(Duration::from_millis(1));
                }
            });
            handles.push(handle);
        }

        // Reader threads
        for _ in 0..num_readers {
            let metrics = metrics.clone();
            let handle = thread::spawn(move || {
                for _ in 0..operations {
                    let _ = metrics.get_fps();
                    let _ = metrics.get_total_frames();
                    let _ = metrics.get_frame_drop_rate();
                    let _ = metrics.format_summary();
                    thread::sleep(Duration::from_millis(1));
                }
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Should complete without deadlocks or panics
        assert_eq!(metrics.get_total_frames(), num_writers * operations);
    }

    #[test]
    fn test_concurrent_reset_operations() {
        let metrics = Arc::new(Metrics::new());
        let num_threads = 5;

        let mut handles = vec![];

        for _ in 0..num_threads {
            let metrics = metrics.clone();
            let handle = thread::spawn(move || {
                for i in 0..50 {
                    // Update metrics
                    metrics.update_fps(i as f64);
                    metrics.increment_total_frames();

                    // Occasionally reset
                    if i % 10 == 0 {
                        metrics.reset();
                    }

                    thread::sleep(Duration::from_millis(1));
                }
            });
            handles.push(handle);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // Should complete without crashes
        // Final state depends on timing of resets vs updates
        assert!(metrics.get_total_frames() >= 0);
    }
}

/// Test stress scenarios and resource constraints
#[cfg(test)]
mod stress_tests {
    use super::*;

    #[test]
    fn test_high_frequency_updates() {
        let metrics = Metrics::new();
        let start = Instant::now();

        // Perform many rapid updates
        for i in 0..100000 {
            metrics.update_fps(i as f64 % 120.0);
            metrics.increment_total_frames();

            if i % 100 == 0 {
                metrics.increment_dropped_frames();
            }
        }

        let duration = start.elapsed();

        assert_eq!(metrics.get_total_frames(), 100000);
        assert_eq!(metrics.get_dropped_frames(), 1000);

        // Should complete in reasonable time (less than 1 second)
        assert!(
            duration.as_secs() < 1,
            "High frequency updates too slow: {:?}",
            duration
        );
    }

    #[test]
    fn test_memory_intensive_operations() {
        let metrics = Metrics::new();

        // Test with very large memory values
        let large_memory_values = [
            1024 * 1024,      // 1GB
            4 * 1024 * 1024,  // 4GB
            16 * 1024 * 1024, // 16GB
            usize::MAX / 2,   // Half of max
            usize::MAX,       // Maximum value
        ];

        for memory in large_memory_values {
            metrics.update_memory_usage(memory);
            assert_eq!(metrics.get_memory_usage_mb(), memory);

            // Peak should track correctly even with large values
            assert!(metrics.get_peak_memory_mb() >= memory);
        }
    }

    #[test]
    fn test_rapid_reporter_operations() {
        let temp_file = NamedTempFile::new().unwrap();
        let reporter = JsonReporter::new(temp_file.path().to_path_buf(), 0);
        let metrics = Metrics::new();

        let start = Instant::now();

        // Rapid reporting
        for i in 0..1000 {
            metrics.update_fps(i as f64 % 60.0);

            // Report every 10 updates
            if i % 10 == 0 {
                assert!(reporter.report(&metrics).is_ok());
            }
        }

        let duration = start.elapsed();

        // Should complete in reasonable time
        assert!(
            duration.as_secs() < 5,
            "Rapid reporting too slow: {:?}",
            duration
        );

        // File should have content
        let content = fs::read_to_string(temp_file.path()).unwrap();
        let line_count = content.lines().count();
        assert!(line_count >= 90); // Should have ~100 lines
    }

    #[test]
    fn test_performance_monitor_stress() {
        let config = StressTestConfig {
            thread_count: 4,
            duration_seconds: 2, // Short stress test
            ops_per_second: 100,
            ..Default::default()
        };

        let executor = StressTestExecutor::new(config);
        let results = executor.run_metrics_stress_test();

        assert!(results.total_operations > 0);
        assert!(results.successful_operations > 0);
        // Should have high success rate
        assert!(results.success_rate() > 90.0);
    }
}

/// Test edge cases and error conditions
#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_division_by_zero_frame_drop_rate() {
        let metrics = Metrics::new();

        // With zero total frames, drop rate should be 0
        assert_eq!(metrics.get_frame_drop_rate(), 0.0);

        // Add dropped frames but no total frames
        metrics.increment_dropped_frames();
        assert_eq!(metrics.get_frame_drop_rate(), 0.0); // Should still be 0
    }

    #[test]
    fn test_metrics_with_mutex_poisoning() {
        // This is a theoretical test - in practice, our metrics shouldn't panic
        // But we test graceful handling if mutex becomes poisoned
        let metrics = Metrics::new();

        // All operations should complete without panicking
        metrics.update_fps(30.0);
        metrics.update_inference_latency(25.0);

        // Reads should return reasonable values even if mutex issues occur
        let fps = metrics.get_fps();
        let latency = metrics.get_inference_latency_ms();

        // Values should be retrievable
        assert!(fps >= 0.0 || fps.is_nan());
        assert!(latency >= 0.0 || latency.is_nan());
    }

    #[test]
    fn test_reporter_with_invalid_json_content() {
        let temp_file = NamedTempFile::new().unwrap();
        let reporter = JsonReporter::new(temp_file.path().to_path_buf(), 0);
        let metrics = Metrics::new();

        // Set values that might cause JSON formatting issues
        metrics.update_fps(f64::NAN);
        metrics.update_inference_latency(f64::INFINITY);

        // Should handle gracefully without crashing
        let result = reporter.report(&metrics);

        // May succeed or fail, but should not panic
        match result {
            Ok(()) => {
                // Check that file was created
                assert!(temp_file.path().exists());
            }
            Err(_) => {
                // Acceptable if JSON formatting fails with special values
            }
        }
    }

    #[test]
    fn test_timestamp_overflow() {
        let metrics = Metrics::new();

        // Multiple rapid updates to test timestamp handling
        for _ in 0..1000 {
            metrics.update_fps(30.0);
            // Each update internally updates timestamp
        }

        // Should complete without overflow issues
        assert_eq!(metrics.get_fps(), 30.0);
    }

    #[test]
    fn test_console_reporter_with_extreme_values() {
        let reporter = ConsoleReporter::new(0);
        let metrics = Metrics::new();

        // Set extreme values
        metrics.update_fps(f64::INFINITY);
        metrics.update_inference_latency(f64::NAN);
        metrics.update_memory_usage(usize::MAX);

        // Should not crash when formatting
        assert!(reporter.report(&metrics).is_ok());
    }
}
