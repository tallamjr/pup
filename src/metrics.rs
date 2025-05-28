//! Performance monitoring and metrics collection for the Pup video processing system
//!
//! This module provides comprehensive performance monitoring capabilities including
//! FPS tracking, latency measurement, memory usage monitoring, and frame drop detection.

use crate::error::{PupError, PupResult};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Core performance metrics
#[derive(Debug)]
pub struct Metrics {
    /// Current frames per second
    pub fps: Mutex<f64>,
    /// Inference latency in milliseconds
    pub inference_latency_ms: Mutex<f64>,
    /// Memory usage in megabytes
    pub memory_usage_mb: AtomicUsize,
    /// Number of dropped frames
    pub dropped_frames: AtomicUsize,
    /// Total frames processed
    pub total_frames: AtomicUsize,
    /// Average processing time per frame in milliseconds
    pub avg_frame_time_ms: Mutex<f64>,
    /// Peak memory usage in megabytes
    pub peak_memory_mb: AtomicUsize,
    /// CPU usage percentage (0-100)
    pub cpu_usage_percent: Mutex<f64>,
    /// GPU utilisation percentage (0-100, if available)
    pub gpu_usage_percent: Mutex<f64>,
    /// Timestamp of last update
    pub last_update: AtomicU64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            fps: Mutex::new(0.0),
            inference_latency_ms: Mutex::new(0.0),
            memory_usage_mb: AtomicUsize::new(0),
            dropped_frames: AtomicUsize::new(0),
            total_frames: AtomicUsize::new(0),
            avg_frame_time_ms: Mutex::new(0.0),
            peak_memory_mb: AtomicUsize::new(0),
            cpu_usage_percent: Mutex::new(0.0),
            gpu_usage_percent: Mutex::new(0.0),
            last_update: AtomicU64::new(0),
        }
    }
}

impl Metrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        Self::default()
    }

    /// Update FPS measurement
    pub fn update_fps(&self, fps: f64) {
        if let Ok(mut fps_guard) = self.fps.lock() {
            *fps_guard = fps;
        }
        self.update_timestamp();
    }

    /// Update inference latency
    pub fn update_inference_latency(&self, latency_ms: f64) {
        if let Ok(mut latency_guard) = self.inference_latency_ms.lock() {
            *latency_guard = latency_ms;
        }
        self.update_timestamp();
    }

    /// Update memory usage and track peak
    pub fn update_memory_usage(&self, usage_mb: usize) {
        self.memory_usage_mb.store(usage_mb, Ordering::Relaxed);
        
        // Update peak memory if current usage is higher
        let current_peak = self.peak_memory_mb.load(Ordering::Relaxed);
        if usage_mb > current_peak {
            self.peak_memory_mb.store(usage_mb, Ordering::Relaxed);
        }
        
        self.update_timestamp();
    }

    /// Increment dropped frames counter
    pub fn increment_dropped_frames(&self) {
        self.dropped_frames.fetch_add(1, Ordering::Relaxed);
        self.update_timestamp();
    }

    /// Increment total frames counter
    pub fn increment_total_frames(&self) {
        self.total_frames.fetch_add(1, Ordering::Relaxed);
        self.update_timestamp();
    }

    /// Update average frame processing time
    pub fn update_frame_time(&self, time_ms: f64) {
        if let Ok(mut time_guard) = self.avg_frame_time_ms.lock() {
            *time_guard = time_ms;
        }
        self.update_timestamp();
    }

    /// Update CPU usage
    pub fn update_cpu_usage(&self, usage_percent: f64) {
        if let Ok(mut cpu_guard) = self.cpu_usage_percent.lock() {
            *cpu_guard = usage_percent;
        }
        self.update_timestamp();
    }

    /// Update GPU utilisation
    pub fn update_gpu_usage(&self, usage_percent: f64) {
        if let Ok(mut gpu_guard) = self.gpu_usage_percent.lock() {
            *gpu_guard = usage_percent;
        }
        self.update_timestamp();
    }

    /// Get current FPS
    pub fn get_fps(&self) -> f64 {
        self.fps.lock().map(|guard| *guard).unwrap_or(0.0)
    }

    /// Get current inference latency
    pub fn get_inference_latency_ms(&self) -> f64 {
        self.inference_latency_ms.lock().map(|guard| *guard).unwrap_or(0.0)
    }

    /// Get current memory usage
    pub fn get_memory_usage_mb(&self) -> usize {
        self.memory_usage_mb.load(Ordering::Relaxed)
    }

    /// Get dropped frames count
    pub fn get_dropped_frames(&self) -> usize {
        self.dropped_frames.load(Ordering::Relaxed)
    }

    /// Get total frames processed
    pub fn get_total_frames(&self) -> usize {
        self.total_frames.load(Ordering::Relaxed)
    }

    /// Get average frame processing time
    pub fn get_avg_frame_time_ms(&self) -> f64 {
        self.avg_frame_time_ms.lock().map(|guard| *guard).unwrap_or(0.0)
    }

    /// Get peak memory usage
    pub fn get_peak_memory_mb(&self) -> usize {
        self.peak_memory_mb.load(Ordering::Relaxed)
    }

    /// Get CPU usage
    pub fn get_cpu_usage_percent(&self) -> f64 {
        self.cpu_usage_percent.lock().map(|guard| *guard).unwrap_or(0.0)
    }

    /// Get GPU utilisation
    pub fn get_gpu_usage_percent(&self) -> f64 {
        self.gpu_usage_percent.lock().map(|guard| *guard).unwrap_or(0.0)
    }

    /// Calculate frame drop rate as percentage
    pub fn get_frame_drop_rate(&self) -> f64 {
        let total = self.get_total_frames();
        let dropped = self.get_dropped_frames();
        
        if total == 0 {
            0.0
        } else {
            (dropped as f64 / total as f64) * 100.0
        }
    }

    /// Check if performance targets are being met
    pub fn check_performance_targets(&self, target_fps: f64, max_latency_ms: f64) -> PupResult<()> {
        let current_fps = self.get_fps();
        let current_latency = self.get_inference_latency_ms();

        if current_fps < target_fps {
            return Err(PupError::PerformanceTarget {
                target_fps,
                actual_fps: current_fps,
            });
        }

        if current_latency > max_latency_ms {
            return Err(PupError::ProcessingTimeout {
                timeout_ms: max_latency_ms as u64,
            });
        }

        Ok(())
    }

    /// Reset all metrics
    pub fn reset(&self) {
        if let Ok(mut fps_guard) = self.fps.lock() {
            *fps_guard = 0.0;
        }
        if let Ok(mut latency_guard) = self.inference_latency_ms.lock() {
            *latency_guard = 0.0;
        }
        if let Ok(mut frame_time_guard) = self.avg_frame_time_ms.lock() {
            *frame_time_guard = 0.0;
        }
        if let Ok(mut cpu_guard) = self.cpu_usage_percent.lock() {
            *cpu_guard = 0.0;
        }
        if let Ok(mut gpu_guard) = self.gpu_usage_percent.lock() {
            *gpu_guard = 0.0;
        }
        
        self.memory_usage_mb.store(0, Ordering::Relaxed);
        self.dropped_frames.store(0, Ordering::Relaxed);
        self.total_frames.store(0, Ordering::Relaxed);
        self.peak_memory_mb.store(0, Ordering::Relaxed);
        self.update_timestamp();
    }

    /// Update the last update timestamp
    fn update_timestamp(&self) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.last_update.store(now, Ordering::Relaxed);
    }

    /// Get formatted metrics summary
    pub fn format_summary(&self) -> String {
        format!(
            "FPS: {:.1} | Latency: {:.1}ms | Memory: {}MB | Dropped: {} ({:.1}%) | CPU: {:.1}% | GPU: {:.1}%",
            self.get_fps(),
            self.get_inference_latency_ms(),
            self.get_memory_usage_mb(),
            self.get_dropped_frames(),
            self.get_frame_drop_rate(),
            self.get_cpu_usage_percent(),
            self.get_gpu_usage_percent()
        )
    }
}

/// Metrics reporter trait for different output formats
pub trait MetricsReporter: Send + Sync {
    /// Report current metrics
    fn report(&self, metrics: &Metrics) -> PupResult<()>;
    
    /// Get reporter name
    fn name(&self) -> &str;
}

/// Console metrics reporter
pub struct ConsoleReporter {
    interval_ms: u64,
    last_report: std::sync::Mutex<Instant>,
}

impl ConsoleReporter {
    /// Create new console reporter with specified interval
    pub fn new(interval_ms: u64) -> Self {
        Self {
            interval_ms,
            last_report: std::sync::Mutex::new(Instant::now()),
        }
    }

    /// Create reporter with 1-second interval
    pub fn default() -> Self {
        Self::new(1000)
    }
}

impl MetricsReporter for ConsoleReporter {
    fn report(&self, metrics: &Metrics) -> PupResult<()> {
        let mut last_report = self.last_report.lock().unwrap();
        let now = Instant::now();
        
        if now.duration_since(*last_report).as_millis() >= self.interval_ms as u128 {
            println!("[METRICS] {}", metrics.format_summary());
            *last_report = now;
        }
        
        Ok(())
    }

    fn name(&self) -> &str {
        "console"
    }
}

/// JSON metrics reporter for file output
pub struct JsonReporter {
    file_path: std::path::PathBuf,
    interval_ms: u64,
    last_report: std::sync::Mutex<Instant>,
}

impl JsonReporter {
    /// Create new JSON reporter
    pub fn new(file_path: std::path::PathBuf, interval_ms: u64) -> Self {
        Self {
            file_path,
            interval_ms,
            last_report: std::sync::Mutex::new(Instant::now()),
        }
    }
}

impl MetricsReporter for JsonReporter {
    fn report(&self, metrics: &Metrics) -> PupResult<()> {
        let mut last_report = self.last_report.lock().unwrap();
        let now = Instant::now();
        
        if now.duration_since(*last_report).as_millis() >= self.interval_ms as u128 {
            let timestamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis();

            let metrics_json = format!(
                r#"{{"timestamp":{},"fps":{:.2},"inference_latency_ms":{:.2},"memory_usage_mb":{},"dropped_frames":{},"total_frames":{},"frame_drop_rate":{:.2},"cpu_usage_percent":{:.2},"gpu_usage_percent":{:.2}}}"#,
                timestamp,
                metrics.get_fps(),
                metrics.get_inference_latency_ms(),
                metrics.get_memory_usage_mb(),
                metrics.get_dropped_frames(),
                metrics.get_total_frames(),
                metrics.get_frame_drop_rate(),
                metrics.get_cpu_usage_percent(),
                metrics.get_gpu_usage_percent()
            );

            // Append to file
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.file_path)
                .map_err(|e| PupError::OutputDirectoryError {
                    path: self.file_path.clone(),
                })?;
            
            writeln!(file, "{}", metrics_json)
                .map_err(|e| PupError::Unexpected(format!("Failed to write metrics: {}", e)))?;

            *last_report = now;
        }
        
        Ok(())
    }

    fn name(&self) -> &str {
        "json"
    }
}

/// Performance monitor that coordinates metrics collection and reporting
pub struct PerformanceMonitor {
    metrics: Arc<Metrics>,
    reporters: Vec<Box<dyn MetricsReporter>>,
    fps_calculator: FpsCalculator,
    memory_monitor: MemoryMonitor,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(Metrics::new()),
            reporters: Vec::new(),
            fps_calculator: FpsCalculator::new(),
            memory_monitor: MemoryMonitor::new(),
        }
    }

    /// Add a metrics reporter
    pub fn add_reporter(&mut self, reporter: Box<dyn MetricsReporter>) {
        self.reporters.push(reporter);
    }

    /// Get shared reference to metrics
    pub fn metrics(&self) -> Arc<Metrics> {
        self.metrics.clone()
    }

    /// Start frame processing timing
    pub fn start_frame(&mut self) -> FrameTimer {
        self.fps_calculator.frame_start();
        FrameTimer::new(self.metrics.clone())
    }

    /// Record inference timing
    pub fn record_inference_time(&self, duration: Duration) {
        self.metrics.update_inference_latency(duration.as_millis() as f64);
    }

    /// Update system metrics (memory, CPU, etc.)
    pub fn update_system_metrics(&mut self) -> PupResult<()> {
        // Update memory usage
        let memory_mb = self.memory_monitor.get_memory_usage_mb()?;
        self.metrics.update_memory_usage(memory_mb);

        // Update FPS
        if let Some(fps) = self.fps_calculator.calculate_fps() {
            self.metrics.update_fps(fps);
        }

        Ok(())
    }

    /// Report metrics to all registered reporters
    pub fn report(&self) -> PupResult<()> {
        for reporter in &self.reporters {
            reporter.report(&self.metrics)?;
        }
        Ok(())
    }

    /// Check if performance targets are being met
    pub fn check_targets(&self, target_fps: f64, max_latency_ms: f64) -> PupResult<()> {
        self.metrics.check_performance_targets(target_fps, max_latency_ms)
    }
}

/// FPS calculator using rolling window
struct FpsCalculator {
    frame_times: Vec<Instant>,
    last_fps_calculation: Instant,
    window_size: usize,
}

impl FpsCalculator {
    fn new() -> Self {
        Self {
            frame_times: Vec::with_capacity(60), // 1 second window at 60fps
            last_fps_calculation: Instant::now(),
            window_size: 60,
        }
    }

    fn frame_start(&mut self) {
        let now = Instant::now();
        self.frame_times.push(now);
        
        // Keep only recent frame times (rolling window)
        if self.frame_times.len() > self.window_size {
            self.frame_times.remove(0);
        }
    }

    fn calculate_fps(&mut self) -> Option<f64> {
        let now = Instant::now();
        
        // Calculate FPS every 500ms
        if now.duration_since(self.last_fps_calculation).as_millis() < 500 {
            return None;
        }

        if self.frame_times.len() < 2 {
            return None;
        }

        let duration = now.duration_since(self.frame_times[0]);
        let fps = (self.frame_times.len() - 1) as f64 / duration.as_secs_f64();
        
        self.last_fps_calculation = now;
        Some(fps)
    }
}

/// Memory usage monitor
struct MemoryMonitor;

impl MemoryMonitor {
    fn new() -> Self {
        Self
    }

    fn get_memory_usage_mb(&self) -> PupResult<usize> {
        // Platform-specific memory usage detection
        #[cfg(target_os = "macos")]
        {
            self.get_memory_usage_macos()
        }
        
        #[cfg(target_os = "linux")]
        {
            self.get_memory_usage_linux()
        }
        
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            // Fallback for unsupported platforms
            Ok(0)
        }
    }

    #[cfg(target_os = "macos")]
    fn get_memory_usage_macos(&self) -> PupResult<usize> {
        use std::process::Command;
        
        let output = Command::new("ps")
            .args(&["-o", "rss=", "-p", &std::process::id().to_string()])
            .output()
            .map_err(|e| PupError::Unexpected(format!("Failed to get memory usage: {}", e)))?;

        let rss_kb = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse::<usize>()
            .unwrap_or(0);
        
        Ok(rss_kb / 1024) // Convert KB to MB
    }

    #[cfg(target_os = "linux")]
    fn get_memory_usage_linux(&self) -> PupResult<usize> {
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
}

/// Frame timing helper
pub struct FrameTimer {
    start_time: Instant,
    metrics: Arc<Metrics>,
}

impl FrameTimer {
    fn new(metrics: Arc<Metrics>) -> Self {
        Self {
            start_time: Instant::now(),
            metrics,
        }
    }

    /// Mark frame as complete and update metrics
    pub fn complete(self) {
        let duration = self.start_time.elapsed();
        self.metrics.update_frame_time(duration.as_millis() as f64);
        self.metrics.increment_total_frames();
    }

    /// Mark frame as dropped
    pub fn drop(self) {
        self.metrics.increment_dropped_frames();
        self.metrics.increment_total_frames();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_metrics_creation() {
        let metrics = Metrics::new();
        assert_eq!(metrics.get_fps(), 0.0);
        assert_eq!(metrics.get_memory_usage_mb(), 0);
        assert_eq!(metrics.get_dropped_frames(), 0);
    }

    #[test]
    fn test_metrics_updates() {
        let metrics = Metrics::new();
        
        metrics.update_fps(30.0);
        assert_eq!(metrics.get_fps(), 30.0);
        
        metrics.update_inference_latency(50.0);
        assert_eq!(metrics.get_inference_latency_ms(), 50.0);
        
        metrics.update_memory_usage(256);
        assert_eq!(metrics.get_memory_usage_mb(), 256);
        assert_eq!(metrics.get_peak_memory_mb(), 256);
        
        // Update with lower memory - peak should remain
        metrics.update_memory_usage(128);
        assert_eq!(metrics.get_memory_usage_mb(), 128);
        assert_eq!(metrics.get_peak_memory_mb(), 256);
    }

    #[test]
    fn test_frame_counting() {
        let metrics = Metrics::new();
        
        metrics.increment_total_frames();
        metrics.increment_total_frames();
        metrics.increment_dropped_frames();
        
        assert_eq!(metrics.get_total_frames(), 2);
        assert_eq!(metrics.get_dropped_frames(), 1);
        assert_eq!(metrics.get_frame_drop_rate(), 50.0);
    }

    #[test]
    fn test_performance_targets() {
        let metrics = Metrics::new();
        
        // Set good performance
        metrics.update_fps(60.0);
        metrics.update_inference_latency(10.0);
        
        // Should pass targets
        assert!(metrics.check_performance_targets(30.0, 50.0).is_ok());
        
        // Should fail FPS target
        metrics.update_fps(20.0);
        assert!(metrics.check_performance_targets(30.0, 50.0).is_err());
        
        // Should fail latency target
        metrics.update_fps(60.0);
        metrics.update_inference_latency(100.0);
        assert!(metrics.check_performance_targets(30.0, 50.0).is_err());
    }

    #[test]
    fn test_metrics_reset() {
        let metrics = Metrics::new();
        
        metrics.update_fps(30.0);
        metrics.update_memory_usage(256);
        metrics.increment_total_frames();
        
        metrics.reset();
        
        assert_eq!(metrics.get_fps(), 0.0);
        assert_eq!(metrics.get_memory_usage_mb(), 0);
        assert_eq!(metrics.get_total_frames(), 0);
        assert_eq!(metrics.get_peak_memory_mb(), 0);
    }

    #[test]
    fn test_console_reporter() {
        let metrics = Metrics::new();
        metrics.update_fps(30.5);
        metrics.update_inference_latency(25.7);
        metrics.update_memory_usage(128);
        
        let reporter = ConsoleReporter::new(0); // No interval limit for testing
        assert!(reporter.report(&metrics).is_ok());
        assert_eq!(reporter.name(), "console");
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();
        monitor.add_reporter(Box::new(ConsoleReporter::new(1000)));
        
        let timer = monitor.start_frame();
        thread::sleep(Duration::from_millis(10));
        timer.complete();
        
        assert!(monitor.update_system_metrics().is_ok());
        assert!(monitor.report().is_ok());
    }

    #[test]
    fn test_frame_timer() {
        let metrics = Arc::new(Metrics::new());
        
        let timer = FrameTimer::new(metrics.clone());
        thread::sleep(Duration::from_millis(10));
        timer.complete();
        
        assert_eq!(metrics.get_total_frames(), 1);
        assert!(metrics.get_avg_frame_time_ms() >= 10.0);
    }

    #[test]
    fn test_fps_calculator() {
        let mut calc = FpsCalculator::new();
        
        // Simulate frames at 30 FPS
        for _ in 0..10 {
            calc.frame_start();
            thread::sleep(Duration::from_millis(33)); // ~30 FPS
        }
        
        thread::sleep(Duration::from_millis(500)); // Wait for calculation interval
        let fps = calc.calculate_fps();
        assert!(fps.is_some());
        let fps_value = fps.unwrap();
        assert!(fps_value > 25.0 && fps_value < 35.0); // Rough FPS range
    }
}