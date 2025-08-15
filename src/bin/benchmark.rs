//! Inference Performance Benchmark Binary
//!
//! Compares CoreML vs CPU execution provider performance for inference timing

use gstpup::inference::{InferenceBackend, OrtBackend};
use serde_json::json;
use std::env;
use std::path::Path;
use std::time::Instant;

const WARMUP_RUNS: usize = 3;
const BENCHMARK_RUNS: usize = 20;

struct BenchmarkResults {
    execution_provider: String,
    timings: Vec<f64>,
    avg_time: f64,
    std_dev: f64,
    min_time: f64,
    max_time: f64,
}

impl BenchmarkResults {
    fn new(execution_provider: String, timings: Vec<f64>) -> Self {
        let avg_time = timings.iter().sum::<f64>() / timings.len() as f64;
        let variance =
            timings.iter().map(|&x| (x - avg_time).powi(2)).sum::<f64>() / timings.len() as f64;
        let std_dev = variance.sqrt();
        let min_time = timings.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_time = timings.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        Self {
            execution_provider,
            timings,
            avg_time,
            std_dev,
            min_time,
            max_time,
        }
    }

    fn to_json(&self) -> serde_json::Value {
        json!({
            "execution_provider": self.execution_provider,
            "avg_time_ms": self.avg_time,
            "std_dev_ms": self.std_dev,
            "min_time_ms": self.min_time,
            "max_time_ms": self.max_time,
            "timings_ms": self.timings
        })
    }
}

fn create_dummy_input() -> Vec<f32> {
    // Create a 640x640x3 RGB image with random-like values
    let size = 3 * 640 * 640;
    let mut input = Vec::with_capacity(size);

    // Generate pseudo-random values in [0, 1] range
    let mut seed = 12345u32;
    for _ in 0..size {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let normalized = (seed as f32) / (u32::MAX as f32);
        input.push(normalized);
    }

    input
}

fn benchmark_execution_provider(
    use_coreml: bool,
    model_path: &Path,
) -> Result<BenchmarkResults, Box<dyn std::error::Error>> {
    let provider_name = if use_coreml { "CoreML" } else { "CPU" };
    println!("🔬 Benchmarking {} Execution Provider", provider_name);

    // Create backend
    let mut backend = if use_coreml {
        let mut backend = OrtBackend::new();
        backend.with_coreml(true);
        backend
    } else {
        OrtBackend::with_cpu_only()
    };

    // Load model
    println!("📦 Loading model: {}", model_path.display());
    backend.load_model(model_path)?;

    // Create input data
    let input_data = create_dummy_input();

    // Warmup runs
    println!("🔥 Warming up with {} runs...", WARMUP_RUNS);
    for i in 0..WARMUP_RUNS {
        let _ = backend.infer(&input_data)?;
        print!(".");
        if (i + 1) % 10 == 0 {
            println!(" {}", i + 1);
        }
    }
    if !WARMUP_RUNS.is_multiple_of(10) {
        println!();
    }

    // Benchmark runs
    println!("⏱️  Running {} benchmark iterations...", BENCHMARK_RUNS);
    let mut timings = Vec::with_capacity(BENCHMARK_RUNS);

    for i in 0..BENCHMARK_RUNS {
        let start = Instant::now();
        let _ = backend.infer(&input_data)?;
        let duration = start.elapsed();
        let time_ms = duration.as_secs_f64() * 1000.0;
        timings.push(time_ms);

        print!(".");
        if (i + 1) % 10 == 0 {
            println!(" {} ({:.1}ms)", i + 1, time_ms);
        }
    }
    if !BENCHMARK_RUNS.is_multiple_of(10) {
        println!();
    }

    let results = BenchmarkResults::new(provider_name.to_string(), timings);

    println!("📊 {} Results:", provider_name);
    println!(
        "   Average: {:.2} ± {:.2} ms",
        results.avg_time, results.std_dev
    );
    println!(
        "   Range:   {:.2} - {:.2} ms",
        results.min_time, results.max_time
    );

    Ok(results)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing with minimal logging for cleaner output
    use tracing_subscriber;
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    println!("🚀 CoreML vs CPU Inference Performance Benchmark");
    println!("================================================");

    // Get model path from command line or use default
    let model_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "models/yolov8n.onnx".to_string());
    let model_path = Path::new(&model_path);

    if !model_path.exists() {
        eprintln!("❌ Error: Model file not found: {}", model_path.display());
        eprintln!("Usage: benchmark [model_path]");
        std::process::exit(1);
    }

    println!("🎯 Model: {}", model_path.display());
    println!(
        "🔧 Configuration: {} warmup + {} benchmark runs",
        WARMUP_RUNS, BENCHMARK_RUNS
    );
    println!();

    // Benchmark CPU execution provider
    let cpu_results = benchmark_execution_provider(false, model_path)?;
    println!();

    // Benchmark CoreML execution provider
    let coreml_results = benchmark_execution_provider(true, model_path)?;
    println!();

    // Calculate speedup
    let speedup = cpu_results.avg_time / coreml_results.avg_time;
    let improvement =
        ((cpu_results.avg_time - coreml_results.avg_time) / cpu_results.avg_time) * 100.0;

    println!("📈 Performance Comparison");
    println!("========================");
    println!(
        "CPU:    {:.2} ± {:.2} ms",
        cpu_results.avg_time, cpu_results.std_dev
    );
    println!(
        "CoreML: {:.2} ± {:.2} ms",
        coreml_results.avg_time, coreml_results.std_dev
    );
    println!("Speedup: {:.2}x faster with CoreML", speedup);
    println!(
        "Improvement: {:.1}% reduction in inference time",
        improvement
    );

    // Output JSON results for Python script
    let results = json!({
        "benchmark_config": {
            "model_path": model_path.display().to_string(),
            "warmup_runs": WARMUP_RUNS,
            "benchmark_runs": BENCHMARK_RUNS
        },
        "cpu": cpu_results.to_json(),
        "coreml": coreml_results.to_json(),
        "comparison": {
            "speedup": speedup,
            "improvement_percent": improvement
        }
    });

    // Write JSON results to file
    let output_file = "benchmark_results.json";
    std::fs::write(output_file, serde_json::to_string_pretty(&results)?)?;
    println!("\n💾 Detailed results saved to: {}", output_file);
    println!("\nBENCHMARK_JSON_START");
    println!("{}", serde_json::to_string(&results)?);
    println!("BENCHMARK_JSON_END");

    Ok(())
}
