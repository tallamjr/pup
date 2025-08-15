#!/usr/bin/env python3
"""
CoreML vs CPU Inference Performance Benchmark

This script runs the benchmark binary and generates matplotlib visualizations.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Configuration
MODEL_PATH = "models/yolov8n.onnx"
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


class InferenceBenchmark:
    """Handles inference timing benchmarks for CPU vs CoreML using dedicated benchmark binary"""

    def __init__(self):
        self.results = None

    def run_benchmark(self) -> dict:
        """Run the benchmark binary and extract JSON results"""
        print("🚀 Starting CoreML vs CPU Inference Benchmark")
        print("=" * 50)

        # Check prerequisites
        model_path = PROJECT_ROOT / MODEL_PATH
        if not model_path.exists():
            print(f"❌ Model not found: {MODEL_PATH}")
            print("Please ensure YOLOv8n model is available")
            return None

        print(f"🎯 Model: {model_path}")
        print("🔨 Building benchmark binary...")

        # Build the benchmark binary
        build_result = subprocess.run(
            ["cargo", "build", "--release", "--bin", "benchmark"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        if build_result.returncode != 0:
            print("❌ Failed to build benchmark binary:")
            print(build_result.stderr)
            return None

        print("✅ Benchmark binary built successfully")
        print("\n🏃 Running benchmark...")

        # Run the benchmark
        benchmark_result = subprocess.run(
            ["cargo", "run", "--release", "--bin", "benchmark", "--", str(model_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        if benchmark_result.returncode != 0:
            print("❌ Benchmark failed:")
            print(benchmark_result.stderr)
            return None

        # Extract JSON results
        output = benchmark_result.stdout
        print(output)  # Show the benchmark output

        # Parse JSON results
        json_start = output.find("BENCHMARK_JSON_START")
        json_end = output.find("BENCHMARK_JSON_END")

        if json_start != -1 and json_end != -1:
            json_data = output[json_start + len("BENCHMARK_JSON_START") : json_end].strip()
            try:
                self.results = json.loads(json_data)
                return self.results
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse JSON results: {e}")
                return None
        else:
            print("❌ No JSON results found in benchmark output")
            return None

    def create_visualizations(self, results: dict):
        """Create matplotlib visualizations of benchmark results"""
        if not results:
            print("❌ No benchmark results to visualize")
            return

        cpu_data = results["cpu"]
        coreml_data = results["coreml"]
        comparison = results["comparison"]

        # Extract timing data
        cpu_times = cpu_data["timings_ms"]
        coreml_times = coreml_data["timings_ms"]

        cpu_mean = cpu_data["avg_time_ms"]
        cpu_std = cpu_data["std_dev_ms"]
        coreml_mean = coreml_data["avg_time_ms"]
        coreml_std = coreml_data["std_dev_ms"]

        speedup = comparison["speedup"]
        improvement = comparison["improvement_percent"]

        # Create visualizations
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(
            "CoreML vs CPU Inference Performance Comparison - YOLOv8n Object Detection", fontsize=16, fontweight="bold"
        )

        # 1. Bar chart comparing average times
        providers = ["CPU", "CoreML"]
        means = [cpu_mean, coreml_mean]
        stds = [cpu_std, coreml_std]
        colors = ["#ff6b6b", "#4ecdc4"]

        bars = ax1.bar(
            providers, means, yerr=stds, capsize=5, color=colors, alpha=0.8, edgecolor="black", linewidth=1.5
        )
        ax1.set_ylabel("Inference Time (ms)", fontsize=12, fontweight="bold")
        ax1.set_title("Average Inference Time Comparison", fontsize=14, fontweight="bold")
        ax1.grid(axis="y", alpha=0.3)
        ax1.set_ylim(0, max(means) * 1.2)

        # Add value labels on bars
        for bar, mean, std in zip(bars, means, stds):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 0.5,
                f"{mean:.1f}±{std:.1f}ms",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
            )

        # 2. Box plot showing distribution
        box_data = [cpu_times, coreml_times]
        bp = ax2.boxplot(box_data, tick_labels=providers, patch_artist=True, widths=0.6)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.5)

        for element in ["whiskers", "fliers", "medians", "caps"]:
            plt.setp(bp[element], color="black", linewidth=1.5)

        ax2.set_ylabel("Inference Time (ms)", fontsize=12, fontweight="bold")
        ax2.set_title("Inference Time Distribution", fontsize=14, fontweight="bold")
        ax2.grid(axis="y", alpha=0.3)

        # 3. Performance comparison metrics (summary)
        ax3.axis("off")

        # Create a nice formatted table
        metrics_text = f"""
PERFORMANCE SUMMARY

CPU Execution Provider:
├─ Average:     {cpu_mean:.1f} ± {cpu_std:.1f} ms
├─ Best time:   {cpu_data['min_time_ms']:.1f} ms
└─ Worst time:  {cpu_data['max_time_ms']:.1f} ms

CoreML Execution Provider:
├─ Average:     {coreml_mean:.1f} ± {coreml_std:.1f} ms
├─ Best time:   {coreml_data['min_time_ms']:.1f} ms
└─ Worst time:  {coreml_data['max_time_ms']:.1f} ms

PERFORMANCE GAINS:
├─ Speedup:     {speedup:.2f}x faster with CoreML
├─ Time saved:  {improvement:.1f}% reduction
└─ Efficiency:  {(1/speedup)*100:.1f}% of original time

BENCHMARK CONFIGURATION:
├─ Warmup runs: {results['benchmark_config']['warmup_runs']}
├─ Test runs:   {results['benchmark_config']['benchmark_runs']}
└─ Model:       YOLOv8n (640x640)
        """

        ax3.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax3.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
        )

        plt.tight_layout()

        # Save plot
        output_path = PROJECT_ROOT / "coreml_vs_cpu_benchmark.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"\n📈 Visualization saved to: {output_path}")

        # Show plot
        plt.show()

        # Print summary
        print("\n" + "=" * 60)
        print("📊 BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        print(f"CPU Execution Provider:    {cpu_mean:.1f} ± {cpu_std:.1f} ms")
        print(f"CoreML Execution Provider: {coreml_mean:.1f} ± {coreml_std:.1f} ms")
        print(f"Performance Improvement:   {speedup:.2f}x faster with CoreML")
        print(f"Time Reduction:            {improvement:.1f}%")
        print(f"Hardware Acceleration:     {100-((coreml_mean/cpu_mean)*100):.1f}% efficiency gain")
        print("=" * 60)


def main():
    """Main benchmark execution"""
    benchmark = InferenceBenchmark()

    try:
        results = benchmark.run_benchmark()
        if results:
            benchmark.create_visualizations(results)
        else:
            print("❌ Benchmark failed to complete")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Benchmark error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
