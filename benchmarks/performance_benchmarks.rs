//! Criterion-based performance benchmarks
//!
//! These benchmarks provide precise performance measurements
//! for validating roadmap optimizations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::Duration;

/// Benchmark letterbox image resizing performance
fn bench_letterbox_resize(c: &mut Criterion) {
    use opencv::{core, imgproc, prelude::*};

    fn letterbox_to_640(src: &core::Mat) -> opencv::Result<core::Mat> {
        let src_size = src.size()?;
        let (orig_w, orig_h) = (src_size.width, src_size.height);

        let scale = if orig_w > orig_h {
            640.0 / orig_w as f64
        } else {
            640.0 / orig_h as f64
        };
        let new_w = (orig_w as f64 * scale).round() as i32;
        let new_h = (orig_h as f64 * scale).round() as i32;

        let mat_type = src.typ();
        let mut dst =
            core::Mat::new_rows_cols_with_default(640, 640, mat_type, core::Scalar::all(0.0))?;

        let mut resized = core::Mat::default();
        imgproc::resize(
            src,
            &mut resized,
            core::Size::new(new_w, new_h),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;

        let x_offset = (640 - new_w) / 2;
        let y_offset = (640 - new_h) / 2;
        let roi_rect = core::Rect::new(x_offset, y_offset, new_w, new_h);
        let mut roi = dst.roi_mut(roi_rect)?;
        resized.copy_to(&mut roi)?;

        Ok(dst)
    }

    let mut group = c.benchmark_group("letterbox_resize");

    let test_cases = vec![
        ("VGA", 640, 480),
        ("HD", 1280, 720),
        ("FullHD", 1920, 1080),
        ("4K", 3840, 2160),
    ];

    for (name, width, height) in test_cases {
        let src = core::Mat::new_rows_cols_with_default(
            height,
            width,
            core::CV_8UC3,
            core::Scalar::all(128.0),
        )
        .unwrap();

        group.throughput(Throughput::Elements((width * height) as u64));
        group.bench_with_input(BenchmarkId::new("letterbox", name), &src, |b, src| {
            b.iter(|| {
                let _result = letterbox_to_640(black_box(src)).unwrap();
            });
        });
    }
    group.finish();
}

/// Benchmark RGB normalization performance
fn bench_rgb_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("rgb_normalization");

    let sizes = vec![
        ("VGA", 640 * 480 * 3),
        ("HD", 1280 * 720 * 3),
        ("FullHD", 1920 * 1080 * 3),
        ("Letterboxed", 640 * 640 * 3),
    ];

    for (name, size) in sizes {
        let input: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        group.throughput(Throughput::Bytes(size as u64));

        // Sequential normalization
        group.bench_with_input(BenchmarkId::new("sequential", name), &input, |b, input| {
            b.iter(|| {
                let _normalized: Vec<f32> =
                    input.iter().map(|&x| black_box(x as f32 / 255.0)).collect();
            });
        });

        // Parallel normalization using rayon
        group.bench_with_input(BenchmarkId::new("parallel", name), &input, |b, input| {
            b.iter(|| {
                use rayon::prelude::*;
                let _normalized: Vec<f32> = input
                    .par_iter()
                    .map(|&x| black_box(x as f32 / 255.0))
                    .collect();
            });
        });
    }
    group.finish();
}

/// Benchmark tensor shape conversion performance
fn bench_tensor_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_conversion");

    // Test HWC to CHW conversion (Height-Width-Channel to Channel-Height-Width)
    let (height, width, channels) = (640, 640, 3);
    let size = height * width * channels;
    let input: Vec<f32> = (0..size).map(|i| i as f32).collect();

    group.throughput(Throughput::Elements(size as u64));

    // Sequential conversion
    group.bench_function("hwc_to_chw_sequential", |b| {
        b.iter(|| {
            let mut output = vec![0.0f32; size];
            for c in 0..channels {
                for h in 0..height {
                    for w in 0..width {
                        let src_idx = h * width * channels + w * channels + c;
                        let dst_idx = c * height * width + h * width + w;
                        output[dst_idx] = black_box(input[src_idx]);
                    }
                }
            }
            black_box(output);
        });
    });

    // Parallel conversion
    group.bench_function("hwc_to_chw_parallel", |b| {
        b.iter(|| {
            use rayon::prelude::*;
            let output: Vec<f32> = (0..size)
                .into_par_iter()
                .map(|dst_idx| {
                    let c = dst_idx / (height * width);
                    let remaining = dst_idx % (height * width);
                    let h = remaining / width;
                    let w = remaining % width;
                    let src_idx = h * width * channels + w * channels + c;
                    black_box(input[src_idx])
                })
                .collect();
            black_box(output);
        });
    });

    group.finish();
}

/// Benchmark memory allocation patterns
fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");

    let frame_size = 1920 * 1080 * 3; // Full HD RGB

    // Standard allocation
    group.bench_function("standard_allocation", |b| {
        b.iter(|| {
            let _buffer = vec![0u8; frame_size];
            black_box(_buffer);
        });
    });

    // Pre-allocated buffer reuse
    group.bench_function("buffer_reuse", |b| {
        let mut buffer = vec![0u8; frame_size];
        b.iter(|| {
            buffer.fill(black_box(128));
            black_box(&buffer);
        });
    });

    // Memory pool simulation
    group.bench_function("memory_pool", |b| {
        struct SimplePool {
            buffers: Vec<Vec<u8>>,
            available: Vec<bool>,
        }

        impl SimplePool {
            fn new(count: usize, size: usize) -> Self {
                Self {
                    buffers: (0..count).map(|_| vec![0u8; size]).collect(),
                    available: vec![true; count],
                }
            }

            fn get_buffer(&mut self) -> Option<&mut Vec<u8>> {
                for (i, available) in self.available.iter_mut().enumerate() {
                    if *available {
                        *available = false;
                        return Some(&mut self.buffers[i]);
                    }
                }
                None
            }

            fn return_buffer(&mut self, buffer_idx: usize) {
                if buffer_idx < self.available.len() {
                    self.available[buffer_idx] = true;
                }
            }
        }

        let mut pool = SimplePool::new(4, frame_size);

        b.iter(|| {
            if let Some(buffer) = pool.get_buffer() {
                buffer.fill(black_box(128));
                black_box(&buffer);
                // In real usage, would return buffer to pool
            }
        });
    });

    group.finish();
}

/// Benchmark detection post-processing
fn bench_detection_postprocessing(c: &mut Criterion) {
    #[derive(Clone, Debug)]
    struct Detection {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        score: f32,
        class_id: i32,
    }

    impl Detection {
        fn area(&self) -> f32 {
            (self.x2 - self.x1) * (self.y2 - self.y1)
        }

        fn iou(&self, other: &Detection) -> f32 {
            let x1 = self.x1.max(other.x1);
            let y1 = self.y1.max(other.y1);
            let x2 = self.x2.min(other.x2);
            let y2 = self.y2.min(other.y2);

            if x2 <= x1 || y2 <= y1 {
                return 0.0;
            }

            let intersection = (x2 - x1) * (y2 - y1);
            let union = self.area() + other.area() - intersection;

            intersection / union
        }
    }

    fn nms(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
        detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        let mut keep = Vec::new();
        let mut suppress = vec![false; detections.len()];

        for i in 0..detections.len() {
            if suppress[i] {
                continue;
            }
            keep.push(detections[i].clone());

            for j in (i + 1)..detections.len() {
                if suppress[j] {
                    continue;
                }

                if detections[i].iou(&detections[j]) > iou_threshold {
                    suppress[j] = true;
                }
            }
        }

        keep
    }

    let mut group = c.benchmark_group("detection_postprocessing");

    // Generate test detections
    let detection_counts = vec![10, 50, 100, 500];

    for count in detection_counts {
        let detections: Vec<Detection> = (0..count)
            .map(|i| Detection {
                x1: (i as f32 * 10.0) % 600.0,
                y1: (i as f32 * 15.0) % 400.0,
                x2: ((i as f32 * 10.0) % 600.0) + 50.0,
                y2: ((i as f32 * 15.0) % 400.0) + 50.0,
                score: 0.5 + (i as f32 * 0.001) % 0.5,
                class_id: i % 80,
            })
            .collect();

        group.throughput(Throughput::Elements(count as u64));

        // Confidence thresholding
        group.bench_with_input(
            BenchmarkId::new("confidence_threshold", count),
            &detections,
            |b, detections| {
                b.iter(|| {
                    let _filtered: Vec<Detection> = detections
                        .iter()
                        .filter(|d| d.score > black_box(0.5))
                        .cloned()
                        .collect();
                });
            },
        );

        // Non-Maximum Suppression
        group.bench_with_input(
            BenchmarkId::new("nms", count),
            &detections,
            |b, detections| {
                b.iter(|| {
                    let _result = nms(black_box(detections.clone()), black_box(0.5));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark GStreamer buffer handling simulation
fn bench_gstreamer_buffers(c: &mut Criterion) {
    let mut group = c.benchmark_group("gstreamer_buffers");

    // Simulate GStreamer buffer operations
    let frame_size = 1920 * 1080 * 3;

    // Copy-based processing (current approach)
    group.bench_function("copy_processing", |b| {
        let src_buffer = vec![128u8; frame_size];
        b.iter(|| {
            // Simulate copying buffer for processing
            let mut dst_buffer = vec![0u8; frame_size];
            dst_buffer.copy_from_slice(black_box(&src_buffer));

            // Simulate processing
            for pixel in dst_buffer.iter_mut() {
                *pixel = (*pixel as f32 / 255.0 * 255.0) as u8;
            }

            black_box(dst_buffer);
        });
    });

    // In-place processing (zero-copy approach)
    group.bench_function("inplace_processing", |b| {
        let mut buffer = vec![128u8; frame_size];
        b.iter(|| {
            // Simulate in-place processing
            for pixel in buffer.iter_mut() {
                *pixel = (*pixel as f32 / 255.0 * 255.0) as u8;
            }
            black_box(&buffer);
        });
    });

    // Memory mapping simulation
    group.bench_function("memory_mapped", |b| {
        let buffer = vec![128u8; frame_size];
        b.iter(|| {
            // Simulate memory-mapped access (just reference, no copy)
            let slice = black_box(&buffer[..]);

            // Simulate read-only processing
            let _sum: u64 = slice.iter().map(|&x| x as u64).sum();
            black_box(_sum);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_letterbox_resize,
    bench_rgb_normalization,
    bench_tensor_conversion,
    bench_memory_allocation,
    bench_detection_postprocessing,
    bench_gstreamer_buffers
);

criterion_main!(benches);
