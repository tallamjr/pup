# `pup`

![ci](https://github.com/tallamjr/pup/actions/workflows/rust.yml/badge.svg)

**Table of Contents**

- [About](#about)
  - [Why Rust?](#why-rust)
- [Architecture](#architecture)
- [Usage](#usage)
- [Model Requirements](#model-requirements)
- [Memory Footprint](#memory-footprint)
- [Cross-compiling](#cross-compiling)
- [Refs](#refs)
- [Licence](#licence)

## About

`pup` is a real-time object detection application built in Rust. It uses YOLOv8
ONNX models for inference on video frames via a GStreamer pipeline, with results
rendered as bounding box overlays directly on the video output.

The core libraries are GStreamer for video pipeline management and ONNX Runtime
(via the `ort` crate) with CoreML acceleration on macOS for high-performance
inference.

### Why Rust?

1. **Cross-compiling for hardware agnostic deployment.** By leveraging the Rust
   toolchain ecosystem, a single codebase can produce executable binaries across
   many different hardware targets.

2. **Full control over memory usage, binary size and optimisation.** Rust as a
   statically typed, memory safe language gives the programmer confidence in how
   memory is being used -- essential for resource-constrained deployment
   settings.

## Architecture

`pup` combines two core libraries into a single real-time inference pipeline:

**GStreamer** handles the entire video lifecycle -- capturing frames from a
webcam or decoding them from a file, converting colour spaces, scaling, and
rendering the final output to a display window. The pipeline uses a `tee`
element to split the video into two parallel branches: one for display and one
for inference. A GStreamer
[pad probe](https://gstreamer.freedesktop.org/documentation/application-development/advanced/pipeline-manipulation.html)
on the display branch draws bounding boxes directly into the video buffer
before it reaches the video sink, so overlays appear with minimal latency.

**ONNX Runtime** (via the [`ort`](https://docs.rs/ort) crate) runs the YOLOv8
model. On macOS, the `ort` session is configured with the CoreML execution
provider, which offloads inference to the Apple Neural Engine or GPU when
available, falling back to CPU transparently. The inference path extracts raw
RGB frame data from an `appsink`, normalises pixel values to `[0.0, 1.0]`,
reshapes from HWC to CHW format, and feeds the resulting `[1, 3, 640, 640]`
tensor into the ORT session. The YOLOv8 output (`[1, 84, 8400]`) is then
post-processed with confidence filtering and non-maximum suppression to produce
the final detection list.

```
                         ┌─────────────┐
                         │  Video Src  │
                         │ (webcam/file)│
                         └──────┬──────┘
                                │
                         ┌──────┴──────┐
                         │  decodebin  │
                         │ videoconvert│
                         │  videoscale │
                         │ capsfilter  │
                         │ (RGB 640x640)│
                         └──────┬──────┘
                                │
                           ┌────┴────┐
                           │   tee   │
                           └────┬────┘
                          ╱            ╲
                   ┌─────┴─────┐  ┌────┴─────┐
                   │  queue 1  │  │  queue 2  │
                   └─────┬─────┘  └────┬─────┘
                         │              │
                   ┌─────┴─────┐  ┌────┴──────┐
                   │  appsink  │  │ pad probe │
                   │ (extract  │  │  (draw    │
                   │  frames)  │  │ overlays) │
                   └─────┬─────┘  └────┬──────┘
                         │              │
                   ┌─────┴─────┐  ┌────┴──────┐
                   │ ORT infer │  │ videosink │
                   │ (YOLOv8)  │  │ (display) │
                   └─────┬─────┘  └───────────┘
                         │
                   ┌─────┴─────┐
                   │ detections│──── shared via Arc<Mutex<>>
                   └───────────┘
```

The inference branch writes its detection results into a shared
`Arc<Mutex<Vec<Detection>>>`. The display branch's pad probe reads those
detections on each frame and renders bounding boxes, class labels, and
confidence scores directly into the pixel buffer before display.

## Usage

```bash
# Process webcam (default)
cargo run --release

# Process video file
cargo run --release -- --input assets/sample.mp4

# Custom model and confidence threshold
cargo run --release -- --model models/yolov8n.onnx --confidence 0.7

# Disable overlays (detection logging only)
cargo run --release -- --no-overlays

# Hide labels or confidence scores on bounding boxes
cargo run --release -- --no-labels --no-confidence

# Enable verbose logging
cargo run --release -- --verbose
```

> **Note:** This assumes Rust is installed on the host system. If not, run:
> `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

> **Model Required:** You need a YOLOv8 ONNX model file. Place `yolov8n.onnx`
> in the `models/` directory.

Running on a webcam with real-time overlays:

<img src="assets/live-demo.png" alt="Live Demo" width="500">

Processing a video file with bounding box annotations:

![Detection Output](assets/output.gif)

Detection information is printed to stdout:

```bash
...
Processing frame 160
generated predictions Tensor[dims 84, 4620; f32]
person: Bbox { xmin: 25.402996, ymin: 281.722, xmax: 37.93538, ymax: 305.6891, confidence: 0.5820799, data: [] }
car: Bbox { xmin: 33.109806, ymin: 284.62817, xmax: 109.44653, ymax: 320.54688, confidence: 0.8652527, data: [] }
truck: Bbox { xmin: 150.58142, ymin: 265.03888, xmax: 198.64215, ymax: 297.8924, confidence: 0.63634956, data: [] }
Processing frame 165
generated predictions Tensor[dims 84, 4620; f32]
person: Bbox { xmin: 29.655272, ymin: 281.85794, xmax: 39.016144, ymax: 299.11453, confidence: 0.56209296, data: [] }
car: Bbox { xmin: 20.964737, ymin: 286.7253, xmax: 101.58455, ymax: 324.42282, confidence: 0.84946674, data: [] }
traffic light: Bbox { xmin: 35.862103, ymin: 219.83334, xmax: 45.31969, ymax: 249.05392, confidence: 0.44971594, data: [] }
...
```

## Model Requirements

`pup` requires a YOLOv8 ONNX model for object detection. The model expects:

- **Input shape:** `[1, 3, 640, 640]` (batch, RGB channels, height, width)
- **Output format:** `[1, 84, #boxes]` where 84 = 4 bounding box coordinates + 80 COCO class scores
- **Location:** Place model files in the `models/` directory

## Memory Footprint

With current optimisation settings, using `--release` sets `opt-level` to 3 for
better speed. The resulting binary is approximately 5 MB:

```bash
$ du -sh target/release/pup
5.7M    pup
```

With additional size optimisations:

```diff
+ [profile.release]
+ codegen-units = 1   # Reduce number of codegen units to increase optimisations.
+ lto = true          # Enable Link Time Optimisation
+ opt-level = "z"     # Optimise for size.
+ panic = "abort"     # Abort on panic
+ strip = true        # Automatically strip symbols from the binary.
```

the binary can be reduced to approximately 3 MB:

```bash
$ du -sh target/release/pup
2.8M    pup
```

See the [Cargo Book](https://doc.rust-lang.org/cargo/reference/profiles.html) or
[min-sized-rust](https://github.com/johnthagen/min-sized-rust) for further
details on binary size optimisation.

### `cargo-bloat` and `cargo-size`

Two useful cargo crates for inspecting where memory is being used:
[`cargo bloat`](https://github.com/RazrFalcon/cargo-bloat) and
[`cargo size`](https://github.com/rust-embedded/cargo-binutils) (syntactic sugar
for the `rust-size` utility from `cargo-binutils`).

```bash
$ rust-size -A target/release/pup
target/release/pup  :
section                 size         addr
__text               1576592   4294984896
__stubs                 1620   4296561488
__stub_helper           1620   4296563108
__const              1017480   4296564736
__gcc_except_tab        6360   4297582216
__cstring                 52   4297588576
__unwind_info          21592   4297588628
__eh_frame             42712   4297610224
__got                     80   4297654272
__const                73816   4297654352
__la_symbol_ptr         1064   4297736192
__data                  1712   4297737256
__thread_vars            672   4297738968
__thread_data             64   4297739640
__thread_bss             392   4297739704
__bss                   1128   4297740096
__common                   4   4297741224
Total                2746960
```

### Runtime

To investigate the runtime memory footprint on macOS using the `leaks` utility:

```bash
$ leaks --atExit -- ./target/release/pup --input assets/sample.mp4

Physical footprint:         95.9M
Physical footprint (peak):  687.3M

Process 24864: 19369 nodes malloced for 34590 KB
Process 24864: 0 leaks for 0 total leaked bytes.
```

Average memory usage of approximately 100 MB with peak usage of approximately 690 MB and
zero memory leaks.

## Cross-compiling

One of the most attractive aspects of Rust is its cross-compilation support.
Currently `pup` has been developed on arm64 macOS:

```bash
$ file target/release/pup
target/release/pup: Mach-O 64-bit executable arm64
```

Cross-compiling from macOS to Linux can be fiddly. Useful resources:

- [rust-cross](https://github.com/japaric/rust-cross) -- general guidance
- [cross-rs](https://github.com/cross-rs/cross) -- Docker-based cross-compilation
- [rust-musl-cross](https://github.com/rust-cross/rust-musl-cross) -- musl-based static linking
- [opencv-rust cross-compilation](https://github.com/twistedfall/opencv-rust/blob/master/INSTALL.md#crosscompilation)

## Refs

- [`ort` crate documentation](https://docs.rs/ort) -- Rust bindings for ONNX Runtime
- [ONNX Runtime execution providers](https://onnxruntime.ai/docs/execution-providers/) -- CoreML, CUDA, TensorRT, etc.
- [GStreamer Rust bindings (`gstreamer-rs`)](https://gitlab.freedesktop.org/gstreamer/gstreamer-rs)
- [Implementing YOLOv8 Object Detection with OpenCV in Rust Using ONNX Models](https://linzichun.com/posts/rust-opencv-onnx-yolov8-detect/)
- [Rust platform support](https://doc.rust-lang.org/rustc/platform-support.html)
- [Cargo Book](https://doc.rust-lang.org/cargo/index.html)
- [Optimise for size](https://docs.rust-embedded.org/book/unsorted/speed-vs-size.html)

## Licence

`pup` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) licence.
