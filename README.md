# `pup` 🦭

![ci](https://github.com/tallamjr/pup/actions/workflows/rust.yml/badge.svg)

**Table of Contents**

- [About](#about)
- [Usage](#usage)
- [Memory Footprint](#memory)
- [Cross-compiling](#cross)
- [License](#license)

# About

`pup` is a small proof-of-concept codebase for running ML models on video data
using rust.

The main libraries are `opencv` for video processing and `candle` as the deep
learning framework.

### _Why?_

You may be asking why rust, when it's far easier to get up and running using the
python equivalent libraries that are out there. While this is absolutely true,
rust has several advantages in this context:

1. **Cross-compiling for hardware agnostic deployment**. By leveraging the rust
   tool-chain ecosystem, we can easily compile an application with a specific
   target in mind. This allows for a single codebase to be used to create
   executable binaries across many different hardware.

2. **Far greater control over memory usage, binary size and level of
   optimizations**. Python is fantastic for flexibility and rapid prototyping, but
   if we want an application that is to be run in resource constrained settings,
   we want full control over the memory footprint and runtime as possible. Rust
   as a statically typed, memory safe, language gives the programmer the power
   to develop with confidence of how memory is being used.

### _What?_

What's going on here then? This repo -- at the time of press -- uses the
[Yolo-V8](https://github.com/huggingface/candle/tree/main/candle-examples/examples/yolo-v8)
model defined in candle and applies it to video frames. The model itself could
of course be switched out for any other object detection architecture, this was
only chosen as a first attempt at combining `opencv` with `candle`.

## Usage

To run the simple example, run:

```bash
cargo run --release -- --video assets/sample.mp4
```

> ❕ This assumes rust is install on the host system. If not one can run: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

<!-- _Note, the first time this is run it will need to download the model weights from huggingface_ -->

This will fire up a window like below and annotate objects in each frame:

![Alt Text](assets/output.gif)

Information about the location of the boxes and the identified objects will be
sent to stdout...

```bash
...
Processing frame 160
generated predictions Tensor[dims 84, 4620; f32]
person: Bbox { xmin: 25.402996, ymin: 281.722, xmax: 37.93538, ymax: 305.6891, confidence: 0.5820799, data: [] }
person: Bbox { xmin: 6.352282, ymin: 284.7604, xmax: 13.664183, ymax: 306.18918, confidence: 0.41667968, data: [] }
car: Bbox { xmin: 33.109806, ymin: 284.62817, xmax: 109.44653, ymax: 320.54688, confidence: 0.8652527, data: [] }
truck: Bbox { xmin: 150.58142, ymin: 265.03888, xmax: 198.64215, ymax: 297.8924, confidence: 0.63634956, data: [] }
Processing frame 165
generated predictions Tensor[dims 84, 4620; f32]
person: Bbox { xmin: 29.655272, ymin: 281.85794, xmax: 39.016144, ymax: 299.11453, confidence: 0.56209296, data: [] }
car: Bbox { xmin: 20.964737, ymin: 286.7253, xmax: 101.58455, ymax: 324.42282, confidence: 0.84946674, data: [] }
truck: Bbox { xmin: 148.0414, ymin: 266.43268, xmax: 194.61363, ymax: 298.94403, confidence: 0.4919662, data: [] }
traffic light: Bbox { xmin: 35.862103, ymin: 219.83334, xmax: 45.31969, ymax: 249.05392, confidence: 0.44971594, data: [] }
Processing frame 170
generated predictions Tensor[dims 84, 4620; f32]
car: Bbox { xmin: 4.899437, ymin: 286.60254, xmax: 92.49835, ymax: 326.64014, confidence: 0.8659615, data: [] }
truck: Bbox { xmin: 144.10123, ymin: 265.35748, xmax: 191.16196, ymax: 298.9926, confidence: 0.6999927, data: [] }
traffic light: Bbox { xmin: 35.988472, ymin: 219.34744, xmax: 45.318306, ymax: 248.43362, confidence: 0.42890158, data: [] }
...
```

## Memory Footprint

With current optimization settings, i.e. using `--release` flag will set
`opt-level` to 3, for better speed. We could optimise for small binary size, but
performance will be slower. See the [Cargo
Book](https://doc.rust-lang.org/cargo/reference/profiles.html) or
https://github.com/johnthagen/min-sized-rust for details on making the resulting
binary smaller or faster. As it stands, the resulting binary is ~ 5Mb

```bash
$ du -sh target/release/pup
5.7M    pup
```

If we use those tips and tricks such as:

```diff
+ [profile.release]
+ codegen-units = 1   # Reduce number of codegen units to increase optimizations.
+ lto = true          # Enable Link Time Optimization
+ opt-level = "z"     # Optimize for size.
+ panic = "abort"     # Abort on panic
+ strip = true        # Automatically strip symbols from the binary.
```

we can get this down further to ~ 3Mb.

```bash
$ cargo clean && cargo +nightly run --release -- --video assets/sample.mp4
$ du -sh target/release/pup
2.8M    pup
```

### `cargo-bloat` & `cargo-size`

Two handy cargo crates for inspecting and profiling for where the memory is
being used up is [`cargo bloat`](https://github.com/RazrFalcon/cargo-bloat) and
[`cargo size`](https://github.com/rust-embedded/cargo-binutils) with the latter
being syntactic-sugar for the `rust-size` utility found in rust-embedded's
`cargo-binutils` crate.

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

With the above numbers in bytes, i.e. 2746960 bytes ~ 2.8Mb.

```bash
$ cargo +nightly bloat --release -n 10
 File  .text     Size        Crate Name
 9.3%  25.9% 399.2KiB          url core::panicking::assert_failed
 0.5%   1.4%  21.3KiB        image jpeg_decoder::decoder::Decoder<R>::decode_internal
 0.4%   1.2%  18.0KiB          pup pup::main
 0.4%   1.1%  17.2KiB    [Unknown] __mh_execute_header
 0.3%   0.9%  13.4KiB  candle_core candle_core::cpu_backend::utils::Map1Any::map
 0.3%   0.8%  12.9KiB clap_builder clap_builder::parser::parser::Parser::get_matches_with
 0.3%   0.8%  12.0KiB   ttf_parser ttf_parser::Face::from_slice
 0.3%   0.8%  11.6KiB       rustls rustls::msgs::handshake::HandshakeMessagePayload::read_version
 0.3%   0.7%  11.0KiB          std std::backtrace_rs::symbolize::gimli::Context::new
 0.3%   0.7%  10.9KiB  candle_core <candle_core::cpu_backend::CpuStorage as candle_core::backend::BackendStorage>::to_dtype
33.4%  92.8%   1.4MiB              And 11025 smaller methods. Use -n N to show more.
36.0% 100.0%   1.5MiB              .text section size, the file size is 4.2MiB
```

**Note**, the above does not include all the optimizations listed previously so
it is re-compiled and the `.text` section size is of the order `4Mb`. But it
gives us a good indication as to where the bulk of the code is being used up.

```bash
$ du -sh target/release/pup
4.2M    target/release/pup
```

We can also look at which crates take up the most space:

```bash
   Compiling pup v0.1.0 ...
   ....
    Finished `release` profile [optimized] target(s) in 1m 39s
    Analyzing target/release/pup

 File  .text     Size Crate
 9.5%  26.9% 420.4KiB url
 7.8%  21.8% 341.8KiB std
 3.5%   9.8% 153.8KiB candle_core
 3.1%   8.8% 137.2KiB [Unknown]
 2.7%   7.6% 118.8KiB rustls
 2.0%   5.8%  90.2KiB clap_builder
 1.9%   5.3%  82.8KiB image
 1.7%   4.9%  76.6KiB pup
 1.6%   4.5%  70.8KiB ring
 1.3%   3.8%  59.1KiB ureq
 1.2%   3.4%  53.9KiB gemm_f16
 0.9%   2.5%  38.4KiB ttf_parser
 0.7%   1.9%  29.0KiB gemm_f32
 0.6%   1.7%  26.4KiB gemm_f64
 0.5%   1.4%  21.9KiB png
 0.4%   1.2%  19.3KiB indicatif
 0.4%   1.2%  18.9KiB jpeg_decoder
 0.4%   1.1%  17.1KiB webpki
 0.4%   1.0%  15.6KiB safetensors
 0.3%   1.0%  15.3KiB miniz_oxide
 4.0%  11.2% 175.6KiB And 76 more crates. Use -n N to show more.
35.5% 100.0%   1.5MiB .text section size, the file size is 4.3MiB

Note: numbers above are a result of guesswork. They are not 100% correct and never will be.

```

We can see the `url` crate is taking up a lot of space which may be linked to
how to pull down weights. If we move to an `ORT` model serving method we could
potentially save here, _but let's see how we go_

## Runtime

To investigate the runtime memory footprint we can use `leaks` utility on macOS.

```bash
$ leaks --atExit -- ./target/release/pup --video assets/sample.mp4

Process:         pup [24864]
Path:            /Users/USER/*/pup
Load Address:    0x104744000
Identifier:      pup
Version:         0
Code Type:       ARM64
Platform:        macOS
Parent Process:  leaks [24863]

Date/Time:       2024-02-12 21:25:46.568 +0000
Launch Time:     2024-02-12 21:25:29.754 +0000
OS Version:      macOS 13.4 (22F66)
Report Version:  7
Analysis Tool:   /Applications/Xcode.app/Contents/Developer/usr/bin/leaks
Analysis Tool Version:  Xcode 14.3.1 (14E300c)

Physical footprint:         95.9M
Physical footprint (peak):  687.3M
Idle exit:                  untracked
----

leaks Report Version: 4.0, multi-line stacks
Process 24864: 19369 nodes malloced for 34590 KB
Process 24864: 0 leaks for 0 total leaked bytes.
```

This gives an average memory usage of ~100Mb and total peak memory usage of
~690Mb.

_Chatgpt summary of Peak Physical vs. Physical:_

> The terms "physical memory footprint" and "peak physical memory footprint" both
> relate to the amount of physical memory (RAM) used by a process or application, but they capture different aspects of memory usage:
>
> 1. **Physical Memory Footprint:**
>
> - The physical memory footprint generally refers to the amount of RAM that a process or application is actively using at a specific point in time or on average during its execution. - It represents the current or average memory consumption and is not necessarily the maximum amount of memory used throughout the entire runtime of the process.
>
> 2. **Peak Physical Memory Footprint:**
>
> - The peak physical memory footprint specifically refers to the maximum amount of RAM that a process or application has used during its entire execution. - It represents the highest point of memory consumption reached by the process or application.
>   In summary, the physical memory footprint gives you an idea of the current or average memory usage, while the peak physical memory footprint highlights the maximum memory usage observed over the entire lifetime of the process or application. Monitoring both metrics is important for understanding how an application utilizes memory resources and for optimizing performance and resource management.

As an aside, here is a snippet of the python example (note, they are not the
same program and this is only show as an illustration)

```bash
Process:         python3.11 [27749]
Path:            /Users/USER/*/python
Load Address:    0x10076c000
Identifier:      python3.11
Version:         0
Code Type:       ARM64
Platform:        macOS
Parent Process:  leaks [27733]

Date/Time:       2024-02-13 08:32:14.002 +0000
Launch Time:     2024-02-13 08:31:57.184 +0000
OS Version:      macOS 13.4 (22F66)
Report Version:  7
Analysis Tool:   /Applications/Xcode.app/Contents/Developer/usr/bin/leaks
Analysis Tool Version:  Xcode 14.3.1 (14E300c)

Physical footprint:         334.4M
Physical footprint (peak):  428.9M
Idle exit:                  untracked
----

leaks Report Version: 4.0, multi-line stacks
Process 27749: 116011 nodes malloced for 38107 KB
Process 27749: 9 leaks for 496 total leaked bytes.

STACK OF 3 INSTANCES OF 'ROOT LEAK: <NSMutableDictionary>':
```

Note the memory leaks 😬, albeit tiny ..

## Cross-compiling (wip) 🚧

One of the most attractive things about using rust is the tools that are
available to cross-compile to various hardware. Currently `pup` has been
developed using an arm64 macOS system.

```bash
$ file target/release/pup

target/release/pup: Mach-O 64-bit executable arm64
```

If we want to instead have an ELF or runnable binary, this can easily done using
cargo and including the right tools in `.cargo/config.toml`:

That said, it is notoriously fiddly to cross compile going from macOS --> Linux.
It is advisable to read over https://github.com/japaric/rust-cross or look into
docker based tools such as https://github.com/cross-rs/cross or
https://github.com/rust-cross/rust-musl-cross

`opencv` cross compilation complications 👇

- https://github.com/twistedfall/opencv-rust/blob/master/INSTALL.md#crosscompilation

## Refs

- https://doc.rust-lang.org/rustc/platform-support.html
- [How do I get a vector of u8 RGB values when using the image crate?](https://stackoverflow.com/a/50821290/4521950)
- [How to create Mat from image loaded from Web](https://github.com/twistedfall/opencv-rust/issues/358#issuecomment-1209076962)
- https://doc.rust-lang.org/cargo/index.html
- [Optimize for size](https://docs.rust-embedded.org/book/unsorted/speed-vs-size.html)
- [Viewing Virtual Memory Usage](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/ManagingMemory/Articles/VMPages.html#//apple_ref/doc/uid/20001985-CJBJFIDD)
- [Memgraph, detection of memory issues](https://blogs.halodoc.io/memgraph-detection-of-memory-issues-on-ios/)
- https://github.com/ToluClassics/candle-tutorial
- [Implementing YOLOv8 Object Detection with OpenCV in Rust Using ONNX Models](https://linzichun.com/posts/rust-opencv-onnx-yolov8-detect/)

## License

`pup` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

let input_shape = input_tensor.shape().to_vec();
let input_data = input_tensor.iter().cloned().collect::<Vec<f32>>();

let inputs = ort::inputs!("input" => (input_shape, input_data)).ok()?;
