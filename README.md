# `pup` 🦭

-----

**Table of Contents**

- [About](#about)
- [Usage](#usage)
- [Memory Footprint](#memory)
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

<!-- _Note, the first time this is run it will need to download the model weights from huggingface_ -->

This will fire up a window like and annotate objects in each frame like so:

![Alt Text](assets/output.gif)

## Memory Footprint

With current optimization settings, i.e. using `--release` flag will set
`opt-level` to 3, for better speed. We could optimise for small binary size, but
performance will be slower. See the [Cargo Book](https://doc.rust-lang.org/cargo/reference/profiles.html) or https://github.com/johnthagen/min-sized-rust for details on making the resulting binary smaller or faster. As it stands, the resulting binary is ~ 5Mb

```bash
$ du -sh target/release/pup
5.7M    pup
```

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

<!-- ## Installation -->

<!-- ```console -->
<!-- pip install pup -->
<!-- ``` -->

## License

`pup` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
