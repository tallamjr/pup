# `pup` 🦭

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

<!-- _Note, the first time this is run it will need to download the model weights from huggingface_ -->

This will fire up a window like and annotate objects in each frame like so:

![Alt Text](assets/output.gif)

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
>   - The physical memory footprint generally refers to the amount of RAM that a process or application is actively using at a specific point in time or on average during its execution.  - It represents the current or average memory consumption and is not necessarily the maximum amount of memory used throughout the entire runtime of the process.
>
> 2. **Peak Physical Memory Footprint:**
>  - The peak physical memory footprint specifically refers to the maximum amount of RAM that a process or application has used during its entire execution.  - It represents the highest point of memory consumption reached by the process or application.
> In summary, the physical memory footprint gives you an idea of the current or average memory usage, while the peak physical memory footprint highlights the maximum memory usage observed over the entire lifetime of the process or application. Monitoring both metrics is important for understanding how an application utilizes memory resources and for optimizing performance and resource management.

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

## License

`pup` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
