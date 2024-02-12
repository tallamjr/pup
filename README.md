# `pup` 🦭

-----

**Table of Contents**

- [About](#about)
- [Usage](#usage)
<!-- - [Installation](#installation) -->
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

3. ...

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


<!-- ## Installation -->

<!-- ```console -->
<!-- pip install pup -->
<!-- ``` -->

## License

`pup` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
