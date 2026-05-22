# dsp-toy

A small, highly unprofessional library for DSP, with simple functionalities for DFT, FFT, STFT, FIR/IIR filtering, and other digital signal-related stuff.  
Provided is a `Signal` struct for owning and easily handling signal data, particularily through usage of the builder pattern (function chaining).  
This project's purpose is for learning, so efficiency, ergonomics, and thourough testing weren't prioritized.


## Requirements
- Rust & Cargo

## Usage
This repo is a Cargo workspace with:
- `dsp_toy` (binary crate)
- `dsp_toy_lib` (library crate)

To use the library as a git dependency, add:
```
[dependencies]
# (...other dependencies)
dsp_toy_lib = { git = "https://github.com/hellothisismynewusername/dsp-toy.git", package = "dsp_toy_lib" }
```
(Or whatever my username is currently)

Then make a `dsp_toy_lib::signal::Signal` using `Signal::from()` (array, slice, or iterator).

For demos of what you can do with this library, see tests in `dsp_toy_lib/src/lib.rs`.
