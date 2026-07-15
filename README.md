# dsp-toy

A small, highly unprofessional WIP library for DSP, with simple functionalities for DFT, FFT, STFT, FIR/IIR filtering, Multivariate Linear Kalman filtering, and other digital signal-related stuff.  
Provided is a `Signal` struct for owning and easily handling signal data, particularily through usage of the builder pattern (function chaining).  
Also present are real-time filters that implement `RealTimeSignalProcessor`.  
This project's purpose is for learning, so efficiency, ergonomics, and thourough testing weren't prioritized.


## Requirements
- Rust & Cargo

## Usage
This repo is a Cargo workspace with:
- `dsp_toy` (binary crate)
- `dsp_toy_lib` (library crate)
With non-rust library usage examples found under `/examples`

### Cargo

To use the library as a git dependency, add:
```
[dependencies]
# (...other dependencies)
dsp_toy_lib = { git = "https://github.com/hellothisismynewusername/dsp-toy.git", package = "dsp_toy_lib" }
```
(Or whatever my username is currently)

Then make a `dsp_toy_lib::signal::Signal` using `Signal::from()` (array, slice, or iterator).

### Non-Rust

C ABI compatible structs and functions are found in `dsp_toy_lib/src/cdylib_stuff.rs`. Currently only supports `FilterIIRPeakBell_f64`.

To use the library as a cdylib, build the library with `cargo build -p dsp_toy_lib --release` and make headers with `cbindgen`.

--- 

For demos of what you can do with this library, see tests in `dsp_toy_lib/src/lib.rs`, foreign examples in `examples`, and code in the `dsp_toy` binary.