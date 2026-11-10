# dsp-toy

A small, highly unprofessional WIP library for signal processing, with simple functionalities for DFT, FFT, STFT, FIR/IIR filtering, Multivariate Linear/Unscented Kalman filtering, and other sampled signal-related stuff.  
Provided is a `Signal` struct for owning and easily handling signal data, particularily through usage of the builder pattern.  
Also present are real-time filters that implement `RealTimeSignalProcessor`.  
This project's purpose is for learning, so efficiency was not prioritized. It is also misnamed, this library is generally designed for sampled analog signals instead of digital.


## Requirements
- Rust & Cargo

## Usage
This repo is a Cargo workspace with:
- `dsp_toy_lib` (library crate)
- `dsp_toy` (binary crate)

And non-rust library usage examples found under [`examples`](examples)

### Cargo

To use the library as a git dependency, add:
```
[dependencies]
# (...other dependencies)
dsp_toy_lib = { git = "https://github.com/hellothisismynewusername/dsp-toy.git", package = "dsp_toy_lib" }
```
(Or whatever my username is currently)

### Non-Rust

C ABI compatible structs and functions are found in [`dsp_toy_lib/src/cdylib.rs`](dsp_toy_lib/src/cdylib.rs) and [`dsp_toy_lib/src/real_time/cdylib.rs`](dsp_toy_lib/src/real_time/cdylib.rs).
Currently only supports `Signal`, `FilterIIRPeakBell_f64`, and DFT-related functions; no Kalman Filtering.

To use the library as a cdylib, build the library with `cargo build -p dsp_toy_lib --release` and make headers with `cbindgen`.

--- 

For demos of what you can do with this library, see tests in [`dsp_toy_lib/tests`](dsp_toy_lib/tests), foreign examples in [`examples`](examples), and code in the [`dsp_toy`](dsp_toy) binary.

### Notable Structs
- `FilterKalmanLinear`, `FilterKalmanLinearComplex`, `FilterKalmanUnscented` - Kalman Filtering, real-time. Currently, `FilterKalmanLinear` returns innovation and returns Result instead of panicking, as opposed to the other two which can panic.
- `Julier` - Sigma points generator function, for KF.
- `Signal` - Offline processing, implements several DFT-related functionalities.
- `FilterIIRPeakBell` - EQing-like peak-bell filtering, real-time.
