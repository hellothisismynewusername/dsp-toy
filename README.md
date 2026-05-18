# dsp-toy

A small, highly unprofessional library for DSP, with simple functionalities for DFT, FFT, STFT, and other digital signal-related stuff.  
Provided is a `Signal` struct for owning and easily handling signal data, particularily through usage of the builder pattern (function chaining).  
This project's purpose is for learning, so efficiency, ergonomics, and thourough testing weren't prioritized.


## Requirements
- Rust & Cargo

## Usage
Add the library crate to your project like so:
```
[dependencies]
# (...other dependencies)
dsp_toy_lib = { git = "https://github.com/hellothisismynewusername/dsp-toy.git" }
```
(Or whatever my username is currently)

Then make a `dsp_toy_lib::signal::Signal` using `Signal::from()` (array, slice, or iterator).

For demos of what you can do with this library, see tests in `src/dsp-toy-lib/lib.rs`.