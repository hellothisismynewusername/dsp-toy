use std::{collections::VecDeque, ptr::{self, null_mut}, slice};

use crate::{math, signal::signal::Signal};

/// Instantiates a `Signal` and provides an opaque pointer.
/// If you intend to work with real-only signals, you may want to use the exposed math functions that work directly with pointers to `f64`.
#[unsafe(no_mangle)]
pub extern "C" fn new_Signal() -> *mut Signal {
    Box::into_raw(Box::new(Signal::new()))
}

/// Instantiates a `Signal` given the provided data, and returns an opaque pointer.
/// If you intend to work with real-only signals, you may want to use the exposed math functions that work directly with pointers to `f64`.
#[unsafe(no_mangle)]
pub extern "C" fn new_from_ptr_Signal(ptr : *const f64, len : usize) -> *mut Signal {
    if ptr.is_null() {
        return null_mut();
    }
    unsafe {
        let data = slice::from_raw_parts(ptr, len);
        return Box::into_raw(Box::from(Signal::from(data)));
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn free_Signal(ptr : *mut Signal) {
    if !ptr.is_null() {
        unsafe {
            let _tmp = Box::from_raw(ptr);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn len_Signal(signal : *const Signal) -> usize {
    if signal.is_null() {
        return usize::MAX;
    }
    unsafe {
        return (*signal).len();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn real_data_Signal(signal : *const Signal, out : *mut f64, out_len : usize) -> bool {
    if signal.is_null() || out.is_null() {
        return false;
    }
    unsafe {
        if out_len < (*signal).len() {
            return false;
        }
        let out_slice = slice::from_raw_parts_mut(out, out_len);
        for (i, val) in (*signal).iter_real().enumerate() {
            out_slice[i] = val;
        }
        return true;
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn imag_data_Signal(signal : *const Signal, out : *mut f64, out_len : usize) -> bool {
    if signal.is_null() || out.is_null() {
        return false;
    }
    unsafe {
        if out_len < (*signal).len() {
            return false;
        }
        let out_slice = slice::from_raw_parts_mut(out, out_len);
        for (i, val) in (*signal).iter_imag().enumerate() {
            out_slice[i] = val;
        }
        return true;
    }
}

/// Resamples by given ratio, returning an error code where `0` means success.
#[unsafe(no_mangle)]
pub extern "C" fn resample_ratio_Signal(signal : *mut *mut Signal, ratio : f64) -> i32 {
    if signal.is_null() {
        return -1;
    }
    if unsafe {
        (*signal).is_null()
    } {
        return -2;
    }

    unsafe {
        let old_ptr = *signal;
        let signal_owned = Box::from_raw(old_ptr);
        let resampled = match signal_owned.resample(Some(ratio), None) {
            Ok(x) => x,
            Err(_) => {
                *signal = old_ptr;
                return -3;
            }
        };

        *signal = Box::into_raw(Box::new(resampled));
        return 0;
    }
}

/// Resamples to given length, returning an error code where `0` means success.
#[unsafe(no_mangle)]
pub extern "C" fn resample_length_Signal(signal : *mut *mut Signal, length : usize) -> i32 {
    if signal.is_null() {
        return -1;
    }
    if unsafe {
        (*signal).is_null()
    } {
        return -2;
    }

    unsafe {
        let old_ptr = *signal;
        let signal_owned = Box::from_raw(old_ptr);
        let resampled = match signal_owned.resample(None, Some(length)) {
            Ok(x) => x,
            Err(_) => {
                *signal = old_ptr;
                return -3;
            }
        };

        *signal = Box::into_raw(Box::new(resampled));
        return 0;
    }
}

/// Returns a dynamic array of `Signal`s, that being the original `signal` split into windows with `window_function` applied to it, whose length is `windows_num`.
/// 
/// `window_function`:
/// - 0: Hann window
/// - 1: Hamming window
/// - 2: Rectangular window
/// - 3: Blackman window
/// 
/// Otherwise, Rectangular window by default.
#[unsafe(no_mangle)]
pub extern "C" fn windows_Signal(
    signal : *const Signal,
    window_size : usize,
    hop_size : usize,
    windows_num : usize,
    window_function : u8,
    symmetric : bool
) -> *mut Signal {
    if signal.is_null() {
        return null_mut();
    }
    let window_f = match window_function {
        0 => math::hann,
        1 => math::hamming,
        2 => math::rectangular,
        3 => math::blackman,
        _ => math::rectangular
    };

    unsafe {
        let mut windows : VecDeque<Signal> = (*signal).windows(window_size, hop_size, windows_num, window_f, symmetric);
        Box::into_raw(windows.make_contiguous().to_vec().into_boxed_slice()) as *mut Signal
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn free_windows_Signal(windows : *mut Signal, windows_len : usize) {
    if !windows.is_null() {
        unsafe {
            drop(Box::from_raw(ptr::slice_from_raw_parts_mut(windows, windows_len)));
        }
    }
}

/// Returns a `Signal` made up of (overlapping) window chunk `Signal`s.
#[unsafe(no_mangle)]
pub extern "C" fn reconstruct_Signal(chunks : *const Signal, chunks_len : usize, hop_size : usize) -> *mut Signal {
    if chunks.is_null() {
        return null_mut();
    }
    unsafe {
        let chunks_slice = slice::from_raw_parts(chunks, chunks_len);
        let tmp = Signal::reconstruct(chunks_slice, hop_size);
        return Box::into_raw(Box::new(tmp));
    }
}