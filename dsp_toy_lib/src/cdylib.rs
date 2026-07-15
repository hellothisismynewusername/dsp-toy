// Expose math functions to C ABI. Complex values are represented as split real and imaginary `f64` arrays.

use std::slice;

use nalgebra::{Complex, ComplexField};

use crate::math;

/// Compute convolution without transforming signals to frequency domain, writing the convolved output to `a` and returning a success flag. Generally less performant than FFT with multiplication.
/// 
/// Real only.
///
/// `a` must point to at least `a_len + b_len - 1` writable elements.
/// 
/// If `false` is returned, one of the provided pointers was null or one of the lengths was zero.
#[unsafe(no_mangle)]
pub extern "C" fn convolve_real(a : *mut f64, a_len : usize, b : *const f64, b_len : usize) -> bool {
    if a.is_null() || b.is_null() || a_len == 0 || b_len == 0 {
        return false;
    }
    unsafe {
        let out_len = match a_len.checked_add(b_len - 1) {
            Some(len) => len,
            None => return false,
        };
        let a_input = slice::from_raw_parts(a, a_len);
        let a_out = slice::from_raw_parts_mut(a, out_len);
        let a_comp : Vec<Complex<f64>> = a_input.iter().map(|x| Complex::<f64>::from(*x)).collect();
        let b_comp : Vec<Complex<f64>> = slice::from_raw_parts(b, b_len).iter().map(|x| Complex::<f64>::from(*x)).collect();

        let convolved = math::convolve(&a_comp, &b_comp);
        for (i, val) in convolved.iter().map(|x| x.real()).enumerate() {
            a_out[i] = val;
        }

        return true;
    }
}

/// Compute convolution without transforming signals to frequency domain, writing the convolved output to `a_real` and `a_imag` and returning a success flag. Generally less performant than FFT with multiplication.
///
/// `a_real` and `a_imag` must point to at least `a_len + b_len - 1` writable elements.
/// 
/// If `false` is returned, one of the provided pointers was null or one of the lengths was zero.
#[unsafe(no_mangle)]
pub extern "C" fn convolve_complex(a_real : *mut f64, a_imag : *mut f64, a_len : usize, b_real : *const f64, b_imag : *mut f64, b_len : usize) -> bool {
    if a_real.is_null() || a_imag.is_null() || b_real.is_null() || b_imag.is_null() || a_len == 0 || b_len == 0 {
        return false;
    }
    unsafe {
        let out_len = match a_len.checked_add(b_len - 1) {
            Some(len) => len,
            None => return false,
        };
        let a_real_input = slice::from_raw_parts(a_real, a_len);
        let a_imag_input = slice::from_raw_parts(a_imag, a_len);
        let a_real_out = slice::from_raw_parts_mut(a_real, out_len);
        let a_imag_out = slice::from_raw_parts_mut(a_imag, out_len);
        let a_comp : Vec<Complex<f64>> = a_real_input
            .iter()
            .zip(a_imag_input.iter())
            .map(|(real, imag)| Complex::<f64>::new(*real, *imag))
            .collect();
        let b_real_input = slice::from_raw_parts(b_real, b_len);
        let b_imag_input = slice::from_raw_parts(b_imag, b_len);
        let b_comp : Vec<Complex<f64>> = b_real_input
            .iter()
            .zip(b_imag_input.iter())
            .map(|(real, imag)| Complex::<f64>::new(*real, *imag))
            .collect();

        let convolved = math::convolve(&a_comp, &b_comp);
        for (i, (real, imag)) in convolved.iter().map(|x| (x.real(), x.imaginary())).enumerate() {
            a_real_out[i] = real;
            a_imag_out[i] = imag;
        }

        return true;
    }
}

/// Compute Discrete Fourier Transform in-place, reading and writing real values through `data` and imaginary values through `imag`, and returning a success flag.
/// 
/// `real` and `imag` are expected to be of the same length.
/// 
/// If `false` is returned, one of the provided pointers was null.
#[unsafe(no_mangle)]
pub extern "C" fn dft(real : *mut f64, imag : *mut f64, len : usize) -> bool {
    if real.is_null() || imag.is_null() {
        return false;
    }
    unsafe {
        let real_slice = slice::from_raw_parts_mut(real, len);
        let imag_slice = slice::from_raw_parts_mut(imag, len);
        let data_comp : Vec<Complex<f64>> = real_slice
            .iter()
            .zip(imag_slice.iter())
            .map(|(r, i)| Complex::<f64>::new(*r, *i))
            .collect();

        let dft = math::dft(&data_comp);

        for (i, val) in dft.iter().enumerate() {
            real_slice[i] = val.real();
            imag_slice[i] = val.imaginary();
        }

        return true;
    }
}

/// Compute Inverse Discrete Fourier Transform in-place, reading and writing real values through `real` and imaginary values through `imag`, and returning a success flag.
/// 
/// `real` and `imag` are expected to be of the same length.
/// 
/// If `false` is returned, one of the provided pointers was null.
#[unsafe(no_mangle)]
pub extern "C" fn idft(real : *mut f64, imag : *mut f64, len : usize) -> bool {
    if real.is_null() || imag.is_null() {
        return false;
    }
    unsafe {
        let real_slice = slice::from_raw_parts_mut(real, len);
        let imag_slice = slice::from_raw_parts_mut(imag, len);
        let data_comp : Vec<Complex<f64>> = real_slice
            .iter()
            .zip(imag_slice.iter())
            .map(|(r, i)| Complex::<f64>::new(*r, *i))
            .collect();

        let idft = math::idft(&data_comp);

        for (i, val) in idft.iter().enumerate() {
            real_slice[i] = val.real();
            imag_slice[i] = val.imaginary();
        }

        return true;
    }
}

/// Compute Radix-2 FFT in-place using Bit-reversal permutation, reading and writing real values through `real` and imaginary values through `imag`, and returning a success flag.
/// 
/// `real` and `imag` are expected to be of the same length.
/// 
/// If `false` is returned, one of the provided pointers was null.
#[unsafe(no_mangle)]
pub extern "C" fn r2fft(real : *mut f64, imag : *mut f64, len : usize) -> bool {
    if real.is_null() || imag.is_null() {
        return false;
    }
    unsafe {
        let real_slice = slice::from_raw_parts_mut(real, len);
        let imag_slice = slice::from_raw_parts_mut(imag, len);
        let mut data_comp : Vec<Complex<f64>> = real_slice
            .iter()
            .zip(imag_slice.iter())
            .map(|(r, i)| Complex::<f64>::new(*r, *i))
            .collect();

        math::r2fft(&mut data_comp);

        for (i, val) in data_comp.iter().enumerate() {
            real_slice[i] = val.real();
            imag_slice[i] = val.imaginary();
        }

        return true;
    }
}

/// Compute Inverse Radix-2 FFT in-place using Bit-reversal permutation, reading and writing real values through `real` and imaginary values through `imag`, and returning a success flag.
/// 
/// `real` and `imag` are expected to be of the same length.
/// 
/// If `false` is returned, one of the provided pointers was null.
#[unsafe(no_mangle)]
pub extern "C" fn ir2fft(real : *mut f64, imag : *mut f64, len : usize) -> bool {
    if real.is_null() || imag.is_null() {
        return false;
    }
    unsafe {
        let real_slice = slice::from_raw_parts_mut(real, len);
        let imag_slice = slice::from_raw_parts_mut(imag, len);
        let mut data_comp : Vec<Complex<f64>> = real_slice
            .iter()
            .zip(imag_slice.iter())
            .map(|(r, i)| Complex::<f64>::new(*r, *i))
            .collect();

        math::ir2fft(&mut data_comp);

        for (i, val) in data_comp.iter().enumerate() {
            real_slice[i] = val.real();
            imag_slice[i] = val.imaginary();
        }

        return true;
    }
}
