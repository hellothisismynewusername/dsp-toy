use std::f64::consts::PI;

use nalgebra::{Complex, RealField};

use crate::{consts::{BLACKMAN_WINDOW_C_0, BLACKMAN_WINDOWS_C_1, BLACKMAN_WINDOWS_C_2, EULER, HAMMING_WINDOW_C_0, HAMMING_WINDOW_C_1}, utility::j};

/// Compute convolution without transforming signals to frequency domain, producing the convolved output. Generally less performant than FFT with multiplication.
pub fn convolve(data : &[Complex<f64>], ir : &[Complex<f64>]) -> Vec<Complex<f64>> {
    let mut out: Vec<_> = vec![Complex::<f64>::from(0.); data.len() + ir.len() - 1];
    for k in 0..out.len() {
        let sum  = data.iter().enumerate().map(|(i, val)| {
            let ir_val = *k
                .checked_sub(i)
                .and_then(|ind| ir.get(ind))
                .unwrap_or(&Complex::<f64>::from(0.));

            *val * ir_val
        }).reduce(|sum, x| {
            sum + x
        }).unwrap_or(Complex::<f64>::from(0.));
        out[k] = sum;
    }

    out
}

// ------------------- FOURIER

/// Compute DFT.
pub fn dft(data : &[Complex<f64>]) -> Vec<Complex<f64>> {
    let mut data_tmp = Vec::with_capacity(data.len());

    for k in 0..data.len() {
        // polar form calculation, but signal will be in cartesian
        let tmp = data.iter().enumerate().map(|(n, val)| {
            *val * Complex::<f64>::from(EULER).powc(
                    Complex::<f64>::from(-1.) * j() * (Complex::<f64>::from(2. * PI) / Complex::<f64>::from(data.len() as f64)) * Complex::<f64>::from(k as f64) * Complex::<f64>::from(n as f64)
                )
        }).reduce(|sum, x| {
            sum + x
        });

        data_tmp.push(tmp.unwrap());
    }

    data_tmp
}

/// Compute Inverse DFT.
pub fn idft(data : &[Complex<f64>]) -> Vec<Complex<f64>> {
    let mut data_tmp : Vec<Complex<f64>> = Vec::with_capacity(data.len());

    for n in 0..data.len() {
        let tmp = data.iter().enumerate().map(|(k, val)| {
            *val * Complex::<f64>::from(EULER).powc(
                j() * (Complex::<f64>::from(2. * PI) / Complex::<f64>::from(data.len() as f64)) * Complex::<f64>::from(k as f64) * Complex::<f64>::from(n as f64)
            )
        }).reduce(|sum, x| {
            sum + x
        }).unwrap() / Complex::<f64>::from(data.len() as f64);

        data_tmp.push(tmp);
    }

    data_tmp
}

/// Compute Radix-2 FFT, using Bit-reversal permutation.
pub fn r2fft(data : &mut [Complex<f64>]) {
    if data.len() > 1 {
        let n = data.len();

        let bits_num = n.trailing_zeros();
        for i in 0..n {
            let ind_new = i.reverse_bits() >> (usize::BITS - bits_num);
            if i < ind_new {
                data.swap(i, ind_new);
            }
        }

        for s in 1..=n.ilog2() {
            let block_len = 1 << s; // same as 2^s
            let half_len = block_len / 2; // 2^(s - 1)

            // step by block_len
            for k in (0..n).step_by(block_len) {
                for j in 0..half_len {
                    let ind_a = k + j;
                    let ind_b = ind_a + half_len;

                    let angle = (2. * PI * j as f64) / block_len as f64;
                    let twiddle_n = Complex::<f64>::new(angle.cos(), -1. * angle.sin());

                    let val_a = data[ind_a];
                    let val_b_scaled = data[ind_b] * twiddle_n;

                    data[ind_a] = val_a + val_b_scaled;
                    data[ind_b] = val_a - val_b_scaled;
                }
            }
        }
    }
}

/// Compute Inverse Radix-2 FFT, using Bit-reversal permutation.
pub fn ir2fft(data : &mut [Complex<f64>]) {
    if data.len() > 1 {
        let n = data.len();

        let bits_num = n.trailing_zeros();
        for i in 0..n {
            let ind_new = i.reverse_bits() >> (usize::BITS - bits_num);
            if i < ind_new {
                data.swap(i, ind_new);
            }
        }

        for s in 1..=n.ilog2() {
            let block_len = 1 << s; // same as 2^s
            let half_len = block_len / 2; // 2^(s - 1)

            // step by block_len
            for k in (0..n).step_by(block_len) {
                for j in 0..half_len {
                    let ind_a = k + j;
                    let ind_b = ind_a + half_len;

                    let angle = (2. * PI * j as f64) / block_len as f64;
                    let twiddle_n = Complex::<f64>::new(angle.cos(), angle.sin());

                    let val_a = data[ind_a];
                    let val_b_scaled = data[ind_b] * twiddle_n;

                    data[ind_a] = val_a + val_b_scaled;
                    data[ind_b] = val_a - val_b_scaled;
                }
            }
        }

        data.iter_mut().for_each(|x| *x = *x / Complex::<f64>::from(n as f64));
    }
}

pub fn r2fft_legacy(data : &[Complex<f64>]) -> Vec<Complex<f64>> {
    if data.len() <= 1 {
        Vec::from(data)
    } else {
        let n = data.len();

        let evens : Box<[Complex<f64>]> = data.iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, x)| *x)
            .collect();
        let odds : Box<[Complex<f64>]>  = data.iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 1)
            .map(|(_, x)| *x)
            .collect();

        let evens_fft = r2fft_legacy(evens.iter().as_slice());
        let odds_fft = r2fft_legacy(odds.iter().as_slice());

        let angle = (-2. * PI) / n as f64;
        let twiddle_n = Complex::<f64>::new(angle.cos(), angle.sin());

        let mut twiddle = Complex::<f64>::from(1.);
        let mut out = vec![Complex::<f64>::from(0.); n];

        for j in 0..(n / 2) {
            out[j] = evens_fft[j] + twiddle * odds_fft[j];
            out[j + n / 2] = evens_fft[j] - twiddle * odds_fft[j];
            twiddle = twiddle * twiddle_n;
        }

        out
    }
}

pub fn ir2fft_legacy(data : &[Complex<f64>]) -> Vec<Complex<f64>> {
    if data.len() <= 1 {
        Vec::from(data)
    } else {
        let n = data.len();

        let evens_fft : Box<[Complex<f64>]> = data.iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 0)
            .map(|(_, x)| *x)
            .collect();
        let odds_fft : Box<[Complex<f64>]>  = data.iter()
            .enumerate()
            .filter(|(i, _)| i % 2 == 1)
            .map(|(_, x)| *x)
            .collect();

        let evens = ir2fft_legacy(evens_fft.iter().as_slice());
        let odds = ir2fft_legacy(odds_fft.iter().as_slice());

        let angle = (-2. * PI) / n as f64;
        let twiddle_n_bar = Complex::<f64>::new(angle.cos(), -1. * angle.sin());

        let mut twiddle_bar = Complex::<f64>::from(1.);
        let mut out = vec![Complex::<f64>::from(0.); n];

        for j in 0..(n / 2) {
            out[j] = (evens[j] + twiddle_bar * odds[j]) / Complex::<f64>::from(2.);
            out[j + n / 2] = (evens[j] - twiddle_bar * odds[j]) / Complex::<f64>::from(2.);
            twiddle_bar = twiddle_bar * twiddle_n_bar;
        }

        out
    }
}

// ----------- WINDOW FUNCTIONS

pub fn hann(n : usize, len : usize) -> f64 {
    0.5 - 0.5 * ((2. * PI * n as f64) / len as f64).cos()
}

pub fn hamming(n : usize, len : usize) -> f64 {
    HAMMING_WINDOW_C_0 - HAMMING_WINDOW_C_1 * ((2. * PI * n as f64) / len as f64).cos()
}

pub fn rectangular(_ : usize, _ : usize) -> f64 {
    1_f64
}

pub fn blackman(n : usize, len : usize) -> f64 {
    BLACKMAN_WINDOW_C_0 - BLACKMAN_WINDOWS_C_1 * ((2. * PI * n as f64) / len as f64).cos() + BLACKMAN_WINDOWS_C_2 * ((4. * PI * n as f64) / len as f64).cos()
}


// ----------- MISC.


pub fn normalize_angle<T>(inp : T) -> T 
where 
    T: RealField + Copy
{
    let tmp = inp % (T::from_i8(2).unwrap() * T::pi());
    if tmp > T::pi() {
        tmp - T::two_pi()
    } else if tmp < T::from_i8(-1).unwrap() * T::pi() {
        tmp + T::two_pi()
    } else {
        tmp
    }
}