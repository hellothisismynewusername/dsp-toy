use std::{collections::VecDeque, f64::consts::PI};

use easy_complex::{Complex64};
use crate::{live_signal_processer::{FilterIIRPeakBell, LiveSignalProcessor}, math};

#[derive(Debug, Clone)]
pub struct Signal {
    pub data : Vec<Complex64>,
}

impl Signal {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
        }
    }

    pub fn from_fn_mut(function : &mut dyn FnMut(usize) -> Complex64, start : usize, end : usize) -> Self {
        let mut tmp = Vec::new();
        for i in start..end {
            tmp.push(function(i));
        }

        Self { data: tmp }
    }

    pub fn from_fn(function : &dyn Fn(usize) -> Complex64, start : usize, end : usize) -> Self {
        let mut tmp = Vec::new();
        for i in start..end {
            tmp.push(function(i));
        }

        Self { data: tmp }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Convolves with signal `other`, consuming and returning `self`.
    pub fn convolve(mut self, other : &Self) -> Self {
        Self { data: math::convolve(&mut self.data, &other.data) }
    }

    /// Performs DFT, producing a frequency domain `Signal`; does not modify/consume `self`.
    pub fn forward_dft(&self) -> Self {
        Self { data: math::dft(&self.data) }
    }

    /// Performs Radix-2 FFT, producing a frequency domain `Signal`; consumes and returns `self`. Fails if length is not a power of 2.
    pub fn radix_2_fft(mut self) -> Result<Self, &'static str> {
        match self.len().is_power_of_two() {
            true => {
                math::r2fft(&mut self.data);
                Ok(self)
            },
            false => Err("Error: Cannot perform radix_2_fft on a signal of length not a power of 2.")
        }
    }

    /// Performs Radix-2 FFT, producing a frequency domain `Signal`; does not modify/consume `self`. Fails if length is not a power of 2.
    pub fn radix_2_fft_new(&self) -> Result<Self, &'static str> {
        match self.len().is_power_of_two() {
            true => {
                let mut data_new = self.data.clone();
                math::r2fft(&mut data_new);
                Ok(Self { data: data_new })
            },
            false => Err("Error: Cannot perform radix_2_fft_new on a signal of length not a power of 2.")
        }
    }

    /// Performs Inverse DFT, producing a time domain `Signal`; does not modify/consume `self`.
    pub fn inverse_dft(&self) -> Self {
        Self { data: math::idft(&self.data) }
    }

    /// Performs Inverse Radix-2 FFT; consumes and returns `self`. Fails if length is not a power of 2.
    pub fn inverse_radix_2_fft(mut self) -> Result<Self, &'static str> {
        match self.len().is_power_of_two() {
            true => {
                math::ir2fft(&mut self.data);
                Ok(self)
            },
            false => Err("Error: Cannot perform inverse_radix_2_fft on a signal of length not a power of 2.")
        }
    }

    /// Performs Inverse Radix-2 FFT, producing a frequency domain `Signal`; does not modify/consume `self`. Fails if length is not a power of 2.
    pub fn inverse_radix_2_fft_new(&self) -> Result<Self, &'static str> {
        match self.len().is_power_of_two() {
            true => {
                let mut data_new = self.data.clone();
                math::ir2fft(&mut data_new);
                Ok(Self { data: data_new })
            },
            false => Err("Error: Cannot perform inverse_radix_2_fft_new on a signal of length not a power of 2.")
        }
    }

    pub fn zero_extend_start_and_end(self, num : usize) -> Self {
        let mut data_tmp : VecDeque<Complex64> = self.data.into();
        for _ in 0..num {
            data_tmp.push_front(Complex64::from(0));
            data_tmp.push_back(Complex64::from(0));
        }
        Self { data: data_tmp.into() }
    }

    pub fn zero_extend_end(mut self, num : usize) -> Self {
        for _ in 0..num {
            self.data.push(Complex64::from(0));
        }
        self
    }

    pub fn zero_extend_end_mut(&mut self, num : usize) -> &Self {
        for _ in 0..num {
            self.data.push(Complex64::from(0));
        }
        self
    }

    pub fn zero_extend_start(self, num : usize) -> Self {
        let mut data_tmp : VecDeque<Complex64> = self.data.into();
        for _ in 0..num {
            data_tmp.push_front(Complex64::from(0));
        }
        Self { data: data_tmp.into() }
    }

    pub fn zero_extend_start_mut(&mut self, num : usize) -> &Self {
        for _ in 0..num {
            self.data.insert(0, Complex64::from(0));
        }
        self
    }

    pub fn iter_real(&self) -> impl Iterator<Item = f64> {
        self.data.iter().map(|x| x.real())
    }

    pub fn iter_imag(&self) -> impl Iterator<Item = f64> {
        self.data.iter().map(|x| x.imag())
    }

    pub fn iter_complex(&self) -> impl Iterator<Item = Complex64> {
        self.data.iter().map(|x| *x)
    }

    /// Crops the signal by `start` and `end`, modifying `self` and returning the cropped Signal.
    pub fn crop(mut self, start : usize, end : usize) -> Self {
        self.data.truncate(end);
        self.data.drain(..start);
        self
    }

    /// Produces a new `Signal` with a copy of the original data from `start` to `end`.
    pub fn crop_new(&self, start : usize, end : usize) -> Self {
        Self { data: self[start..end].to_vec() }
    }

    /// Applies `window_function`, mutating `self`.
    pub fn apply_window(mut self, window_function : impl Fn(usize, usize) -> f64, symmetric : bool) -> Self {
        let len = if symmetric {
            self.len() - 1
        } else {
            self.len()
        };
        for (i, val) in self.data.iter_mut().enumerate() {
            *val = *val * Complex64::from(window_function(i, len));
        }
        self
    }

    /// Concatenates `other` to `self`, returning the concatenated signal.
    pub fn concat(mut self, other : &Signal) -> Self {
        self.data.extend_from_slice(&other.data);
        self
    }

    /// Overlaps `other` onto `self`, performing element-wise addition beginning at index `offset`. 
    /// 
    /// Modifies and consumes `self` and returns the final signal.
    pub fn overlap(mut self, other : &Signal, offset : usize) -> Self {
        let overlap_len = self.len() - offset;
        self.data.resize(self.len() + (other.len() - overlap_len), Complex64::from(0));

        for i in offset..self.data.len() {
            self[i] = self[i] + other[i - offset];
        }

        self
    }

    /// Returns a `VecDeque<Signal>` containing windowed segments of the signal.
    pub fn windows(
        &self,
        window_size : usize,
        hop_size : usize,
        windows_num : usize,
        window_function : impl Fn(usize, usize) -> f64,
        symmetric : bool
    ) -> VecDeque<Signal> {
        let mut out = VecDeque::new();
        for i in 0..windows_num {
            out.push_back(self.crop_new(hop_size * i, hop_size * i + window_size).apply_window(&window_function, symmetric));
        }
        out
    }

    /// Produces a `Signal` from (overlapping) window `Signal`s.
    pub fn reconstruct(chunks : &[Signal], hop_size : usize) -> Self {
        let mut out = Signal::new();

        for (i, chunk) in chunks.iter().enumerate() {
            out = out.overlap(chunk, hop_size * i);
        }

        out
    }

    /// Resample to a new length with sinc interpolation by converting to frequency domain. 
    pub fn resample(self, ratio : Option<f64>, new_len : Option<usize>) -> Result<Self, &'static str> {
        if (ratio.is_none() && new_len.is_none()) || (ratio.is_some() && new_len.is_some()) {
            Err("Error: resampling where both ratio and new_len were supplied")
        } else {
            let n = self.len();

            let len = if ratio.is_some() { // use ratio
                (ratio.unwrap() * self.len() as f64).round() as usize
            } else { // use new_len
                new_len.unwrap()
            };

            let zeroes = len as i64 - self.len() as i64;

            if zeroes == 0 {
                return Ok(self);
            }

            let mut freq = if self.len().is_power_of_two() {
                self.radix_2_fft().expect("Error: resampling unexpected error during fft")
            } else {
                self.forward_dft()
            };

            let out = if zeroes > 0 {
                if freq.len() % 2 == 0 {
                    let nyquist_bin = freq.len() / 2;
                    let nyquist = freq[nyquist_bin];
                    freq[nyquist_bin] = nyquist / Complex64::from(2.);

                    for _ in 0..zeroes {
                        freq.data.insert(nyquist_bin + 1, Complex64::from(0.));
                    }

                    freq[nyquist_bin + zeroes as usize] = nyquist / Complex64::from(2.);
                    
                    let gain_mult = 1. + (zeroes as f64 / n as f64);

                    if freq.len().is_power_of_two() {
                        freq.inverse_radix_2_fft()? * gain_mult
                    } else {
                        freq.inverse_dft() * gain_mult
                    }
                } else {
                    let halfway = freq.len() / 2;
                    for _ in 0..zeroes {
                        freq.data.insert(halfway + 1, Complex64::from(0.));
                    }

                    let gain_mult = 1. + (zeroes as f64 / n as f64);

                    freq.inverse_dft() * gain_mult
                }
            } else {
                if len % 2 == 0 {
                    let k = len / 2 - 1;

                    let pos_last = freq[k + 1];
                    let neg_first = freq[n - k - 1];

                    for _ in 0..zeroes.abs() {
                        freq.data.remove(k + 1);
                    }

                    freq[k + 1] = pos_last + neg_first;

                    let gain_mult = 1. + (zeroes as f64 / n as f64);

                    if freq.len().is_power_of_two() {
                        freq.inverse_radix_2_fft()? * gain_mult
                    } else {
                        freq.inverse_dft() * gain_mult
                    }
                } else {
                    let k = (len - 1) / 2; // how many bins on each side to keep.

                    for _ in 0..zeroes.abs() {
                        freq.data.remove(k + 1);
                    }

                    let gain_mult = 1. + (zeroes as f64 / n as f64);

                    freq.inverse_dft() * gain_mult
                }
            };

            Ok(out)
        }
    }

    /// Performs an EQing filter, returning the new filtered signal. Real only, so this is useful for audio signals.
    /// - `bands.0` (band_frequency): 0 ≤ band_frequency ≤ NYQUIST_FREQ, in Hz
    /// - `bands.1` (band_gain): in Hz
    /// - `bands.2` (band_q): 0 < band_q
    pub fn iir_filter_peak_bell_real(&self, bands : &[(f64, f64, f64)], sample_rate : usize) -> Self {
        let mut filter = FilterIIRPeakBell::new_real(bands, sample_rate);
        let mut data = Vec::with_capacity(self.len());
        for sample in self.iter_real() {
            data.push(Complex64::from(filter.process_sample(sample)))
        }
        Self {
            data: data
        }
    }

    /// Performs an EQing filter, returning the new filtered signal, use real version for audio signals.
    /// - `bands.0` (band_frequency): 0 ≤ band_frequency ≤ NYQUIST_FREQ, in Hz
    /// - `bands.1` (band_gain): in Hz
    /// - `bands.2` (band_q): 0 < band_q
    pub fn iir_filter_peak_bell_imag(&self, bands : &[(f64, f64, f64)], sample_rate : usize) -> Self {
        let mut filter = FilterIIRPeakBell::new(bands, sample_rate);
        let mut data = Vec::with_capacity(self.len());
        for sample in self.iter_complex() {
            data.push(filter.process_sample(sample))
        }
        Self {
            data: data
        }
    }
}