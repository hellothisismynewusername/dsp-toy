use std::{collections::VecDeque, f64::consts::PI, ops::{Add, Mul}};

use nalgebra::{Complex, ComplexField};

use crate::{math, real_time::real_time_signal_processer::RealTimeSignalProcessor};

/// Stateful Peak-bell IIR filter.
/// Use `FilterIIRPeakBell<f64>` for audio, otherwise you can use `FilterIIRPeakBell<Complex<f64>>`
pub struct FilterIIRPeakBell<T> {
    inps : VecDeque<T>,
    outs : VecDeque<T>,
    b_coeffs : Vec<T>,
    a_coeffs : Vec<T>
}

impl FilterIIRPeakBell<Complex<f64>> {
    /// For audio, use `FilterIIRPeakBell<f64>`
    /// Create a peak-bell filter 
    pub fn new(bands : &[(f64, f64, f64)], sample_rate : usize) -> Self {
        // get poles and zeroes from the bands
        let mut poles = Vec::with_capacity(bands.len() * 2);
        let mut zeroes = Vec::with_capacity(bands.len() * 2);
        for (band_freq, band_gain, band_q) in bands.iter() {
            let angle = 2. * PI * (*band_freq / sample_rate as f64);
            let thickness = angle.sin() / (2. * band_q);
            let amplitude = 10_f64.powf(band_gain / 40.);
            
            let pole_radius = f64::sqrt((1. - thickness / amplitude) / (1. + thickness / amplitude));
            let zero_radius = f64::sqrt((1. - thickness * amplitude) / (1. + thickness * amplitude));

            poles.push(Complex::<f64>::new(pole_radius * angle.cos(), pole_radius * angle.sin()));
            zeroes.push(Complex::<f64>::new(zero_radius * angle.cos(), zero_radius * angle.sin()));
        }

        // polynomial expansion with convolution
        let mut feedforward_coeffs = vec![Complex::<f64>::from(1.)];
        let mut feedback_coeffs = vec![Complex::<f64>::from(1.)];

        for zero in zeroes {
            let tmp_binomial = [Complex::<f64>::from(1.), Complex::<f64>::from(-1.) * zero];
            feedforward_coeffs = math::convolve(feedforward_coeffs.as_slice(), &tmp_binomial);
        }
        for pole in poles {
            let tmp_binomial = [Complex::<f64>::from(1.), Complex::<f64>::from(-1.) * pole];
            feedback_coeffs = math::convolve(feedback_coeffs.as_slice(), &tmp_binomial);
        }

        // remove imaginary components and normalization for both lists
        //let feedforward_first = feedforward_coeffs[0].real();
        let feedback_first = feedback_coeffs[0].real();
        let feedforward_coeffs_norm : Vec<Complex<f64>> = feedforward_coeffs
            .iter()
            .map(|x| *x / feedback_first)
            .collect();
        // also, flip signs for after-first entries in feedbacks and remove first entry
        let feedback_coeffs_norm : Vec<Complex<f64>> = feedback_coeffs
            .iter()
            .map(|x| Complex::<f64>::from(-1.) * *x / feedback_first)
            .enumerate()
            .filter(|(i, _)| *i > 0)
            .map(|(_, x)| x)
            .collect();

        let max_prev_inps = feedforward_coeffs_norm.len();
        let max_prev_outs = feedback_coeffs_norm.len();
        Self {
            inps: VecDeque::from(vec![Complex::<f64>::from(0.); max_prev_inps]),
            outs: VecDeque::from(vec![Complex::<f64>::from(0.); max_prev_outs]),
            b_coeffs: feedforward_coeffs_norm,
            a_coeffs: feedback_coeffs_norm
        }
    }
}

impl FilterIIRPeakBell<f64> {
    /// Audio-focused; negative frequencies should be expected to be mirrored counterparts.
    pub fn new_real(bands : &[(f64, f64, f64)], sample_rate : usize) -> Self {
        // get poles and zeroes from the bands, and their respective complex conjugates.
        let mut poles = Vec::with_capacity(bands.len() * 2);
        let mut zeroes = Vec::with_capacity(bands.len() * 2);
        for (band_freq, band_gain, band_q) in bands.iter() {
            let angle = 2. * PI * (*band_freq / sample_rate as f64);
            let thickness = angle.sin() / (2. * band_q);
            let amplitude = 10_f64.powf(band_gain / 40.);
            
            let pole_radius = f64::sqrt((1. - thickness / amplitude).clamp(0., f64::MAX) / (1. + thickness / amplitude));
            let zero_radius = f64::sqrt((1. - thickness * amplitude).clamp(0., f64::MAX) / (1. + thickness * amplitude));

            poles.push(Complex::<f64>::new(pole_radius * angle.cos(), pole_radius * angle.sin()));
            poles.push(Complex::<f64>::new(pole_radius * angle.cos(), -1. * pole_radius * angle.sin()));
            zeroes.push(Complex::<f64>::new(zero_radius * angle.cos(), zero_radius * angle.sin()));
            zeroes.push(Complex::<f64>::new(zero_radius * angle.cos(), -1. * zero_radius * angle.sin()));
        }

        // polynomial expansion with convolution
        let mut feedforward_coeffs = vec![Complex::<f64>::from(1.)];
        let mut feedback_coeffs = vec![Complex::<f64>::from(1.)];

        for zero in zeroes {
            let tmp_binomial = [Complex::<f64>::from(1.), Complex::<f64>::from(-1.) * zero];
            feedforward_coeffs = math::convolve(feedforward_coeffs.as_slice(), &tmp_binomial);
        }
        for pole in poles {
            let tmp_binomial = [Complex::<f64>::from(1.), Complex::<f64>::from(-1.) * pole];
            feedback_coeffs = math::convolve(feedback_coeffs.as_slice(), &tmp_binomial);
        }

        // manual sanity check, ensure there is no imaginary components in both lists
        // println!("feedforwards\t{:?}", feedforward_coeffs);
        // println!("feedbacks\t{:?}", feedback_coeffs);

        // remove imaginary components and normalization for both lists
        //let feedforward_first = feedforward_coeffs[0].real();
        let feedback_first = feedback_coeffs[0].real();
        let feedforward_coeffs_norm : Vec<f64> = feedforward_coeffs
            .iter()
            .map(|x| x.real())
            .map(|x| x / feedback_first)
            .collect();
        // also, flip signs for after-first entries in feedbacks and remove first entry
        let feedback_coeffs_norm : Vec<f64> = feedback_coeffs
            .iter()
            .map(|x| x.real())
            .map(|x| -1. * x / feedback_first)
            .enumerate()
            .filter(|(i, _)| *i > 0)
            .map(|(_, x)| x)
            .collect();

        let max_prev_inps = feedforward_coeffs_norm.len();
        let max_prev_outs = feedback_coeffs_norm.len();
        Self {
            inps: VecDeque::from(vec![0.; max_prev_inps]),
            outs: VecDeque::from(vec![0.; max_prev_outs]),
            b_coeffs: feedforward_coeffs_norm,
            a_coeffs: feedback_coeffs_norm
        }
    }
}

impl<T> RealTimeSignalProcessor<T, T> for FilterIIRPeakBell<T>
where 
    T: Add<Output = T> + Mul<Output = T> + PartialEq + Copy
{
    /// Performs an EQing filter given the next sample, returning the affected sample.
    /// - `bands.0` (band_frequency): 0 ≤ band_frequency ≤ NYQUIST_FREQ, in Hz
    /// - `bands.1` (band_gain): in Hz
    /// - `bands.2` (band_q): 0 < band_q
    fn process_sample(&mut self, val : T) -> T {
        self.inps.pop_front();
        self.inps.push_back(val);

        let inps = self.inps.make_contiguous().iter().as_slice();
        let outs = self.outs.make_contiguous().iter().as_slice();

        let tmp_b : T = self.b_coeffs
            .iter()
            .enumerate().map(|(i, val)| *val * inps[inps.len() - 1 - i])
            .reduce(|sum, x| sum + x)
            .unwrap();
        let tmp_a : T = self.a_coeffs
            .iter()
            .enumerate()
            .map(|(i, val)| *val * outs[outs.len() - 1 - i])
            .reduce(|sum, x| sum + x)
            .unwrap();

        let out = tmp_b + tmp_a;

        self.outs.pop_front();
        self.outs.push_back(out);

        out
    }
}
