use std::{collections::VecDeque, f64::consts::PI, fmt::Display, ops::{Add, Mul, Sub}, process::Output};

use easy_complex::{Complex, Complex64};
use crate::consts::{EQUALITY_ACCURACY, EULER, HAMMING_WINDOW_ALPHA, HAMMING_WINDOW_BETA};

#[derive(Debug, Clone)]
pub struct Signal {
    pub data : Vec<Complex64>,
}

impl Signal {
    pub fn new() -> Signal {
        Signal {
            data: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Performs DFT, producing a frequency domain `Signal`; does not modify/consume `self`.
    pub fn forward_dft(&self) -> Signal {
        let mut data_tmp = Vec::new();

        for k in 0..self.len() {
            // polar form calculation, but signal will be in cartesian
            let tmp = self.data.iter().enumerate().map(|(n, val)| {
                *val * Complex64::from(EULER).powc(
                        Complex64::from(-1.) * j() * (Complex64::from(2. * PI) / Complex64::from(self.len() as f64)) * Complex64::from(k as f64) * Complex64::from(n as f64)
                    )
            }).reduce(|sum, x| {
                sum + x
            });

            data_tmp.push(tmp.unwrap());
        }
        Signal { data: data_tmp }
    }

    /// Performs Radix-2 FFT, producing a frequency domain `Signal`; does not modify/consume `self`.
    pub fn radix_2_fft(&self) -> Result<Signal, ()> {
        match self.len().is_power_of_two() {
            true => Ok(Signal { data: Self::r2fft(&self.data).to_vec() }),
            false => Err(())
        }
    }

    fn r2fft(data : &[Complex64]) -> Vec<Complex64> {
        if data.len() <= 1 {
            Vec::from(data)
        } else {
            let n = data.len();

            let evens : Box<[Complex64]> = data.iter()
                .enumerate()
                .filter(|(i, _)| i % 2 == 0)
                .map(|(_, x)| *x)
                .collect();
            let odds : Box<[Complex64]>  = data.iter()
                .enumerate()
                .filter(|(i, _)| i % 2 == 1)
                .map(|(_, x)| *x)
                .collect();

            let evens_fft = Self::r2fft(evens.iter().as_slice());
            let odds_fft = Self::r2fft(odds.iter().as_slice());

            let angle = (-2. * PI) / n as f64;
            let twiddle_n = Complex64::new(angle.cos(), angle.sin());

            let mut twiddle = Complex64::from(1);
            let mut out = vec![Complex64::from(0); n];

            for j in 0..(n / 2) {
                out[j] = evens_fft[j] + twiddle * odds_fft[j];
                out[j + n / 2] = evens_fft[j] - twiddle * odds_fft[j];
                twiddle = twiddle * twiddle_n;
            }

            out
        }
    }

    /// Performs Inverse DFT, producing a time domain `Signal`; does not modify/consume `self`.
    pub fn inverse_dft(&self) -> Signal {
        let mut data_tmp : Vec<Complex64> = Vec::new();

        for n in 0..self.len() {
            let tmp = self.data.iter().enumerate().map(|(k, val)| {
                *val * Complex64::from(EULER).powc(
                    j() * (Complex64::from(2. * PI) / Complex64::from(self.len() as f64)) * Complex64::from(k as f64) * Complex64::from(n as f64)
                )
            }).reduce(|sum, x| {
                sum + x
            }).unwrap() / Complex64::from(self.len() as f64);

            data_tmp.push(tmp);
        }

        Signal { data: data_tmp }
    }

    /// Performs Inverse Radix-2 FFT, producing a frequency domain `Signal`; does not modify/consume `self`.
    pub fn inverse_radix_2_fft(&self) -> Result<Signal, ()> {
        match self.len().is_power_of_two() {
            true => Ok(Signal { data: Self::ir2fft(&self.data).to_vec() }),
            false => Err(())
        }
    }

    fn ir2fft(data : &[Complex64]) -> Vec<Complex64> {
        if data.len() <= 1 {
            Vec::from(data)
        } else {
            let n = data.len();

            let evens_fft : Box<[Complex64]> = data.iter()
                .enumerate()
                .filter(|(i, _)| i % 2 == 0)
                .map(|(_, x)| *x)
                .collect();
            let odds_fft : Box<[Complex64]>  = data.iter()
                .enumerate()
                .filter(|(i, _)| i % 2 == 1)
                .map(|(_, x)| *x)
                .collect();

            let evens = Self::ir2fft(evens_fft.iter().as_slice());
            let odds = Self::ir2fft(odds_fft.iter().as_slice());

            let angle = (-2. * PI) / n as f64;
            let twiddle_n_bar = Complex64::new(angle.cos(), -1. * angle.sin());

            let mut twiddle_bar = Complex64::from(1);
            let mut out = vec![Complex64::from(0); n];

            for j in 0..(n / 2) {
                out[j] = (evens[j] + twiddle_bar * odds[j]) / Complex64::from(2);
                out[j + n / 2] = (evens[j] - twiddle_bar * odds[j]) / Complex64::from(2);
                twiddle_bar = twiddle_bar * twiddle_n_bar;
            }

            out
        }
    }

    pub fn zero_extend_start_and_end(self, num : usize) -> Self {
        let mut data_tmp : VecDeque<Complex64> = self.data.into();
        for _ in 0..num {
            data_tmp.push_front(Complex64::from(0));
            data_tmp.push_back(Complex64::from(0));
        }
        Signal { data: data_tmp.into() }
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
        Signal { data: data_tmp.into() }
    }

    pub fn zero_extend_start_mut(&mut self, num : usize) -> &Self {
        for _ in 0..num {
            self.data.insert(0, Complex64::from(0));
        }
        self
    }

    /// Crops the signal by `start` and `end`, modifying `self` and returning the cropped Signal.
    pub fn crop(mut self, start : usize, end : usize) -> Self {
        self.data.truncate(end);
        self.data.drain(..start);
        self
    }

    /// Produces a new `Signal` with a copy of the original data from `start` to `end`.
    pub fn crop_new(&self, start : usize, end : usize) -> Self {
        Signal { data: self.data[start..end].to_vec() }
    }

    /// Applies the Hann window to the `Signal`, returning the consumed `Signal`.
    pub fn hann_window(mut self, symmetric : bool) -> Self {
        let len = if symmetric {
            self.len() - 1
        } else {
            self.len()
        };
        let hann = |n : usize| 0.5 - 0.5 * ((2. * PI * n as f64) / len as f64).cos();
        for (i, val) in self.data.iter_mut().enumerate() {
            *val = *val * Complex64::from(hann(i));
        }
        self
    }

    /// Applies the Hann window to the `Signal`, mutating `self` and returning a reference.
    pub fn hann_window_mut(&mut self, symmetric : bool) -> &Self {
        let len = if symmetric {
            self.len() - 1
        } else {
            self.len()
        };
        let hann = |n : usize| 0.5 - 0.5 * ((2. * PI * n as f64) / len as f64).cos();
        for (i, val) in self.data.iter_mut().enumerate() {
            *val = *val * Complex64::from(hann(i));
        }
        self
    }

    /// Applies the Hamming window to the `Signal`, returning the consumed `Signal`.
    pub fn hamming_window(mut self, symmetric : bool) -> Self {
        let len = if symmetric {
            self.len() - 1
        } else {
            self.len()
        };
        let hamming = |n : usize| HAMMING_WINDOW_ALPHA - HAMMING_WINDOW_BETA * ((2. * PI * n as f64) / len as f64).cos();
        for (i, val) in self.data.iter_mut().enumerate() {
            *val = *val * Complex64::from(hamming(i));
        }
        self
    }

    /// Applies the Hamming window to the `Signal`, mutating `self` and returning a reference.
    pub fn hamming_window_mut(&mut self, symmetric : bool) -> &Self {
        let len = if symmetric {
            self.len() - 1
        } else {
            self.len()
        };
        let hamming = |n : usize| HAMMING_WINDOW_ALPHA - HAMMING_WINDOW_BETA * ((2. * PI * n as f64) / len as f64).cos();
        for (i, val) in self.data.iter_mut().enumerate() {
            *val = *val * Complex64::from(hamming(i));
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
            self.data[i] = self.data[i] + other.data[i - offset];
        }

        self
    }

    pub fn windows(
        &self,
        window_size : usize,
        hop_size : usize,
        windows_num : usize,
        window_function : impl Fn(Self, bool) -> Self, symmetric : bool
    ) -> Result<VecDeque<Signal>, ()> {
        let mut out = VecDeque::new();
        for i in 0..windows_num {
            out.push_back(window_function(self.crop_new(hop_size * i, hop_size * i + window_size), symmetric));
        }

        Ok(out)
    }
}

impl From<&[Complex64]> for Signal {
    fn from(data: &[Complex64]) -> Self {
        Signal { data: Vec::from(data) }
    }
}

impl From<&[f64]> for Signal {
    fn from(data: &[f64]) -> Self {
        let tmp = data.iter().map(|x| Complex64::from(*x));
        Signal { data: Vec::from_iter(tmp) }
    }
}

impl From<&[isize]> for Signal {
    fn from(data: &[isize]) -> Self {
        let tmp = data.iter().map(|x| Complex64::from(*x as f64));
        Signal { data: Vec::from_iter(tmp) }
    }
}

impl<'a> Add<&Signal> for &'a mut Signal {
    type Output = &'a mut Signal;

    /// Add `rhs` to `self`, returning the element-wise sum. Importantly, `self` is mutated but not consumed; it holds the sum.
    fn add(self, rhs: &Signal) -> Self::Output {
        if rhs.len() != self.len() {
            eprintln!("Adding signals of different length ({} and {})", self.len(), rhs.len());
        }
        for i in 0..usize::max(self.data.len(), rhs.data.len()) {
            self.data[i] = self.data[i] + rhs.data[i];
        }

        self
    }
}

impl Add<&Signal> for Signal {
    type Output = Signal;

    /// Add `rhs` to `self`, consuming `self` and returning the element-wise sum.
    fn add(mut self, rhs: &Signal) -> Self::Output {
        if rhs.len() != self.len() {
            eprintln!("Adding signals of different length ({} and {})", self.len(), rhs.len());
        }
        for i in 0..usize::max(self.data.len(), rhs.data.len()) {
            self.data[i] = self.data[i] + rhs.data[i];
        }

        self
    }
}

impl<'a> Mul<&Signal> for &'a mut Signal {
    type Output = &'a mut Signal;

    /// Multiply `self` by `rhs`, returning the element-wise product. Importantly, `self` is mutated but not consumed; it holds the product.
    fn mul(self, rhs: &Signal) -> Self::Output {
        if rhs.len() != self.len() {
            eprintln!("Adding signals of different length ({} and {})", self.len(), rhs.len());
        }
        for i in 0..usize::max(self.data.len(), rhs.data.len()) {
            self.data[i] = self.data[i] * rhs.data[i];
        }

        self
    }
}

impl Mul<&Signal> for Signal {
    type Output = Signal;

    /// Multiply `self` by `rhs`, consuming `self` and returning the element-wise product.
    fn mul(mut self, rhs: &Signal) -> Self::Output {
        if rhs.len() != self.len() {
            eprintln!("Adding signals of different length ({} and {})", self.len(), rhs.len());
        }
        for i in 0..usize::max(self.data.len(), rhs.data.len()) {
            self.data[i] = self.data[i] * rhs.data[i];
        }

        self
    }
}

impl<'a, T> Mul<T> for &'a mut Signal
where 
    T: Add<Output = T> + Sub<Output = T> + Copy + PartialOrd + From<i8> + Into<Complex64>,
    Complex64: From<T>
{
    type Output = &'a mut Signal;

    /// Scalar multiplies entries in `self` by `rhs`, mutating and returning a mutable reference.
    fn mul(self, rhs: T) -> Self::Output {
        self.data = self.data.iter().map(|x| *x * Complex64::from(rhs)).collect();
        self
    }
}

impl<T> Mul<T> for Signal
where 
    T: Add<Output = T> + Sub<Output = T> + Copy + PartialOrd + From<i8> + Into<Complex64>,
    Complex64: From<T>
{
    type Output = Signal;

    /// Scalar multiplies entries in `self` by `rhs`, consuming and returning the final `Signal`.
    fn mul(mut self, rhs: T) -> Self::Output {
        self.data = self.data.iter().map(|x| *x * Complex64::from(rhs)).collect();
        self
    }
}

impl PartialEq for Signal {
    /// Checks if `self` is roughly equal to `other`, to 3 decimal places.
    fn eq(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).all(|(a, b)| {
            let a_real_approx = round_to_place(a.real(), EQUALITY_ACCURACY);
            let a_imag_approx = round_to_place(a.imag(), EQUALITY_ACCURACY);
            let b_real_approx = round_to_place(b.real(), EQUALITY_ACCURACY);
            let b_imag_approx = round_to_place(b.imag(), EQUALITY_ACCURACY);

            // println!("{a_real_approx} == {b_real_approx}\t&&\t{a_imag_approx} == {b_imag_approx}");

            (a_real_approx == b_real_approx) && (a_imag_approx == b_imag_approx)
        })
    }

    /// Checks if `self` is not roughly equal to `other`, to 3 decimal places.
    fn ne(&self, other: &Self) -> bool {
        !(self == other)
    }
}

impl Display for Signal {
    /// Display the entries of the signal data, in polar form by default.
    /// 
    /// - `.*` (precision) flag affects to what place values are rounded to. Default is 3 decimal places.
    /// - `#` (alternate) flag prints in cartesian form.
    /// - `+` (plus) flag removes the `* e ^ _j` part, polar form only.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = "[".to_string();
        let round_place = f.precision();
        for (i, val ) in self.data.iter().enumerate() {
            if i != 0 {
                out += ", ";
            }

            if !f.alternate() {
                let mut mag = f64::sqrt(val.real().powf(2.) + val.imag().powf(2.));
                let mut phase = f64::atan2(val.imag(), val.real());

                // if the magnitude is basically zero, force the phase to zero
                if mag < 0.00001 {
                    mag = 0.0;
                    phase = 0.0;
                } else {
                    mag = match round_place {
                        Some(x) => round_to_place(mag, x),
                        None => round_to_place(mag, 3),
                    };
                    phase = match round_place {
                        Some(x) => round_to_place(phase, x),
                        None => round_to_place(phase, 3)
                    };
                }

                out += &*("".to_string() + &*mag.to_string());
                if !f.sign_plus() {
                    out += &*(" * e^".to_string() + &*phase.to_string() + "j");
                }
            } else {
                let real = match round_place {
                    Some(x) => round_to_place(val.real(), x),
                    None => round_to_place(val.real(), 3)
                };
                let imag = match round_place {
                    Some(x) => round_to_place(val.imag(), x),
                    None => round_to_place(val.imag(), 3)
                };
                out += &*(real.to_string() + " + " + &*imag.to_string() + "j")
            }
        }
        out += "]";
        
        write!(f, "{}", out)
    }
}

fn round_to_place(num : f64, place : usize) -> f64 {
    let factor = 10_f64.powi(place as i32);
    (num * factor).round() / factor
}

fn j() -> Complex64 {
    Complex64::new(0., 1.)
}