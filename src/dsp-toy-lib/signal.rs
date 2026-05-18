use std::{collections::VecDeque, fmt::Display, ops::{Add, Div, Mul, Sub}};

use easy_complex::{Complex64};
use crate::{math, utility::equality_accuracy};
use crate::utility::{round_to_place};

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

    /// Convolves with signal `other`, consuming and returning `self`.
    pub fn convolve(mut self, other : &Self) -> Self {
        Signal { data: math::convolve(&mut self.data, &other.data) }
    }

    /// Performs DFT, producing a frequency domain `Signal`; does not modify/consume `self`.
    pub fn forward_dft(&self) -> Self {
        Signal { data: math::dft(&self.data) }
    }

    /// Performs Radix-2 FFT, producing a frequency domain `Signal`; consumes and returns `self`. Fails if length is not a power of 2.
    pub fn radix_2_fft(mut self) -> Result<Self, ()> {
        match self.len().is_power_of_two() {
            true => {
                math::r2fft(&mut self.data);
                Ok(self)
            },
            false => Err(())
        }
    }

    /// Performs Radix-2 FFT, producing a frequency domain `Signal`; does not modify/consume `self`. Fails if length is not a power of 2.
    pub fn radix_2_fft_new(&self) -> Result<Self, ()> {
        match self.len().is_power_of_two() {
            true => {
                let mut data_new = self.data.clone();
                math::r2fft(&mut data_new);
                Ok(Signal { data: data_new })
            },
            false => Err(())
        }
    }

    /// Performs Inverse DFT, producing a time domain `Signal`; does not modify/consume `self`.
    pub fn inverse_dft(&self) -> Self {
        Signal { data: math::idft(&self.data) }
    }

    /// Performs Inverse Radix-2 FFT; consumes and returns `self`. Fails if length is not a power of 2.
    pub fn inverse_radix_2_fft(mut self) -> Result<Self, ()> {
        match self.len().is_power_of_two() {
            true => {
                math::ir2fft(&mut self.data);
                Ok(self)
            },
            false => Err(())
        }
    }

    /// Performs Inverse Radix-2 FFT, producing a frequency domain `Signal`; does not modify/consume `self`. Fails if length is not a power of 2.
    pub fn inverse_radix_2_fft_new(&self) -> Result<Self, ()> {
        match self.len().is_power_of_two() {
            true => {
                let mut data_new = self.data.clone();
                math::ir2fft(&mut data_new);
                Ok(Signal { data: data_new })
            },
            false => Err(())
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
        for (i, val) in self.data.iter_mut().enumerate() {
            *val = *val * Complex64::from(math::hann(i, len));
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
        for (i, val) in self.data.iter_mut().enumerate() {
            *val = *val * Complex64::from(math::hann(i, len));
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
        for (i, val) in self.data.iter_mut().enumerate() {
            *val = *val * Complex64::from(math::hamming(i, len));
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
        for (i, val) in self.data.iter_mut().enumerate() {
            *val = *val * Complex64::from(math::hamming(i, len));
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

// -------------- TRAIT IMPLEMENTATIONS

impl<T> From<&[T]> for Signal
where 
    T: Add<Output = T> + Sub<Output = T> + Copy + PartialOrd + From<i8> + Into<Complex64>,
    Complex64: From<T>
{
    fn from(data: &[T]) -> Self {
        let tmp = data.iter().map(|x| Complex64::from(*x));
        Signal { data: Vec::from_iter(tmp) }
    }
}

impl<T, const N : usize> From<[T; N]> for Signal
where 
    T: Add<Output = T> + Sub<Output = T> + Copy + PartialOrd + From<i8> + Into<Complex64>,
    Complex64: From<T>
{
    fn from(value: [T; N]) -> Self {
        Signal { data: value.iter().map(|x| Complex64::from(*x)).collect() }
    }
}

impl<T> FromIterator<T> for Signal
where 
    T: Add<Output = T> + Sub<Output = T> + Copy + PartialOrd + From<i8> + Into<Complex64>,
    Complex64: From<T>
{
    fn from_iter<A: IntoIterator<Item = T>>(iter: A) -> Self {
        let tmp : Vec<Complex64> = iter.into_iter().map(|x| Complex64::from(x)).collect();
        Signal { data: tmp }
    }
}

impl<'a> Add<&Signal> for &'a mut Signal {
    type Output = &'a mut Signal;

    /// Add `rhs` to `self`, returning the element-wise sum. Importantly, `self` is mutated but not consumed; it holds the sum.
    fn add(self, rhs: &Signal) -> Self::Output {
        if rhs.len() != self.len() {
            eprintln!("Adding signals of different length ({} and {})", self.len(), rhs.len());
        }
        for i in 0..usize::min(self.data.len(), rhs.data.len()) {
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
        for i in 0..usize::min(self.data.len(), rhs.data.len()) {
            self.data[i] = self.data[i] + rhs.data[i];
        }

        self
    }
}

impl<'a> Sub<&Signal> for &'a mut Signal {
    type Output = &'a mut Signal;

    /// Subtract `rhs` from `self`, returning the element-wise difference. Importantly, `self` is mutated but not consumed; it holds the difference.
    fn sub(self, rhs: &Signal) -> Self::Output {
        if rhs.len() != self.len() {
            eprintln!("Subtracting signals of different length ({} and {})", self.len(), rhs.len());
        }
        for i in 0..usize::min(self.data.len(), rhs.data.len()) {
            self.data[i] = self.data[i] - rhs.data[i];
        }

        self
    }
}

impl Sub<&Signal> for Signal {
    type Output = Signal;

    /// Subtract `rhs` from `self`, consuming `self` and returning the element-wise difference.
    fn sub(mut self, rhs: &Signal) -> Self::Output {
        if rhs.len() != self.len() {
            eprintln!("Subtracting signals of different length ({} and {})", self.len(), rhs.len());
        }
        for i in 0..usize::min(self.data.len(), rhs.data.len()) {
            self.data[i] = self.data[i] - rhs.data[i];
        }

        self
    }
}

impl<'a> Mul<&Signal> for &'a mut Signal {
    type Output = &'a mut Signal;

    /// Multiply `self` by `rhs`, returning the element-wise product. Importantly, `self` is mutated but not consumed; it holds the product.
    fn mul(self, rhs: &Signal) -> Self::Output {
        if rhs.len() != self.len() {
            eprintln!("Multiplying signals of different length ({} and {})", self.len(), rhs.len());
        }
        for i in 0..usize::min(self.data.len(), rhs.data.len()) {
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
            eprintln!("Multiplying signals of different length ({} and {})", self.len(), rhs.len());
        }
        for i in 0..usize::min(self.data.len(), rhs.data.len()) {
            self.data[i] = self.data[i] * rhs.data[i];
        }

        self
    }
}

impl<'a> Div<&Signal> for &'a mut Signal {
    type Output = &'a mut Signal;

    /// Divide `self` by `rhs`, returning the element-wise quotient. Importantly, `self` is mutated but not consumed; it holds the quotient.
    fn div(self, rhs: &Signal) -> Self::Output {
        if rhs.len() != self.len() {
            eprintln!("Dividing signals of different length ({} and {})", self.len(), rhs.len());
        }
        for i in 0..usize::min(self.data.len(), rhs.data.len()) {
            self.data[i] = self.data[i] / rhs.data[i];
        }

        self
    }
}

impl Div<&Signal> for Signal {
    type Output = Signal;

    /// Divide `self` by `rhs`, consuming `self` and returning the element-wise quotient.
    fn div(mut self, rhs: &Signal) -> Self::Output {
        if rhs.len() != self.len() {
            eprintln!("Dividing signals of different length ({} and {})", self.len(), rhs.len());
        }
        for i in 0..usize::min(self.data.len(), rhs.data.len()) {
            self.data[i] = self.data[i] / rhs.data[i];
        }

        self
    }
}

// Scalar multiplication
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

// Scalar multiplication
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
            let a_real_approx = round_to_place(a.real(), equality_accuracy());
            let a_imag_approx = round_to_place(a.imag(), equality_accuracy());
            let b_real_approx = round_to_place(b.real(), equality_accuracy());
            let b_imag_approx = round_to_place(b.imag(), equality_accuracy());

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