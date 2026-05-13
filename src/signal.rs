use std::{f64::consts::PI, fmt::Display, ops::{Add, Mul}};

use easy_complex::Complex64;
use crate::consts::EULER;

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

    /// Computes DFT, producing a new `Signal`; does not modify/consume `self`.
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

    pub fn inverse_dft(&self) -> Signal {
        let mut data_tmp : Vec<Complex64> = Vec::new();

        todo!()
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
        for i in 0..self.data.len() {
            self.data[i] = self.data[i] + rhs.data[i];
        }

        self
    }
}

impl Add<&Signal> for Signal {
    type Output = Signal;

    /// Add `rhs` to `self`, consuming `self` and returning the element-wise sum.
    fn add(mut self, rhs: &Signal) -> Self::Output {
        for i in 0..self.data.len() {
            self.data[i] = self.data[i] + rhs.data[i];
        }

        self
    }
}

impl<'a> Mul<&Signal> for &'a mut Signal {
    type Output = &'a mut Signal;

    /// Multiply `self` by `rhs`, returning the element-wise product. Importantly, `self` is mutated but not consumed; it holds the product.
    fn mul(self, rhs: &Signal) -> Self::Output {
        for i in 0..self.data.len() {
            self.data[i] = self.data[i] * rhs.data[i];
        }

        self
    }
}

impl Mul<&Signal> for Signal {
    type Output = Signal;

    /// Multiply `self` by `rhs`, consuming `self` and returning the element-wise product.
    fn mul(mut self, rhs: &Signal) -> Self::Output {
        for i in 0..self.data.len() {
            self.data[i] = self.data[i] * rhs.data[i];
        }

        self
    }
}

impl PartialEq for Signal {
    /// Checks if `self` is roughly equal to `other`, to 3 decimal places.
    fn eq(&self, other: &Self) -> bool {
        self.data.iter().zip(other.data.iter()).all(|(a, b)| {
            let a_real_approx = round_to_place(a.real(), 3);
            let a_imag_approx = round_to_place(a.imag(), 3);
            let b_real_approx = round_to_place(b.real(), 3);
            let b_imag_approx = round_to_place(b.imag(), 3);

            (a_real_approx == b_real_approx) && (a_imag_approx == b_imag_approx)
        })
    }

    /// Checks if `self` is not roughly equal to `other`, to 3 decimal places.
    fn ne(&self, other: &Self) -> bool {
        !(self == other)
    }
}

impl Display for Signal {
    /// Display the entries of the signal data, to an accuracy of 3 decimal places.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = "[".to_string();
        for (i, val ) in self.data.iter().enumerate() {
            if i != 0 {
                out += ", ";
            }
            let mut mag = f64::sqrt(val.real().powf(2.) + val.imag().powf(2.));
            let mut phase = f64::atan2(val.imag(), val.real());
            mag = round_to_place(mag, 3);
            phase = round_to_place(phase, 3);
            out += &*("".to_string() + &*mag.to_string() + " * e^" + &*phase.to_string() + "j");
        }
        out += "]";
        
        write!(f, "{}", out)
    }
}

fn round_to_place(num : f64, place : i32) -> f64 {
    let factor = 10_f64.powi(place);
    (num * factor).round() / factor
}

fn j() -> Complex64 {
    Complex64::new(0., 1.)
}