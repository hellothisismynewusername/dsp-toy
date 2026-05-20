use std::{fmt::Display, ops::{Add, Div, Index, IndexMut, Mul, Range, Sub}};

use easy_complex::{Complex64};
use crate::{signal::Signal, utility::equality_accuracy};
use crate::utility::{round_to_place};

impl Index<usize> for Signal {
    type Output = Complex64;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<usize> for Signal {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Index<Range<usize>> for Signal {
    type Output = [Complex64];
    fn index(&self, index: Range<usize>) -> &Self::Output {
        &self.data[index]
    }
}

impl IndexMut<Range<usize>> for Signal {
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        &mut self.data[index]
    }
}

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
            self[i] = self[i] + rhs[i];
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
            self[i] = self[i] + rhs[i];
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
            self[i] = self[i] - rhs[i];
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
            self[i] = self[i] - rhs[i];
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
            self[i] = self[i] * rhs[i];
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
            self[i] = self[i] * rhs[i];
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
            self[i] = self[i] / rhs[i];
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
            self[i] = self[i] / rhs[i];
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
    /// - `+` (plus) flag hides the exponential in polar form, or the imaginary part in cartesian.
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
                out += &*(real.to_string());
                if !f.sign_plus() {
                    out += &*(" + ".to_string() + &*imag.to_string() + "j");
                }
            }
        }
        out += "]";
        
        write!(f, "{}", out)
    }
}