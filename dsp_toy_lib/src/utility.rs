use litemap::LiteMap;
use nalgebra::{Complex, ComplexField, RealField, SMatrix};

use crate::consts::EQUALITY_ACCURACY;

pub fn round_to_place(num : f64, place : usize) -> f64 {
    let factor = 10_f64.powi(place as i32);
    (num * factor).round() / factor
}

pub fn j() -> Complex<f64> {
    Complex::<f64>::new(0., 1.)
}

pub fn equality_accuracy() -> usize {
    *EQUALITY_ACCURACY.get_or_init(|| 2)
}

/// This is essentially a coefficient matrix with `entry_multipliers_powers` defining the powers of the variable at each entry (so 0 / not in the map to not affect that entry).
#[derive(Debug)]
pub struct SMatrixTimes<T, const R : usize, const C : usize>
{
    pub matrix : SMatrix<T, R, C>,
    pub entry_multipliers_powers : Option<LiteMap<(usize, usize), T>>
}

impl<T, const R : usize, const C : usize> SMatrixTimes<T, R, C>
where 
    T: ComplexField + Copy,
{
    pub fn new(matrix : SMatrix<T, R, C>, num_multiplied : usize) -> Self {
        SMatrixTimes {
            matrix: matrix,
            entry_multipliers_powers: {
                if num_multiplied > 0 {
                    Some(LiteMap::with_capacity(num_multiplied))
                } else {
                    None
                }
            }
        }
    }

    pub fn new_with_litemap(matrix : SMatrix<T, R, C>, map : LiteMap<(usize, usize), T>) -> Self {
        SMatrixTimes {
            matrix: matrix,
            entry_multipliers_powers: Some(map)
        }
    }
}

// only going to work on integers
impl<T, const R : usize, const C : usize> SMatrixTimes<T, R, C>
where 
    T: ComplexField + Copy,
    i32: From<T>
{
    /// `entry_multiplier_value` is like plugging in the dt value.
    pub fn multiply_entries_int(&mut self, entry_multiplier_value : T) {
        if self.entry_multipliers_powers.is_none() {
            return;
        }
        for (pos, curr_power) in self.entry_multipliers_powers.as_ref().unwrap().iter() {
            self.matrix[*pos] = self.matrix[*pos] * entry_multiplier_value.powi(i32::from(*curr_power))
        }
    }
}

// floating point
impl<T, const R : usize, const C : usize> SMatrixTimes<T, R, C>
where 
    T: RealField + Copy
{
    /// `entry_multiplier_value` is like plugging in the dt value.
    pub fn multiply_entries_float(&self, entry_multiplier_value : T) -> SMatrix<T, R, C> {
        if self.entry_multipliers_powers.is_none() {
            return self.matrix;
        }
        let mut out = self.matrix.clone();
        for (pos, curr_power) in self.entry_multipliers_powers.as_ref().unwrap().iter() {
            out[*pos] = self.matrix[*pos] * entry_multiplier_value.powf(*curr_power)
        }

        out
    }
}

// for when T is just complex and we have to do `.powc()`
impl<T, const R : usize, const C : usize> SMatrixTimes<T, R, C>
where 
    T: ComplexField + Copy,
{
    /// `entry_multiplier_value` is like plugging in the dt value.
    pub fn multiply_entries_complex(&mut self, entry_multiplier_value : T) {
        if self.entry_multipliers_powers.is_none() {
            return;
        }
        for (pos, curr_power) in self.entry_multipliers_powers.as_ref().unwrap().iter() {
            self.matrix[*pos] = self.matrix[*pos] * entry_multiplier_value.powc(*curr_power)
        }
    }
}