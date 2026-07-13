use nalgebra::Complex;

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