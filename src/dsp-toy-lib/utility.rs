use easy_complex::Complex64;

pub fn round_to_place(num : f64, place : usize) -> f64 {
    let factor = 10_f64.powi(place as i32);
    (num * factor).round() / factor
}

pub fn j() -> Complex64 {
    Complex64::new(0., 1.)
}