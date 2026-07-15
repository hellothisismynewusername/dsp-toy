use std::sync::OnceLock;

pub(crate) const EULER : f64 = 2.718281828459045;

pub static  EQUALITY_ACCURACY : OnceLock<usize> = OnceLock::new();

pub(crate) const HAMMING_WINDOW_C_0 : f64 = 0.5434782608695652;
pub(crate) const HAMMING_WINDOW_C_1 : f64 = 0.45652173913043476;
pub(crate) const BLACKMAN_WINDOW_C_0 : f64 = 0.42;
pub(crate) const BLACKMAN_WINDOWS_C_1 : f64 = 0.5;
pub(crate) const BLACKMAN_WINDOWS_C_2 : f64 = 0.08;