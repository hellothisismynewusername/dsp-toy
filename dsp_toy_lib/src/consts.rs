use std::sync::OnceLock;

pub(crate) const EULER : f64 = 2.718281828459045;

pub static  EQUALITY_ACCURACY : OnceLock<usize> = OnceLock::new();

pub(crate) const HAMMING_WINDOW_ALPHA : f64 = 0.5434782608695652;
pub(crate) const HAMMING_WINDOW_BETA : f64 = 0.45652173913043476;