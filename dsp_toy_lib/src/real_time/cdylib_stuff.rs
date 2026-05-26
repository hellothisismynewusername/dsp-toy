use std::ptr::{null_mut};

use crate::real_time::{filters::filter_iir_peak_bell::FilterIIRPeakBell, real_time_signal_processer::RealTimeSignalProcessor};

impl From<&Tuple3F64> for (f64, f64, f64) {
    fn from(value: &Tuple3F64) -> Self {
        (value.a, value.b, value.c)
    }
}

#[repr(C)]
pub struct Tuple3F64 {
    pub a : f64,
    pub b : f64,
    pub c : f64
}

/// Instantiate a `FilterIIRPeakBell<f64>` with bands' frequency defined by Tuple3F64 a, gain Tuple3F64 b, and Q factor Tuple3F64 c.
#[unsafe(no_mangle)]
pub extern "C" fn new_FilterIIRPeakBellF64(bands_ptr : *const Tuple3F64, bands_count : usize, sample_rate : usize) -> *mut FilterIIRPeakBell<f64> {
    if bands_ptr.is_null() || bands_count == 0 {
        return null_mut();
    }

    unsafe {
        let bands_tmp = std::slice::from_raw_parts(bands_ptr, bands_count);

        let bands : Vec<(f64, f64, f64)> = bands_tmp.iter().map(|x| {
            let tmp : (f64, f64, f64) = x.into();
            tmp
        }).collect();
        let filter = Box::new(FilterIIRPeakBell::new_real(bands.as_slice(), sample_rate));

        return Box::into_raw(filter);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn free_FilterIIRPeakBellF64(ptr : *mut FilterIIRPeakBell<f64>) {
    if !ptr.is_null() {
        unsafe {
            let _tmp = Box::from_raw(ptr);
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn process_sample_FilterIIRPeakBellF64(filter : *mut FilterIIRPeakBell<f64>, sample : f64) -> f64 {
    if filter.is_null() {
        return -1. * f64::MAX;
    }
    unsafe {
        return (*filter).process_sample(sample);
    }
}