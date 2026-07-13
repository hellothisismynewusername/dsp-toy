use std::usize;

use nalgebra::{Complex, SMatrix};

use crate::real_time::real_time_signal_processer::RealTimeSignalProcessor;

pub struct FilterKalmanLinear<T, const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize> {
    state_dim : usize,
    measure_dim : usize,
    control_dim : usize,
    prev_state_vector : SMatrix<T, STATE_DIM, 1>,
    prev_estimate_covariance : SMatrix<T, STATE_DIM, STATE_DIM>,
}

impl<const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize> RealTimeSignalProcessor<f64> for FilterKalmanLinear<f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
    fn process_sample(&mut self, inp : f64) -> f64 {
        todo!()
    }
}

impl<const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize> RealTimeSignalProcessor<Complex<f64>> for FilterKalmanLinear<Complex<f64>, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
    fn process_sample(&mut self, inp : Complex<f64>) -> Complex<f64> {
        
        todo!()
    }
}