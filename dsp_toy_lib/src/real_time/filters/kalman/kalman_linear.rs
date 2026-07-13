use std::usize;

use nalgebra::{Complex, SMatrix};

use crate::real_time::real_time_signal_processer::RealTimeSignalProcessor;

/// T should be Complex<f64> or f64
pub struct FilterKalmanLinear<T, const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize> {
    state_dim : usize,
    measure_dim : usize,
    control_dim : usize,
    state_vector : SMatrix<T, STATE_DIM, 1>,
    estimate_covariance : SMatrix<T, STATE_DIM, STATE_DIM>,
    state_transition : SMatrix<T, STATE_DIM, STATE_DIM>,
    control : SMatrix<T, STATE_DIM, CONTROL_DIM>,
    process_noise_covariance : Option<SMatrix<T, STATE_DIM, STATE_DIM>>
}

impl<const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize> FilterKalmanLinear<f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
    pub fn new() -> Self {
        // init then run predict
        todo!()
    }

    /// Returns `(extrapolate_state, extrapolate_covariance)`
    fn predict(&mut self, control_vector : SMatrix<f64, CONTROL_DIM, 1>) -> (SMatrix<f64, STATE_DIM, 1>, SMatrix<f64, STATE_DIM, STATE_DIM>) {
        let extrapolate_state: SMatrix<f64, STATE_DIM, 1> = self.state_transition * self.state_vector + self.control * control_vector;
        let extrapolate_covariance: SMatrix<f64, STATE_DIM, STATE_DIM> = if let Some(process_noise_covariance) = self.process_noise_covariance {
            self.state_transition * self.estimate_covariance * self.state_transition.transpose() + process_noise_covariance
        } else {
            self.state_transition * self.estimate_covariance * self.state_transition.transpose()
        };

        (extrapolate_state, extrapolate_covariance)
    }
}

impl<KalmanInput, const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize> RealTimeSignalProcessor<KalmanInput, f64> for FilterKalmanLinear<f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
    fn process_sample(&mut self, inp : KalmanInput) -> f64 {
        // Correct first



        // Prediction stage & save vars



        todo!()
    }
}

impl<KalmanInput, const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize> RealTimeSignalProcessor<KalmanInput, Complex<f64>> for FilterKalmanLinear<Complex<f64>, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
    fn process_sample(&mut self, inp : KalmanInput) -> Complex<f64> {
        todo!()
    }
}