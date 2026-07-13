use std::usize;

use nalgebra::{Complex, SMatrix};

use crate::real_time::{filters::kalman::kalman_input::KalmanInput, real_time_signal_processer::RealTimeSignalProcessor};

/// T should be Complex<f64> or f64
pub struct FilterKalmanLinear<T, const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize> {
    state_dim : usize,
    measure_dim : usize,
    control_dim : usize,
    state_vector : SMatrix<T, STATE_DIM, 1>,
    estimate_covariance : SMatrix<T, STATE_DIM, STATE_DIM>,
    state_transition : SMatrix<T, STATE_DIM, STATE_DIM>,
    control : SMatrix<T, STATE_DIM, CONTROL_DIM>,
    process_noise_covariance : Option<SMatrix<T, STATE_DIM, STATE_DIM>>,
    observation : SMatrix<T, MEASURE_DIM, STATE_DIM>,
    measure_covariance : SMatrix<T, MEASURE_DIM, MEASURE_DIM>
}

impl<const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize> FilterKalmanLinear<f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
    pub fn init(&mut self, control_vector : Option<SMatrix<f64, CONTROL_DIM, 1>>) {
        let (state, estimate_covariance) = self.predict(control_vector);
        self.state_dim = STATE_DIM;
        self.measure_dim = MEASURE_DIM;
        self.control_dim = CONTROL_DIM;
        self.state_vector = state;
        self.estimate_covariance = estimate_covariance;
    }

    /// Returns `(extrapolate_state, extrapolate_covariance)`
    fn predict(&mut self, control_vector : Option<SMatrix<f64, CONTROL_DIM, 1>>) -> (SMatrix<f64, STATE_DIM, 1>, SMatrix<f64, STATE_DIM, STATE_DIM>) {
        let extrapolate_state: SMatrix<f64, STATE_DIM, 1> = if let Some(control_vector_u) = control_vector {
            self.state_transition * self.state_vector + self.control * control_vector_u
        }  else {
            self.state_transition * self.state_vector
        };
        let extrapolate_estimate_covariance: SMatrix<f64, STATE_DIM, STATE_DIM> = if let Some(process_noise_covariance) = self.process_noise_covariance {
            self.state_transition * self.estimate_covariance * self.state_transition.transpose() + process_noise_covariance
        } else {
            self.state_transition * self.estimate_covariance * self.state_transition.transpose()
        };

        (extrapolate_state, extrapolate_estimate_covariance)
    }
}

impl<const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize>
    RealTimeSignalProcessor<KalmanInput<f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM>, SMatrix<f64, STATE_DIM, 1>>
    for
    FilterKalmanLinear<f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {

    fn process_sample(&mut self, inp : KalmanInput<f64, STATE_DIM, MEASURE_DIM, CONTROL_DIM>) -> SMatrix<f64, STATE_DIM, 1> {
        
        // Prediction

        let (extrapolate_state, extrapolate_estimate_covariance) = self.predict(inp.control_vector);
        self.state_vector = extrapolate_state;
        self.estimate_covariance = extrapolate_estimate_covariance;
        
        // Correct

        let mut innovation_covariance = self.observation * self.estimate_covariance * self.observation.transpose() + self.measure_covariance;
        if !innovation_covariance.try_inverse_mut() {
            // temporary
            panic!("uh oh not invertible");
        }
        let kalman_gain = self.estimate_covariance * self.observation.transpose() * innovation_covariance;

        let updated_state = self.state_vector + kalman_gain * (inp.measurement_vector - self.observation * self.state_vector);
        self.state_vector = updated_state;
        
        let tmp_upper = (kalman_gain * self.observation).upper_triangle();
        let mut i_minus_tmp_symm = (SMatrix::identity() - tmp_upper);
        i_minus_tmp_symm.fill_lower_triangle_with_upper_triangle();
        let updated_estimate_covariance = i_minus_tmp_symm * self.estimate_covariance;
        self.estimate_covariance = updated_estimate_covariance;

        updated_state
    }
}

// impl<KalmanInput, const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize> RealTimeSignalProcessor<KalmanInput, Complex<f64>> for FilterKalmanLinear<Complex<f64>, STATE_DIM, MEASURE_DIM, CONTROL_DIM> {
//     fn process_sample(&mut self, inp : KalmanInput) -> Complex<f64> {
//         todo!()
//     }
// }