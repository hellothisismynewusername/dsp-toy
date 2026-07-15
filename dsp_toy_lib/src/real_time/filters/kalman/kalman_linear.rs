use std::usize;

use nalgebra::{RealField, SMatrix};

use crate::{real_time::{filters::kalman::kalman_input::KalmanInput, real_time_signal_processer::RealTimeSignalProcessor}, utility::{SMatrixTimes}};

pub struct FilterKalmanLinear<T, const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize> {
    /// `STATE_DIM` x `1`
    pub state_vector : SMatrix<T, STATE_DIM, 1>,
    /// `STATE_DIM` x `STATE_DIM`
    pub estimate_covariance : SMatrix<T, STATE_DIM, STATE_DIM>,
    /// `MEASURE_DIM` x `STATE_DIM`
    pub observation : SMatrix<T, MEASURE_DIM, STATE_DIM>,
    /// `MEASURE_DIM` x `MEASURE_DIM`
    pub measure_covariance : SMatrix<T, MEASURE_DIM, MEASURE_DIM>,

    /// `STATE_DIM` x `STATE_DIM`
    pub state_transition : SMatrixTimes<T, STATE_DIM, STATE_DIM>,
    /// `STATE_DIM` x `CONTROL_DIM`
    pub control : Option<SMatrixTimes<T, STATE_DIM, CONTROL_DIM>>,
    /// `STATE_DIM` x `STATE_DIM`
    pub process_noise_covariance : Option<SMatrixTimes<T, STATE_DIM, STATE_DIM>>,
}

// This is for when T is integer or floating point. Not Complex. An int-only version can be made, which would in theory have a better performance than this.
impl<T, const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize> FilterKalmanLinear<T, STATE_DIM, MEASURE_DIM, CONTROL_DIM>
where 
    T: RealField + Copy + Clone
{
    pub fn init(&mut self, input : &KalmanInput<T, STATE_DIM, MEASURE_DIM, CONTROL_DIM>) {
        let (state, estimate_covariance) = self.predict_internal(input);
        self.state_vector = state;
        self.estimate_covariance = estimate_covariance;
    }

    pub fn predict(&mut self, input : &KalmanInput<T, STATE_DIM, MEASURE_DIM, CONTROL_DIM>) {
        let (state, estimate_covariance) = self.predict_internal(input);
        self.state_vector = state;
        self.estimate_covariance = estimate_covariance;
    }

    /// Returns `(extrapolate_state, extrapolate_covariance)`.
    fn predict_internal(&mut self, input : &KalmanInput<T, STATE_DIM, MEASURE_DIM, CONTROL_DIM>) -> (SMatrix<T, STATE_DIM, 1>, SMatrix<T, STATE_DIM, STATE_DIM>) {
        // handle needed changes regarding dynamically changing values in the matrices
        let state_transition = if let Some(delta_time) = input.delta_time {
            self.state_transition.multiply_entries_float(delta_time)
        } else {
            self.state_transition.matrix
        };
        let control_o = if self.control.is_some() {
            if let Some(delta_time) = input.delta_time {
                Some(self.control.as_mut().unwrap().multiply_entries_float(delta_time))
            } else {
                Some(self.control.as_mut().unwrap().matrix)
            }
        } else {
            None
        };
        let process_noise_covariance_o = if self.process_noise_covariance.is_some() {
            if let Some(delta_time) = input.delta_time {
                Some(self.process_noise_covariance.as_mut().unwrap().multiply_entries_float(delta_time))
            } else {
                Some(self.process_noise_covariance.as_mut().unwrap().matrix)
            }
        } else {
            None
        };

        // only bother adding control_matrix * control_vector if both are Some.
        let control_vector = input.control_vector;
        let extrapolate_state: SMatrix<T, STATE_DIM, 1> = if let Some(control_vector_u) = control_vector {
            if let Some(control) = &control_o.as_ref() {
                state_transition * self.state_vector + *control * control_vector_u
            } else {
                state_transition * self.state_vector
            }
        }  else {
            state_transition * self.state_vector
        };
        let extrapolate_estimate_covariance: SMatrix<T, STATE_DIM, STATE_DIM> = if let Some(process_noise_covariance) = process_noise_covariance_o.as_ref() {
            state_transition * self.estimate_covariance * state_transition.transpose() + process_noise_covariance
        } else {
            state_transition * self.estimate_covariance * state_transition.transpose()
        };

        (extrapolate_state, extrapolate_estimate_covariance)
    }
}

impl<T, const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize>
    RealTimeSignalProcessor<&KalmanInput<T, STATE_DIM, MEASURE_DIM, CONTROL_DIM>, SMatrix<T, STATE_DIM, 1>>
    for
    FilterKalmanLinear<T, STATE_DIM, MEASURE_DIM, CONTROL_DIM>
where 
    T: RealField + Copy + Clone
{
    fn process_sample(&mut self, input : &KalmanInput<T, STATE_DIM, MEASURE_DIM, CONTROL_DIM>) -> SMatrix<T, STATE_DIM, 1> {
        
        // Prediction

        let (extrapolate_state, extrapolate_estimate_covariance) = self.predict_internal(&input);
        self.state_vector = extrapolate_state;
        self.estimate_covariance = extrapolate_estimate_covariance;
        
        // Correct

        let mut innovation_covariance = self.observation * self.estimate_covariance * self.observation.transpose() + self.measure_covariance;
        if !innovation_covariance.try_inverse_mut() {
            panic!("Innovation covariance couldn't be inverted");
        }
        let kalman_gain = self.estimate_covariance * self.observation.transpose() * innovation_covariance;

        let updated_state = self.state_vector + kalman_gain * (input.measurement_vector - self.observation * self.state_vector);
        self.state_vector = updated_state;
        
        // Simplified Covariance Update Equation
        let mut updated_estimate_covariance = (SMatrix::identity() - (kalman_gain * self.observation)) * self.estimate_covariance;
        updated_estimate_covariance.fill_lower_triangle_with_upper_triangle(); // force symmetrical, a tiny floating point mismatch could ruin everything.
        self.estimate_covariance = updated_estimate_covariance;

        updated_state
    }
}