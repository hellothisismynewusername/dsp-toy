use std::marker::PhantomData;

use nalgebra::{RealField, SMatrix};

use crate::{real_time::{filters::kalman::{kalman_input::KalmanInput, sigma_points_functions::sigma_points_function::SigmaPointsFunction}, real_time_signal_processer::RealTimeSignalProcessor}, utility::SMatrixTimes};

/// Control matrix isn't used anymore, it can be accounted for in the state transition function.
/// 
/// So, user will have to add their own custom behaviours to the state transition function, but `T` is still passed in as a time variable.
/// This is to uphold the passed-in `delta_time` from `KalmanInput`.
pub struct FilterKalmanUnscented<T, TimeStepType, const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize, const N_OUT : usize, FStateTransition, FObservation, FSigmaPointsFunction>
where 
    TimeStepType: RealField,
    FStateTransition: Fn(SMatrix<T, STATE_DIM, 1>, TimeStepType) -> SMatrix<T, STATE_DIM, 1>,
    FObservation: Fn(SMatrix<T, STATE_DIM, 1>) -> SMatrix<T, MEASURE_DIM, 1>,
    FSigmaPointsFunction: SigmaPointsFunction<T, STATE_DIM, N_OUT>
{
    /// `STATE_DIM` x `1`
    pub state_vector : SMatrix<T, STATE_DIM, 1>,
    /// `STATE_DIM` x `STATE_DIM`
    pub estimate_covariance : SMatrix<T, STATE_DIM, STATE_DIM>,
    
    /// `MEASURE_DIM` x `MEASURE_DIM`
    pub measure_covariance : SMatrix<T, MEASURE_DIM, MEASURE_DIM>,

    /// `STATE_DIM` x `STATE_DIM`
    pub process_noise_covariance : Option<SMatrixTimes<T, TimeStepType, STATE_DIM, STATE_DIM>>,

    /// aka Measurement Function
    pub observation : FObservation,
    /// State transition function (Vector of size `STATE_DIM`, `TimeStepType`) -> Vector of size `STATE_DIM`
    pub state_transition : FStateTransition,

    pub sigma_points_function : FSigmaPointsFunction,

    pub _phantom : PhantomData<TimeStepType>
}

/// Real-only (`T` impl `RealField`)
impl<T, TimeStepType, const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize, const N_OUT : usize, FStateTransition, FObservation, FSigmaPointsFunction>
    FilterKalmanUnscented<T, TimeStepType, STATE_DIM, MEASURE_DIM, CONTROL_DIM, N_OUT, FStateTransition, FObservation, FSigmaPointsFunction>
where
    TimeStepType: RealField + Copy + Clone + From<usize>,
    T: RealField + Copy + Clone + From<TimeStepType>,
    FStateTransition: Fn(SMatrix<T, STATE_DIM, 1>, TimeStepType) -> SMatrix<T, STATE_DIM, 1>,
    FObservation: Fn(SMatrix<T, STATE_DIM, 1>) -> SMatrix<T, MEASURE_DIM, 1>,
    FSigmaPointsFunction: SigmaPointsFunction<T, STATE_DIM, N_OUT>
{
    pub fn init(&mut self, input : &KalmanInput<T, TimeStepType, STATE_DIM, MEASURE_DIM, CONTROL_DIM>) {
        let (state, estimate_covariance, _) = self.predict_internal(input);
        self.state_vector = state;
        self.estimate_covariance = estimate_covariance;
    }

    pub fn predict(&mut self, input : &KalmanInput<T, TimeStepType, STATE_DIM, MEASURE_DIM, CONTROL_DIM>) {
        let (state, estimate_covariance, _) = self.predict_internal(input);
        self.state_vector = state;
        self.estimate_covariance = estimate_covariance;
    }

    /// Returns `(extrapolate_state, extrapolate_covariance)`.
    fn predict_internal(&mut self, input : &KalmanInput<T, TimeStepType, STATE_DIM, MEASURE_DIM, CONTROL_DIM>)
    -> (SMatrix<T, STATE_DIM, 1>, SMatrix<T, STATE_DIM, STATE_DIM>, [SMatrix<T, STATE_DIM, 1>; N_OUT])
    where 
        FSigmaPointsFunction: SigmaPointsFunction<T, STATE_DIM, N_OUT>
    {
        // update matrices moving forward in time
        let process_noise_covariance_o = if self.process_noise_covariance.is_some() {
            if let Some(delta_time) = input.delta_time {
                Some(self.process_noise_covariance.as_ref().unwrap().multiply_entries_float(delta_time))
            } else {
                Some(self.process_noise_covariance.as_ref().unwrap().matrix)
            }
        } else {
            None
        };

        let sigmas = self.sigma_points_function.generate_sigma_points();
        let weights_mean = self.sigma_points_function.generate_weights_mean();
        let weights_covariance = self.sigma_points_function.generate_weights_covariance();

        let time_step = if input.delta_time.is_some() {
            input.delta_time.unwrap()
        } else {
            TimeStepType::from(1)
        };
        
        let new_sigmas : [SMatrix<T, STATE_DIM, 1>; N_OUT] = sigmas.map(|sigma| (self.state_transition)(sigma, time_step));

        // intellisense seems to have trouble with .sum() with the matrices
        let new_state = weights_mean
            .iter()
            .zip(new_sigmas)
            .map(|(w, s)| s * *w)
            .fold(SMatrix::<T, STATE_DIM, 1>::zeros(), |acc, val| acc + val);

        let new_estimate_covariance = if let Some(process_noise_covariance) = process_noise_covariance_o {
            process_noise_covariance + weights_covariance.iter().zip(new_sigmas).map(|(w, s)| {
                (s - new_state) * (s - new_state).transpose() * *w
            })
            .fold(SMatrix::<T, STATE_DIM, STATE_DIM>::zeros(), |acc, val| acc + val)
        } else {
            weights_covariance.iter().zip(new_sigmas).map(|(w, s)| {
                (s - new_state) * (s - new_state).transpose() * *w
            })
            .fold(SMatrix::<T, STATE_DIM, STATE_DIM>::zeros(), |acc, val| acc + val)
        };

        (new_state, new_estimate_covariance, new_sigmas)
    }
}

impl<T, TimeStepType, const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize, const N_OUT : usize, FStateTransition, FObservation, FSigmaPointsFunction>
    RealTimeSignalProcessor<&KalmanInput<T, TimeStepType, STATE_DIM, MEASURE_DIM, CONTROL_DIM>, SMatrix<T, STATE_DIM, 1>>
for
    FilterKalmanUnscented<T, TimeStepType, STATE_DIM, MEASURE_DIM, CONTROL_DIM, N_OUT, FStateTransition, FObservation, FSigmaPointsFunction>
where 
    TimeStepType: RealField + Copy + Clone + From<usize>,
    T: RealField + Copy + Clone + From<TimeStepType>,
    FStateTransition: Fn(SMatrix<T, STATE_DIM, 1>, TimeStepType) -> SMatrix<T, STATE_DIM, 1>,
    FObservation: Fn(SMatrix<T, STATE_DIM, 1>) -> SMatrix<T, MEASURE_DIM, 1>,
    FSigmaPointsFunction: SigmaPointsFunction<T, STATE_DIM, N_OUT>
{
    fn process_sample(&mut self, inp : &KalmanInput<T, TimeStepType, STATE_DIM, MEASURE_DIM, CONTROL_DIM>) -> SMatrix<T, STATE_DIM, 1> {
        
        // Prediction

        let (state, estimate_covariance, sigmas) = self.predict_internal(&inp);
        self.state_vector = state;
        self.estimate_covariance = estimate_covariance;

        // Correct

        let measurement_sigmas = sigmas.map(|s| (self.observation)(s));

        todo!()
    }
}