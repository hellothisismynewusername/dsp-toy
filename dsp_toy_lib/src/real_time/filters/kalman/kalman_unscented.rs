use nalgebra::{Const, MatrixView, RealField, SMatrix};

use crate::{real_time::{filters::kalman::{kalman_input::KalmanInput, sigma_points_functions::sigma_points_function::SigmaPointsFunction}, real_time_signal_processer::RealTimeSignalProcessor}, utility::SMatrixTimes};

/// Control matrix isn't used anymore, it can be accounted for in the state transition function.
/// 
/// So, user will have to add their own custom behaviours to the state transition function, but `T` is still passed in as a time variable.
/// This is to uphold the passed-in `delta_time` from `KalmanInput`.
/// 
/// To do controlling in state transition function aside from the optional control input, you could use an `Rc<RefCell<T>>` pattern.
pub struct FilterKalmanUnscented<T, TimeStepType, const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize, const N_OUT : usize, FStateTransition, FObservation, FSigmaPointsFunction, FStateMean, FMeasureMean, FResidualZ, FResidualX, FAddX>
where 
    TimeStepType: RealField,
    FStateTransition: FnMut(SMatrix<T, STATE_DIM, 1>, TimeStepType, Option<SMatrix<T, CONTROL_DIM, 1>>) -> SMatrix<T, STATE_DIM, 1>,
    FObservation: Fn(SMatrix<T, STATE_DIM, 1>) -> SMatrix<T, MEASURE_DIM, 1>,
    FSigmaPointsFunction: SigmaPointsFunction<T, STATE_DIM, N_OUT>,
    FStateMean: Fn([SMatrix<T, STATE_DIM, 1>; N_OUT], [T; N_OUT]) -> SMatrix<T, STATE_DIM, 1>,
    FMeasureMean: Fn([SMatrix<T, MEASURE_DIM, 1>; N_OUT], [T; N_OUT]) -> SMatrix<T, MEASURE_DIM, 1>,
    FResidualZ: Fn(SMatrix<T, MEASURE_DIM, 1>, SMatrix<T, MEASURE_DIM, 1>) -> SMatrix<T, MEASURE_DIM, 1>,
    FResidualX: Fn(SMatrix<T, STATE_DIM, 1>, SMatrix<T, STATE_DIM, 1>) -> SMatrix<T, STATE_DIM, 1>,
    FAddX: Fn(SMatrix<T, STATE_DIM, 1>, MatrixView<'_, T, Const<STATE_DIM>, Const<1>, Const<1>, Const<STATE_DIM>>) -> SMatrix<T, STATE_DIM, 1> + Copy
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
    /// Sigma Generator Function
    pub sigma_generator_function: FSigmaPointsFunction,
    /// Weighted average function. State sigma points vectors[], Mean weights[] -> Mean state vector.
    /// 
    /// `Fn([SMatrix<T, STATE_DIM, 1>; N_OUT], [T; N_OUT]) -> SMatrix<T, STATE_DIM, 1>`
    pub state_mean_function : FStateMean,
    /// Weighted average function. Measurement sigma point vectors[], Mean weights[] -> Mean measurement vector.
    /// 
    /// `Fn([SMatrix<T, MEASURE_DIM, 1>; N_OUT], [T; N_OUT]) -> SMatrix<T, MEASURE_DIM, 1>`
    pub measure_mean_function : FMeasureMean,
    /// `Fn(SMatrix<T, MEASURE_DIM, 1>, SMatrix<T, MEASURE_DIM, 1>) -> SMatrix<T, MEASURE_DIM, 1>`
    pub residual_z_function : FResidualZ,
    /// `Fn(SMatrix<T, STATE_DIM, 1>, SMatrix<T, STATE_DIM, 1>) -> SMatrix<T, STATE_DIM, 1>`
    pub residual_x_function : FResidualX,
    /// Function to add to state, to, for example, allow for custom wraparound logic if needed. `bool` parameter is `true` if subtraction is being done.
    /// 
    /// `Fn(SMatrix<T, STATE_DIM, 1>, MatrixView<'_, T, Const<STATE_DIM>, Const<1>, Const<1>, Const<STATE_DIM>>) -> SMatrix<T, STATE_DIM, 1>`
    pub add_state_function: FAddX
}

/// Real-only (`T` impl `RealField`)
impl<T, TimeStepType, const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize, const N_OUT : usize, FStateTransition, FObservation, FSigmaPointsFunction, FStateMean, FMeasureMean, FResidualZ, FResidualX, FAddX>
    FilterKalmanUnscented<T, TimeStepType, STATE_DIM, MEASURE_DIM, CONTROL_DIM, N_OUT, FStateTransition, FObservation, FSigmaPointsFunction, FStateMean, FMeasureMean, FResidualZ, FResidualX, FAddX>
where
    TimeStepType: RealField + Copy + Clone,
    T: RealField + Copy + Clone + From<TimeStepType>,
    FStateTransition: FnMut(SMatrix<T, STATE_DIM, 1>, TimeStepType, Option<SMatrix<T, CONTROL_DIM, 1>>) -> SMatrix<T, STATE_DIM, 1>,
    FObservation: Fn(SMatrix<T, STATE_DIM, 1>) -> SMatrix<T, MEASURE_DIM, 1>,
    FSigmaPointsFunction: SigmaPointsFunction<T, STATE_DIM, N_OUT>,
    FStateMean: Fn([SMatrix<T, STATE_DIM, 1>; N_OUT], [T; N_OUT]) -> SMatrix<T, STATE_DIM, 1>,
    FMeasureMean: Fn([SMatrix<T, MEASURE_DIM, 1>; N_OUT], [T; N_OUT]) -> SMatrix<T, MEASURE_DIM, 1>,
    FResidualZ: Fn(SMatrix<T, MEASURE_DIM, 1>, SMatrix<T, MEASURE_DIM, 1>) -> SMatrix<T, MEASURE_DIM, 1>,
    FResidualX: Fn(SMatrix<T, STATE_DIM, 1>, SMatrix<T, STATE_DIM, 1>) -> SMatrix<T, STATE_DIM, 1>,
    FAddX: Fn(SMatrix<T, STATE_DIM, 1>, MatrixView<'_, T, Const<STATE_DIM>, Const<1>, Const<1>, Const<STATE_DIM>>) -> SMatrix<T, STATE_DIM, 1> + Copy
{
    pub fn init(&mut self, input : &KalmanInput<T, TimeStepType, STATE_DIM, MEASURE_DIM, CONTROL_DIM>) {
        let (state, estimate_covariance, _, _ , _) = self.predict_internal(input);
        self.state_vector = state;
        self.estimate_covariance = estimate_covariance;
    }

    pub fn predict(&mut self, input : &KalmanInput<T, TimeStepType, STATE_DIM, MEASURE_DIM, CONTROL_DIM>) {
        let (state, estimate_covariance, _, _, _) = self.predict_internal(input);
        self.state_vector = state;
        self.estimate_covariance = estimate_covariance;
    }

    /// Returns `(extrapolate_state, extrapolate_covariance)`.
    fn predict_internal(&mut self, input : &KalmanInput<T, TimeStepType, STATE_DIM, MEASURE_DIM, CONTROL_DIM>)
    -> (SMatrix<T, STATE_DIM, 1>, SMatrix<T, STATE_DIM, STATE_DIM>, [SMatrix<T, STATE_DIM, 1>; N_OUT], [T; N_OUT], [T; N_OUT])
    where 
        FSigmaPointsFunction: SigmaPointsFunction<T, STATE_DIM, N_OUT>
    {
        // update matrices moving forward in time
        let process_noise_covariance_o = if self.process_noise_covariance.is_some() {
            if let Some(input_process_noise_covariance) = input.process_noise_covariance {
                Some(input_process_noise_covariance)
            } else {
                if let Some(delta_time) = input.delta_time {
                    Some(self.process_noise_covariance.as_ref().unwrap().multiply_entries_float(delta_time))
                } else {
                    Some(self.process_noise_covariance.as_ref().unwrap().matrix)
                }
            }
        } else {
            None
        };

        let sigmas = self.sigma_generator_function.generate_sigma_points(self.state_vector, self.estimate_covariance, self.add_state_function);
        let w_m = self.sigma_generator_function.generate_w_m();
        let w_c = self.sigma_generator_function.generate_w_c();

        let time_step = if input.delta_time.is_some() {
            input.delta_time.unwrap()
        } else {
            TimeStepType::from_usize(1).unwrap()
        };
        
        let new_sigmas : [SMatrix<T, STATE_DIM, 1>; N_OUT] = sigmas.map(|sigma| (self.state_transition)(sigma, time_step, input.control_vector));

        // intellisense seems to have trouble with .sum() with the matrices
        let new_state = (self.state_mean_function)(new_sigmas, w_m);

        let new_estimate_covariance = if let Some(process_noise_covariance) = process_noise_covariance_o {
            process_noise_covariance + w_c.iter().zip(new_sigmas).map(|(w, s)| {
                let res = (self.residual_x_function)(s, new_state);
                res * res.transpose() * *w
            })
            .fold(SMatrix::zeros(), |acc, val| acc + val)
        } else {
            w_c.iter().zip(new_sigmas).map(|(w, s)| {
                let res = (self.residual_x_function)(s, new_state);
                res * res.transpose() * *w
            })
            .fold(SMatrix::zeros(), |acc, val| acc + val)
        };

        (new_state, new_estimate_covariance, new_sigmas, w_m, w_c)
    }
}

impl<T, TimeStepType, const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize, const N_OUT : usize, FStateTransition, FObservation, FSigmaPointsFunction, FStateMean, FMeasureMean, FResidualZ, FResidualX, FAddX>
    RealTimeSignalProcessor<&KalmanInput<T, TimeStepType, STATE_DIM, MEASURE_DIM, CONTROL_DIM>, SMatrix<T, STATE_DIM, 1>>
for
    FilterKalmanUnscented<T, TimeStepType, STATE_DIM, MEASURE_DIM, CONTROL_DIM, N_OUT, FStateTransition, FObservation, FSigmaPointsFunction, FStateMean, FMeasureMean, FResidualZ, FResidualX, FAddX>
where 
    TimeStepType: RealField + Copy + Clone,
    T: RealField + Copy + Clone + From<TimeStepType>,
    FStateTransition: FnMut(SMatrix<T, STATE_DIM, 1>, TimeStepType, Option<SMatrix<T, CONTROL_DIM, 1>>) -> SMatrix<T, STATE_DIM, 1>,
    FObservation: Fn(SMatrix<T, STATE_DIM, 1>) -> SMatrix<T, MEASURE_DIM, 1>,
    FSigmaPointsFunction: SigmaPointsFunction<T, STATE_DIM, N_OUT>,
    FStateMean: Fn([SMatrix<T, STATE_DIM, 1>; N_OUT], [T; N_OUT]) -> SMatrix<T, STATE_DIM, 1>,
    FMeasureMean: Fn([SMatrix<T, MEASURE_DIM, 1>; N_OUT], [T; N_OUT]) -> SMatrix<T, MEASURE_DIM, 1>,
    FResidualZ: Fn(SMatrix<T, MEASURE_DIM, 1>, SMatrix<T, MEASURE_DIM, 1>) -> SMatrix<T, MEASURE_DIM, 1>,
    FResidualX: Fn(SMatrix<T, STATE_DIM, 1>, SMatrix<T, STATE_DIM, 1>) -> SMatrix<T, STATE_DIM, 1>,
    FAddX: Fn(SMatrix<T, STATE_DIM, 1>, MatrixView<'_, T, Const<STATE_DIM>, Const<1>, Const<1>, Const<STATE_DIM>>) -> SMatrix<T, STATE_DIM, 1> + Copy
{
    fn process_sample(&mut self, inp : &KalmanInput<T, TimeStepType, STATE_DIM, MEASURE_DIM, CONTROL_DIM>) -> SMatrix<T, STATE_DIM, 1> {
        
        // Prediction

        let (state, estimate_covariance, sigmas, w_m, w_c) = self.predict_internal(&inp);
        self.state_vector = state;
        self.estimate_covariance = estimate_covariance;

        // Correct

        let measurement_sigmas = sigmas.map(|s| (self.observation)(s));

        let mean_measurement = (self.measure_mean_function)(measurement_sigmas, w_m);

        let mut covariance_measurement = self.measure_covariance + w_c
            .iter()
            .zip(measurement_sigmas)
            .map(|(w, s)| {
                let res = (self.residual_z_function)(s, mean_measurement);
                res * res.transpose() * *w
            })
            .fold(SMatrix::zeros(), |acc, val| acc + val);

        let innovation = (self.residual_z_function)(inp.measurement_vector, mean_measurement);

        // compute cross covariance of state and measurement
        let cross_covariance = w_c
            .iter()
            .zip(sigmas)
            .zip(measurement_sigmas)
            .map(|((w, s_y), s_z)| {
                let res_x = (self.residual_x_function)(s_y, state);
                let res_z = (self.residual_z_function)(s_z, mean_measurement);
                res_x * res_z.transpose() * *w
            })
            .fold(SMatrix::zeros(), |acc, val| acc + val);

        if !covariance_measurement.try_inverse_mut() {
            panic!("Innovation covariance couldn't be inverted");
        }

        let kalman_gain = cross_covariance * covariance_measurement;

        let updated_state = (self.add_state_function)(state, (kalman_gain * innovation).as_view());
        self.state_vector = updated_state;

        // K * P_z = P_xz, substitute in to fix the mutated P_z problem
        let updated_estimate_covariance = estimate_covariance - cross_covariance * kalman_gain.transpose();

        self.estimate_covariance = updated_estimate_covariance;

        updated_state
    }
}