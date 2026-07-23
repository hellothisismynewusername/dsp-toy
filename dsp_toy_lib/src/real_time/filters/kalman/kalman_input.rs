use nalgebra::SMatrix;

/// T is the type that's in matrices. Can be Complex.
/// `process_noise_covariance` is there if you want to update Q during the run.
/// No need to use if you don't have to change anything on the fly and nothing relies on time step
pub struct KalmanInput<T, TimeStepType, const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize> {
    /// `MEASURE_DIM` x `1`
    pub measurement_vector : SMatrix<T, MEASURE_DIM, 1>,
    /// `CONTROL_DIM` x `1`
    pub control_vector : Option<SMatrix<T, CONTROL_DIM, 1>>,
    /// `STATE_DIM` x `STATE_DIM`
    pub process_noise_covariance : Option<SMatrix<T, STATE_DIM, STATE_DIM>>,
    pub delta_time : Option<TimeStepType>
}