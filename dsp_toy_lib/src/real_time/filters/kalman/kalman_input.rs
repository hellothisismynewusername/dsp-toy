use nalgebra::SMatrix;

/// T is the type that's in matrices. Can be Complex<f64> or f64.
/// `process_noise_covariance` is there if you want to update Q during the run.
pub struct KalmanInput<T, const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize> {
    pub measurement_vector : SMatrix<T, MEASURE_DIM, 1>,
    pub control_vector : Option<SMatrix<T, CONTROL_DIM, 1>>,
    pub process_noise_covariance : Option<SMatrix<T, STATE_DIM, STATE_DIM>>
}