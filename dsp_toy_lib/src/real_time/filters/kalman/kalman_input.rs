use nalgebra::SMatrix;

/// T should be Complex<f64> or f64
pub struct KalmanInput<T, const STATE_DIM : usize, const MEASURE_DIM : usize, const CONTROL_DIM : usize> {
    measurement_vector : SMatrix<T, MEASURE_DIM, 1>,
    input_vector : Option<SMatrix<T, CONTROL_DIM, 1>>,
    process_noise_covariance : Option<SMatrix<T, STATE_DIM, STATE_DIM>>
}