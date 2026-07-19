use nalgebra::{ComplexField, SMatrix};

use crate::real_time::filters::kalman::sigma_points_functions::sigma_points_function::SigmaPointsFunction;

/// For UKF additive noise models, so `N` = `STATE_DIM`.
/// 
/// Generally, `N_OUT` = `2 * (STATE_DIM) + 1`, where `N_OUT` is the number of points generated.
pub struct Julier<T, const STATE_DIM : usize, const N_OUT : usize> {
    pub state_distribution_mean_vector : SMatrix<T, STATE_DIM, 1>,
    pub state_covariance : SMatrix<T, STATE_DIM, STATE_DIM>,
    pub kappa : T
}

impl<T, const STATE_DIM : usize, const N_OUT : usize> SigmaPointsFunction<T, STATE_DIM, N_OUT> for Julier<T, STATE_DIM, N_OUT>
where 
    T: ComplexField + Copy + From<usize>
{
    fn generate_sigma_points(&self) -> [SMatrix<T, STATE_DIM, 1>; N_OUT] {
        let mut out = [self.state_distribution_mean_vector; N_OUT];

        let matrix_tmp = (self.state_covariance * (T::from(STATE_DIM) + self.kappa))
            .cholesky()
            .expect("Cholesky failed unexpectedly during sigma_points_julier()")
            .unpack();

        // matrix_tmp is a lower triangle; use ith col instead of ith row
        for i in 1..(STATE_DIM + 1) {
            out[i] = self.state_distribution_mean_vector + matrix_tmp.column(i);
            out[i + STATE_DIM] = self.state_distribution_mean_vector - matrix_tmp.column(i);
        }

        out
    }

    /// Same weights as mean weights
    fn generate_weights_covariance(&self) -> [T; N_OUT] {
        let mut out = [T::from(1) /  (T::from(2) * (T::from(STATE_DIM) + self.kappa)); N_OUT];

        out[0] = self.kappa / (T::from(STATE_DIM) + self.kappa);
        
        out
    }

    /// Same weights as covariance weights
    fn generate_weights_mean(&self) -> [T; N_OUT] {
        let mut out = [T::from(1) /  (T::from(2) * (T::from(STATE_DIM) + self.kappa)); N_OUT];

        out[0] = self.kappa / (T::from(STATE_DIM) + self.kappa);
        
        out
    }
}