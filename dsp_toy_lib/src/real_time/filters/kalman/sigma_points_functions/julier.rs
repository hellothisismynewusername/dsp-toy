use std::ops::Neg;

use nalgebra::{ComplexField, Const, MatrixView, RealField, SMatrix, Scalar};

use crate::real_time::filters::kalman::sigma_points_functions::sigma_points_function::SigmaPointsFunction;

/// For UKF additive noise models, so `N` = `STATE_DIM`.
/// 
/// Generally, `N_OUT` = `2 * (STATE_DIM) + 1`, where `N_OUT` is the number of points generated.
pub struct Julier<T, const STATE_DIM : usize, const N_OUT : usize> {
    pub kappa : T
}

impl<T, const STATE_DIM : usize, const N_OUT : usize> Julier<T, STATE_DIM, N_OUT>
where 
    T: ComplexField + Copy
{
    pub fn new(kappa : T) -> Julier<T, STATE_DIM, N_OUT> {
        Julier { kappa: kappa }
    }
}

impl<T, const STATE_DIM : usize, const N_OUT : usize> SigmaPointsFunction<T, STATE_DIM, N_OUT> for Julier<T, STATE_DIM, N_OUT>
where 
    T: ComplexField + Copy
{
    fn generate_sigma_points<FAddX>(
        &self,
        state_distribution_mean_vector : SMatrix<T, STATE_DIM, 1>,
        state_covariance : SMatrix<T, STATE_DIM, STATE_DIM>,
        state_add_function : FAddX
    ) -> [SMatrix<T, STATE_DIM, 1>; N_OUT]
    where 
        FAddX: Fn(SMatrix<T, STATE_DIM, 1>, MatrixView<'_, T, Const<STATE_DIM>, Const<1>, Const<1>, Const<STATE_DIM>>) -> SMatrix<T, STATE_DIM, 1>
    {
        let mut out = [state_distribution_mean_vector; N_OUT];

        let matrix_tmp = (state_covariance * (T::from_usize(STATE_DIM).unwrap() + self.kappa))
            .cholesky()
            .expect("Cholesky failed unexpectedly during sigma_points_julier()")
            .unpack();

        // matrix_tmp is a lower triangle; use ith col instead of ith row
        for i in 0..STATE_DIM {
            out[i] = state_add_function(state_distribution_mean_vector, matrix_tmp.column(i));
            out[i + STATE_DIM] = state_add_function(state_distribution_mean_vector, matrix_tmp.column(i).neg().as_view());
        }

        out
    }

    /// Same weights as mean weights
    fn generate_w_c(&self) -> [T; N_OUT] {
        let mut out = [T::one() /  (T::from_usize(2).unwrap() * (T::from_usize(STATE_DIM).unwrap() + self.kappa)); N_OUT];

        out[0] = self.kappa / (T::from_usize(STATE_DIM).unwrap() + self.kappa);
        
        out
    }

    /// Same weights as covariance weights
    fn generate_w_m(&self) -> [T; N_OUT] {
        let mut out = [T::one() /  (T::from_usize(2).unwrap() * (T::from_usize(STATE_DIM).unwrap() + self.kappa)); N_OUT];

        out[0] = self.kappa / (T::from_usize(STATE_DIM).unwrap()  + self.kappa);
        
        out
    }
}