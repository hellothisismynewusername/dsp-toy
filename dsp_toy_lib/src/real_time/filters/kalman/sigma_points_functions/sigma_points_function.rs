use nalgebra::{Const, MatrixView, SMatrix};

pub trait SigmaPointsFunction<T, const DIM : usize, const N_POINTS : usize> {
    fn generate_sigma_points<FAddX>(
        &self,
        state_distribution_mean_vector : SMatrix<T, DIM, 1>,
        state_covariance : SMatrix<T, DIM, DIM>,
        add_state_function : FAddX
    ) -> [SMatrix<T, DIM, 1>; N_POINTS]
    where 
        FAddX: Fn(SMatrix<T, DIM, 1>, MatrixView<'_, T, Const<DIM>, Const<1>, Const<1>, Const<DIM>>) -> SMatrix<T, DIM, 1>
    ;
    fn generate_w_m(&self) -> [T; N_POINTS];
    fn generate_w_c(&self) -> [T; N_POINTS];
}