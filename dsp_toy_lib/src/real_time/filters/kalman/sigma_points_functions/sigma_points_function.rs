use nalgebra::{SMatrix};

pub trait SigmaPointsFunction<T, const DIM : usize, const N_POINTS : usize> {
    fn generate_sigma_points(& self) -> [SMatrix<T, DIM, 1>; N_POINTS];
    fn generate_weights_mean(& self) -> [T; N_POINTS];
    fn generate_weights_covariance(& self) -> [T; N_POINTS];
}