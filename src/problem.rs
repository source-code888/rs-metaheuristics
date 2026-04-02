use nalgebra::DVector;

pub trait Solvable<T, R>
where
    R: FromSeed + Individual<T>,
{
    fn solve(&self, individual: &R) -> f64;
}

pub trait Individual<T> {
    fn solution_vector(&self) -> &[T];

    fn position_vector(&self) -> &DVector<f64>;

    fn fitness(&self) -> f64;

    fn check_if_goes_beyond_bounds(&mut self, l_bound: f64, u_bound: f64);

    fn update_position_vector(&mut self, new: DVector<f64>);

    fn update_fitness(&mut self, new: f64);
}

pub trait FromSeed
where
    Self: Sized,
{
    fn from_seed<F>(size: usize, func: F) -> Vec<Self>
    where
        F: Fn() -> Self;

    fn zeros(dim: usize) -> Self;
}
