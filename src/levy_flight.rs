use std::f64::consts::PI;

use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use statrs::function::gamma::gamma;

pub(crate) fn levy_flight(dim: usize, beta: f64, alfa: f64) -> Vec<f64> {
    let normal = Normal::new(0f64, 1f64).unwrap();
    (0..dim)
        .into_par_iter()
        .map(|_| {
            let sigma_u = ((gamma(1f64 + beta) * (PI * beta / 2f64).sin())
                / (gamma((1f64 + beta) / 2f64) * beta * 2f64.powf((beta - 1f64) / 2f64)))
            .powf(1f64 / beta);
            let sigma_v = 1f64;
            let mut rng = rand::rng();
            let u: f64 = normal.sample(&mut rng) * sigma_u;
            let v: f64 = normal.sample(&mut rng) * sigma_v;
            (u / v.abs().powf(1f64 / beta)) * alfa
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levy_flight() {
        let result = levy_flight(10, 1.5, 0.1);
        assert_eq!(result.len(), 10);
    }
}
