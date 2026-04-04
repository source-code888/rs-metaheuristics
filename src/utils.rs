use std::f64::consts::PI;

use nalgebra::DVector;
use rand::RngExt;
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

pub(crate) fn sign(x: DVector<f64>) -> DVector<f64> {
    x.map(|v| v.signum())
}

pub(crate) fn exclusive_usize(l_bound: usize, u_bound: usize, exclude: Vec<usize>) -> usize {
    let mut rng = rand::rng();
    let mut index: usize = rng.random_range(l_bound..u_bound);
    while exclude.contains(&index) {
        index = rng.random_range(l_bound..u_bound);
    }
    index
}
