use crate::problem::{FromSeed, Individual, Solvable};
use nalgebra::DVector;
use rand::{RngExt, rng};
use std::f64::consts::{E, PI};
use std::ops::Add;

use rayon::prelude::*;

mod whale;

pub struct Bounds(pub f64, pub f64);

pub struct Algorithm<T, R>
where
    R: Send + Sync + Clone + FromSeed + Individual<T> + Ord,
{
    pub(crate) problem: Box<dyn Solvable<T, R> + Send + Sync>,
    pub(crate) bounds: Bounds,
    pub(crate) maximization: bool,
    pub(crate) max_iterations: usize,
    pub(crate) pool_size: usize,
    pub(crate) whale_dim: usize,
    pub(crate) whales: Vec<R>,
    pub(crate) best_whale: R,
}

impl<T, R> Algorithm<T, R>
where
    R: Send + Sync + Clone + FromSeed + Individual<T> + Ord,
{
    pub fn new<F>(
        problem: Box<dyn Solvable<T, R> + Send + Sync>,
        bounds: Bounds,
        maximization: bool,
        max_iterations: usize,
        pool_size: usize,
        whale_dim: usize,
        seed_fn: F,
    ) -> Self
    where
        F: Fn() -> R,
    {
        Self {
            problem,
            whales: R::from_seed(pool_size, seed_fn),
            bounds,
            maximization,
            max_iterations,
            pool_size,
            best_whale: R::zeros(whale_dim),
            whale_dim,
        }
    }

    pub fn solve_problem_for_each_whale(&mut self) {
        self.whales.par_iter_mut().for_each(|w| {
            w.check_if_goes_beyond_bounds(self.bounds.0, self.bounds.1);
            let fit = self.problem.solve(w);
            w.update_fitness(fit)
        })
    }

    pub fn sort_whales(&mut self) {
        if self.maximization {
            self.whales.par_sort_by(|a, b| b.partial_cmp(a).unwrap())
        } else {
            self.whales.par_sort()
        }
    }

    pub fn solve(&mut self) {
        self.sort_whales();
        self.best_whale = self.whales[0].clone();
        let dim = self.whale_dim;
        for t in 0..self.max_iterations {
            let a: f64 = 2f64 - t as f64 * 2f64 / self.max_iterations as f64;
            let aux_whales = self.whales.clone();
            self.whales.par_iter_mut().for_each(|w| {
                let position = w.position_vector();
                let r: DVector<f64> = DVector::from_fn(dim, |_, _| {
                    rng().random_range(self.bounds.0..=self.bounds.1)
                });
                let coefficient_a: DVector<f64> = (&r * a).add_scalar(a);
                let coefficient_c: DVector<f64> = 2f64 * r;
                let l: f64 = rng().random_range(-1f64..=1f64);
                let p: f64 = rng().random_range(0f64..=1f64);
                let best_position = self.best_whale.position_vector();
                if p < 0.5 {
                    if coefficient_a.norm() < 1f64 {
                        let d_coefficient: DVector<f64> =
                            (coefficient_c.component_mul(best_position) - position).abs();
                        let new_pos: DVector<f64> =
                            best_position - coefficient_a.component_mul(&d_coefficient);
                        w.update_position_vector(new_pos);
                    } else {
                        let random_pos: usize =
                            rng().random_range(0f64..self.pool_size as f64) as usize;
                        let r_whale = aux_whales[random_pos].position_vector();

                        let d_coefficient: DVector<f64> =
                            (coefficient_c.component_mul(r_whale) - position).abs();
                        let new_pos: DVector<f64> =
                            r_whale - coefficient_a.component_mul(&d_coefficient);
                        w.update_position_vector(new_pos);
                    }
                } else {
                    let d_coefficient: DVector<f64> = (best_position - position).abs();
                    let new_pos: DVector<f64> =
                        best_position.add(d_coefficient * E.powf(l) * (2f64 * PI * l).cos());
                    w.update_position_vector(new_pos);
                }
            });
            self.solve_problem_for_each_whale();
            self.sort_whales();
            let current_gen_best = self.whales[0].clone();
            let curr_best_fit = self.best_whale.fitness();
            let curr_gen_best_fit = current_gen_best.fitness();
            if (curr_gen_best_fit < curr_best_fit && !self.maximization)
                || (curr_gen_best_fit > curr_best_fit && self.maximization)
            {
                self.best_whale = current_gen_best;
            }
        }
    }
}
