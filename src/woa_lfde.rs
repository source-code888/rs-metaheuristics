use crate::problem::{FromSeed, Individual, ProblemBounds, Solvable};
use crate::utils::{exclusive_usize, levy_flight, sign};
use nalgebra::DVector;
use rand::{RngExt, rng};
use rayon::prelude::*;
use std::{
    f64::consts::{E, PI},
    ops::{Add, Sub},
};

pub struct WoaLfde<T, R>
where
    R: Send + Sync + Clone + FromSeed + Individual<T> + Ord,
{
    pub(crate) problem: Box<dyn Solvable<T, R> + Send + Sync>,
    pub(crate) bounds: ProblemBounds,
    pub(crate) maximization: bool,
    pub(crate) max_iterations: usize,
    pub(crate) pool_size: usize,
    pub(crate) whale_dim: usize,
    pub(crate) whales: Vec<R>,
    pub(crate) best_whale: R,
}

impl<T, R> WoaLfde<T, R>
where
    R: Send + Sync + Clone + FromSeed + Individual<T> + Ord,
{
    pub fn new<F>(
        problem: Box<dyn Solvable<T, R> + Send + Sync>,
        bounds: ProblemBounds,
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
            self.whales
                .par_iter_mut()
                .enumerate()
                .for_each(|(flag, w)| {
                    let position = w.position_vector();
                    let best_position = self.best_whale.position_vector();
                    let r: DVector<f64> = DVector::from_fn(dim, |_, _| {
                        rng().random_range(self.bounds.0..=self.bounds.1)
                    });
                    let coefficient_a: DVector<f64> = (&r * a).add_scalar(a);
                    let coefficient_c: DVector<f64> = 2f64 * r;
                    let l: f64 = rng().random_range(-1f64..=1f64);
                    let p: f64 = rng().random_range(0f64..=1f64);
                    if p < 0.5f64 {
                        if coefficient_a.norm() < 1f64 {
                            let d_coefficient: DVector<f64> =
                                (coefficient_c.component_mul(best_position) - position).abs();
                            let new_pos: DVector<f64> =
                                best_position - coefficient_a.component_mul(&d_coefficient);
                            w.update_position_vector(new_pos);
                        } else {
                            let random_pos: usize = exclusive_usize(0, self.pool_size, vec![flag]);
                            let r_whale: &DVector<f64> = aux_whales[random_pos].position_vector();
                            let sg: DVector<f64> = sign(DVector::from_vec(
                                (0..dim)
                                    .into_par_iter()
                                    .map(|_| rng().random_range(0f64..1f64) - 0.5f64)
                                    .collect::<Vec<f64>>(),
                            ));
                            let levy: DVector<f64> =
                                DVector::from_vec(levy_flight(dim, 1.5f64, 0.1f64));
                            let new_pos: DVector<f64> = r_whale.add(
                                sg.component_mul(&levy)
                                    .component_mul(&r_whale.sub(position)),
                            );
                            w.update_position_vector(new_pos);
                        }
                    } else {
                        let d_coefficient: DVector<f64> = (best_position - position).abs();
                        let new_pos: DVector<f64> =
                            best_position.add(d_coefficient * E.powf(l) * (2f64 * PI * l).cos());
                        w.update_position_vector(new_pos);
                    }
                    // Here I just want to use ranked order value, I do not want to clamp the position
                    w.check_if_goes_beyond_bounds(f64::MIN, f64::MAX);
                    let fitness = self.problem.solve(&w);
                    w.update_fitness(fitness);
                    if (flag + 1) % 3 == 0 && fitness >= self.best_whale.fitness() {
                        // DE current_to_best
                        let position = w.position_vector();
                        let f: f64 = rng().random_range(0f64..=2f64);
                        let r1: usize = exclusive_usize(0, self.pool_size, vec![flag]);
                        let r2: usize = exclusive_usize(0, self.pool_size, vec![flag, r1]);
                        let r1_w = aux_whales[r1].position_vector();
                        let r2_w = aux_whales[r2].position_vector();
                        let j_rand: usize = rng().random_range(0usize..dim);
                        let mut_vec: DVector<f64> = position.add(
                            best_position
                                .sub(position)
                                .scale(f)
                                .add(r1_w.sub(r2_w).scale(f)),
                        );
                        let mixed: Vec<f64> = mut_vec
                            .iter()
                            .enumerate()
                            .map(|(j, x)| {
                                let r: f64 = rng().random_range(0f64..=1f64);
                                if j == j_rand || r < 0.6f64 {
                                    *x
                                } else {
                                    position[j]
                                }
                            })
                            .collect::<Vec<f64>>();
                        let mut new_whale: R = w.clone();
                        new_whale.update_position_vector(DVector::from_vec(mixed));
                        new_whale.check_if_goes_beyond_bounds(f64::MIN, f64::MAX);
                        let new_whale_fitness = self.problem.solve(&new_whale);
                        new_whale.update_fitness(new_whale_fitness);
                        if new_whale_fitness <= fitness {
                            w.update_position_vector(new_whale.position_vector().clone());
                        }
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
