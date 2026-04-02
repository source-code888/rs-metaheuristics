use crate::problem::{FromSeed, Individual};
use nalgebra::DVector;
use rand::{RngExt, rng};
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct Whale {
    pub(crate) position: DVector<f64>,
    pub(crate) fitness: f64,
    pub(crate) sequence: DVector<usize>,
}

impl Whale {
    pub fn ranked_order_value(&mut self) {
        let mut indexed_positions: Vec<(usize, &f64)> = self.position.iter().enumerate().collect();
        indexed_positions.sort_by(|a, b| a.1.total_cmp(b.1));
        let mut result: DVector<usize> = DVector::zeros(self.sequence.len());
        for (i, (original_index, _)) in indexed_positions.iter().enumerate() {
            result[*original_index] = self.sequence[i];
        }
        self.sequence = result;
    }

    pub fn with_random_components(
        dim: usize,
        lower_bound: f64,
        upper_bound: f64,
        sequence: DVector<usize>,
    ) -> Self {
        let position: DVector<f64> =
            DVector::from_fn(dim, |_, _| rng().random_range(lower_bound..=upper_bound));
        let mut whale = Self {
            position,
            sequence,
            fitness: 0f64,
        };
        whale.ranked_order_value();
        whale
    }

    #[allow(unused)]
    pub fn new(position: DVector<f64>, sequence: DVector<usize>, fitness: f64) -> Self {
        Self {
            position,
            sequence,
            fitness,
        }
    }
}

impl Individual<usize> for Whale {
    fn solution_vector(&self) -> &[usize] {
        self.sequence.as_slice()
    }

    fn position_vector(&self) -> &DVector<f64> {
        &self.position
    }

    fn fitness(&self) -> f64 {
        self.fitness
    }
    fn check_if_goes_beyond_bounds(&mut self, l_bound: f64, u_bound: f64) {
        let pos = self.position.as_mut_slice();
        pos.par_iter_mut().for_each(|p| {
            if *p < l_bound {
                *p = l_bound;
            } else if *p > u_bound {
                *p = u_bound;
            }
        });
        self.ranked_order_value();
    }

    fn update_position_vector(&mut self, new: DVector<f64>) {
        self.position = new
    }

    fn update_fitness(&mut self, new: f64) {
        self.fitness = new
    }
}

impl PartialEq for Whale {
    fn eq(&self, other: &Self) -> bool {
        self.fitness == other.fitness
            && self.sequence == other.sequence
            && self.position == other.position
    }
}

impl PartialOrd for Whale {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }

    fn lt(&self, other: &Self) -> bool {
        self.fitness < other.fitness
    }

    fn le(&self, other: &Self) -> bool {
        self.fitness <= other.fitness
    }

    fn gt(&self, other: &Self) -> bool {
        self.fitness > other.fitness
    }

    fn ge(&self, other: &Self) -> bool {
        self.fitness >= other.fitness
    }
}
impl Eq for Whale {}

impl Ord for Whale {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.fitness.total_cmp(&other.fitness)
    }
}

impl FromSeed for Whale {
    fn from_seed<F>(size: usize, func: F) -> Vec<Whale>
    where
        F: Fn() -> Self,
    {
        (0..size).map(|_| func()).collect::<Vec<Whale>>()
    }

    fn zeros(dim: usize) -> Whale {
        Self {
            position: DVector::zeros(dim),
            sequence: DVector::zeros(dim),
            fitness: 0f64,
        }
    }
}
