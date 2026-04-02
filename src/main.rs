use crate::jssp::whale::Whale;
use crate::jssp::{Instance, JobShopSchedulingProblem};
use crate::problem::Solvable;
use crate::woa::{Algorithm, Bounds};
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::io::Error;
use std::time::Instant;

mod jssp;
mod problem;
mod woa;
fn main() -> Result<(), Error> {
    let start = Instant::now();
    let threads = std::thread::available_parallelism()?.get();
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();
    let lower_bound = 0f64;
    let upper_bound = 1f64;
    let pool_size = 40usize;
    let maximization = false;
    let max_iterations = 300usize;
    let executions: usize = 1000;
    println!("Available threads: {threads}, Will execute {executions} tasks.");
    let mut bests: Vec<Whale> = (0..executions)
        .into_par_iter()
        .map(move |_| {
            let instance: JobShopSchedulingProblem =
                JobShopSchedulingProblem::from_instance(Instance::LA07).unwrap();
            let whale_dim: usize = instance.n_jobs * instance.n_machines;
            let mut solution = Algorithm::new(
                Box::new(instance.clone()),
                Bounds(lower_bound, upper_bound),
                maximization,
                max_iterations,
                pool_size,
                whale_dim,
                move || {
                    let sequence = instance.generate_base_sequence();
                    let mut whale: Whale = Whale::with_random_components(
                        whale_dim,
                        lower_bound,
                        upper_bound,
                        sequence.clone(),
                    );
                    whale.fitness = instance.solve(&whale);
                    whale
                },
            );
            solution.solve();
            solution.best_whale
        })
        .collect::<Vec<Whale>>();
    bests.par_sort();
    let sum: f64 = bests.par_iter().map(|x| x.fitness).sum::<f64>();
    println!("Avg fitness: {}", sum / executions as f64);
    println!("Best fitness: {}", bests[0].fitness);
    println!("Worst fitness: {}", bests[executions - 1].fitness);
    println!("Execution time: {:?}", start.elapsed());
    Ok(())
}
