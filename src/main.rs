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
    let lower_bound = -1f64;
    let upper_bound = 1f64;
    let pool_size = 40usize;
    let maximization = false;
    let max_iterations = 800usize;
    let executions: usize = 100;
    println!("Available threads: {threads}, Will execute {executions} tasks.");
    let mut bests: Vec<Whale> = (0..executions)
        .into_par_iter()
        .map(move |_| {
            let instance: JobShopSchedulingProblem =
                JobShopSchedulingProblem::from_instance(Instance::LA02).unwrap();
            let whale_dim: usize = (instance.n_jobs * instance.n_machines) as usize;
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
    bests.par_iter().for_each(|w| {
        println!("{}", w.fitness);
    });
    println!("Execution time: {:?}", start.elapsed());
    Ok(())
}
