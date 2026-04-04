use super::{Instance, JobShopSchedulingProblem};
use crate::jssp::whale::Whale;
use crate::problem::{Individual, Solvable};
use nalgebra::DVector;
use std::io::Error;

#[test]
fn test_1() {
    let instance: Instance = Instance::TEST01;
    let instance: Result<JobShopSchedulingProblem, Error> =
        JobShopSchedulingProblem::from_instance(instance);
    assert!(instance.is_ok());
    let instance = instance.unwrap();
    assert_eq!(
        instance,
        JobShopSchedulingProblem::new(
            Instance::TEST01,
            vec![vec![0, 2, 1], vec![1, 0, 2], vec![2, 1, 0]],
            vec![
                vec![10f64, 5f64, 15f64],
                vec![8f64, 15f64, 20f64],
                vec![15f64, 10f64, 9f64]
            ],
            3,
            3
        )
    );
}

#[test]
fn test_2() {
    let mut whale: Whale = Whale::new(
        DVector::from_vec(vec![0.1, 0.5, 0.2, 0.8, 0.3]),
        DVector::from_vec(vec![0, 1, 2, 3, 4]),
        0f64,
    );
    whale.ranked_order_value();
    assert_eq!(whale.sequence, DVector::from_vec(vec![0, 3, 1, 4, 2]),)
}

#[test]
fn test_3() {
    let instance: Instance = Instance::TEST01;
    let instance: Result<JobShopSchedulingProblem, Error> =
        JobShopSchedulingProblem::from_instance(instance);
    assert!(instance.is_ok());
    let instance = instance.unwrap();
    let whale = Whale::new(
        DVector::zeros(9),
        DVector::from_vec(vec![0, 1, 1, 2, 0, 2, 1, 0, 2]),
        0f64,
    );
    assert_eq!(instance.solve(&whale), 45f64)
}

#[test]
fn test_4() {
    let mut whale: Whale = Whale::new(
        DVector::from_vec(vec![-2.65, 0.5, 0.2, 0.8, 0.3]),
        DVector::from_vec(vec![0, 1, 2, 3, 4]),
        0f64,
    );
    whale.check_if_goes_beyond_bounds(-1f64, 1f64);
    assert_eq!(
        whale.position,
        DVector::from_vec(vec![-1f64, 0.5, 0.2, 0.8, 0.3]),
    )
}
