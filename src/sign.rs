use nalgebra::DVector;

pub(crate) fn sign(x: DVector<f64>) -> DVector<f64> {
    x.map(|v| v.signum())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sign() {
        assert_eq!(
            sign(DVector::from_vec(vec![-1f64, 2f64, 0f64])),
            sign(DVector::from_vec(vec![-1f64, 1f64, 0f64]))
        )
    }
}
