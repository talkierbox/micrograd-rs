// Gradient descent optimizer
use super::optimizer::Optimizer;

pub struct GradientDescentOptimizer {
    name: &'static str,
    lr: f32
}

impl Optimizer for GradientDescentOptimizer {
    fn step(&mut self, params: &mut [f32], grads: &[f32]) {
        assert_eq!(params.len(), grads.len(), "Parameters are not same size as the gradient!");
        for (p, &g) in params.iter_mut().zip(grads.iter()) {
            *p -= self.lr * g;
        }

        return ();
    }
}