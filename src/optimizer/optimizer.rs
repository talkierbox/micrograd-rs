pub trait Optimizer {
    fn step(&mut self, params: &mut [f32], grads: &[f32]);
}