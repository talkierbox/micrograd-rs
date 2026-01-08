use crate::nn::Sequential;
use crate::loss::loss_function::LossFunction;

pub struct SGD {
    pub lr: f32,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
    
    /// Perform a single training step: forward, loss, backward, update
    /// Returns the loss value
    pub fn step<L: LossFunction>(
        &self,
        model: &mut Sequential,
        input: &[f32],
        target: &[f32],
        loss_fn: &L,
    ) -> f32 {
        model.zero_grad();
        
        let pred = model.forward(input);
        let loss = loss_fn.compute_loss(&pred, target);
        
        let grad = loss_fn.compute_gradient(&pred, target);
        model.backward(&grad);
        
        // Update parameters
        let grads: Vec<f32> = model.gradients().iter().map(|&&g| g).collect();
        for (param, grad) in model.parameters_mut().into_iter().zip(grads.iter()) {
            *param -= self.lr * grad;
        }
        
        loss
    }
}
