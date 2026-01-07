pub trait LossFunction {
    /// Compute the loss between predictions and targets
    /// Returns the loss value (scalar)
    fn compute_loss(&self, pred: &[f32], target: &[f32]) -> f32;
    
    /// Compute the gradient of the loss with respect to predictions
    /// Returns a vector of gradients (same length as pred/target)
    fn compute_gradient(&self, pred: &[f32], target: &[f32]) -> Vec<f32>;
}