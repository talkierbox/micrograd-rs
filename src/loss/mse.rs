use super::loss_function::LossFunction;

pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        MSELoss
    }
}

impl LossFunction for MSELoss {
    fn compute_loss(&self, pred: &[f32], target: &[f32]) -> f32 {
        assert_eq!(pred.len(), target.len(), "Shape of predictions does not match shape of target!");
        
        let mut sum_squared_error = 0.0;
        for i in 0..pred.len() {
            let diff = pred[i] - target[i];
            sum_squared_error += diff * diff;
        }
        
        sum_squared_error / pred.len() as f32
    }
    
    fn compute_gradient(&self, pred: &[f32], target: &[f32]) -> Vec<f32> {
        assert_eq!(pred.len(), target.len(), "Shape of predictions does not match shape of target!");
        
        let n = pred.len() as f32;
        pred.iter()
            .zip(target.iter())
            .map(|(p, t)| 2.0 * (p - t) / n)
            .collect()
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        MSELoss
    }
}
