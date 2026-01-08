use super::super::util::dot::flat_matrix_vector_dot;
use std::sync::atomic::{AtomicU64, Ordering};

// Global seed for weight initialization
static SEED: AtomicU64 = AtomicU64::new(42);

pub struct Linear {
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: Vec<f32>,       // Size: input_dim * output_dim (row-major)
    pub biases: Vec<f32>,        // Size: output_dim
    pub grad_weights: Vec<f32>,
    pub grad_biases: Vec<f32>,
    pub bias_enabled: bool,
    cached_input: Option<Vec<f32>>,
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        // Xavier initialization
        let limit = (6.0_f32 / (input_dim + output_dim) as f32).sqrt();
        
        // LCG random using global seed
        let mut seed = SEED.fetch_add(1, Ordering::Relaxed);
        let mut rand = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = ((seed >> 33) as f32) / (u32::MAX as f32);
            val * 2.0 * limit - limit
        };
        
        let weights: Vec<f32> = (0..input_dim * output_dim).map(|_| rand()).collect();
        
        Self {
            input_dim,
            output_dim,
            weights,
            biases: vec![0.0; output_dim],
            grad_weights: vec![0.0; input_dim * output_dim],
            grad_biases: vec![0.0; output_dim],
            bias_enabled: true,
            cached_input: None,
        }
    }
    
    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_dim, "Input dimension mismatch");
        
        let mut output = vec![0.0; self.output_dim];
        
        for i in 0..self.output_dim {
            let offset = i * self.input_dim;
            let dot = flat_matrix_vector_dot(&self.weights, input, offset)
                .expect("Matrix-vector dot failed");
            output[i] = dot + if self.bias_enabled { self.biases[i] } else { 0.0 };
        }
        
        self.cached_input = Some(input.to_vec());
        output
    }
    
    pub fn backward(&mut self, grad_output: &[f32]) -> Vec<f32> {
        assert_eq!(grad_output.len(), self.output_dim, "Gradient dimension mismatch");
        
        let input = self.cached_input.as_ref()
            .expect("Must call forward before backward");
        
        // Gradient w.r.t. biases
        if self.bias_enabled {
            for i in 0..self.output_dim {
                self.grad_biases[i] += grad_output[i];
            }
        }
        
        // Gradient w.r.t. weights
        for i in 0..self.output_dim {
            for j in 0..self.input_dim {
                self.grad_weights[i * self.input_dim + j] += grad_output[i] * input[j];
            }
        }
        
        // Gradient w.r.t. input (W^T @ grad_output)
        let mut grad_input = vec![0.0; self.input_dim];
        for j in 0..self.input_dim {
            for i in 0..self.output_dim {
                grad_input[j] += self.weights[i * self.input_dim + j] * grad_output[i];
            }
        }
        
        grad_input
    }
    
    pub fn zero_grad(&mut self) {
        self.grad_weights.fill(0.0);
        self.grad_biases.fill(0.0);
    }
}
