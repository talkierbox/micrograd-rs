use super::linear::Linear;
use super::activation::{ReLU, Tanh};

// Type-erased layer so we can store different layer types in one Vec
enum Layer {
    Linear(Linear),
    ReLU(ReLU),
    Tanh(Tanh),
}

impl Layer {
    fn input_dim(&self) -> usize {
        match self {
            Layer::Linear(l) => l.input_dim,
            Layer::ReLU(r) => r.dim,
            Layer::Tanh(t) => t.dim,
        }
    }
    
    fn output_dim(&self) -> usize {
        match self {
            Layer::Linear(l) => l.output_dim,
            Layer::ReLU(r) => r.dim,
            Layer::Tanh(t) => t.dim,
        }
    }
    
    fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        match self {
            Layer::Linear(l) => l.forward(input),
            Layer::ReLU(r) => r.forward(input),
            Layer::Tanh(t) => t.forward(input),
        }
    }
    
    fn backward(&mut self, grad: &[f32]) -> Vec<f32> {
        match self {
            Layer::Linear(l) => l.backward(grad),
            Layer::ReLU(r) => r.backward(grad),
            Layer::Tanh(t) => t.backward(grad),
        }
    }
    
    fn zero_grad(&mut self) {
        match self {
            Layer::Linear(l) => l.zero_grad(),
            Layer::ReLU(r) => r.zero_grad(),
            Layer::Tanh(t) => t.zero_grad(),
        }
    }
}

// A sequential neural network with a PyTorch-like builder API.
pub struct Sequential {
    layers: Vec<Layer>,
    current_dim: usize,
}

impl Sequential {
    pub fn new(input_dim: usize) -> Self {
        Self {
            layers: Vec::new(),
            current_dim: input_dim,
        }
    }
    
    pub fn linear(mut self, output_dim: usize) -> Self {
        let layer = Linear::new(self.current_dim, output_dim);
        self.layers.push(Layer::Linear(layer));
        self.current_dim = output_dim;
        self
    }
    
    pub fn relu(mut self) -> Self {
        let layer = ReLU::new(self.current_dim);
        self.layers.push(Layer::ReLU(layer));
        self
    }
    
    pub fn tanh(mut self) -> Self {
        let layer = Tanh::new(self.current_dim);
        self.layers.push(Layer::Tanh(layer));
        self
    }
    
    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let mut current = input.to_vec();
        for layer in &mut self.layers {
            current = layer.forward(&current);
        }
        current
    }
    
    pub fn backward(&mut self, grad_output: &[f32]) -> Vec<f32> {
        let mut current = grad_output.to_vec();
        for layer in self.layers.iter_mut().rev() {
            current = layer.backward(&current);
        }
        current
    }
    
    pub fn zero_grad(&mut self) {
        for layer in &mut self.layers {
            layer.zero_grad();
        }
    }
    
    pub fn input_dim(&self) -> usize {
        self.layers.first().map(|l| l.input_dim()).unwrap_or(self.current_dim)
    }
    
    pub fn output_dim(&self) -> usize {
        self.current_dim
    }
    
    // Get all trainable parameters (weights and biases from Linear layers)
    pub fn parameters(&self) -> Vec<&f32> {
        let mut params = Vec::new();
        for layer in &self.layers {
            if let Layer::Linear(l) = layer {
                params.extend(l.weights.iter());
                if l.bias_enabled {
                    params.extend(l.biases.iter());
                }
            }
        }
        params
    }
    
    // Get mutable references to all trainable parameters
    pub fn parameters_mut(&mut self) -> Vec<&mut f32> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            if let Layer::Linear(l) = layer {
                params.extend(l.weights.iter_mut());
                if l.bias_enabled {
                    params.extend(l.biases.iter_mut());
                }
            }
        }
        params
    }
    
    // Get all gradients corresponding to parameters
    pub fn gradients(&self) -> Vec<&f32> {
        let mut grads = Vec::new();
        for layer in &self.layers {
            if let Layer::Linear(l) = layer {
                grads.extend(l.grad_weights.iter());
                if l.bias_enabled {
                    grads.extend(l.grad_biases.iter());
                }
            }
        }
        grads
    }
}
