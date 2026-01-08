pub struct Tanh {
    pub dim: usize,
    cached_output: Option<Vec<f32>>,
}

impl Tanh {
    pub fn new(dim: usize) -> Self {
        Self { dim, cached_output: None }
    }
    
    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.dim, "Input dimension mismatch");
        let output: Vec<f32> = input.iter().map(|&x| x.tanh()).collect();
        self.cached_output = Some(output.clone());
        output
    }
    
    pub fn backward(&mut self, grad_output: &[f32]) -> Vec<f32> {
        let output = self.cached_output.as_ref()
            .expect("Must call forward before backward");
        // d/dx tanh(x) = 1 - tanh(x)^2
        output.iter()
            .zip(grad_output.iter())
            .map(|(&t, &g)| g * (1.0 - t * t))
            .collect()
    }
    
    pub fn zero_grad(&mut self) {
    }
}
