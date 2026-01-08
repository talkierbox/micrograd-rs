pub struct ReLU {
    pub dim: usize,
    cached_input: Option<Vec<f32>>,
}

impl ReLU {
    pub fn new(dim: usize) -> Self {
        Self { dim, cached_input: None }
    }
    
    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.dim, "Input dimension mismatch");
        self.cached_input = Some(input.to_vec());
        input.iter().map(|&x| x.max(0.0)).collect()
    }
    
    pub fn backward(&mut self, grad_output: &[f32]) -> Vec<f32> {
        let input = self.cached_input.as_ref()
            .expect("Must call forward before backward");
        input.iter()
            .zip(grad_output.iter())
            .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
            .collect()
    }
    
    pub fn zero_grad(&mut self) {
    }
}
