use crate::nn::nn_module::NNModule;

pub struct ReLU<const DIM: usize> {
    cached_input: Option<[f32; DIM]>,
}

impl<const DIM: usize> ReLU<DIM> {
    pub fn new() -> Self {
        Self { cached_input: None }
    }
}

impl<const DIM: usize> NNModule<DIM, DIM> for ReLU<DIM> {
    fn _validate_dimensions(&self) {}

    fn input_dim(&self) -> usize {
        DIM
    }

    fn output_dim(&self) -> usize {
        DIM
    }

    fn forward(&mut self, input: &[f32; DIM]) -> [f32; DIM] {
        self.cached_input = Some(*input);
        let mut output = [0.0; DIM];
        for i in 0..DIM {
            output[i] = if input[i] > 0.0 { input[i] } else { 0.0 };
        }
        output
    }

    fn backward(&mut self, grad_output: &[f32; DIM]) -> [f32; DIM] {
        let input = self.cached_input.expect("Must call forward before backward");
        let mut grad_input = [0.0; DIM];
        for i in 0..DIM {
            grad_input[i] = if input[i] > 0.0 { grad_output[i] } else { 0.0 };
        }
        grad_input
    }

    fn zero_grad(&mut self) {
        self.cached_input = None;
    }
}

