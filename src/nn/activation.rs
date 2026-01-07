use super::nn_module::NNModule;

pub enum Activation {
    Tanh,
    ReLU
}
// Activation function needs to take input dimension and output dimension
pub struct ActivationFunction<const INPUT_DIM: usize, const OUTPUT_DIM: usize> {
    pub activation: Activation,
    pub input_dim: usize,
    pub output_dim: usize,
    pub grad: [f32; OUTPUT_DIM],
    pub cached_input: Option<[f32; INPUT_DIM]>
}

fn tanh(x: f32) -> f32 {
    x.tanh()
}

impl<const INPUT_DIM: usize, const OUTPUT_DIM: usize> NNModule<INPUT_DIM, OUTPUT_DIM> for ActivationFunction<INPUT_DIM, OUTPUT_DIM> {
    fn _validate_dimensions(&self) {
        assert_eq!(self.input_dim, INPUT_DIM, "Input dim must match const generic");
        assert_eq!(self.output_dim, OUTPUT_DIM, "Output dim must match const generic");
    }
    fn input_dim(&self) -> usize {
        self._validate_dimensions();
        INPUT_DIM
    }
    fn output_dim(&self) -> usize {
        self._validate_dimensions();
        OUTPUT_DIM
    }

    fn forward(&mut self, input: &[f32; INPUT_DIM]) -> [f32; OUTPUT_DIM] {
        self._validate_dimensions();
        assert_eq!(INPUT_DIM, OUTPUT_DIM, "Activation functions require INPUT_DIM == OUTPUT_DIM");

        let mut output = [0.0; OUTPUT_DIM];
        
        // Run the activation
        match self.activation {
            Activation::Tanh => {
                for i in 0..INPUT_DIM {
                    output[i] = tanh(input[i]);
                }
            },
            Activation::ReLU => {
                for i in 0..INPUT_DIM {
                    output[i] = if input[i] > 0.0 { input[i] } else { 0.0 };
                }
            },
        };

        self.cached_input = Some(input.clone());

        output
    }

    fn backward(&mut self, grad_output: &[f32; OUTPUT_DIM]) -> [f32; INPUT_DIM] {
        self._validate_dimensions();
        todo!("Implement backward pass for activations")
    }

    fn zero_grad(&mut self) -> () {
        self.grad.fill(0.0);
        self.cached_input = None;
        return ();
    }

}