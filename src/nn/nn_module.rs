// TODO: Make sure this supports activation functions and the linear layers
pub trait NNModule<const INPUT_DIM: usize, const OUTPUT_DIM: usize> {
    fn _validate_dimensions(&self);

    /// The input dimension for this module
    fn input_dim(&self) -> usize;

    /// The output dimension for this module
    fn output_dim(&self) -> usize;

    /// Forward pass
    fn forward(&mut self, input: &[f32; INPUT_DIM]) -> [f32; OUTPUT_DIM];

    /// Backward pass: takes array of gradient wrt output, returns gradient wrt input
    fn backward(&mut self, grad_output: &[f32; OUTPUT_DIM]) -> [f32; INPUT_DIM];

    fn zero_grad(&mut self) -> ();
}