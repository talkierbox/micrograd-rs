use super::nn_module::NNModule;

pub struct LinearLayer<const INPUT_DIM: usize, const OUTPUT_DIM: usize> {
    pub weights: Vec<f32>,  // Size: INPUT_DIM * OUTPUT_DIM
    pub grad_weights: Vec<f32>,     // Size: INPUT_DIM * OUTPUT_DIM
    pub grad_biases: [f32; OUTPUT_DIM],
    pub biases: [f32; OUTPUT_DIM],
    pub bias_enabled: bool,
    pub cached_input: Option<[f32; INPUT_DIM]>
}

impl<const INPUT_DIM: usize, const OUTPUT_DIM: usize> LinearLayer<INPUT_DIM, OUTPUT_DIM> {
    pub fn new() -> Self {
        Self {
            weights: vec![0.0; INPUT_DIM * OUTPUT_DIM],
            grad_weights: vec![0.0; INPUT_DIM * OUTPUT_DIM],
            grad_biases: [0.0; OUTPUT_DIM],
            biases: [0.0; OUTPUT_DIM],
            bias_enabled: true,
            cached_input: None            
        }
    }
}

// TODO: Parallelize this
fn dot(matrix: &[f32], vec: &[f32], start_idx: usize) -> Result<f32, &'static str> {
    let mut res: f32 = 0.0;

    for i in 0..vec.len() {
        if start_idx + i >= matrix.len() {
            return Err("Index out of bounds of the matrix!");
        }

        res += vec[i] * matrix[start_idx + i];
    }

    Ok(res)
}

impl<const INPUT_DIM: usize, const OUTPUT_DIM: usize> NNModule<INPUT_DIM, OUTPUT_DIM> for LinearLayer<INPUT_DIM, OUTPUT_DIM> {
    fn _validate_dimensions(&self) {
        assert_eq!(self.weights.len(), INPUT_DIM * OUTPUT_DIM, "Weights must match dimensions");
        assert_eq!(self.grad_weights.len(), INPUT_DIM * OUTPUT_DIM, "Grad must match dimensions");
        assert_eq!(self.biases.len(), OUTPUT_DIM, "Biases must match dimensions");
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
        // Matrix-Vector Multiply 
        let mut res = [0.0; OUTPUT_DIM];

        let num_rows: usize = OUTPUT_DIM;
        let num_cols: usize = INPUT_DIM;

        // TODO: Parallelize this
        for i in 0..num_rows {
            // Get this row and dot product it with input 
            let offset = i * num_cols;
            res[i] = dot(&self.weights, input, offset)
                .expect("Matrix-vector multiplication bounds check failed - this indicates a programming error")
                + (if self.bias_enabled { self.biases[i] } else {0.0});
        }       

        self.cached_input = Some(input.clone());
        
        res // return our result
    }

    fn backward(&mut self, grad_output: &[f32; OUTPUT_DIM]) -> [f32; INPUT_DIM] {
        self._validate_dimensions();
        todo!("Implement this!")
    }

    fn zero_grad(&mut self) -> () {
        self.grad_weights.fill(0.0);
        self.grad_biases.fill(0.0);
        self.cached_input = None;
    }
}
