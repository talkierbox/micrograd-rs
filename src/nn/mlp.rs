use super::activation::ActivationType;
use super::nn_module::NNModule;

pub struct MLPSpecification {
    input_dim: usize,
    hidden_dims: Vec<usize>,
    output_dim: usize,
    activation: ActivationType,
}

// Type-erased module that can store modules with different dimensions
trait ErasedModule {
    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;
    fn forward_dyn(&mut self, input: &[f32]) -> Vec<f32>;
    fn backward_dyn(&mut self, grad_output: &[f32]) -> Vec<f32>;
    fn zero_grad(&mut self);
}

// Wrapper to erase the const generics
struct ModuleWrapper<const IN: usize, const OUT: usize> {
    module: Box<dyn NNModule<IN, OUT>>,
}

impl<const IN: usize, const OUT: usize> ErasedModule for ModuleWrapper<IN, OUT> {
    fn input_dim(&self) -> usize {
        IN
    }

    fn output_dim(&self) -> usize {
        OUT
    }

    fn forward_dyn(&mut self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), IN, "Input length must match module input dimension");
        let input_array: [f32; IN] = input.try_into().unwrap();
        let output_array = self.module.forward(&input_array);
        output_array.to_vec()
    }

    fn backward_dyn(&mut self, grad_output: &[f32]) -> Vec<f32> {
        assert_eq!(grad_output.len(), OUT, "Gradient length must match module output dimension");
        let grad_array: [f32; OUT] = grad_output.try_into().unwrap();
        let input_grad_array = self.module.backward(&grad_array);
        input_grad_array.to_vec()
    }

    fn zero_grad(&mut self) {
        self.module.zero_grad();
    }
}

pub struct MLP<const INPUT_DIM: usize, const OUTPUT_DIM: usize> {
    components: Vec<Box<dyn ErasedModule>>,
    spec: MLPSpecification   
}

impl <const INPUT_DIM: usize, const OUTPUT_DIM: usize> MLP<INPUT_DIM, OUTPUT_DIM> { 
    pub fn new(spec: MLPSpecification) -> Self {
        assert_eq!(spec.input_dim, INPUT_DIM, "Input dim must match const generic");
        assert_eq!(spec.output_dim, OUTPUT_DIM, "Output dim must match const generic");

        // Also make sure to implement the weight initialization using Xavier
        todo!("Implement this!");
    }

    pub fn forward(&mut self, inputs: [f32; INPUT_DIM]) -> [f32; OUTPUT_DIM] {
        NNModule::forward(self, &inputs)
    }
}

// Allow the MLP to be used as a NNModule (i.e. to chain MLPs together)
impl <const INPUT_DIM: usize, const OUTPUT_DIM: usize> NNModule<INPUT_DIM, OUTPUT_DIM> for MLP<INPUT_DIM, OUTPUT_DIM> {
    fn _validate_dimensions(&self) {
        // Validation is done at construction time
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
        // Chain forward through all components
        let mut current = input.to_vec();
        for component in self.components.iter_mut() {
            // Validate that dimensions match
            assert_eq!(current.len(), component.input_dim(), 
                "Component input dimension mismatch during forward pass");
            current = component.forward_dyn(&current);
        }
        // Convert final output to array
        assert_eq!(current.len(), OUTPUT_DIM, 
            "Final output dimension mismatch");
        current.try_into().unwrap()
    }

    fn backward(&mut self, grad_output: &[f32; OUTPUT_DIM]) -> [f32; INPUT_DIM] {
        self._validate_dimensions();
        // Chain backward through all components in reverse order
        let mut current = grad_output.to_vec();
        for component in self.components.iter_mut().rev() {
            // Validate that dimensions match
            assert_eq!(current.len(), component.output_dim(), 
                "Component output dimension mismatch during backward pass");
            current = component.backward_dyn(&current);
        }
        // Convert final gradient to array
        assert_eq!(current.len(), INPUT_DIM, 
            "Final gradient dimension mismatch");
        current.try_into().unwrap()
    }

    fn zero_grad(&mut self) -> () {
        self._validate_dimensions();
        for component in self.components.iter_mut() {
            component.zero_grad();
        }
        return ();
    }
}