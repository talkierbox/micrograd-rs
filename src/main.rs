mod nn;
mod loss;
mod util;
mod optimizer;

use nn::Sequential;
use loss::mse::MSELoss;
use optimizer::SGD;

fn main() {
    // XOR dataset
    let data = vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ];
    
    // Small MLP for XOR: 2 -> 8 -> 1
    let mut model = Sequential::new(2)
        .linear(8)
        .tanh()
        .linear(1);
    
    let loss_fn = MSELoss::new();
    let optimizer = SGD::new(0.1);
    
    // Training loop
    for epoch in 0..1000 {
        let mut total_loss = 0.0;
        
        for (input, target) in &data {
            total_loss += optimizer.step(&mut model, input, target, &loss_fn);
        }
        
        if epoch % 100 == 0 {
            println!("Epoch {:4}: loss = {:.6}", epoch, total_loss / 4.0);
        }
    }
    
    // Test the trained model
    println!("\nResults:");
    for (input, target) in &data {
        let pred = model.forward(input);
        println!("  {:?} -> {:.3} (expected {})", input, pred[0], target[0]);
    }
}
