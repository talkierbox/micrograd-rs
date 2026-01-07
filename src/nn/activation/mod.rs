mod tanh;
mod relu;

pub use tanh::Tanh;
pub use relu::ReLU;

/// For use in specs/configs where you need to pick an activation type
#[derive(Clone, Copy)]
pub enum ActivationType {
    Tanh,
    ReLU,
}

