//! Symplectic optimization for agent dynamics.
//!
//! Provides momentum-based optimizers that preserve phase-space volume,
//! analogous to symplectic integrators in Hamiltonian mechanics.

pub mod optimizer;

use num_traits::Float;

/// Scalar trait for optimization.
pub trait Scalar: Float + std::fmt::Debug + 'static {}
impl<T: Float + std::fmt::Debug + 'static> Scalar for T {}
