//! # symplectic-opt
//!
//! Symplectic geometry for optimization — Hamiltonian systems, symplectic integrators,
//! and natural gradient descent.
//!
//! Symplectic geometry provides the mathematical framework for structure-preserving
//! optimization. By respecting the symplectic structure of phase space, we obtain
//! integrators that conserve energy over long time horizons — a property directly
//! analogous to stable training in machine learning.
//!
//! ## Modules
//!
//! - [`symplectic`] — Symplectic matrix operations (J matrix, symplecticity tests)
//! - [`hamiltonian`] — Hamiltonian systems and Hamilton's equations
//! - [`integrator`] — Symplectic integrators (Euler, Störmer-Verlet, leapfrog)
//! - [`conservation`] — Conservation law tracking (energy drift, angular momentum)
//! - [`natural_gradient`] — Natural gradient via symplectic / Fisher structure

pub mod symplectic;
pub mod hamiltonian;
pub mod integrator;
pub mod conservation;
pub mod natural_gradient;

pub use symplectic::{SymplecticMatrix, is_symplectic, symplectic_inverse, canonical_form};
pub use hamiltonian::{HamiltonianSystem, separable_hamiltonian, energy, hamiltons_equations};
pub use integrator::{symplectic_euler, stormer_verlet, leapfrog};
pub use conservation::ConservationTracker;
pub use natural_gradient::{fisher_information, natural_gradient};
