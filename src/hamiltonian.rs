//! Hamiltonian systems and Hamilton's equations.
//!
//! A Hamiltonian system is defined by a scalar function H(p, q) (the Hamiltonian)
//! where p are momenta and q are coordinates. The time evolution follows:
//!
//! ```text
//! dq/dt =  ∂H/∂p
//! dp/dt = -∂H/∂q
//! ```
//!
//! For separable systems H(p,q) = T(p) + V(q), these simplify to Newtonian mechanics.

/// A Hamiltonian system defined by H(p, q).
///
/// The Hamiltonian is stored as a boxed function for flexibility.
pub struct HamiltonianSystem {
    /// The Hamiltonian function H(p, q) → ℝ.
    h: Box<dyn Fn(&[f64], &[f64]) -> f64>,
    /// Gradient ∂H/∂p (computed numerically if not provided analytically).
    dh_dp: Box<dyn Fn(&[f64], &[f64]) -> Vec<f64>>,
    /// Gradient ∂H/∂q (computed numerically if not provided analytically).
    dh_dq: Box<dyn Fn(&[f64], &[f64]) -> Vec<f64>>,
}

impl HamiltonianSystem {
    /// Create a Hamiltonian system from explicit Hamiltonian and gradient functions.
    ///
    /// # Arguments
    /// * `h` - The Hamiltonian H(p, q)
    /// * `dh_dp` - Partial derivative ∂H/∂p
    /// * `dh_dq` - Partial derivative ∂H/∂q
    pub fn new(
        h: impl Fn(&[f64], &[f64]) -> f64 + 'static,
        dh_dp: impl Fn(&[f64], &[f64]) -> Vec<f64> + 'static,
        dh_dq: impl Fn(&[f64], &[f64]) -> Vec<f64> + 'static,
    ) -> Self {
        Self {
            h: Box::new(h),
            dh_dp: Box::new(dh_dp),
            dh_dq: Box::new(dh_dq),
        }
    }

    /// Evaluate the Hamiltonian at (p, q).
    pub fn hamiltonian(&self, p: &[f64], q: &[f64]) -> f64 {
        (self.h)(p, q)
    }

    /// Compute ∂H/∂p at (p, q).
    pub fn grad_p(&self, p: &[f64], q: &[f64]) -> Vec<f64> {
        (self.dh_dp)(p, q)
    }

    /// Compute ∂H/∂q at (p, q).
    pub fn grad_q(&self, p: &[f64], q: &[f64]) -> Vec<f64> {
        (self.dh_dq)(p, q)
    }
}

/// Create a separable Hamiltonian system H(p, q) = T(p) + V(q).
///
/// For separable systems, the gradients simplify:
/// - ∂H/∂p = ∂T/∂p
/// - ∂H/∂q = ∂V/∂q
///
/// The kinetic gradient and potential gradient are computed numerically.
pub fn separable_hamiltonian(
    t: fn(&[f64]) -> f64,
    v: fn(&[f64]) -> f64,
) -> HamiltonianSystem {
    let eps = 1e-7;

    let h = move |p: &[f64], q: &[f64]| t(p) + v(q);

    let dh_dp = move |p: &[f64], _q: &[f64]| {
        numerical_gradient(p, eps, |x| t(x))
    };

    let dh_dq = move |_p: &[f64], q: &[f64]| {
        numerical_gradient(q, eps, |x| v(x))
    };

    HamiltonianSystem::new(h, dh_dp, dh_dq)
}

/// Compute the energy (Hamiltonian) of a system at state (p, q).
pub fn energy(system: &HamiltonianSystem, p: &[f64], q: &[f64]) -> f64 {
    system.hamiltonian(p, q)
}

/// Evaluate Hamilton's equations at (p, q).
///
/// Returns (dq/dt, dp/dt) = (∂H/∂p, -∂H/∂q).
pub fn hamiltons_equations(
    system: &HamiltonianSystem,
    p: &[f64],
    q: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let dq_dt = system.grad_p(p, q);
    let dp_dt = system.grad_q(p, q).iter().map(|&x| -x).collect();
    (dq_dt, dp_dt)
}

/// Central finite-difference gradient.
fn numerical_gradient(x: &[f64], eps: f64, f: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    x.iter()
        .enumerate()
        .map(|(i, _)| {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] += eps;
            xm[i] -= eps;
            (f(&xp) - f(&xm)) / (2.0 * eps)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_separable_hamiltonian() {
        // H = ½|p|² + ½|q|² (harmonic oscillator)
        let t = |p: &[f64]| -> f64 { p.iter().map(|x| 0.5 * x * x).sum() };
        let v = |q: &[f64]| -> f64 { q.iter().map(|x| 0.5 * x * x).sum() };
        let sys = separable_hamiltonian(t, v);

        let p = vec![1.0, 0.0];
        let q = vec![0.0, 1.0];
        let h = energy(&sys, &p, &q);
        assert!((h - 1.0).abs() < 1e-6, "Expected H=1.0, got {}", h);
    }

    #[test]
    fn test_hamiltons_equations_harmonic() {
        let t = |p: &[f64]| -> f64 { p.iter().map(|x| 0.5 * x * x).sum() };
        let v = |q: &[f64]| -> f64 { q.iter().map(|x| 0.5 * x * x).sum() };
        let sys = separable_hamiltonian(t, v);

        let p = vec![0.0];
        let q = vec![1.0];
        let (dq, dp) = hamiltons_equations(&sys, &p, &q);
        // dq/dt = ∂H/∂p = p = 0
        assert!(dq[0].abs() < 1e-5, "dq/dt should be ~0, got {}", dq[0]);
        // dp/dt = -∂H/∂q = -q = -1
        assert!((dp[0] + 1.0).abs() < 1e-5, "dp/dt should be ~-1, got {}", dp[0]);
    }
}
