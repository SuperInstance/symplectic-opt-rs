//! Symplectic integrators for Hamiltonian systems.
//!
//! Standard integrators (like forward Euler) do not preserve the symplectic structure
//! of phase space, causing energy to drift over time. Symplectic integrators preserve
//! the symplectic 2-form exactly, yielding bounded energy error over arbitrarily long
//! simulations.
//!
//! # Available Integrators
//!
//! - **Symplectic Euler** — First-order, symplectic, cheap per step.
//! - **Störmer-Verlet** (a.k.a. leapfrog) — Second-order, symplectic, time-reversible.
//!   The workhorse of molecular dynamics and Hamiltonian Monte Carlo.

use crate::hamiltonian::HamiltonianSystem;

/// Symplectic (semi-implicit) Euler integrator for a separable Hamiltonian system.
///
/// Update rule for H = T(p) + V(q):
/// ```text
/// p_{n+1} = p_n - dt · ∂V/∂q(q_n)
/// q_{n+1} = q_n + dt · ∂T/∂p(p_{n+1})
/// ```
///
/// First-order accurate and symplectic.
///
/// # Arguments
/// * `system` — The Hamiltonian system
/// * `p` — Initial momenta
/// * `q` — Initial coordinates
/// * `dt` — Time step
/// * `steps` — Number of integration steps
///
/// # Returns
/// Vector of (p, q) states at each step (including initial condition).
pub fn symplectic_euler(
    system: &HamiltonianSystem,
    p: &[f64],
    q: &[f64],
    dt: f64,
    steps: usize,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let n = p.len();
    assert_eq!(n, q.len(), "p and q must have same dimension");

    let mut trajectory = Vec::with_capacity(steps + 1);
    let mut p = p.to_vec();
    let mut q = q.to_vec();
    trajectory.push((p.clone(), q.clone()));

    for _ in 0..steps {
        // p update: p_{n+1} = p_n - dt * ∂V/∂q(q_n) = p_n - dt * ∂H/∂q(q_n)
        let dhdq = system.grad_q(&p, &q);
        for i in 0..n {
            p[i] -= dt * dhdq[i];
        }

        // q update: q_{n+1} = q_n + dt * ∂T/∂p(p_{n+1}) = q_n + dt * ∂H/∂p(p_{n+1})
        let dhdp = system.grad_p(&p, &q);
        for i in 0..n {
            q[i] += dt * dhdp[i];
        }

        trajectory.push((p.clone(), q.clone()));
    }

    trajectory
}

/// Störmer-Verlet (velocity Verlet) integrator.
///
/// Second-order accurate, symplectic, and time-reversible. The update rule:
/// ```text
/// p_{n+1/2} = p_n - (dt/2) · ∂V/∂q(q_n)
/// q_{n+1}   = q_n + dt · ∂T/∂p(p_{n+1/2})
/// p_{n+1}   = p_{n+1/2} - (dt/2) · ∂V/∂q(q_{n+1})
/// ```
///
/// This is the go-to integrator for long-time energy-stable simulations.
///
/// # Arguments
/// * `system` — The Hamiltonian system
/// * `p` — Initial momenta
/// * `q` — Initial coordinates
/// * `dt` — Time step
/// * `steps` — Number of integration steps
///
/// # Returns
/// Vector of (p, q) states at each step (including initial condition).
pub fn stormer_verlet(
    system: &HamiltonianSystem,
    p: &[f64],
    q: &[f64],
    dt: f64,
    steps: usize,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    let n = p.len();
    assert_eq!(n, q.len(), "p and q must have same dimension");

    let mut trajectory = Vec::with_capacity(steps + 1);
    let mut p = p.to_vec();
    let mut q = q.to_vec();
    trajectory.push((p.clone(), q.clone()));

    for _ in 0..steps {
        // Half-step momentum
        let dhdq = system.grad_q(&p, &q);
        for i in 0..n {
            p[i] -= 0.5 * dt * dhdq[i];
        }

        // Full-step position
        let dhdp = system.grad_p(&p, &q);
        for i in 0..n {
            q[i] += dt * dhdp[i];
        }

        // Half-step momentum
        let dhdq = system.grad_q(&p, &q);
        for i in 0..n {
            p[i] -= 0.5 * dt * dhdq[i];
        }

        trajectory.push((p.clone(), q.clone()));
    }

    trajectory
}

/// Leapfrog integrator — an alias for Störmer-Verlet.
///
/// The names are used interchangeably in the literature. "Leapfrog" emphasizes
/// the staggered grid interpretation where positions and momenta are defined at
/// alternating half-integer time steps.
pub fn leapfrog(
    system: &HamiltonianSystem,
    p: &[f64],
    q: &[f64],
    dt: f64,
    steps: usize,
) -> Vec<(Vec<f64>, Vec<f64>)> {
    stormer_verlet(system, p, q, dt, steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hamiltonian::separable_hamiltonian;

    fn harmonic_system() -> HamiltonianSystem {
        let t = |p: &[f64]| -> f64 { p.iter().map(|x| 0.5 * x * x).sum() };
        let v = |q: &[f64]| -> f64 { q.iter().map(|x| 0.5 * x * x).sum() };
        separable_hamiltonian(t, v)
    }

    #[test]
    fn test_stormer_verlet_energy_conservation() {
        let sys = harmonic_system();
        let p = vec![1.0];
        let q = vec![0.0];
        let dt = 0.001;
        let steps = 10_000;

        let traj = stormer_verlet(&sys, &p, &q, dt, steps);
        let e0 = sys.hamiltonian(&traj[0].0, &traj[0].1);
        let ef = sys.hamiltonian(&traj[steps].0, &traj[steps].1);
        let drift = (ef - e0).abs();
        assert!(drift < 1e-6, "Energy drift too large: {}", drift);
    }

    #[test]
    fn test_symplectic_euler_runs() {
        let sys = harmonic_system();
        let p = vec![1.0];
        let q = vec![0.0];
        let traj = symplectic_euler(&sys, &p, &q, 0.1, 100);
        assert_eq!(traj.len(), 101);
    }

    #[test]
    fn test_leapfrog_equals_stormer_verlet() {
        let sys = harmonic_system();
        let p = vec![0.5, -0.3];
        let q = vec![1.0, 0.7];
        let sv = stormer_verlet(&sys, &p, &q, 0.01, 50);
        let lf = leapfrog(&sys, &p, &q, 0.01, 50);
        for (s, l) in sv.iter().zip(lf.iter()) {
            for i in 0..p.len() {
                assert!((s.0[i] - l.0[i]).abs() < 1e-15);
                assert!((s.1[i] - l.1[i]).abs() < 1e-15);
            }
        }
    }
}
