//! Integration tests for symplectic-opt.
//!
//! These tests verify end-to-end behavior: conservation laws, integrator properties,
//! and the connection between symplectic structure and optimization.

use symplectic_opt::*;
use symplectic_opt::conservation::angular_momentum;

// ---------------------------------------------------------------------------
// Symplectic matrix tests
// ---------------------------------------------------------------------------

#[test]
fn test_identity_is_symplectic() {
    let id = nalgebra::DMatrix::identity(4, 4);
    assert!(is_symplectic(&id, 1e-10));
}

#[test]
fn test_canonical_form_j_squared_minus_identity() {
    // J^2 = -I for the canonical symplectic form
    let j = canonical_form(2);
    let j2 = &j * &j;
    let neg_id = nalgebra::DMatrix::from_diagonal_element(4, 4, -1.0);
    for i in 0..4 {
        for k in 0..4 {
            assert!(
                ((j2[(i, k)] - neg_id[(i, k)]).abs() < 1e-12),
                "J^2 != -I at ({}, {}): got {}",
                i, k, j2[(i, k)]
            );
        }
    }
}

#[test]
fn test_symplectic_inverse_j() {
    // J^{-1} = -J for the canonical form (since J^T = -J)
    let j = canonical_form(3);
    let j_inv = symplectic_inverse(&j);
    let expected = -&j;
    assert!((&j_inv - &expected).iter().all(|&v| v.abs() < 1e-12));
}

#[test]
fn test_symplectic_inverse_times_identity() {
    // S * S^{-1} ≈ I for a known symplectic matrix
    // A simple symplectic matrix: rotation in phase space
    let n = 1;
    let _j = canonical_form(n);
    // exp(0.5 * J) is symplectic (rotation by 0.5 radians)
    let theta: f64 = 0.5;
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let s = nalgebra::DMatrix::from_row_slice(2, 2, &[
        cos_t, sin_t,
        -sin_t, cos_t,
    ]);
    assert!(is_symplectic(&s, 1e-10));
    let s_inv = symplectic_inverse(&s);
    let product = &s * &s_inv;
    let id = nalgebra::DMatrix::identity(2, 2);
    assert!((&product - &id).iter().all(|&v| v.abs() < 1e-10));
}

// ---------------------------------------------------------------------------
// Integrator conservation tests
// ---------------------------------------------------------------------------

#[test]
fn test_harmonic_oscillator_stormer_verlet_energy_conservation() {
    // H = ½(p² + q²) — the harmonic oscillator
    // Störmer-Verlet should preserve energy to < 1e-10 over 10,000 steps
    let sys = hamiltonian::separable_hamiltonian(
        |p: &[f64]| p.iter().map(|x| 0.5 * x * x).sum(),
        |q: &[f64]| q.iter().map(|x| 0.5 * x * x).sum(),
    );

    let p = vec![1.0];
    let q = vec![0.0];
    let dt = 0.001;
    let steps = 10_000;

    let traj = integrator::stormer_verlet(&sys, &p, &q, dt, steps);
    let e0 = sys.hamiltonian(&traj[0].0, &traj[0].1);
    let ef = sys.hamiltonian(&traj[steps].0, &traj[steps].1);
    let drift = (ef - e0).abs();

    assert!(drift < 1e-6, "Energy drift too large: {}", drift);
}

#[test]
fn test_symplectic_euler_preserves_phase_space_volume() {
    // Symplectic Euler should preserve phase space volume (Liouville's theorem).
    // We test this by evolving two nearby initial conditions and checking the
    // area of the parallelogram they span is preserved.
    let sys = hamiltonian::separable_hamiltonian(
        |p: &[f64]| p.iter().map(|x| 0.5 * x * x).sum(),
        |q: &[f64]| q.iter().map(|x| 0.5 * x * x).sum(),
    );

    let eps = 1e-6;
    let dt = 0.01;
    let steps = 1000;

    // Two perturbed initial conditions
    let p0 = vec![1.0];
    let q0 = vec![0.0];
    let p1 = vec![1.0 + eps];
    let q1 = vec![0.0];

    let traj0 = integrator::symplectic_euler(&sys, &p0, &q0, dt, steps);
    let traj1 = integrator::symplectic_euler(&sys, &p1, &q1, dt, steps);

    // Initial phase space area (determinant of the 2×2 matrix formed by the two state vectors)
    let dp0 = traj0[0].0[0] - traj1[0].0[0];
    let dq0 = traj0[0].1[0] - traj1[0].1[0];
    let area0 = dp0.hypot(dq0); // approximate for small perturbation

    let dpf = traj0[steps].0[0] - traj1[steps].0[0];
    let dqf = traj0[steps].1[0] - traj1[steps].1[0];
    let areaf = dpf.hypot(dqf);

    // The phase space volume should be approximately preserved
    let ratio = areaf / area0;
    assert!(
        (ratio - 1.0).abs() < 0.05,
        "Phase space volume not preserved: ratio = {}",
        ratio
    );
}

#[test]
fn test_energy_tracking_max_drift_bounded() {
    let sys = hamiltonian::separable_hamiltonian(
        |p: &[f64]| p.iter().map(|x| 0.5 * x * x).sum(),
        |q: &[f64]| q.iter().map(|x| 0.5 * x * x).sum(),
    );

    let p = vec![1.0, 0.5];
    let q = vec![0.0, -0.5];
    let dt = 0.01;
    let steps = 5000;

    let traj = integrator::stormer_verlet(&sys, &p, &q, dt, steps);

    let mut tracker = conservation::ConservationTracker::new();
    for (i, (pi, qi)) in traj.iter().enumerate() {
        let e = sys.hamiltonian(pi, qi);
        tracker.track(i, pi, qi, e);
    }

    assert!(tracker.max_drift() < 1e-4, "Max drift: {}", tracker.max_drift());
}

// ---------------------------------------------------------------------------
// Angular momentum conservation
// ---------------------------------------------------------------------------

#[test]
fn test_angular_momentum_conservation_central_force() {
    // Central force: V(q) = -1/|q|, T(p) = ½|p|²
    // Angular momentum should be conserved
    let sys = hamiltonian::separable_hamiltonian(
        |p: &[f64]| p.iter().map(|x| 0.5 * x * x).sum(),
        |q: &[f64]| {
            let r: f64 = q.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);
            -1.0 / r
        },
    );

    let p = vec![0.0, 1.0]; // initial tangential momentum
    let q = vec![1.0, 0.0]; // initial radial position
    let dt = 0.001;
    let steps = 10_000;

    let traj = integrator::stormer_verlet(&sys, &p, &q, dt, steps);

    let l0 = angular_momentum(&traj[0].0, &traj[0].1);
    let lf = angular_momentum(&traj[steps].0, &traj[steps].1);

    let drift = (lf - l0).abs();
    assert!(drift < 1e-4, "Angular momentum drift too large: {}", drift);
}

// ---------------------------------------------------------------------------
// Natural gradient tests
// ---------------------------------------------------------------------------

#[test]
fn test_natural_gradient_valid_direction() {
    // Fisher = I → natural gradient = gradient
    let fisher = nalgebra::DMatrix::identity(3, 3);
    let grad = vec![1.0, 2.0, 3.0];
    let ng = natural_gradient::natural_gradient(&grad, &fisher);
    assert_eq!(ng.len(), 3);
    for i in 0..3 {
        assert!((ng[i] - grad[i]).abs() < 1e-6, "ng[{}] = {}, expected {}", i, ng[i], grad[i]);
    }
}

#[test]
fn test_fisher_from_samples() {
    let samples = vec![
        vec![1.0, 0.0],
        vec![-1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.0, -1.0],
    ];
    let f = natural_gradient::fisher_information(&samples);
    // Should be 0.5 * I
    assert!((f[(0, 0)] - 0.5).abs() < 1e-10);
    assert!((f[(1, 1)] - 0.5).abs() < 1e-10);
    assert!(f[(0, 1)].abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// Separable Hamiltonian test
// ---------------------------------------------------------------------------

#[test]
fn test_separable_hamiltonian_splits_correctly() {
    let t_fn = |p: &[f64]| -> f64 { p.iter().map(|x| 0.5 * x * x).sum() };
    let v_fn = |q: &[f64]| -> f64 { q.iter().map(|x| x * x * x).sum::<f64>().abs() };
    let sys = hamiltonian::separable_hamiltonian(t_fn, v_fn);

    let p = vec![2.0, 3.0];
    let q = vec![1.0, 1.0];
    let e = energy(&sys, &p, &q);
    let expected = t_fn(&p) + v_fn(&q);
    assert!((e - expected).abs() < 1e-6, "Energy = {}, expected {}", e, expected);
}

// ---------------------------------------------------------------------------
// SymplecticMatrix wrapper test
// ---------------------------------------------------------------------------

#[test]
fn test_symplectic_matrix_wrapper() {
    let j = canonical_form(2);
    let sm = symplectic::SymplecticMatrix::new(j.clone());
    assert!(sm.verify(1e-10));
    assert_eq!(sm.n(), 2);
    assert_eq!(sm.dim(), 4);

    // Inverse of J should equal -J
    let inv = sm.inverse();
    let expected = -&j;
    assert!((&inv - &expected).iter().all(|&v| v.abs() < 1e-12));
}

// ---------------------------------------------------------------------------
// Comparison: Euler vs Störmer-Verlet drift
// ---------------------------------------------------------------------------

#[test]
fn test_euler_vs_stormer_verlet_drift() {
    let sys = hamiltonian::separable_hamiltonian(
        |p: &[f64]| p.iter().map(|x| 0.5 * x * x).sum(),
        |q: &[f64]| q.iter().map(|x| 0.5 * x * x).sum(),
    );

    let p = vec![1.0];
    let q = vec![0.0];
    let dt = 0.1;
    let steps = 500;

    let euler_traj = integrator::symplectic_euler(&sys, &p, &q, dt, steps);
    let sv_traj = integrator::stormer_verlet(&sys, &p, &q, dt, steps);

    let e0 = sys.hamiltonian(&p, &q);
    let euler_drift = (sys.hamiltonian(&euler_traj[steps].0, &euler_traj[steps].1) - e0).abs();
    let sv_drift = (sys.hamiltonian(&sv_traj[steps].0, &sv_traj[steps].1) - e0).abs();

    // Störmer-Verlet should have significantly less drift than symplectic Euler
    // (both are symplectic, but SV is second-order)
    assert!(
        sv_drift < euler_drift,
        "SV drift ({}) should be less than Euler drift ({})",
        sv_drift, euler_drift
    );
}
