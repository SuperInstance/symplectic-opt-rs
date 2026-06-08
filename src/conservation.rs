//! Conservation law tracking for Hamiltonian simulations.
//!
//! Symplectic integrators preserve energy to within bounded oscillations (no secular drift).
//! This module provides tools to track energy, angular momentum, and other conserved
//! quantities across a trajectory.

/// Tracker for conservation laws during a Hamiltonian simulation.
///
/// Records energy at each time step and provides analysis of drift, boundedness,
/// and angular momentum conservation.
#[derive(Clone, Debug)]
pub struct ConservationTracker {
    /// Recorded energy values at each step.
    energies: Vec<f64>,
    /// Recorded (step, p, q) for computing other conserved quantities.
    states: Vec<(usize, Vec<f64>, Vec<f64>)>,
}

impl ConservationTracker {
    /// Create a new, empty conservation tracker.
    pub fn new() -> Self {
        Self {
            energies: Vec::new(),
            states: Vec::new(),
        }
    }

    /// Record a (step, p, q, energy) observation.
    pub fn track(&mut self, step: usize, p: &[f64], q: &[f64], energy: f64) {
        self.energies.push(energy);
        self.states.push((step, p.to_vec(), q.to_vec()));
    }

    /// Return the energy drift (deviation from initial energy) at each recorded step.
    pub fn energy_drift(&self) -> Vec<f64> {
        if self.energies.is_empty() {
            return Vec::new();
        }
        let e0 = self.energies[0];
        self.energies.iter().map(|e| e - e0).collect()
    }

    /// Return the maximum absolute energy drift over all recorded steps.
    pub fn max_drift(&self) -> f64 {
        self.energy_drift().iter().map(|d| d.abs()).fold(0.0_f64, f64::max)
    }

    /// Compute the relative energy drift: |ΔE| / |E₀|.
    pub fn relative_drift(&self) -> f64 {
        if self.energies.is_empty() {
            return 0.0;
        }
        let e0 = self.energies[0];
        if e0.abs() < 1e-15 {
            return self.max_drift();
        }
        self.max_drift() / e0.abs()
    }

    /// Return all recorded energy values.
    pub fn energies(&self) -> &[f64] {
        &self.energies
    }

    /// Return the number of recorded states.
    pub fn len(&self) -> usize {
        self.energies.len()
    }

    /// Whether any states have been recorded.
    pub fn is_empty(&self) -> bool {
        self.energies.is_empty()
    }
}

/// Compute the angular momentum L = Σ (q_i × p_i) for 2D systems (scalar).
///
/// For a 2D system with coordinates (q₁, q₂) and momenta (p₁, p₂):
/// L = q₁p₂ - q₂p₁
///
/// For higher-dimensional systems, this computes the z-component of angular momentum
/// using the first two coordinates.
pub fn angular_momentum(p: &[f64], q: &[f64]) -> f64 {
    assert!(p.len() >= 2 && q.len() >= 2, "Need at least 2D for angular momentum");
    q[0] * p[1] - q[1] * p[0]
}

/// Compute the total angular momentum vector L for a 3D system.
///
/// L = q × p
pub fn angular_momentum_3d(p: &[f64], q: &[f64]) -> Vec<f64> {
    assert!(p.len() >= 3 && q.len() >= 3, "Need at least 3D");
    vec![
        q[1] * p[2] - q[2] * p[1],
        q[2] * p[0] - q[0] * p[2],
        q[0] * p[1] - q[1] * p[0],
    ]
}

impl Default for ConservationTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_tracking_drift_bounded() {
        let mut tracker = ConservationTracker::new();
        let e0 = 1.0;
        for i in 0..100 {
            let e = e0 + 1e-12 * (i as f64).sin();
            tracker.track(i, &[], &[], e);
        }
        assert!(tracker.max_drift() < 1e-10);
        assert_eq!(tracker.len(), 100);
    }

    #[test]
    fn test_angular_momentum_2d() {
        let p = vec![0.0, 1.0];
        let q = vec![1.0, 0.0];
        let l = angular_momentum(&p, &q);
        assert!((l - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_angular_momentum_3d() {
        let p = vec![0.0, 1.0, 0.0];
        let q = vec![1.0, 0.0, 0.0];
        let l = angular_momentum_3d(&p, &q);
        // q × p = (0, 0, -1)... let's compute:
        // L_x = q_y*p_z - q_z*p_y = 0*0 - 0*1 = 0
        // L_y = q_z*p_x - q_x*p_z = 0*0 - 1*0 = 0
        // L_z = q_x*p_y - q_y*p_x = 1*1 - 0*0 = 1
        assert!((l[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_empty_tracker() {
        let tracker = ConservationTracker::new();
        assert!(tracker.is_empty());
        assert_eq!(tracker.energy_drift().len(), 0);
        assert_eq!(tracker.max_drift(), 0.0);
    }
}
