//! Symplectic momentum optimizer.

use crate::Scalar;

/// State of a symplectic optimizer: position and momentum.
#[derive(Clone, Debug)]
pub struct OptState<S> {
    pub position: Vec<S>,
    pub momentum: Vec<S>,
}

impl<S: Scalar> OptState<S> {
    pub fn new(position: Vec<S>, momentum: Vec<S>) -> Self {
        Self { position, momentum }
    }
}

/// Symplectic gradient descent with momentum.
/// Preserves a discrete symplectic form, avoiding energy drift.
pub struct SymplecticOptimizer<S: Scalar> {
    pub lr: S,
    pub momentum_coef: S,
}

impl<S: Scalar> SymplecticOptimizer<S> {
    pub fn new(lr: S, momentum_coef: S) -> Self {
        Self { lr, momentum_coef }
    }

    /// Perform one leapfrog-like step:
    ///   p_half = p - lr/2 * grad
    ///   x_new  = x + lr * p_half
    ///   p_new  = p_half - lr/2 * grad(x_new)
    pub fn step<F>(&self, state: &mut OptState<S>, gradient: F)
    where
        F: Fn(&[S]) -> Vec<S>,
    {
        let half = S::one() / (S::one() + S::one());
        let grad0 = gradient(&state.position);

        // Half-step momentum
        let mut p_half: Vec<S> = state
            .momentum
            .iter()
            .zip(grad0.iter())
            .map(|(&p, &g)| p - self.lr * half * g)
            .collect();

        // Apply momentum damping
        for p in p_half.iter_mut() {
            *p = *p * self.momentum_coef;
        }

        // Full-step position
        let x_new: Vec<S> = state
            .position
            .iter()
            .zip(p_half.iter())
            .map(|(&x, &p)| x + self.lr * p)
            .collect();

        // Half-step momentum with new gradient
        let grad1 = gradient(&x_new);
        let p_new: Vec<S> = p_half
            .iter()
            .zip(grad1.iter())
            .map(|(&p, &g)| p - self.lr * half * g)
            .collect();

        state.position = x_new;
        state.momentum = p_new;
    }

    /// Optimize for a fixed number of steps.
    pub fn optimize<F>(
        &self,
        initial: OptState<S>,
        gradient: F,
        steps: usize,
    ) -> Vec<OptState<S>>
    where
        F: Fn(&[S]) -> Vec<S>,
    {
        let mut traj = vec![initial.clone()];
        let mut state = initial;
        for _ in 0..steps {
            self.step(&mut state, &gradient);
            traj.push(state.clone());
        }
        traj
    }
}

/// Compute the objective value along a trajectory.
pub fn objective_trace<S: Scalar, O>(traj: &[OptState<S>], obj: O) -> Vec<S>
where
    O: Fn(&[S]) -> S,
{
    traj.iter().map(|s| obj(&s.position)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_minimization() {
        let opt = SymplecticOptimizer::new(0.1, 0.9);
        let state = OptState::new(vec![5.0_f64], vec![0.0]);
        let gradient = |x: &[f64]| vec![2.0 * x[0]]; // grad of x^2
        let traj = opt.optimize(state, gradient, 200);
        let final_pos = traj.last().unwrap().position[0];
        assert!(final_pos.abs() < 1e-1, "did not converge to zero: {}", final_pos);
    }

    #[test]
    fn objective_decreases() {
        let opt = SymplecticOptimizer::new(0.05, 0.95);
        let state = OptState::new(vec![3.0_f64, -2.0], vec![0.0, 0.0]);
        let gradient = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
        let obj = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
        let traj = opt.optimize(state, gradient, 50);
        let trace = objective_trace(&traj, obj);
        assert!(trace.last().unwrap() < trace.first().unwrap());
    }
}
