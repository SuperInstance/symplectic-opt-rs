# INTEGRATION.md — symplectic-opt-rs × conservation-law-rs × renormalization-group-rs

**Symplectic optimization** uses momentum updates that preserve phase-space
volume, avoiding the energy drift common in naive gradient descent. It
connects to Hamiltonian dynamics for agent state evolution and to
renormalization group for multi-scale parameter tuning.

## Synergy Map

```
conservation-law-rs          symplectic-opt-rs              renormalization-group-rs
┌──────────────────┐        ┌──────────────────────┐       ┌─────────────────────┐
│ SymplecticIntegr  │◄──────►│ SymplecticOptimizer  │◄─────►│ BetaFunction        │
│ AgentState        │        │ OptState             │       │ flow_rk4            │
│ total_energy      │        │ step                 │       │ FixedPoint          │
│ verify_noether    │        │ optimize             │       │ scaling_ratios      │
└──────────────────┘        │ objective_trace      │       └─────────────────────┘
                            └──────────────────────┘
```

## Key Insight

Standard gradient descent on agent parameters is like explicit Euler:
energy (objective) drifts and oscillations grow. Symplectic optimization
uses a leapfrog-style update that keeps the objective bounded, exactly as
symplectic integrators keep physical energy bounded. When combined with RG
flow, you can optimize parameters at coarse scale first, then refine.

## Example 1: Agent Parameter Optimization with Symplectic Steps

Optimize an agent's control parameters while tracking a Lagrangian energy.

```rust
use symplectic_opt::optimizer::{SymplecticOptimizer, OptState, objective_trace};
use conservation_law::lagrangian::{AgentState, MechanicalLagrangian, total_energy};

fn optimize_agent_parameters() {
    // Initial parameter vector [bias, gain]
    let state = OptState::new(vec![2.0_f64, -1.0], vec![0.0, 0.0]);
    let opt = SymplecticOptimizer::new(0.05, 0.9);

    // Gradient of a quadratic cost: bias^2 + gain^2
    let gradient = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];

    // Optimize for 200 steps
    let traj = opt.optimize(state, gradient, 200);

    // Track objective
    let obj = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
    let trace = objective_trace(&traj, obj);
    println!("Initial cost: {:.4}, Final cost: {:.4}", trace[0], trace.last().unwrap());

    // Verify with Lagrangian energy analogy
    let lagrangian = MechanicalLagrangian {
        mass: 1.0,
        potential_fn: |q: &[f64; 1]| 0.5 * q[0] * q[0],
    };
    let agent_state = AgentState::new([traj.last().unwrap().position[0]], [0.0]);
    println!("Parameter energy: {:.4}", total_energy(&lagrangian, &agent_state));
}
```

## Example 2: Multi-Scale Optimization via RG Flow

Coarse-grain the objective landscape, optimize at low resolution, then
refine with symplectic steps.

```rust
use symplectic_opt::optimizer::{SymplecticOptimizer, OptState};
use renormalization_group::coarse_grain::{coarse_grain, CoarseGrainConfig, AggregationMethod};
use renormalization_group::flow::BetaFunction;

fn multiscale_optimize(initial: &[f64]) -> Vec<f64> {
    // Step 1: Coarse-grain the parameter vector
    let config = CoarseGrainConfig {
        block_size: 2,
        method: AggregationMethod::Mean,
    };
    let coarse = coarse_grain(initial, &config);
    println!("Coarse parameters: {:?}", coarse);

    // Step 2: Optimize at coarse scale with RG flow
    let beta = BetaFunction::new(|g: &[f64]| {
        g.iter().map(|&x| -0.1 * x).collect()
    });
    let coarse_traj = beta.flow_rk4(&coarse, 20, 0.1);
    let coarse_opt = coarse_traj.points.last().unwrap().clone();

    // Step 3: Refine with symplectic optimizer
    let mut fine = initial.to_vec();
    for (i, &c) in coarse_opt.iter().enumerate() {
        if i < fine.len() {
            fine[i] = c;
        }
    }

    let opt = SymplecticOptimizer::new(0.01, 0.95);
    let state = OptState::new(fine, vec![0.0; initial.len()]);
    let gradient = |x: &[f64]| x.iter().map(|&v| 2.0 * v).collect::<Vec<f64>>();
    let refined = opt.optimize(state, gradient, 100);

    refined.last().unwrap().position.clone()
}
```

## Example 3: Compare Symplectic vs Naive Gradient Descent

Show that symplectic optimization avoids energy drift on a harmonic
oscillator-like objective.

```rust
use symplectic_opt::optimizer::{SymplecticOptimizer, OptState, objective_trace};

fn compare_methods() {
    let obj = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
    let grad = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];

    // Symplectic
    let sym = SymplecticOptimizer::new(0.1, 0.9);
    let sym_traj = sym.optimize(OptState::new(vec![3.0, 4.0], vec![0.0, 0.0]), grad, 50);
    let sym_trace = objective_trace(&sym_traj, obj);

    // Naive Euler: x -= lr * grad (no momentum half-step)
    let mut euler_pos = vec![3.0_f64, 4.0];
    let lr = 0.1;
    let mut euler_trace = vec![obj(&euler_pos)];
    for _ in 0..50 {
        let g = grad(&euler_pos);
        euler_pos[0] -= lr * g[0];
        euler_pos[1] -= lr * g[1];
        euler_trace.push(obj(&euler_pos));
    }

    println!("Symplectic final: {:.6}", sym_trace.last().unwrap());
    println!("Euler final:      {:.6}", euler_trace.last().unwrap());
}
```

## Cargo.toml Wiring

```toml
[dependencies]
symplectic-opt = { git = "https://github.com/SuperInstance/symplectic-opt-rs" }
conservation-law = { git = "https://github.com/SuperInstance/conservation-law-rs" }
renormalization-group = { git = "https://github.com/SuperInstance/renormalization-group-rs" }
```

## Design Patterns

### Pattern: Hamiltonian Learning Rate Scheduling

Treat the learning rate as a time-dependent parameter in a Hamiltonian
system, evolving it via symplectic steps rather than ad-hoc decay:

```rust
use symplectic_opt::optimizer::{SymplecticOptimizer, OptState};

fn hamiltonian_lr_schedule(initial_lr: f64, steps: usize) -> Vec<f64> {
    let mut lr_state = OptState::new(vec![initial_lr], vec![0.0]);
    let opt = SymplecticOptimizer::new(0.01, 0.99);

    // Gradient pulls LR toward zero (cooling)
    let grad = |x: &[f64]| vec![-0.1 * x[0]];

    let mut schedule = vec![initial_lr];
    for _ in 0..steps {
        opt.step(&mut lr_state, &grad);
        schedule.push(lr_state.position[0].max(1e-4));
    }
    schedule
}
```

This produces smooth, non-oscillatory cooling curves that preserve
optimization stability across long training runs.
