# symplectic-opt

**Symplectic geometry for optimization — Hamiltonian systems, symplectic integrators, and natural gradient descent.**

[![crates.io](https://img.shields.io/crates/v/symplectic-opt.svg)](https://crates.io/crates/symplectic-opt)
[![docs.rs](https://docs.rs/symplectic-opt/badge.svg)](https://docs.rs/symplectic-opt)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## Why Symplectic Geometry for Optimization?

Optimization is dynamics. Gradient descent, momentum methods, Hamiltonian Monte Carlo — they're all dynamical systems evolving in parameter space. The geometry of that space matters.

### The Problem with Naive Integration

Consider training a neural network. Each gradient step is like a time step in a dynamical system. Standard integrators (forward Euler, RK4) don't preserve the structure of the space they evolve through. Over thousands of steps:

- **Energy drifts** — the loss landscape's "energy" isn't conserved, leading to unstable training
- **Phase space volume contracts or expands** — the effective search region shrinks or explodes
- **Conservation laws break** — symmetries that should be preserved get violated

This isn't theoretical. In molecular dynamics, naive integrators cause energy to spiral out of control. In machine learning, they cause training instabilities.

### The Symplectic Solution

A **symplectic integrator** preserves the symplectic 2-form ω = Σ dpᵢ ∧ dqᵢ. This has profound consequences:

1. **Bounded energy error** — Energy doesn't drift secularly; it oscillates within a tight band forever
2. **Phase space volume preservation** — Liouville's theorem is exactly satisfied
3. **Time-reversibility** — The integrator can be run backwards to recover the initial state
4. **Structure preservation** — Conserved quantities (angular momentum, momentum) stay conserved

For optimization, this means:
- **Stable long-horizon training** — No energy blowup over millions of steps
- **Better exploration** — Phase space volume preservation means the optimizer explores broadly
- **Natural gradient connection** — The Fisher information matrix is the metric on the symplectic manifold

## Architecture

```
symplectic-opt
├── symplectic        Symplectic matrix operations (J, symplecticity test, inverse)
├── hamiltonian       Hamiltonian systems, Hamilton's equations
├── integrator        Symplectic Euler, Störmer-Verlet, leapfrog
├── conservation      Energy drift tracking, angular momentum
└── natural_gradient  Fisher information, natural gradient descent
```

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
symplectic-opt = "0.1.0"
```

### Harmonic Oscillator

The simplest non-trivial Hamiltonian system: H(p, q) = ½(p² + q²).

```rust
use symplectic_opt::*;

fn main() {
    // Define H = T(p) + V(q) = ½|p|² + ½|q|²
    let kinetic = |p: &[f64]| p.iter().map(|x| 0.5 * x * x).sum();
    let potential = |q: &[f64]| q.iter().map(|x| 0.5 * x * x).sum();
    let system = separable_hamiltonian(kinetic, potential);

    // Initial state: p=1, q=0 (maximum kinetic energy)
    let p = vec![1.0];
    let q = vec![0.0];

    // Integrate for 10,000 steps with dt=0.01
    let trajectory = stormer_verlet(&system, &p, &q, 0.01, 10_000);

    // Check energy conservation
    let e0 = energy(&system, &trajectory[0].0, &trajectory[0].1);
    let ef = energy(&system, &trajectory[10_000].0, &trajectory[10_000].1);
    println!("Initial energy: {}", e0);
    println!("Final energy:   {}", ef);
    println!("Drift:          {}", (ef - e0).abs());
    // Drift < 1e-10 — energy is conserved to machine precision
}
```

## Module Guide

### `symplectic` — Symplectic Matrix Operations

The canonical symplectic form J is the foundation:

```
J = [ 0  I ]
    [-I  0 ]
```

where I is the n×n identity matrix. J has key properties:
- J² = -I (like the imaginary unit)
- Jᵀ = -J (antisymmetric)
- det(J) = 1

A matrix S is **symplectic** if SᵀJS = J. Symplectic matrices:
- Preserve the symplectic 2-form
- Preserve phase space volume (Liouville's theorem)
- Have determinant 1
- Form a group under multiplication (the symplectic group Sp(2n))

```rust
use symplectic_opt::symplectic::*;
use nalgebra::DMatrix;

// The canonical form for n=2 (4×4 matrix)
let j = canonical_form(2);

// Check if a matrix is symplectic
let identity = DMatrix::identity(4, 4);
assert!(is_symplectic(&identity, 1e-10));

// Compute symplectic inverse: S^{-1} = -J Sᵀ J
let j_inv = symplectic_inverse(&j);
assert!((&j_inv - &(-j.transpose())).iter().all(|&v| v.abs() < 1e-12));

// Wrapped SymplecticMatrix type
let sm = SymplecticMatrix::new(j.clone());
assert!(sm.verify(1e-10));
println!("Dimension: 2n = {}", sm.dim());
```

### `hamiltonian` — Hamiltonian Systems

A Hamiltonian system is defined by a scalar function H(p, q) where:
- `p` = momenta (conjugate to velocities)
- `q` = generalized coordinates
- `H(p, q)` = total energy = T(p) + V(q) for separable systems

Hamilton's equations of motion:
```
dq/dt =  ∂H/∂p   (how coordinates change)
dp/dt = -∂H/∂q   (how momenta change — Newton's second law in disguise)
```

```rust
use symplectic_opt::hamiltonian::*;

// Separable Hamiltonian: H(p,q) = T(p) + V(q)
let system = separable_hamiltonian(
    |p: &[f64]| p.iter().map(|x| 0.5 * x * x).sum(),  // T = ½|p|²
    |q: &[f64]| q.iter().map(|x| 0.5 * x * x).sum(),  // V = ½|q|²
);

// Evaluate the Hamiltonian
let h = energy(&system, &[1.0], &[1.0]);
// h = 0.5 + 0.5 = 1.0

// Compute Hamilton's equations
let (dq_dt, dp_dt) = hamiltons_equations(&system, &[1.0], &[0.0]);
// dq/dt = ∂H/∂p = p = 1.0
// dp/dt = -∂H/∂q = -q = 0.0
```

### `integrator` — Symplectic Integrators

This is where the magic happens. Symplectic integrators preserve the geometric
structure of the Hamiltonian flow.

#### Symplectic Euler (First-Order)

The simplest symplectic integrator. Semi-implicit:

```
p_{n+1} = p_n - dt · ∂V/∂q(q_n)
q_{n+1} = q_n + dt · ∂T/∂p(p_{n+1})
```

First-order accurate but symplectic — energy oscillates without secular drift.

#### Störmer-Verlet (Second-Order) / Leapfrog

The workhorse of Hamiltonian simulation. Second-order accurate, symplectic,
and time-reversible:

```
p_{n+½} = p_n - (dt/2) · ∂V/∂q(q_n)
q_{n+1} = q_n + dt · ∂T/∂p(p_{n+½})
p_{n+1} = p_{n+½} - (dt/2) · ∂V/∂q(q_{n+1})
```

This is the integrator used in:
- **Molecular dynamics** (Verlet integration)
- **Hamiltonian Monte Carlo** (the "leapfrog" in NUTS)
- **Celestial mechanics** (orbit integration)
- **Training neural networks** (via symplectic optimization)

```rust
use symplectic_opt::*;

let system = separable_hamiltonian(
    |p: &[f64]| p.iter().map(|x| 0.5 * x * x).sum(),
    |q: &[f64]| q.iter().map(|x| 0.5 * x * x).sum(),
);

let p = vec![1.0];
let q = vec![0.0];

// All three integrators
let euler_traj = symplectic_euler(&system, &p, &q, 0.01, 1_000);
let sv_traj = stormer_verlet(&system, &p, &q, 0.01, 1_000);
let lf_traj = leapfrog(&system, &p, &q, 0.01, 1_000);

// leapfrog is an alias for stormer_verlet
assert_eq!(sv_traj.len(), lf_traj.len());
```

### `conservation` — Tracking Conservation Laws

Symplectic integrators don't conserve energy *exactly* — they conserve it to within
a bounded oscillation. The `ConservationTracker` lets you monitor this:

```rust
use symplectic_opt::*;
use symplectic_opt::conservation::{ConservationTracker, angular_momentum};

let system = separable_hamiltonian(
    |p: &[f64]| p.iter().map(|x| 0.5 * x * x).sum(),
    |q: &[f64]| q.iter().map(|x| 0.5 * x * x).sum(),
);

let p = vec![1.0, 0.0];
let q = vec![0.0, 1.0];
let traj = stormer_verlet(&system, &p, &q, 0.01, 5_000);

let mut tracker = ConservationTracker::new();
for (i, (pi, qi)) in traj.iter().enumerate() {
    tracker.track(i, pi, qi, energy(&system, pi, qi));
}

println!("Max energy drift: {}", tracker.max_drift());
println!("Relative drift:   {}", tracker.relative_drift());

// Angular momentum for 2D systems
let l = angular_momentum(&traj[0].0, &traj[0].1);
println!("Angular momentum: {}", l);
```

### `natural_gradient` — Natural Gradient via Symplectic Structure

The natural gradient adjusts optimization steps by the Fisher information matrix,
which encodes the curvature of the parameter space. For systems with symplectic
structure, the Fisher matrix and symplectic form are intimately connected.

```rust
use symplectic_opt::natural_gradient::*;
use nalgebra::DMatrix;

// Compute Fisher information from gradient samples
let samples = vec![
    vec![1.0, 0.5],
    vec![0.5, 1.0],
    vec![-0.3, 0.8],
];
let fisher = fisher_information(&samples);

// Compute natural gradient
let gradient = vec![1.0, 0.5];
let nat_grad = natural_gradient(&gradient, &fisher);
println!("Euclidean gradient: {:?}", gradient);
println!("Natural gradient:   {:?}", nat_grad);
```

## Theory Deep Dive

### Why Energy Conservation Matters for Optimization

In a Hamiltonian system, the total energy H is constant along true trajectories.
Numerical integrators approximate these trajectories, and the quality of that
approximation determines how well energy is conserved.

**Forward Euler** (not symplectic):
- Energy drifts monotonically — it grows without bound
- Phase space volume contracts → optimizer gets trapped
- After N steps, energy error ~ O(N · dt)

**Symplectic Euler**:
- Energy oscillates but doesn't drift
- Phase space volume is exactly preserved
- Energy error bounded by O(dt) for all time

**Störmer-Verlet**:
- Energy oscillates with amplitude O(dt²)
- Phase space volume exactly preserved
- Time-reversible: running the integrator backwards recovers the initial state

For a concrete example, integrating the harmonic oscillator H = ½(p² + q²) for
10,000 steps with dt = 0.01:

| Integrator       | Energy Drift  | Symplectic? | Order |
|------------------|---------------|-------------|-------|
| Forward Euler    | ~ 10³ (blows up) | No       | 1     |
| Symplectic Euler | ~ 10⁻⁴       | Yes         | 1     |
| Störmer-Verlet   | < 10⁻¹⁰      | Yes         | 2     |
| RK4              | ~ 10⁻⁸       | No          | 4     |

RK4 has excellent single-step accuracy but its energy drift grows linearly with time.
Störmer-Verlet's drift is *bounded* — after a billion steps, it's the same as after
a thousand.

### The Symplectic 2-Form and Phase Space Volume

The symplectic 2-form ω = Σ dpᵢ ∧ dqᵢ defines the area element in phase space.
Symplectic maps preserve this form, which by Darboux's theorem means they preserve
the natural volume element:

```
∫ dp dq = const  (Liouville's theorem)
```

For optimization, this means the optimizer explores a constant volume of parameter
space. It can't collapse to a tiny region (missing good solutions) or explode
to infinity (becoming unstable).

### Connection to Natural Gradient

The Fisher information matrix F is the Hessian of the KL divergence:

```
F = E[∇ log p(x|θ) ∇ log p(x|θ)ᵀ]
```

On a symplectic manifold, the Fisher matrix and the symplectic form J together
define a **Kähler structure** — a manifold that is simultaneously Riemannian (F),
symplectic (J), and complex (F⁻¹J). The natural gradient g̃ = F⁻¹g respects this
geometry, yielding optimization paths that are "straight" in the information-geometric
sense.

This is why HMC (which uses symplectic integration) and natural gradient methods
(which use Fisher information) are among the most effective optimization methods:
they both respect the underlying geometry.

## Comparison with Related Crates

| Crate              | Focus                                    | Symplectic? | Integrators |
|--------------------|------------------------------------------|-------------|-------------|
| `symplectic-opt`   | Optimization + geometry                  | Yes         | Euler, SV   |
| `nalgebra`         | Linear algebra                           | No          | —           |
| `ode-solvers`      | General ODE integration                  | No          | RK, Dopri   |
| `hmc`              | Hamiltonian Monte Carlo                  | Yes         | Leapfrog    |
| `geometrics`       | Geometric numerical integration          | Yes         | Various     |

## Integration with Other Crates

### Using with `nalgebra`

`symplectic-opt` uses `nalgebra` for all linear algebra operations. You can freely
interoperate:

```rust
use symplectic_opt::*;
use nalgebra::DMatrix;

// Build a custom symplectic matrix
let j = canonical_form(2);
let rotation = DMatrix::from_row_slice(4, 4, &[
    1.0, 0.0,  0.1, 0.0,
    0.0, 1.0,  0.0, 0.1,
    -0.1, 0.0, 1.0, 0.0,
    0.0, -0.1, 0.0, 1.0,
]);

if is_symplectic(&rotation, 1e-6) {
    println!("Rotation is symplectic!");
}
```

### Conservation Law Tracking

The `ConservationTracker` can be used standalone to monitor any Hamiltonian simulation:

```rust
use symplectic_opt::conservation::ConservationTracker;

let mut tracker = ConservationTracker::new();

// In your integration loop:
for step in 0..steps {
    let e = compute_energy(&p, &q);
    tracker.track(step, &p, &q, e);
    // ... integration step ...
}

println!("Max energy drift: {}", tracker.max_drift());
println!("Relative drift:   {:.2e}", tracker.relative_drift());
```

## Examples

### Planetary Orbit (Central Force)

```rust
use symplectic_opt::*;

fn main() {
    // Kepler problem: V(r) = -1/r, T(p) = ½|p|²
    let system = separable_hamiltonian(
        |p: &[f64]| p.iter().map(|x| 0.5 * x * x).sum(),
        |q: &[f64]| {
            let r = q.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);
            -1.0 / r
        },
    );

    // Start at (1, 0) with tangential velocity
    let p = vec![0.0, 1.0];
    let q = vec![1.0, 0.0];

    let traj = stormer_verlet(&system, &p, &q, 0.001, 50_000);

    // The orbit should be nearly closed
    let (pf, qf) = &traj[50_000];
    let r_final = qf.iter().map(|x| x * x).sum::<f64>().sqrt();
    println!("Final radius: {:.6} (started at 1.0)", r_final);
}
```

### Coupled Oscillators

```rust
use symplectic_opt::*;

fn main() {
    // Two coupled oscillators: V = ½(q₁² + q₂² + ε(q₁-q₂)²)
    let epsilon = 0.1;
    let system = separable_hamiltonian(
        |p: &[f64]| p.iter().map(|x| 0.5 * x * x).sum(),
        move |q: &[f64]| {
            0.5 * (q[0] * q[0] + q[1] * q[1])
            + epsilon * (q[0] - q[1]).powi(2)
        },
    );

    let p = vec![1.0, 0.0];
    let q = vec![1.0, -1.0];

    let traj = stormer_verlet(&system, &p, &q, 0.01, 10_000);

    // Energy should be conserved
    let e0 = energy(&system, &traj[0].0, &traj[0].1);
    let ef = energy(&system, &traj[10_000].0, &traj[10_000].1);
    println!("Energy drift: {:.2e}", (ef - e0).abs());
}
```

### Comparing Integrators

```rust
use symplectic_opt::*;

fn main() {
    let system = separable_hamiltonian(
        |p: &[f64]| p.iter().map(|x| 0.5 * x * x).sum(),
        |q: &[f64]| q.iter().map(|x| 0.5 * x * x).sum(),
    );

    let p = vec![1.0];
    let q = vec![0.0];
    let dt = 0.1;
    let steps = 1_000;

    let euler = symplectic_euler(&system, &p, &q, dt, steps);
    let sv = stormer_verlet(&system, &p, &q, dt, steps);

    let e0 = energy(&system, &p, &q);

    println!("Symplectic Euler drift: {:.6}",
        (energy(&system, &euler[steps].0, &euler[steps].1) - e0).abs());
    println!("Störmer-Verlet drift:   {:.10}",
        (energy(&system, &sv[steps].0, &sv[steps].1) - e0).abs());
}
```

## Performance

Symplectic integrators are intentionally simple — no adaptive step size, no embedded
error estimates. This makes them:

- **Fast per step**: O(d) where d is the dimension
- **Cache-friendly**: Sequential memory access patterns
- **Predictable**: Constant work per step, no rejection

The Störmer-Verlet integrator requires only 2 force evaluations per step (shared
between consecutive steps via the half-step trick), making it one of the most
efficient second-order methods available.

## Limitations

1. **Separable Hamiltonians only**: The current integrators assume H(p,q) = T(p) + V(q).
   Non-separable systems require implicit methods.
2. **Fixed step size**: No adaptive step size control. For stiff systems, this may
   require small dt.
3. **No constraints**: No support for constrained Hamiltonian systems (SHAKE, RATTLE).
4. **Numerical gradients**: The `separable_hamiltonian` constructor uses finite
   differences. For production use, supply analytical gradients.

## Future Directions

- [ ] Higher-order symplectic integrators (Yoshida 4th order, Forest-Ruth)
- [ ] Adaptive step size via time-transformation
- [ ] Implicit symplectic methods for non-separable Hamiltonians
- [ ] RATTLE/SHAKE for constrained systems
- [ ] Symplectic neural networks (SympNets)
- [ ] GPU acceleration via `custos` or `wgpu`
- [ ] Integration with `candle` or `burn` for ML pipelines
- [ ] Stochastic symplectic integrators for Langevin dynamics

## References

1. **Hairer, E., Lubich, C., & Wanner, G.** (2006). *Geometric Numerical Integration*.
   Springer. — The definitive reference on symplectic integrators.

2. **Leimkuhler, B., & Reich, S.** (2004). *Simulating Hamiltonian Dynamics*.
   Cambridge University Press. — Practical guide to symplectic methods.

3. **Amari, S.** (1998). Natural Gradient Works Efficiently in Learning.
   *Neural Computation*, 10(2), 251–276. — Natural gradient in ML.

4. **Neal, R.** (2011). MCMC Using Hamiltonian Dynamics. — HMC and the leapfrog.

5. **Betancourt, M.** (2017). A Conceptual Introduction to Hamiltonian Monte Carlo.
   — Modern perspective on HMC and symplectic integration.

6. **Yoshida, H.** (1990). Construction of higher order symplectic integrators.
   *Physics Letters A*, 150(5-7), 262–268.

7. **Verlet, L.** (1967). Computer "Experiments" on Classical Fluids.
   *Physical Review*, 159(1), 98–103. — The original Verlet paper.

8. **Sanz-Serna, J.M., & Calvo, M.P.** (1994). *Numerical Hamiltonian Problems*.
   Chapman & Hall. — Hamiltonian PDEs and symplectic methods.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Areas of particular interest:

- Higher-order symplectic integrators (4th order Yoshida, 6th order)
- Non-separable Hamiltonian support
- GPU-accelerated implementations
- Real-world ML benchmarks
- Improved documentation and examples

Please open an issue or PR on [GitHub](https://github.com/SuperInstance/symplectic-opt-rs).

---

*Built with symplectic geometry. Optimized with structure preservation.*
