# Kolmogorov Flow with Neural ODEs, Part 1: Fixed `ν`

Learn the dynamics of a chaotic 2D turbulent flow as a continuous-time latent ODE. Write a pseudospectral Navier-Stokes solver to generate ground truth, then train an encoder / latent neural ODE / decoder to predict the vorticity field forward in time, and benchmark it against the trivial baselines the way a weather model gets benchmarked. In this section, we treat `ν` as a fixed parameter, the second part of this project is to see if the model can learn to adapt to varying `ν`.

## The flow (`kolmogorov_flow.py`)

Kolmogorov flow is 2D incompressible Navier-Stokes in a periodic box driven by a steady, unidirectional, sinusoidal body force. Here, we treat that force is `100·cos(4y)` added to the vorticity equation, which corresponds to a physical body force of `-25·sin(4y)` in x (one curl earlier). We have a 64x64 grid, with `L = 2π`, `ν = 0.1`.

We solve it in vorticity-streamfunction form, with the whole system collapsing to one scalar field evolving by `∂ω/∂t + (u·∇)ω = ν∇²ω + f`. The tricky part is that the equation for `ω` contains `u`, but `u` isn't a state variable, so we have to recover velocity from vorticity by solving `∇²ψ = -ω`. That is a global elliptic solve, and it's the reason the solver lives in Fourier space: the Laplacian is diagonal there, so the entire inversion is `omega_hat / K2`. One division.

Our solver is pseudospectral, in the standard sense: linear terms (derivatives, the Poisson solve, viscous damping as `-νk²ω̂`) computed exactly in Fourier space, and the nonlinear advection term computed in physical space, because a product there beats a convolution sum in Fourier space by ~100x at this resolution. However, we need to alias, since a product of modes `k₁` and `k₂` creates content at `k₁+k₂`, which wraps around past Nyquist and masquerades as a resolved mode. Therefore `mask` implements Orszag's 2/3 rule on the product, keeping `|k| ≤ 21`.

At these parameters, the Reynolds number `Re = U/(νk) ≈ 39`, where laminar Kolmogorov flow loses stability above `Re ≈ √2`. So the flow never settles onto the laminar profile; it breaks down into a chaotic state, which is the point.

## The model (`model.py`)

- **Encoder**: 3 stride-2 convolutions, 1→16→32→64 channels, taking 64x64 down to 8x8. The latent is 64·8·8 = 4096 numbers against a 4096-pixel input, so it's break-even rather than compressive.
- **ODEFunc**: 3 convolutions 64→128→128→64, plus a learned spatial forcing field, integrated by `torchdiffeq.odeint` (dopri5, `rtol=1e-3`, `atol=1e-6`).
- **Decoder**: three rounds of nearest-neighbour upsample followed by convolution, back to 64x64.

345,857 parameters. Every convolution uses `padding_mode='circular'`, as fits the periodic boundary conditions.

## Evaluation (`evals.py`)

A single window-averaged MSE is close to useless here, for two reasons, so the benchmark reports more. 

We provide some baseline reference points, since "MSE = 0.41" means nothing until you know that predicting the climatology (the per-pixel time mean, ignoring the input entirely) scores 0.926 and persistence (which assumes that nothing changes) scores 0.855. Persistence crosses climatology at `t≈0.67`, which measures how fast the flow decorrelates.

Alongside MSE I also report anomaly correlation (ACC, not Antarctic Circumpolar Current!): subtract the climatology from both prediction and truth, then correlate the anomalies. This separates pattern skill from amplitude, which MSE conflates. The `ACC = 0.6` line is the operational weather-forecasting convention for useful skill.

Also generated per checkpoint: a vorticity montage (truth / prediction / error at five lead times, on one shared symmetric color scale, since per-panel autoscaling hides amplitude collapse) and enstrophy spectra, which are the tests for a prediction that has gone smooth keeps its low-k power and loses the high-k tail.

Results can be found in `benchmarks/epoch_N/`.

## Results

Window-mean MSE across checkpoints, against climatology 0.9257 and persistence 0.8553:

| epoch | 0 | 15 | 30 | 45 | 60 | 75 |
|---|---|---|---|---|---|---|
| MSE | 0.5386 | 0.3042 | 0.2464 | 0.2187 | 0.2050 | 0.1995 |

The final model, by lead time (`skill` = MSE / climatology MSE; below 1 beats the mean field):

| t | MSE | skill | ACC |
|---|---|---|---|
| 0.00 | 0.0293 | 0.032 | 0.984 |
| 0.25 | 0.0823 | 0.089 | 0.955 |
| 0.50 | 0.1216 | 0.131 | 0.932 |
| 1.00 | 0.2675 | 0.289 | 0.846 |
| 1.45 | 0.4126 | 0.448 | 0.756 |

It beats climatology at every lead time and never approaches the ACC = 0.6 floor, ending at 0.756. The `t = 0` value of 0.0293 is pure autoencoder reconstruction error, which is the floor under everything else.