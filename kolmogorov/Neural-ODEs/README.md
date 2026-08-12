# Kolmogorov Flow with Neural ODEs, Part 1: Fixed `ν`

Learn the dynamics of a chaotic 2D turbulent flow as a continuous-time latent ODE. Write a pseudospectral Navier-Stokes solver to generate ground truth, then train an encoder / latent neural ODE / decoder to predict the vorticity field forward in time, and benchmark it against the trivial baselines the way a weather model gets benchmarked. In this section, we treat `ν` as a fixed parameter, the second part of this project is to see if the model can learn to adapt to varying `ν`.

## The flow (`v1/kolmogorov_flow.py`)

Kolmogorov flow is 2D incompressible Navier-Stokes in a periodic box driven by a steady, unidirectional, sinusoidal body force. Here, we treat that force as `100·cos(4y)` added to the vorticity equation, which corresponds to a physical body force of `-25·sin(4y)` in x (one curl earlier). We have a 64x64 grid, with `L = 2π`, `ν = 0.1`.

We solve it in vorticity-streamfunction form, with the whole system collapsing to one scalar field evolving by `∂ω/∂t + (u·∇)ω = ν∇²ω + f`. The tricky part is that the equation for `ω` contains `u`, but `u` isn't a state variable, so we have to recover velocity from vorticity by solving `∇²ψ = -ω`. That is a global elliptic solve, and it's the reason the solver lives in Fourier space: the Laplacian is diagonal there, so the entire inversion is `omega_hat / K2`.

Our solver is pseudospectral, in the standard sense: linear terms (derivatives, the Poisson solve, viscous damping as `-νk²ω̂`) are computed exactly in Fourier space, and the nonlinear advection term computed in physical space, because a product there beats a convolution sum in Fourier space. However, we need to alias, since a product of modes `k₁` and `k₂` creates content at `k₁+k₂`, which wraps around past Nyquist and masquerades as a resolved mode. Therefore `mask` implements Orszag's 2/3 rule on the product, keeping `|k| ≤ 21`.

At these parameters, the Reynolds number `Re = U/(νk) ≈ 39`, where laminar Kolmogorov flow loses stability above `Re ≈ √2`. So the flow never settles onto the laminar profile; it breaks down into a chaotic state, which is the point.

## The model (`v1/model.py`)

- **Encoder**: 3 stride-2 convolutions, 1→16→32→64 channels, taking 64x64 down to 8x8. The latent is 64·8·8 = 4096 numbers against a 4096-pixel input, creating an even latent space.
- **ODEFunc**: 3 convolutions 64→128→128→64, plus a learned spatial forcing field, integrated by `torchdiffeq.odeint` (dopri5, `rtol=1e-3`, `atol=1e-6`).
- **Decoder**: three rounds of nearest-neighbour upsample followed by convolution, back to 64x64.

345,857 parameters. Every convolution uses `padding_mode='circular'`, as fits the periodic boundary conditions.

## Evaluation (`v1/evals.py`)

A single window-averaged MSE is close to useless here, for two reasons, so the benchmark reports more. 

We provide some baseline reference points, since "MSE = 0.41" means nothing until you know that predicting the climatology (the per-pixel time mean, ignoring the input entirely) scores 0.926 and persistence (which assumes that nothing changes) scores 0.855. Persistence crosses climatology at `t≈0.67`, which measures how fast the flow decorrelates.

Alongside MSE I also report anomaly correlation (ACC, not Antarctic Circumpolar Current!): subtract the climatology from both prediction and truth, then correlate the anomalies. This separates pattern skill from amplitude, which MSE conflates. The `ACC = 0.6` line is the operational weather-forecasting convention for useful skill.

Also generated per checkpoint: a vorticity montage (truth / prediction / error at five lead times, on one shared symmetric color scale, since per-panel autoscaling hides amplitude collapse) and enstrophy spectra, which are the tests for a prediction that has gone smooth keeps its low-k power and loses the high-k tail.

Results can be found in `v1/benchmarks/epoch_N/`.

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

---

# Kolmogorov Flow with Neural ODEs, Part 2: Learning `ν`

Part 1 learned one flow at one viscosity. Part 2 asks whether the same architecture can learn a family of flows, taking `ν` as an input and adapting its dynamics to it. This problem is closer to operator learning than to fitting a single trajectory. An interesting question that arises is not just if the loss goes down; it's whether the model uses the `ν` you hand it, or quietly infers everything from the initial condition and ignores the input entirely. 

## The data (`v2/kolmogorov_flow.py`)

Same pseudospectral solver as Part 1 (`100·cos(4y)` forcing, 64x64, `L = 2π`, Orszag 2/3 dealiasing). The only structural change is that `ν` moves from a module-level constant to an argument, and datagen sweeps it.

- **Train**: `ν ∈ {0.03, 0.043, 0.06, 0.088, 0.12}`, 75 initial conditions each, 375 runs.
- **Test**: `ν ∈ {0.025, 0.0375, 0.06, 0.095, 0.15}`, 10 each.

The test sweep is built to separate four questions that a single held-out set would conflate: `0.06` is a training value, so it isolates generalization over initial conditions; `0.0375` and `0.095` interpolate between training values; `0.025` extrapolates below the band and `0.15` above it. 

Both splits run `T = 15` over 150 snapshots, so `dt ≈ 0.1`, and the model trains on 30-frame windows. Each window is then divided by the standard deviation of its own first frame rather than by a global `σ`. Vorticity amplitude scales roughly like `1/ν`, so a global scale would let the model read viscosity straight off the magnitude of `x0` and never consult the `ν` input at all, and it would leave the losses incomparable. 

## The model (`v2/model.py`)

- **Encoder**: 2 stride-2 convolutions, 1→32→64, taking 64x64 to a 64x16x16 latent — 16,384 numbers against a 4,096-pixel input, so 4x expansion rather than Part 1's break-even 8x8. This was necessary for the neural ode to learn well. 
- **ODEFunc**: 3 convolutions 64→128→128→64, a learned spatial forcing field, and FiLM conditioning on `ν`, where an MLP produces a per-channel scale and shift, applied as `(1 + tanh(γ)) · net(z) + forcing + β`. Note that the scale is bounded, since otherwise integration failed with explosive stiffness due to the growth of `γ`.
- **Decoder**: 64→32 at 16x16, upsample, 32→16 at 32x32, upsample, 16→1 at 64x64.

403,329 parameters, all convolutions `padding_mode='circular'`.

### Symmetry augmentation

Training windows get a random circular shift: any of 64 in `x`, and multiples of 16 in `y`. Both are exact symmetries, as the forcing has no `x` dependence, and `cos(4(y+δ)) = cos(4y)` requires `δ = πk/2`, which is 16 cells on this grid. A shifted trajectory is a genuine solution of the same equation at the same `ν`, so these are 256 real training examples per window, not perturbations, helping us to avoid overfitting.

### Integration

We use the midpoint method with `step_size = 0.1`, which divides the output spacing so every requested time is landed on exactly rather than interpolated to. This replaced adaptive `dopri5` because the function evaluations were becoming prohibitively expensive and memory was exploding. 

## Evaluation (`v2/evals.py`)

Everything from Part 1, plus three test subjects aimed at the conditioning question.

**Per-`ν` scoring.** Every metric broken out by viscosity, with `seen`/`unseen` marked, so interpolation and extrapolation can be read separately.

**The shuffled-`ν` ablation.** Feed the correct `x0` but permute `ν` within the batch. Shuffling rather than substituting a constant keeps the marginal distribution exactly as trained, so a degradation can't be blamed on an out-of-distribution input. This test distinguishes real conditioning from a model that ignores the input.

**The `ν` response sweep.** Hold one `x0` fixed, sweep `ν`, and compare against what the data actually does. The direction test is the fraction of enstrophy above `k = 8`, which falls monotonically with `ν` in the data (0.0418 → 0.0056, 7.5x) and is invariant to the per-window normalization.

Results in `v2/benchmarks/epoch_N/` and `v2/benchmarks/best/`.

## Results

Window-mean MSE across checkpoints, against climatology 0.9858 and persistence 1.0003:

| epoch | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 75 |
|---|---|---|---|---|---|---|---|---|---|
| MSE | 0.6362 | 0.5459 | 0.4996 | 0.4860 | 0.4687 | 0.4642 | 0.4583 | 0.4602 | 0.4607 |

Best checkpoint at epoch 49, MSE 0.4546. By lead time:

| t | skill | ACC |
|---|---|---|
| 0.0 | 0.019 | 0.991 |
| 0.5 | 0.279 | 0.849 |
| 1.0 | 0.390 | 0.782 |
| 1.5 | 0.492 | 0.714 |
| 2.0 | 0.581 | 0.652 |
| 2.9 | 0.712 | 0.548 |

It beats climatology at every lead time and holds `ACC ≥ 0.6` out to `t = 2.50`. The `t = 0` value of 0.0185 MSE is pure autoencoder reconstruction. As skill that is 0.019 against Part 1's 0.032: the 16x16 latent cuts the reconstruction floor by roughly 40%, and the encoder now captures 98% of the field's variance before any dynamics are applied.

Per viscosity at the best checkpoint:

| `ν` | seen | skill | ACC | wrong-`ν` penalty |
|---|---|---|---|---|
| 0.025 | no | 0.590 | 0.629 | +73.2% |
| 0.0375 | no | 0.620 | 0.601 | +70.7% |
| 0.06 | yes | 0.574 | 0.638 | +56.0% |
| 0.095 | no | 0.425 | 0.753 | +63.7% |
| 0.15 | no | 0.060 | 0.970 | +856.2% |

Note that skill is not comparable across `ν`, since low `ν` is intrinsically harder due to more turbulent flow. The comparison that means something is against the trained value: extrapolating to 0.025 and 0.0375 gives 0.590 and 0.620 against 0.574 at the seen `ν = 0.06`.

Also note that the `ν` conditioning is real and grows with training:

| epoch | 0 | 20 | 40 | 60 | best |
|---|---|---|---|---|---|
| shuffled-`ν` penalty | +10.8% | +23.8% | +34.8% | +39.6% | +36.0% |

Feeding the model the wrong viscosity costs 36% on pooled MSE and 56–73% on the chaotic groups. 

However, the model is not foolproof, as it blurs at long lead times: small-scale enstrophy at `ν = 0.025` sits at 0.05x of truth by the end of the window, rising to 0.84x at `ν = 0.13`.

Note that this is not an architectural limit, as at `t = 0` the autoencoder reproduces 98% of the field, so encoder and decoder can represent those scales perfectly well. The rollout loses them, and this is a byproduct of MSE. Past the predictability horizon of a chaotic flow, the MSE-minimizing forecast is the conditional mean, which is smooth. The model is hedging its guesses, and the model cannot move past that, as sharp long-range fields need a spectral or adversarial term in the loss.