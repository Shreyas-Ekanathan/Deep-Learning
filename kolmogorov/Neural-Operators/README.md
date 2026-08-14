# Kolmogorov Flow with Neural Operators, Part 3: Fourier Neural Operator

Part 2 ended with a diagnosis rather than a result: the latent neural ODE beat climatology at every lead time but went smooth at long range, and the reason was the loss, not the network. Part 3 replaces that architecture with a Fourier Neural Operator, while keeping the same data and evals. The comparison raises two questions: 1) do we beat the neural ODE, and 2) do we achieve the goal of operator learning, namely transfer to an unseen grid?

The short answers are no and yes. The FNO loses the pooled MSE comparison by about 5%, and it runs on a 128x128 grid at a 2.2% cost having only ever seen 64x64.

## The model (`v1/fno.py`)

- **Lifting**: a 1x1 convolution, 1 → 16 channels, at full 64x64 resolution.
- **Four Fourier layers**, 16 → 16 channels, keeping 12 modes per axis.
- **Projection**: 1x1 convolutions 16 → 64 → 1 with a GELU between.

954,209 parameters, with 589,824 spectral, 262,144 in the learned forcing fields, 99,968 across the four `ν` embedders, and 2,273 in the pointwise convolutions. Against Part 2's 403,329 that is 2.4x larger, which is worth holding in mind when reading the comparison.

There is no encoder and no downsampling. Part 2 shrank to a 16x16 latent because `odeint` had to be affordable; there is no integrator here and nothing to make affordable, and the mode truncation already does the job a stride-2 encoder was doing.

### The spectral convolution

The layer is a learned version of what the solver already does. Part 1's solver lives in Fourier space because the Laplacian is diagonal there: the derivatives are `1j·k`, the viscous term is `-νk²ω̂`, the Poisson inversion is `ω̂/k²`. Every one of those is a per-mode multiplication. So the layer transforms, multiplies each retained wavenumber by its **own** learned complex matrix `R_k` of shape (in_channels × out_channels), and transforms back:

```python
torch.einsum('bixy,ioxy->boxy', block, torch.view_as_complex(weights))
```

The channel index is contracted; the two frequency indices ride along untouched, since mode `k` never talks to mode `k'`. Note that the weights must be complex, not real, as a real multiplier can only rescale a mode's amplitude, which buys damping like `-νk²ω̂` but never transport, and phase rotation in Fourier space is translation in physical space. The solver's own `1j·KX` could not be represented by a real weight.

Running in parallel is a 1x1 convolution on the input. Truncation makes the spectral path blind above `k = 11`, so the pointwise path is the only route by which anything above the cutoff survives. The two are summed and then activated, giving `σ(K(x) + W(x))`.

That split is the same one the solver makes: linear operations where they are diagonal (Fourier), nonlinear operations where products are cheap (physical). A pseudospectral step, learned.

### The forcing

We keep Part 2's learned spatial forcing field, one per layer, and for the same reason as everything else here: changing the forcing mechanism and the dynamics core at once would confound the comparison. A pure FNO is exactly translation-equivariant, since spectral convolutions and pointwise operations both commute with a circular shift, so it structurally cannot represent the `y`-dependent drive without something to break the symmetry.

This is also the one part of the model defined per pixel rather than per wavenumber, and the only part that does not transfer across grids for free.

### Training

30-frame windows, stride 5 (9,375 windows), batch 64, Adam at 1e-3 annealed by cosine to 1e-5 over 50 epochs, weight decay 1e-4 on everything except the `ν` embedders and the forcing fields. The model predicts the increment, `x ← x + step(x)`, and the trajectory is composed autoregressively at `Δt = 0.1`.

Note that each rollout step is wrapped in `torch.utils.checkpoint`. Storing activations for 29 steps × 4 spectral blocks costs about 10.8 GB, which on a 16 GB machine means swap: the first attempt slowed progressively and stalled with 12.5 GB of swap in use. Recomputing in the backward pass costs 0.09 GB, a 120x reduction, and made training faster.

## Evaluation (`v1/evals.py`)

Ported from Part 2 and unchanged in substance: per-lag MSE against climatology and persistence, anomaly correlation, vorticity montages, enstrophy spectra, per-`ν` scoring with seen/unseen marked, the shuffled-`ν` ablation, and the `ν` response sweep.

One thing does not carry over. Part 2's lag 0 was `decoder(encoder(x0))` and cost 0.0185 MSE, the autoencoder reconstruction floor under everything else. The FNO works in physical space and seeds its rollout with `x0` itself, so lag 0 is exactly 0 MSE and exactly 1.0 ACC. That is an architectural difference rather than a result, and the lag 0 row is not comparable between the two parts.

Added for Part 3 is `report_transfer`, which scores the same checkpoint at 64x64 and at 128x128 through the same code path. The learned forcing is moved onto the finer grid by zero-padding its spectrum, which is exact rather than approximate for a band-limited field, as it preserves the field's rms to four decimals.

## Results

Window-mean MSE across checkpoints, against climatology 0.9858 and persistence 1.0003:

| epoch | 0 | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|
| MSE | 0.5962 | 0.4938 | 0.4781 | 0.4843 | 0.4818 | 0.4854 |

Best checkpoint at epoch 35, MSE 0.4758. Test loss was effectively flat from epoch 19 onward while train fell from 0.523 to 0.442, so the last twenty epochs were fitting rather than learning.

By lead time:

| t | MSE | skill | ACC |
|---|---|---|---|
| 0.0 | 0.0000 | 0.000 | 1.000 |
| 0.5 | 0.3206 | 0.330 | 0.819 |
| 1.0 | 0.4047 | 0.411 | 0.767 |
| 1.5 | 0.5024 | 0.504 | 0.704 |
| 2.0 | 0.5861 | 0.589 | 0.643 |
| 2.9 | 0.7084 | 0.713 | 0.542 |

It beats climatology at every lead time and holds `ACC ≥ 0.6` out to `t = 2.30`.

### Against Part 2

| `ν` | seen | FNO skill | v2 skill | FNO ACC | wrong-`ν` penalty |
|---|---|---|---|---|---|
| 0.025 | no | 0.602 | 0.590 | 0.620 | +122.9% |
| 0.0375 | no | 0.646 | 0.620 | 0.574 | +107.5% |
| 0.06 | yes | 0.605 | 0.574 | 0.613 | +74.4% |
| 0.095 | no | 0.466 | 0.425 | 0.726 | +49.4% |
| 0.15 | no | 0.055 | 0.060 | 0.972 | +992.5% |

The neural ODE wins on four of five viscosities and on the pooled number, 0.4546 against 0.4758, so roughly 5% with 42% of the parameters. The FNO's only win is `ν = 0.15`, the most laminar and least chaotic case.

That result is more interesting than it looks, since an earlier checkpoint scored per lead time showed the two models crossing: the FNO was the better short-horizon operator and the worse long-horizon one, with the crossover near `t ≈ 1.0`. Part 2 trained against 30-frame windows and therefore optimised long-rollout MSE directly, and past the predictability horizon the MSE-minimising forecast is the smooth conditional mean.

Conditioning, on the other hand, is unambiguously stronger. Shuffling `ν` across the test set costs +42.3% against Part 2's +36%, and the mismatched-`ν` penalties on the chaotic groups run 74–123% where Part 2 saw 56–73%. Measured directly, the output difference between `ν = 0.025` and `ν = 0.15` on identical input grows from 1.6% at initialisation to 129% at convergence.

### Resolution transfer

This is the experiment Part 2 cannot enter. A stride-2 convolution is defined in grid cells, so the neural ODE is welded to the resolution it was trained on, but a spectral weight is indexed by wavenumber, and wavenumber 3 means "three oscillations across the box" on any grid. Nothing in the model is indexed by cell offset, so the same weights can be evaluated on a finer sampling of the same field as we do.

| t | 64x64 skill | 128x128 skill | 64x64 ACC | 128x128 ACC |
|---|---|---|---|---|
| 0.3 | 0.303 | 0.307 | 0.835 | 0.833 |
| 1.5 | 0.504 | 0.511 | 0.704 | 0.700 |
| 2.9 | 0.713 | 0.722 | 0.542 | 0.533 |

Window-mean skill 0.483 at 64x64 and 0.493 at 128x128, a 2.2% cost for four times the pixels, with ACC tracking within 0.01 at every lead time.

The spectral check is what makes this a result rather than a number. At `t = 2.9` the enstrophy above `k = 21`, outside the 64x64 dealiasing band entirely, so content the model has never seen in any form, is 0.0005 in truth and 0.0003 in the prediction. The model is not aliasing or inventing structure there, which is the failure that would have made the MSE agreement hollow.

Note however that the 128x128 truth carries only 0.05% of its enstrophy above `k = 21`, which means 64x64 was genuinely well resolved and the finer grid adds almost no new physics. So this demonstrates invariance to how finely a representable function is sampled, not extrapolation to scales the model has no weights for. The model retains 12 modes; a flow with real energy at `k = 30` would transfer badly regardless.

### The model finds the forcing

Part 2 handed the network a free per-pixel field and let it fit whatever it wanted. Inspecting what it fitted:

| layer | rms | power at `kx=0` | dominant `ky` at `kx=0` |
|---|---|---|---|
| 0 | 0.0589 | 88.8% | 4, 8, 12 |
| 1 | 0.0416 | 69.1% | 4, 8 |
| 2 | 0.0312 | 55.4% | 4, 8 |
| 3 | 0.0213 | 36.3% | 4, 8 |

Every layer's dominant peak is at `(kx=0, ky=4)`, which is exactly `cos(4y)`, with the second at `(0, 8)`, the harmonic the nonlinearity generates from it. The first layer puts 88.8% of its power at `kx = 0`, having discovered that the forcing is `x`-independent.

The symmetry augmentation is what makes this possible: random `x` rolls make any `kx ≠ 0` forcing inconsistent across samples, and 16-cell `y` rolls admit only `ky ∈ 4ℤ`. Note that those shifts are redundant for a pure FNO, which is already exactly equivariant, but they start mattering the moment a free spatial field is introduced. The concentration decays with depth, so the deeper layers are fitting structure the physics does not have.

### Where it fails

The same place Part 2 failed, and now visible in the weights rather than only in the output. Mean spectral weight magnitude against wavenumber, layer 0 through 3:

| k | 0 | 2 | 4 | 6 | 8 | 11 |
|---|---|---|---|---|---|---|
| layer 0 | 0.0197 | 0.0488 | 0.0295 | 0.0145 | 0.0096 | 0.0027 |
| layer 3 | 0.0258 | 0.0636 | 0.0499 | 0.0347 | 0.0181 | 0.0057 |

Magnitude falls by 11–18x from `k = 2` to `k = 11` across the four layers. The upper third of the band is effectively dead: at `k = 8` the weights sit at under 30% of their peak, and at `k = 11` under 10%. Twenty-nine-step MSE offers no reason to model scales that are unpredictable past `t ≈ 1.5`, so those weights decay toward zero and the forecast hedges. (Depth helps: layer 3 reaches consistently higher than layer 0.)

The `ν` response sweep shows the same thing with a sharper edge than Part 2 managed. Enstrophy above `k = 8`, as a fraction of truth:

| `ν` | 0.025 | 0.04 | 0.06 | 0.09 | 0.13 |
|---|---|---|---|---|---|
| vs truth | 0.15x | 0.17x | 0.24x | 0.45x | 1.83x |

The deficit is not uniform, it is strongly `ν`-dependent, since the model retains 15% of the small-scale enstrophy at the most turbulent viscosity and overshoots by 1.83x at the most laminar. It blurs hardest exactly where the flow is most chaotic, which is where the small scales carry the most information.

This is not an architectural limit. Lag 0 is exact, the spectral layers can represent `k ≤ 11` perfectly well, and the resolution transfer shows the machinery handles finer grids without complaint. It is what minimising mean squared error over a chaotic rollout asks for. Fixing it needs a different objective, for example a spectral term in the loss, or a denoising formulation like PDE-Refiner, which is Part 4.

# To Come: PDE Refiner...
