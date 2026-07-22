# Acrobot Control (from scratch)

A progression of control tasks on the underactuated Acrobot (torque only on the second joint), everything hand-rolled, with networks, losses, training loops, and a custom continuous-action env. Convention, as per acrobot environment, is that `θ1 = 0` hangs down, **upright is `θ1 = π, θ2 = 0`**.

The files build up in difficulty, from "reach the top" to "swing up *and* balance there." The first three are warm-ups solved quickly; the last — continuous stabilization — is the main effort and the focus below, finished off with an LQR handoff (`hybrid_controller.py`) that finally sticks the balance.

## The warm-ups

- **`dqn.py` — DQN swing-up** on `Acrobot-v1` (reach the top; discrete 3-action torque). Online + target net, replay buffer, ε-greedy, Bellman targets. Solved (~75 greedy steps to goal).
- **`ppo.py` — PPO swing-up**, same task. Actor-critic, clipped surrogate, Monte-Carlo returns, entropy bonus. Solved (~77 steps) — a deliberate value-based → policy-gradient contrast against the DQN.
- **`stabilization.py` — first balance attempt**, discrete torque + an energy/LQR reward. Swings up but *can't hold* the top: 3 discrete torque levels are too coarse for the fine corrections balancing needs. This failure is what motivated going continuous.

## The main task: continuous swing-up + balance (`stabilization_v2.py`)

Continuous-torque PPO that must **swing up and then stabilize** at the upright — a single policy doing both an energy-pumping and a balancing job. This is genuinely hard (an unstable equilibrium on an underactuated system), and most of the work was in the reward design.

### Infrastructure

- **Continuous env** by subclassing `AcrobotEnv` (`gym.Wrapper` isn't sufficient); `Box` action space, own 500-step truncation.
- **GAE** (λ=0.95) instead of raw MC returns.
- **Restart distribution**: ~20% of training episodes start *near the top*, so the policy actually experiences the balance basin instead of only ever arriving there at speed. Help the model learn behavior in the balancing regime.
- Policy outputs a `tanh`-bounded mean; `log_std` is a state-independent `nn.Parameter`.
- **Eval metric**: fraction of steps in the tight target set (`‖dev‖<0.3, ‖v‖<1`), greedy (mean action).

### Reward design journey

The bulk of the effort. Each reward fixed the previous one's failure mode:

1. **Two-branch** (LQR near top, `−(E−E*)²` elsewhere) → squared energy gave ~1e10 losses; thrashes through the top.
2. **Hybrid** (dense height + tiny two-branch) → still whirls: a height reward is *farmed by fly-bys* through the top.
3. **Height-only + restart distribution** → whirl persists, and a too-strong velocity penalty could make it *avoid* the top entirely.
4. **Dropped the branch** → `height + (−|E−E*|) − torque`. Cleaner, but the two weights kept fighting.
5. **Sign-flip velocity trick** (reward speed when low, penalize when high) → elegant but *reward-hackable*: unbounded velocity reward → spin forever in the lower half.
6. **Energy-matching + target bonus** → `+2` when in a low-velocity **and** high-position set. Un-farmable (a fly-by has high velocity), makes the hold the unique optimum. Reaches the top and loosely hovers.
7. **Gated LQR well** (`−c·up·‖v‖²  −c·up·‖dev‖`), coefficients tuned `0.01 → 0.25` — this is what turned the hover into a hold.
8. **Bounded the terms**: capped the energy penalty (`E` contains kinetic energy ~`v²`, which blew it up to 100+/step), capped the velocity penalty, and sharpened the gate to `up²` so the strong penalties live only at the very top and don't tax the swing-up pass. Bounding the reward stabilized the critic (loss went from *climbing* to converging).

**Final reward:** `2·height − 0.1·min(|E−E*|,60) − 0.25·up²·min(‖v‖²,40) − 0.25·up²·‖dev‖ − 0.001·τ²`, plus `+2` inside the target set. Per-step max is `+6`, achieved uniquely at upright-and-still → **not gameable**.

### Results

- All videos of training are shown in the corresponding videos of training folder.
- Particularly interesting is the training progression of the continuous stabilization model, it really starts to learn
what its doing around epoch 40. 
- End results showed that it could keep around 30% of the steps in the target regime (see iterations 140, 145, 150, where we see the model really learning to hold itself up as opposed to prior results). The model struggles with stabilization (roughly 20 step vertical bursts), but clearly has learned and made progress.

## Sticking the landing: LQR handoff (`hybrid_controller.py`)

Evaluating the trained policy over 200 episodes exposed the real limit: the ~30% fraction is accumulated from **short bursts** — longest continuous hold ~16 steps, and *zero* episodes hold for ≥100 steps in a row. It reaches the top and dwells briefly but never truly locks in. That's fundamental: the upright is an unstable equilibrium, and one end-to-end policy learning both the global swing-up and the razor-thin local balance is asking a lot.

The classical fix is a **hybrid controller**: let the learned policy do the swing-up (which it's great at), and hand off to a small analytic **LQR** balance controller once near the top. This is purely an *execution-time* switch — no retraining, no reward change (unlike the earlier discontinuous *reward*, which hurt learning, a discontinuity in the *control law* is standard and fine):

- Linearize the dynamics about the upright (numerically, via finite differences on the env), solve the continuous-time Riccati equation for the gain `K`.
- Far from the top → learned PPO policy. Near the top (`‖dev‖<0.3, ‖v‖<3`) → `τ = −K·[dev; vel]`, with hysteresis (drop back to the policy if it falls past `‖dev‖>0.6`).

Result — the handoff converts brief dwelling into a genuine indefinite hold:

| metric (200 eps) | pure PPO | PPO swing-up + LQR |
|---|---|---|
| fraction in target | 0.29 | 0.33 |
| longest continuous hold | ~16 steps | **~150 (median), 400 max** |
| episodes holding ≥100 steps | **0%** | **~58%** |

The fraction barely moves (and its variance jumps) because performance is now bimodal — when the LQR catches, it holds dead-vertical for most of the episode; when the swing-up arrives too fast to catch, it misses that episode. The longest-hold metric is the honest one, and it goes from "never holds" to "holds indefinitely in the majority of episodes." Demo rollouts in `final_hybrid_videos/`.

Takeaway: end-to-end RL nails the hard nonlinear swing-up; a two-line optimal-control law nails the local balance the policy couldn't. Using each tool for the part it's best at beats forcing one to do both.
