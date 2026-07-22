# Acrobot Control (from scratch)

Control tasks on the underactuated Acrobot (torque only on the second joint). Everything is hand-rolled: networks, losses, training loops, and a custom continuous-action env. Convention (from the Acrobot env): `θ1 = 0` hangs down, upright is `θ1 = π, θ2 = 0`.

The files go from "reach the top" to "swing up and balance there." The first three are quick warm-ups. The last one, continuous stabilization, is the main effort, finished with an LQR handoff that actually holds the balance.

## Warm-ups

- `dqn.py`: DQN swing-up on `Acrobot-v1` (discrete 3-action torque). Online + target net, replay buffer, epsilon-greedy, Bellman targets. Solved, ~75 greedy steps to goal.
- `ppo.py`: PPO swing-up, same task. Actor-critic, clipped surrogate, MC returns, entropy bonus. Solved, ~77 steps. A value-based vs policy-gradient contrast with the DQN.
- `stabilization.py`: first balance attempt with discrete torque and an energy/LQR reward. Swings up but can't hold. 3 torque levels are too coarse for the fine corrections balancing needs. This is what motivated going continuous.

## Main task: continuous swing-up + balance (`stabilization_v2.py`)

Continuous-torque PPO that has to swing up and then stabilize at the top with one policy. Hard problem: the upright is an unstable equilibrium on an underactuated system. Most of the work was in the reward, with some refactors to PPO itself.

Setup:

- Continuous env by subclassing `AcrobotEnv` (a `gym.Wrapper` doesn't propagate state). `Box` action space, own 500-step truncation.
- GAE (lambda 0.95) instead of raw MC returns.
- Restart distribution: ~40% of training episodes start near the top, so the policy actually sees the balance region instead of only ever arriving at speed.
- Policy outputs a tanh-bounded mean; `log_std` is a state-independent `nn.Parameter`.
- Grad-norm clipping (0.5) and a moderate epoch count to keep on-policy updates stable.
- Eval metric: fraction of steps in the tight target set (`‖dev‖<0.3, ‖v‖<1`), greedy (mean action).

### Reward iterations

Each reward fixed the previous one's failure:

1. Two-branch (LQR near top, `-(E-E*)²` elsewhere). Squared energy gave ~1e10 losses and it thrashed through the top.
2. Dense height + tiny two-branch. Still whirls; a height reward gets farmed by fly-bys through the top.
3. Height-only + restart distribution. Whirl persists, and a too-strong velocity penalty can make it avoid the top.
4. Dropped the branch: `height + (-|E-E*|) - torque`. Cleaner, but the weights kept fighting.
5. Sign-flip velocity trick (reward speed low, penalize high). Reward-hackable: unbounded velocity reward means it spins forever down low.
6. Energy matching + target bonus. `+2` inside a low-velocity and high-position set. A fly-by has high velocity so it can't farm the bonus. Reaches the top and hovers.
7. Gated LQR well (`-c·up·‖v‖² - c·up·‖dev‖`), coefficient tuned `0.01` to `0.25`. This turned the hover into a hold.
8. Bounded every term. Capped the energy penalty (`E` includes kinetic energy so it blew up past 100/step), capped the velocity penalty, and sharpened the gate to `up²` so the strong penalties only act at the very top and don't tax the swing-up. Bounding the reward stopped the critic loss from climbing.

Final reward: `2·height - 0.1·min(|E-E*|,60) - 0.25·up²·min(‖v‖²,40) - 0.25·up²·‖dev‖ - 0.001·τ²`, plus `+2` inside the target set. Per-step max is `+6`, hit only at upright-and-still, so it's not gameable.

### Results

Videos are in the training-videos folder. The continuous model starts to hold itself up around iteration 40, and by iterations 140-150 it reliably swings up and stays near vertical.

Over 200 episodes it keeps ~0.30 of steps in the target set, up from 0.17 and 0.25 on earlier runs. The honest catch: this is a limit cycle, not a real hold. Longest continuous hold is 16 steps and no episode holds for 100+ steps in a row. It dwells near the top in short bursts rather than locking in, which is about the ceiling for one end-to-end policy on this equilibrium.

## LQR handoff (`hybrid_controller.py`)

The fix for the brief-dwell problem is a hybrid controller. The learned policy does the swing-up. Near the top it hands off to an analytic LQR balance controller. This is an execution-time switch only: no retraining and no reward change. (A discontinuity in the control law is standard and fine, unlike the discontinuous reward that hurt training earlier.)

- Linearize the dynamics about the upright numerically (finite differences on the env), then solve the continuous Riccati equation for the gain `K`.
- Far from the top, use the PPO policy. Near the top (`‖dev‖<0.3, ‖v‖<3`), use `τ = -K·[dev; vel]`. Hysteresis drops back to the policy past `‖dev‖>0.6`.

Over 200 episodes:

| metric | pure PPO | PPO + LQR |
|---|---|---|
| fraction in target | 0.29 | 0.33 |
| longest continuous hold | 16 steps | 150 median, 400 max |
| episodes holding 100+ steps | 0% | 58% |

The fraction barely moves and its variance jumps because performance is now bimodal: when the LQR catches, it holds dead vertical for most of the episode; when the swing-up arrives too fast, it misses that episode. The longest-hold row is the honest one and goes from "never holds" to "holds indefinitely most of the time." Demo rollouts are in `final_hybrid_videos/`.

Takeaway: RL handles the hard nonlinear swing-up, and a small optimal-control law handles the local balance the policy couldn't. Each tool does the part it's best at.
