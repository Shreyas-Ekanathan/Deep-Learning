#hybrid controller that uses LQR linearization near hte top
#idea based on https://arxiv.org/pdf/2012.11663

import os
import numpy as np
import torch
import torch.nn as nn
import imageio
import scipy.linalg as sla
from gymnasium import spaces
from gymnasium.envs.classic_control.acrobot import AcrobotEnv, wrap, bound, rk4

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "best_stabilizer.pt")

class StabilizeAcrobot(AcrobotEnv):
    def __init__(self, max_steps=500, render_mode=None, max_torque=2.0):
        super().__init__(render_mode=render_mode)
        self.max_steps = max_steps
        self._t = 0
        self.max_torque = max_torque
        self.action_space = spaces.Box(low=-max_torque, high=max_torque, shape=(1,), dtype=np.float32)

    def reset(self, **kwargs):
        self._t = 0
        return super().reset(**kwargs)

    def step(self, a):
        s = self.state
        torque = float(np.clip(a, -self.max_torque, self.max_torque))
        ns = rk4(self._dsdt, np.append(s, torque), [0, self.dt])
        ns[0] = wrap(ns[0], -np.pi, np.pi)
        ns[1] = wrap(ns[1], -np.pi, np.pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        if self.render_mode == "human":
            self.render()
        self._t += 1
        return self._get_ob(), 0.0, False, self._t >= self.max_steps, {}

class PPO(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, state):
        mu = self.layers(state)
        return torch.cat([mu, self.log_std.expand_as(mu)], dim=-1)

def wrapf(a):
    return ((a + np.pi) % (2 * np.pi)) - np.pi

def in_target(state):
    theta1, theta2, dtheta1, dtheta2 = state
    dev = np.array([wrapf(theta1 - np.pi), wrapf(theta2)])
    vel = np.array([dtheta1, dtheta2])
    return np.linalg.norm(dev) < 0.3 and np.linalg.norm(vel) < 1

def lqr_gain():
    #linearize about the upright with finite differences, then solve the riccati equation
    e = AcrobotEnv()
    e.reset()
    x0 = np.array([np.pi, 0.0, 0.0, 0.0])
    def f(x, tau):
        return np.array(e._dsdt(np.array([x[0], x[1], x[2], x[3], tau]))[:4], dtype=float)
    eps = 1e-6
    A = np.zeros((4, 4))
    for j in range(4):
        dx = np.zeros(4); dx[j] = eps
        A[:, j] = (f(x0 + dx, 0.0) - f(x0 - dx, 0.0)) / (2 * eps)
    B = ((f(x0, eps) - f(x0, -eps)) / (2 * eps)).reshape(4, 1)
    Q = np.diag([10.0, 10.0, 1.0, 1.0])
    R = np.array([[1.0]])
    P = sla.solve_continuous_are(A, B, Q, R)
    return np.linalg.inv(R) @ B.T @ P

def run_episode(env, controller, K, record=False):
    obs, _ = env.reset()
    frames = [env.render()] if record else None
    good = total = cur = longest = 0
    balancing = False
    done = False
    while not done:
        theta1, theta2, dtheta1, dtheta2 = env.unwrapped.state
        dev = np.array([wrapf(theta1 - np.pi), wrapf(theta2)])
        vel = np.array([dtheta1, dtheta2])
        dn, vn = np.linalg.norm(dev), np.linalg.norm(vel)

        #hand off to LQR once close and slow enough, fallback to the policy if we fall away
        if not balancing and dn < 0.3 and vn < 3.0:
            balancing = True
        elif balancing and dn > 0.6:
            balancing = False

        if balancing:
            tau = float((-K @ np.array([dev[0], dev[1], vel[0], vel[1]]))[0])
        else:
            with torch.no_grad():
                mu, _ = controller(torch.tensor(obs, dtype=torch.float32))
            tau = float(mu)

        obs, _, term, trunc, _ = env.step(tau)
        if record:
            frames.append(env.render())
        total += 1
        if in_target(env.unwrapped.state):
            good += 1; cur += 1; longest = max(longest, cur)
        else:
            cur = 0
        done = term or trunc
    return good / total, longest, frames

ckpt = torch.load(CKPT, map_location="cpu")
controller = PPO(6)
controller.load_state_dict(ckpt["controller"])
controller.eval()

K = lqr_gain()

np.random.seed(0)
torch.manual_seed(0)
env = StabilizeAcrobot()
env.reset(seed=0)

N = 200
fracs, holds = [], []
for _ in range(N):
    frac, longest, _ = run_episode(env, controller, K)
    fracs.append(frac)
    holds.append(longest)
fracs, holds = np.array(fracs), np.array(holds)
print(f"over {N} episodes:")
print(f"  fraction in target: {fracs.mean():.3f} +/- {fracs.std():.3f}")
print(f"  longest hold: median {np.median(holds):.0f}, max {holds.max()}")
print(f"  held >=100 steps in {(holds >= 100).mean() * 100:.0f}% of episodes")

vid_dir = os.path.join(HERE, "final_hybrid_videos")
os.makedirs(vid_dir, exist_ok=True)
venv = StabilizeAcrobot(render_mode="rgb_array")
for i in range(5):
    _, _, frames = run_episode(venv, controller, K, record=True)
    imageio.mimsave(os.path.join(vid_dir, f"hybrid_{i}.mp4"), frames, fps=30, macro_block_size=1)
venv.close()
print(f"saved 5 videos to {vid_dir}")
