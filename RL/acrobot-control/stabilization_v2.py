#try to control the acrobot near the top of its swing
#needs some edits to the acrobot env itself

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import imageio
from gymnasium import spaces
from gymnasium.envs.classic_control.acrobot import AcrobotEnv, wrap, bound, rk4

# new acrobot class
#lets make the model continuous now -> it can take continuous actions (space is no longer discrete)
class StabilizeAcrobot(AcrobotEnv):
    def __init__(self, max_steps=500, render_mode=None, max_torque = 1.0):
        super().__init__(render_mode=render_mode)
        self.max_steps = max_steps
        self._t = 0
        #pull params from normal class
        u = self.unwrapped
        self.m1, self.m2 = u.LINK_MASS_1, u.LINK_MASS_2
        self.l1 = u.LINK_LENGTH_1
        self.lc1, self.lc2 = u.LINK_COM_POS_1, u.LINK_COM_POS_2
        self.I1, self.I2 = u.LINK_MOI, u.LINK_MOI
        self.g = 9.8
        self.max_torque = max_torque
        self.action_space = spaces.Box( #continuous space
            low=-max_torque, high=max_torque, shape=(1,), dtype=np.float32
        )


    def reset(self, **kwargs):
        self._t = 0
        return super().reset(**kwargs)

    def step(self, a):
        s = self.state
        torque = float(np.clip(a, -self.max_torque, self.max_torque)) #make torque continuous
        s_augmented = np.append(s, torque) #same as normal
        ns = rk4(self._dsdt, s_augmented, [0, self.dt])
        ns[0] = wrap(ns[0], -np.pi, np.pi)
        ns[1] = wrap(ns[1], -np.pi, np.pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        if self.render_mode == "human":
            self.render()
        self._t += 1
        return self._get_ob(), self._reward(self.state, torque), False, self._t >= self.max_steps, {}
    
    def _reward(self, state, torque):
        theta1, theta2, dtheta1, dtheta2 = state
        
        def wrap(angle):
            return ((angle + np.pi) % (2 * np.pi)) - np.pi #keep it in range [-pi, pi]

        deviation = np.array([wrap(theta1 - np.pi), wrap(theta2)])
        velocity = np.array([dtheta1, dtheta2])
     
        V = (-self.m1 * self.g * self.lc1 * np.cos(theta1) - self.m2 * self.g * (self.l1 * np. cos(theta1) + 
                                                                                self.lc2 * np.cos(theta1 + theta2)))
        M11 = (self.m1 * self.lc1 ** 2 + self.m2 * (self.l1 ** 2 + self.lc2 ** 2 + 2 * self.l1 * self.lc2 * np.cos(theta2)) 
                + self.I1 + self.I2)
        
        M12 = self.m2 * (self.lc2 ** 2 + self.l1 * self.lc2 * np.cos(theta2)) + self.I2
        M22 = self.m2 * self.lc2 ** 2 + self.I2
        K = 0.5 * (M11 * dtheta1 ** 2 + 2 * M12 * dtheta1 * dtheta2 + M22 * dtheta2 ** 2)
        E  = V + K
        E_target = self.m1 * self.g * self.lc1 + self.m2 * self.g * (self.l1 + self.lc2)
        energy_reward = -0.1 * np.minimum(np.abs(E - E_target), 60) #cap the reward
        
        height = -np.cos(theta1) - np.cos(theta1 + theta2) 
        height_reward = height * 2
        
        up = (height + 2) / 4
        top_gate = up ** 2 
        velocity_penalty = 0.25 * top_gate * np.minimum(np.linalg.norm(velocity) ** 2, 40)
        deviation_penalty = 0.25 * top_gate * np.linalg.norm(deviation)

        reward = height_reward + energy_reward - velocity_penalty - deviation_penalty - 0.001 * torque ** 2

        if (np.linalg.norm(deviation) < 0.5 and np.linalg.norm(velocity) < 0.8):
            reward += 2 #bonus for landing in an ideal area 

        return reward

class PPO(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1), #output the mean
            nn.Tanh() #scale to (-1, 1), which is the legal range of torques
        )
        
        self.log_std = nn.Parameter(torch.zeros(1))
    
    def forward(self, state):
        mu = self.layers(state) #logits per action
        log_std = self.log_std.expand_as(mu) 
        return torch.cat([mu, log_std], dim = -1)

class CRITIC(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1), #one output, predicts the baseline
        )
    
    def forward(self, state):
        return self.layers(state) #logits per action


input_shape = 6
controller = PPO(input_shape)
critic = CRITIC(input_shape)

def loss(state, logits, old_logits, critic_target, advantage, entropy, epsilon, c1, c2):
    #actor loss terms
    baseline = critic(state)
    prob = torch.exp(logits - old_logits)    
    actor_loss = (torch.minimum(prob * advantage, torch.clip(prob, 1 - epsilon, 1 + epsilon) * advantage)).mean() # PPO update
    
    #critic loss terms
    critic_loss = ((baseline.squeeze(-1) - critic_target) ** 2).mean() #MSE, get close to the true value
        
    return -actor_loss + c1 * critic_loss - c2 * entropy

#training loop now
def gen_traj_with_labels():
    traj = []
    state, info = env.reset()
    if np.random.random() < 0.3:
        #teach policy to learn how to balance near the top part of the time
        theta1_deviation = np.random.uniform(-0.15, 0.15)
        theta2_deviation = np.random.uniform(-0.15, 0.15)
        v1 = np.random.uniform(-0.75, 0.75)
        v2 = np.random.uniform(-0.75, 0.75)
        env.state = np.array([np.pi + theta1_deviation, theta2_deviation, v1, v2])
        state = env._get_ob()
    done = False
    with torch.no_grad():
        while not done:
            mu, log_std = controller(torch.tensor(state, dtype=torch.float32))
            log_std = torch.clamp(log_std, -5, 2)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            value = critic(torch.tensor(state, dtype=torch.float32))
            traj.append((state, action, reward, dist.log_prob(action).sum(-1), value))
            state = next_state
        
    #now we need to backtrack to find the critic targets
    gamma = 0.995
    final_data = []
    V_prev = 0
    advantage_prev = 0
    for data in reversed(traj):
        #swap to a GAE scheme
        state, action, reward, old_log_prob, value = data
        delta_t = reward + gamma * V_prev - value
        advantage_t = delta_t + gamma * 0.95 * advantage_prev
        return_t = advantage_t + value
        advantage_prev = advantage_t
        V_prev = value
        final_data.append((state, action, old_log_prob, advantage_t, return_t))
    
    return final_data

class data_loader(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        state, action, old_log_prob, advantage, ret = self.samples[i]
        return (torch.tensor(state, dtype=torch.float32),
                torch.tensor(float(action), dtype=torch.float32),
                torch.tensor(float(old_log_prob), dtype=torch.float32),
                torch.tensor(float(advantage), dtype=torch.float32),
                torch.tensor(float(ret), dtype=torch.float32))

num_trajectories = 150
num_iters = 151
optimizer = torch.optim.Adam(list(controller.parameters()) + list(critic.parameters()), lr=2e-3)
num_epochs = 8
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters, eta_min=5e-5)
env = StabilizeAcrobot()
eval_env = StabilizeAcrobot()

best_fraction = -1.0  # track the best eval so far so we can checkpoint improvements
ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_stabilizer.pt")

for iter in range(num_iters):
    #generate trajectoreis now
    data = []
    for traj in range(num_trajectories):
        data.extend(gen_traj_with_labels())
        
    #normalize advantages
    advs = np.array([float(d[3]) for d in data], dtype=np.float32)
    mean, std = advs.mean(), advs.std()
    data = [(s, a, old_log_prob, (float(adv) - mean) / (std + 1e-8), ret) for (s, a, old_log_prob, adv, ret) in data]

    #wrap data in a loader
    loader = DataLoader(data_loader(data), batch_size=128, shuffle=True)
    for epoch in range(num_epochs):
        avg_loss = 0
        for sample in loader:            
            state, action, old_log_prob, advantage, returns = sample
            out = controller(state) 
            mu = out[:, 0]
            log_std = out[:, 1]
            log_std = torch.clamp(log_std, -5, 2)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mu, std)
            entropy = dist.entropy().mean() #for loss function
            new_logprob = dist.log_prob(action)
            optimizer.zero_grad()
            l = loss(state, new_logprob, old_log_prob, returns, advantage, entropy, 0.2, 0.4, 0.01)
            avg_loss += l
            l.backward()
            torch.nn.utils.clip_grad_norm_(list(controller.parameters()) + list(critic.parameters()), 0.5)
            optimizer.step()
        print(f"Iteration {iter}, Epoch {epoch}, Average loss: {avg_loss / (len(loader))}")
            
    scheduler.step()
    #epsilon greedy performance analysis
    #this eval is every iteration
    
    eval_episodes = 10
    total_steps = 0
    for _ in range(eval_episodes):
        eval_state, _ = eval_env.reset()
        good_steps = 0
        all_steps = 0
        eval_done = False
        
        def check_step(state):
            theta1, theta2, dtheta1, dtheta2 = state
            
            def wrap(angle):
                return ((angle + np.pi) % (2 * np.pi)) - np.pi #keep it in range [-pi, pi]

            deviation = np.array([wrap(theta1 - np.pi), wrap(theta2)])
            velocity = np.array([dtheta1, dtheta2])
     
            if (np.linalg.norm(deviation) < 0.3 and np.linalg.norm(velocity) < 1): 
                return True
        
            return False

        while not eval_done:
            with torch.no_grad():
                mu, _ = controller(torch.tensor(eval_state, dtype=torch.float32))
            eval_state, _, term, trunc, _ = eval_env.step(mu)
            all_steps += 1
            if (check_step(eval_env.unwrapped.state)): good_steps += 1
            eval_done = term or trunc
            
        total_steps += good_steps / all_steps
    avg_steps = total_steps / eval_episodes
        
    print(f"Iteration {iter}, Average fraction of steps in target regime = {avg_steps}")

    #checkpoint whenever we hit a new best eval fraction
    if avg_steps > best_fraction:
        best_fraction = avg_steps
        torch.save({
            "controller": controller.state_dict(),
            "critic": critic.state_dict(),
            "iter": iter,
            "fraction": avg_steps,
        }, ckpt_path)
        print(f"  -> new best ({avg_steps:.4f}), saved to {ckpt_path}")

    #save video of performance every 5 iterations
    if (iter % 5 == 0):
        video_env = StabilizeAcrobot(render_mode="rgb_array")
        vs, _ = video_env.reset()
        frames = [video_env.render()]
        done = False
        while not done:
            with torch.no_grad():
                mu, _ = controller(torch.tensor(vs, dtype=torch.float32))
            vs, _, term, trunc, _ = video_env.step(mu)
            frames.append(video_env.render())
            done = term or trunc
        video_env.close()
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "continuous_stabilization_training_videos")
        os.makedirs(out_dir, exist_ok=True)
        imageio.mimsave(os.path.join(out_dir, f"continuous_stabilization_ppo_iter_{iter}.mp4"), frames, fps=30, macro_block_size=1)
