#try to control the acrobot near the top of its swing
#needs some edits to the acrobot env itself

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import gymnasium as gym
import numpy as np
import os
import imageio
from gymnasium.envs.classic_control import AcrobotEnv

# new acrobot class
class StabilizeAcrobot(gym.Wrapper):
    def __init__(self, max_steps=500, render_mode=None):
        super().__init__(AcrobotEnv(render_mode=render_mode))
        self.max_steps = max_steps
        self._t = 0
        #pull params from normal class
        u = self.unwrapped
        self.m1, self.m2 = u.LINK_MASS_1, u.LINK_MASS_2
        self.l1 = u.LINK_LENGTH_1
        self.lc1, self.lc2 = u.LINK_COM_POS_1, u.LINK_COM_POS_2
        self.I1, self.I2 = u.LINK_MOI, u.LINK_MOI
        self.g = 9.8

    def reset(self, **kwargs):
        self._t = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self._t += 1
        state  = self.unwrapped.state # [theta1, theta2, dtheta1, dtheta2]
        torque = self.unwrapped.AVAIL_TORQUE[action] # applied torque
        reward = self._reward(state, torque)
        truncated = self._t >= self.max_steps
        return obs, reward, False, truncated, info

    def _reward(self, state, torque):
        theta1, theta2, dtheta1, dtheta2 = state
        
        def wrap(angle):
            return ((angle + np.pi) % (2 * np.pi)) - np.pi #keep it in range [-pi, pi]

        deviation = np.array([wrap(theta1 - np.pi), wrap(theta2)])
        velocity = np.array([dtheta1, dtheta2])
        lambda_u = 0.9
     
        if (np.linalg.norm(deviation) < 0.75 and np.linalg.norm(velocity) < 2.0): 
            # we are close to the top, want to encourage the model to stay here
            Q = np.diag([10, 10])
            return -deviation.T @ Q @ deviation - lambda_u * torque ** 2 - 0.6 * np.linalg.norm(velocity) ** 2
        else:
            #need to find kinetic, potential energy
            #push towards that energy level
            V = (-self.m1 * self.g * self.lc1 * np.cos(theta1) - self.m2 * self.g * (self.l1 * np. cos(theta1) + 
                                                                                    self.lc2 * np.cos(theta1 + theta2)))
            M11 = (self.m1 * self.lc1 ** 2 + self.m2 * (self.l1 ** 2 + self.lc2 ** 2 + 2 * self.l1 * self.lc2 * np.cos(theta2)) 
                    + self.I1 + self.I2)
            
            M12 = self.m2 * (self.lc2 ** 2 + self.l1 * self.lc2 * np.cos(theta2)) + self.I2
            M22 = self.m2 * self.lc2 ** 2 + self.I2
            K = 0.5 * (M11 * dtheta1 ** 2 + 2 * M12 * dtheta1 * dtheta2 + M22 * dtheta2 ** 2)
            E  = V + K
            E_target = self.m1 * self.g * self.lc1 + self.m2 * self.g * (self.l1 + self.lc2)
            return -(E - E_target)**2 - lambda_u * torque ** 2

class PPO(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_shape), #one output per action, will become a probability distribution
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.layers(state) #logits per action

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
output_shape = 3
controller = PPO(input_shape, output_shape)
critic = CRITIC(input_shape)

def loss(state, action, logits, old_logits, critic_target, advantage, epsilon, c1, c2):
    #actor loss terms
    baseline = critic(state)
    action_probs = logits.gather(1, action.unsqueeze(1)).squeeze(1) 
    prob = torch.exp(torch.log(action_probs) - old_logits) 
    actor_loss = (torch.minimum(prob * advantage, torch.clip(prob, 1 - epsilon, 1 + epsilon) * advantage)).mean() # PPO update
    
    #critic loss terms
    critic_loss = ((baseline.squeeze(-1) - critic_target) ** 2).mean() #MSE, get close to the true value
    
    #entropy
    entropy = (logits * torch.log(logits)).sum(dim=1).mean()
    
    return -actor_loss + c1 * critic_loss + c2 * entropy

#training loop now
def gen_traj_with_labels():
    traj = []
    state, info = env.reset()
    done = False
    with torch.no_grad():
        while not done:
            dist = torch.distributions.Categorical(probs=controller(torch.tensor(state, dtype=torch.float32)))
            action = dist.sample()
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            value = critic(torch.tensor(state, dtype=torch.float32))
            traj.append((state, action, reward, dist.log_prob(action), value))
            state = next_state
        
    #now we need to backtrack to find the critic targets
    gamma = 0.995
    final_data = []
    return_prev = 0
    for data in reversed(traj):
        state, action, reward, old_log_prob, value = data
        return_t = reward + gamma * return_prev
        advantage_t = return_t - value
        final_data.append((state, action, old_log_prob, advantage_t, return_t))
        return_prev = return_t
    
    return final_data

class data_loader(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        state, action, old_log_prob, advantage, ret = self.samples[i]
        return (torch.tensor(state, dtype=torch.float32),
                torch.tensor(int(action), dtype=torch.long),
                torch.tensor(float(old_log_prob), dtype=torch.float32),
                torch.tensor(float(advantage), dtype=torch.float32),
                torch.tensor(float(ret), dtype=torch.float32))

num_trajectories = 100
num_iters = 50
optimizer = torch.optim.Adam(list(controller.parameters()) + list(critic.parameters()), lr=1e-3)
num_epochs = 10
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters, eta_min=1e-5)
env = StabilizeAcrobot()
eval_env = StabilizeAcrobot()

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
            logits = controller(state)
            optimizer.zero_grad()
            l = loss(state, action, logits, old_log_prob, returns, advantage, 0.2, 0.5, 0.01)
            avg_loss += l
            l.backward()
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
                eval_action = int(controller(torch.tensor(eval_state, dtype=torch.float32)).argmax())
            eval_state, _, term, trunc, _ = eval_env.step(eval_action)
            all_steps += 1
            if (check_step(eval_env.unwrapped.state)): good_steps += 1
            eval_done = term or trunc
            
        total_steps += good_steps / all_steps
    avg_steps = total_steps / eval_episodes
        
    print(f"Iteration {iter}, Average fraction of steps in target regime = {avg_steps}")

    #save video of performance every 5 iterations
    if (iter % 5 == 0):
        video_env = StabilizeAcrobot(render_mode="rgb_array")
        vs, _ = video_env.reset()
        frames = [video_env.render()]
        done = False
        while not done:
            with torch.no_grad():
                a = int(controller(torch.tensor(vs, dtype=torch.float32)).argmax())
            vs, _, term, trunc, _ = video_env.step(a)
            frames.append(video_env.render())
            done = term or trunc
        video_env.close()
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stabilization_training_videos")
        os.makedirs(out_dir, exist_ok=True)
        imageio.mimsave(os.path.join(out_dir, f"stabilization_ppo_iter_{iter}.mp4"), frames, fps=30, macro_block_size=1)
