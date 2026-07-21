import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import gymnasium as gym
import numpy as np
import random
import os
import copy
import imageio
from collections import deque

class DQN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        #the role of the DQN is to take in the state and predict the Q value
        #Q value encompasses current reward and discounted future reward
        #continuous requires DQN
        self.layers = nn.Sequential(
            nn.Linear(input_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_shape) #one output per action
        )
    
    def forward(self, state):
        return self.layers(state)

input_shape = 6
output_shape = 3
controller = DQN(input_shape, output_shape)
target_net = copy.deepcopy(controller)

def loss(state, action, reward, next_state, done, gamma):
    state = torch.tensor(state, dtype=torch.float32)
    next_state = torch.tensor(next_state, dtype=torch.float32)
    q = controller(state)[action] #value of the action we just took                                  
    with torch.no_grad():
        target = reward + gamma * (1 - done) * target_net(next_state).max() #what we should have done ideally
    return (q - target) ** 2

#training loop now
optimizer = torch.optim.Adam(controller.parameters(), lr=1e-3)
num_epochs = 100
samples_per_epoch = 1000
batch_size = 128
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
update_delay = 100 #update every 100 samples, 10 updates per epoch

buffer = deque(maxlen = 100000)
#need to warm up the buffer
env = gym.make("Acrobot-v1")
eval_env = gym.make("Acrobot-v1")
warmup_samples_size = 5000
state, info = env.reset(seed=0)
for i in range(warmup_samples_size):
    action = env.action_space.sample() #random action for now
    next_state, reward, terminated, truncated, info = env.step(action)
    buffer.append((state, action, reward, next_state, terminated))
    state = next_state
    if terminated or truncated: #new episode
        state, info = env.reset()

iter = 0
gamma = 0.995
for epoch in range(num_epochs):
    avg_loss = 0
    epsilon = 1 - epoch / num_epochs
    for sample in range(samples_per_epoch):
        if (iter % update_delay == 0):
            target_net = copy.deepcopy(controller)
            
        if random.random() < epsilon:
            action = env.action_space.sample() # explore
        else:
            with torch.no_grad():
                action = int(controller(torch.tensor(state, dtype=torch.float32)).argmax()) #take best possible option
        
        next_state, reward, terminated, truncated, info = env.step(action)
        buffer.append((state, action, reward, next_state, terminated))
        state = next_state
        if terminated or truncated: #new episode
            state, info = env.reset()

        l = 0
        samples = random.sample(buffer, batch_size)
        optimizer.zero_grad()
        for sample in samples:
            b_state, b_action, b_reward, b_next_state, b_terminated = sample
            l += loss(b_state, b_action, b_reward, b_next_state, b_terminated, gamma)

        avg_loss += l
        l.backward()
        optimizer.step()
        iter += 1
        
    scheduler.step()
    #epsilon greedy performance analysis
    #this eval is every epoch
    
    eval_episodes = 10
    total_steps = 0
    for _ in range(eval_episodes):
        eval_state, _ = eval_env.reset()
        steps = 0
        eval_done = False
        while not eval_done:
            with torch.no_grad():
                eval_action = int(controller(torch.tensor(eval_state, dtype=torch.float32)).argmax())
            eval_state, _, term, trunc, _ = eval_env.step(eval_action)
            steps += 1
            eval_done = term or trunc
        total_steps += steps
    avg_steps = total_steps / eval_episodes
        
    print(f"Epoch {epoch}, Average Loss = {avg_loss / (batch_size * samples_per_epoch)}, Greedy average steps over {eval_episodes} eps = {avg_steps}")

    #save video of performance every 5 epochs
    if (epoch % 5 == 0):
        video_env = gym.make("Acrobot-v1", render_mode="rgb_array")
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
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_videos")
        os.makedirs(out_dir, exist_ok=True)
        imageio.mimsave(os.path.join(out_dir, f"dqn_epoch_{epoch}.mp4"), frames, fps=30, macro_block_size=1)

# Epoch 90, Average Loss = 0.5549750328063965, Greedy average steps over 10 eps = 72.1
# Epoch 91, Average Loss = 0.555826723575592, Greedy average steps over 10 eps = 76.0
# Epoch 92, Average Loss = 0.559714138507843, Greedy average steps over 10 eps = 69.1
# Epoch 93, Average Loss = 0.5620923638343811, Greedy average steps over 10 eps = 78.7
# Epoch 94, Average Loss = 0.5611153841018677, Greedy average steps over 10 eps = 69.5
# Epoch 95, Average Loss = 0.5626959800720215, Greedy average steps over 10 eps = 71.3
# Epoch 96, Average Loss = 0.5708425641059875, Greedy average steps over 10 eps = 79.4
# Epoch 97, Average Loss = 0.5718831419944763, Greedy average steps over 10 eps = 74.5
# Epoch 98, Average Loss = 0.5735864043235779, Greedy average steps over 10 eps = 75.5
# Epoch 99, Average Loss = 0.5746353268623352, Greedy average steps over 10 eps = 68.8