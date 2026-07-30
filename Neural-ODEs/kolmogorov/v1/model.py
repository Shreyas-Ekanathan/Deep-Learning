import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchdiffeq import odeint
import os

HERE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#explore a neural ODE based design for learning nonlinaer dynamics

class ODEFunc(nn.Module):
    def __init__(self, streams):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(streams, streams * 2, 3, padding = 1, padding_mode = 'circular'),
            nn.Tanh(),
            nn.Conv2d(streams * 2, streams * 2, 3, padding = 1, padding_mode = 'circular'),
            nn.Tanh(),
            nn.Conv2d(streams * 2, streams, 3, padding = 1, padding_mode = 'circular')
        )
        
        self.forcing = nn.Parameter(torch.zeros(streams, 8, 8))

    def forward(self, t, z):
        return self.net(z) + self.forcing
    
class NODE(nn.Module):
    def __init__(self, streams):
        super().__init__()
        #take inspiration from U-net for encoder/decoder logic
        #input is the condition at time t0, and we want to rollout the trajectory correctly
        self.encoder = nn.Sequential(
            nn.Conv2d(streams, 16, 3, stride = 2, padding = 1, padding_mode = 'circular'),  #we have periodic boundary cond
            nn.Tanh(),
            nn.Conv2d(16, 32, 3, stride = 2, padding = 1, padding_mode = 'circular'),
            nn.Tanh(),
            nn.Conv2d(32, 64, 3 , stride = 2, padding = 1, padding_mode = 'circular'),
        )
                
        self.ode_func = ODEFunc(64)
        
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'nearest'), #upsample -> conv vs conv2d bc we want circular padding
            nn.Conv2d(64, 32, 3, padding = 1, padding_mode = 'circular'),
            nn.Tanh(),
            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(32, 16, 3, padding = 1, padding_mode = 'circular'),
            nn.Tanh(),
            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(16, streams, 3, padding = 1, padding_mode = 'circular')
        ) #symmetric for output
        
    def forward(self, x0, t):
        z0 = self.encoder(x0)

        z_traj = odeint(self.ode_func, z0, t, rtol = 1e-3, atol = 1e-6, options = {'dtype': torch.float32}) #solve the diffeq, integrate in latent space
        
        x_hat = self.decoder(z_traj.reshape(-1, *z0.shape[1:])) #output
        return x_hat.reshape(len(t), z0.shape[0], *x_hat.shape[1:]).transpose(0, 1) #fix batch first data

class KolmogorovDataset(Dataset):
    def __init__(self, all_runs, window_len, stride, dt, sigma):
        self.window_len = window_len
        self.windows = []  
        num_runs, num_snapshots = all_runs.shape[:2]
        for r in range(num_runs):
            for start in range(0, num_snapshots - window_len + 1, stride):
                self.windows.append((r, start))
                
        self.data = all_runs / sigma
        self.t = torch.arange(window_len) * dt

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        r, start = self.windows[idx]
        traj = self.data[r, start:start+self.window_len]
        x0 = traj[0]
        return x0, traj, self.t
    
if __name__ == "__main__":
    runs = torch.load(os.path.join(HERE, "kolmogorov_train_dataset.pt"))
    runs = runs.unsqueeze(2) #add a channel dim
    sigma = runs.std()
    dataset = KolmogorovDataset(runs, 30, 10, 0.05, sigma)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    test_traj = torch.load(os.path.join(HERE, "kolmogorov_test_dataset.pt")) #2 eval trajectories
    test_traj = test_traj.unsqueeze(2) #add a channel dim
    test_dataset = KolmogorovDataset(test_traj, 30, 20, 0.05, sigma)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    num_epochs = 75
    model = NODE(1).to(device)
    print(f"training on {device}")
    t_grid = dataset.t.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    loss_func = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        avg_loss = 0
        for x0, traj, _ in train_loader:
            x0, traj = x0.to(device), traj.to(device)
            predicted_traj = model(x0, t_grid)
            optimizer.zero_grad()
            loss = loss_func(predicted_traj, traj)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            avg_test_loss = 0
            for x0, traj, _ in test_loader:
                x0, traj = x0.to(device), traj.to(device)
                predicted_traj = model(x0, t_grid)
                loss = loss_func(predicted_traj, traj)
                avg_test_loss += loss.item()
            
            
        print(f"Epoch {epoch + 1}, Average train loss = {avg_loss / len(train_loader)}, Average test loss = {avg_test_loss / len(test_loader)}")
    
        if (epoch % 15 == 0):
            torch.save(model.state_dict(), os.path.join(HERE, f"model_epoch_{epoch}.pth"))
    
    torch.save(model.state_dict(), os.path.join(HERE, f"model_epoch_{num_epochs}.pth"))
