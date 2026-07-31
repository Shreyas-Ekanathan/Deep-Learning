import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchdiffeq import odeint
from collections import defaultdict
import os

HERE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#explore a neural ODE based design for learning nonlinaer dynamics

#v2 update: we are learning how to integrate nu into the model, closer to some sort of operator learning

#given from the distribution we built for this
NU_LOG_CENTER = 2.8134
NU_LOG_SCALE = 0.6931

def standardise_nu(nu):
    return (torch.log(nu) + NU_LOG_CENTER) / NU_LOG_SCALE

def unstandardise_nu(value):
    #for labels & reports
    return float(np.exp(value * NU_LOG_SCALE - NU_LOG_CENTER))

class ODEFunc(nn.Module):
    def __init__(self, streams):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(streams, streams * 2, 3, padding = 1, padding_mode = 'circular'),
            nn.Tanh(),
            nn.Conv2d(streams * 2, streams * 2, 3, padding = 1, padding_mode = 'circular'),
            nn.Tanh(),
            nn.Conv2d(streams * 2, streams, 3, padding = 1, padding_mode = 'circular'),
        )
        self.streams = streams
        
        self.forcing = nn.Parameter(torch.zeros(streams, 16, 16))
        
        self.nu_embedding = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 256),
            nn.Tanh(),
            nn.Linear(256, 2 * streams) #output a scaling and a shift
        )

    def forward(self, t, z, nu):
        nu = nu.unsqueeze(-1)
        nu_embedding = self.nu_embedding(nu)
        nu_embedding = nu_embedding.view(*nu_embedding.shape, 1, 1) # (B, streams * 2, 1, 1)
        # we want the first streams values to be scalings, and the next streams values to be shifts
        streams = self.streams
        nu_scaling = nu_embedding[:, :streams, :, :]
        nu_shift = nu_embedding[:, streams:, :, :]
        return (1 + torch.tanh(nu_scaling)) * self.net(z) + self.forcing + nu_shift
    
class NODE(nn.Module):
    def __init__(self, streams):
        super().__init__()
        #take inspiration from U-net for encoder/decoder logic
        #input is the condition at time t0, and we want to rollout the trajectory correctly
        self.encoder = nn.Sequential(
            nn.Conv2d(streams, 32, 3, stride = 2, padding = 1, padding_mode = 'circular'),  #we have periodic boundary cond
            nn.Tanh(),
            nn.Conv2d(32, 64, 3, stride = 2, padding = 1, padding_mode = 'circular'),
        )
                
        self.ode_func = ODEFunc(64)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding = 1, padding_mode = 'circular'), 
            nn.Tanh(),
            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(32, 16, 3, padding = 1, padding_mode = 'circular'),
            nn.Tanh(),
            nn.Upsample(scale_factor = 2, mode = 'nearest'),
            nn.Conv2d(16, streams, 3, padding = 1, padding_mode = 'circular'),
        ) #symmetric for output

    def forward(self, x0, t, nu):
        z0 = self.encoder(x0)

        #bound the run since dopri was becoming ridiculously expensive per solve
        z_traj = odeint(lambda s, w: self.ode_func(s, w, nu), z0, t, method = 'midpoint', options = {'step_size': 0.1}) #solve the diffeq, integrate in latent space

        x_hat = self.decoder(z_traj.reshape(-1, *z0.shape[1:])) #output
        return x_hat.reshape(len(t), z0.shape[0], *x_hat.shape[1:]).transpose(0, 1) #fix batch first data

class KolmogorovDataset(Dataset):
    def __init__(self, all_runs, window_len, stride, dt, data_aug):
        runs = all_runs["omega"].unsqueeze(2)
        nus = all_runs["nu"]
        self.window_len = window_len
        self.windows = []  
        num_runs, num_snapshots = runs.shape[:2]
        for r in range(num_runs):
            for start in range(0, num_snapshots - window_len + 1, stride):
                self.windows.append((nus[r], r, start))
                
        self.data = runs
        self.t = torch.arange(window_len) * dt
        self.data_aug = data_aug

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        nu, r, start = self.windows[idx]
        traj = self.data[r, start : start + self.window_len]
        if self.data_aug:
            dx = torch.randint(0, 64, ()).item()
            dy = 16 * torch.randint(0, 4, ()).item()
            traj = torch.roll(traj, shifts = (dx, dy), dims = (-2, -1))
        scale = traj[0].std()
        traj = traj / scale
        x0 = traj[0]
        nu_out = standardise_nu(nu) #preprocess to [-1, 1] for training
        return nu_out, x0, traj, self.t, scale #scale is only for evals, no other real purpose
    
if __name__ == "__main__":
    runs = torch.load(os.path.join(HERE, "kolmogorov_train_dataset.pt"))
    dataset = KolmogorovDataset(runs, 30, 10, 0.10, True)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    test_traj = torch.load(os.path.join(HERE, "kolmogorov_test_dataset.pt")) #eval trajectories
    test_dataset = KolmogorovDataset(test_traj, 30, 20, 0.10, False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    num_epochs = 75
    model = NODE(1).to(device)
    print(f"training on {device}")
    t_grid = dataset.t.to(device)

    nu_params = list(model.ode_func.nu_embedding.parameters())
    nu_param_ids = {id(p) for p in nu_params}
    other_params = [p for p in model.parameters() if id(p) not in nu_param_ids]
    optimizer = torch.optim.Adam([
        {"params": other_params, "weight_decay": 1e-4},
        {"params": nu_params, "weight_decay": 0.0}, #dont decay the viscosity embedder
    ], lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    loss_func = nn.MSELoss()
    eval_loss_func = nn.MSELoss(reduction = 'none')

    best_test_loss = float("inf")
    best_epoch = None

    for epoch in range(num_epochs):
        model.train()
        avg_loss = 0
        for nu, x0, traj, _, _ in train_loader:
            x0, traj, nu = x0.to(device), traj.to(device), nu.to(device)
            predicted_traj = model(x0, t_grid, nu)
            optimizer.zero_grad()
            loss = loss_func(predicted_traj, traj)
            avg_loss += loss.item()
            loss.backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            loss_sums = defaultdict(float)
            loss_counts = defaultdict(int)
            for nu, x0, traj, _, _ in test_loader:
                x0, traj, nu = x0.to(device), traj.to(device), nu.to(device)
                predicted_traj = model(x0, t_grid, nu)
                per_example = eval_loss_func(predicted_traj, traj).flatten(1).mean(1)
                for nu_val, l in zip(nu.tolist(), per_example.tolist()):
                    #unstandardize nu for data reporting
                    key = round(unstandardise_nu(nu_val), 4)
                    loss_sums[key] += l
                    loss_counts[key] += 1

            test_by_nu = {k: loss_sums[k] / loss_counts[k] for k in sorted(loss_sums)}
            avg_test_loss = sum(loss_sums.values()) / sum(loss_counts.values())

        by_nu_str = ", ".join(f"nu={k:g}: {v:.5f}" for k, v in test_by_nu.items())

        is_best = avg_test_loss < best_test_loss
        if is_best:
            best_test_loss = avg_test_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(HERE, "model_best.pth"))
            with open(os.path.join(HERE, "model_best.txt"), "w") as fh:
                fh.write(f"epoch {best_epoch}\ntest loss {best_test_loss:.6f}\n"
                         f"by nu {by_nu_str}\n")

        print(f"Epoch {epoch + 1}, Average train loss = {avg_loss / len(train_loader)}, "
              f"Average test loss = {avg_test_loss}{'  <- best so far' if is_best else ''}, "
              f"by nu -> {by_nu_str}")

        if (epoch % 10 == 0):
            torch.save(model.state_dict(), os.path.join(HERE, f"model_epoch_{epoch}.pth"))

    torch.save(model.state_dict(), os.path.join(HERE, f"model_epoch_{num_epochs}.pth"))
    print(f"\nbest test loss {best_test_loss:.6f} at epoch {best_epoch} "
          f"(saved as model_best.pth); final epoch {num_epochs} was {avg_test_loss:.6f}")
