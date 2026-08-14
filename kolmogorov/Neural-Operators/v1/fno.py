import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchdiffeq import odeint
from collections import defaultdict
import os
import torch.fft as fourier
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

class Fourier_Layer(nn.Module):
    def __init__(self, in_streams, out_streams, modes1, modes2, grid):
        super().__init__()
        self.in_streams = in_streams
        self.out_streams = out_streams
        self.modes1 = modes1 #x axis
        self.modes2 = modes2 #y axis

        scale = 1 / (in_streams * out_streams)
        self.weights_pos = nn.Parameter(scale * torch.rand(in_streams, out_streams, modes1, modes2, 2)) #positive modes
        self.weights_neg = nn.Parameter(scale * torch.rand(in_streams, out_streams, modes1, modes2, 2)) #negative modes
        #two separate weights for sake of cleanliness in slicing

        self.lin = nn.Conv2d(in_streams, out_streams, 1)
        
        self.nu_embedding = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),
            nn.Linear(64, 256),
            nn.Tanh(),
            nn.Linear(256, 2 * out_streams) #output a scaling and a shift
        )

        self.forcing = nn.Parameter(torch.zeros(out_streams, grid, grid)) #grid sized forcing


    def spectral_mult(self, block, weights):
        #blcok is (B, 20, 16, 16)
        #weights is (20, 20, 16, 16)
        #we want the result to be B, 20, 16, 16, so we kill the first axis of weights and do a multiply
        return torch.einsum('bixy,ioxy->boxy', block, torch.view_as_complex(weights))

    def forward(self, x, nu):
        B, C, H, W = x.shape
        x_hat = fourier.rfft2(x)

        #only write to relevant modes, all others stay 0
        out = torch.zeros(B, self.out_streams, H, W // 2 + 1, dtype = x_hat.dtype, device = x.device)
        out[:, :, :self.modes1, :self.modes2] = self.spectral_mult(x_hat[:, :, :self.modes1, :self.modes2], self.weights_pos)
        out[:, :, -self.modes1:, :self.modes2] = self.spectral_mult(x_hat[:, :, -self.modes1:, :self.modes2], self.weights_neg)

        out = fourier.irfft2(out, s = (H, W)) #get back to original shape
        
        #viscosity stuff
        nu = nu.unsqueeze(-1)
        nu_embedding = self.nu_embedding(nu)
        nu_embedding = nu_embedding.view(*nu_embedding.shape, 1, 1) # (B, streams * 2, 1, 1)
        nu_scaling = nu_embedding[:, :self.out_streams, :, :]
        nu_shift = nu_embedding[:, self.out_streams:, :, :]

        return F.gelu((out + self.lin(x)) * (1 + torch.tanh(nu_scaling)) + nu_shift + self.forcing)
    
class FNO(nn.Module):
    def __init__(self, streams, grid):
        super().__init__()
        #same encoder/decoder logic for FNO, but no neuralode, instead linear transforms
        self.encoder = nn.Conv2d(streams, 16, 1, stride = 1)

        self.fourier = nn.ModuleList([
            Fourier_Layer(16, 16, 12, 12, grid),
            Fourier_Layer(16, 16, 12, 12, grid),
            Fourier_Layer(16, 16, 12, 12, grid),
            Fourier_Layer(16, 16, 12, 12, grid)
        ])
                        
        self.decoder = nn.Sequential(
            nn.Conv2d(16, 64, 1, stride = 1),
            nn.GELU(),
            nn.Conv2d(64, streams, 1)
        )

    def step(self, x, nu):
        #one dt = 0.1 step of the learned operator, predicting the increment
        out = self.encoder(x)
        for layer in self.fourier:
            out = layer(out, nu)
        out = self.decoder(out)
        return x + out

    def forward(self, x0, t, nu):
        rollout = [x0]
        for i in range(len(t) - 1): #learn to take steps of 0.1
            if self.training:
                x0 = checkpoint(self.step, x0, nu, use_reentrant = False)
            else:
                x0 = self.step(x0, nu)
            rollout.append(x0)
        return torch.stack(rollout, dim=1)


#given from the distribution we built for this
NU_LOG_CENTER = 2.8134
NU_LOG_SCALE = 0.6931

def standardise_nu(nu):
    return (torch.log(nu) + NU_LOG_CENTER) / NU_LOG_SCALE

def unstandardise_nu(value):
    #for labels and reports
    return float(np.exp(value * NU_LOG_SCALE - NU_LOG_CENTER))

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
    dataset = KolmogorovDataset(runs, 30, 5, 0.10, True)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    test_traj = torch.load(os.path.join(HERE, "kolmogorov_test_dataset.pt")) #eval trajectories
    test_dataset = KolmogorovDataset(test_traj, 30, 20, 0.10, False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    num_epochs = 50
    model = FNO(1, 64).to(device)
    print(f"training on {device}")
    t_grid = dataset.t.to(device)
    test_t_grid = test_dataset.t.to(device)
    
    #one embedder and one forcing field per fourier layer now
    no_decay = [p for layer in model.fourier for p in layer.nu_embedding.parameters()]
    no_decay += [layer.forcing for layer in model.fourier]
    no_decay_ids = {id(p) for p in no_decay}
    other_params = [p for p in model.parameters() if id(p) not in no_decay_ids]
    optimizer = torch.optim.Adam([
        {"params": other_params, "weight_decay": 1e-4},
        {"params": no_decay, "weight_decay": 0.0}, #dont decay the viscosity embedder or the learned forcing
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
        with torch.no_grad(): #eval trajectories are longer
            loss_sums = defaultdict(float)
            loss_counts = defaultdict(int)
            for nu, x0, traj, _, _ in test_loader:
                x0, traj, nu = x0.to(device), traj.to(device), nu.to(device)
                predicted_traj = model(x0, test_t_grid, nu)
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
