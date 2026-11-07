import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from collections import defaultdict
import os
import torch.fft as fourier
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#now trying to update our forward pass to include PDe refiner's diffusion-like step
# learn to retain the low freq modes

class Fourier_Layer(nn.Module):
    def __init__(self, in_streams, out_streams, modes1, modes2, grid, k_dim = 8):
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
        
        self.embedding = nn.Sequential( #now does k and nu
            nn.Linear(1 + k_dim, 64),
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

    def forward(self, x, nu, k):
        B, C, H, W = x.shape
        x_hat = fourier.rfft2(x)

        #only write to relevant modes, all others stay 0
        out = torch.zeros(B, self.out_streams, H, W // 2 + 1, dtype = x_hat.dtype, device = x.device)
        out[:, :, :self.modes1, :self.modes2] = self.spectral_mult(x_hat[:, :, :self.modes1, :self.modes2], self.weights_pos)
        out[:, :, -self.modes1:, :self.modes2] = self.spectral_mult(x_hat[:, :, -self.modes1:, :self.modes2], self.weights_neg)

        out = fourier.irfft2(out, s = (H, W)) #get back to original shape
        
        #viscosity stuff
        nu = torch.cat([nu.unsqueeze(-1), k], dim = -1) #add the timestep to the viscosity embedding
        nu_embedding = self.embedding(nu)
        nu_embedding = nu_embedding.view(*nu_embedding.shape, 1, 1) # (B, streams * 2, 1, 1)
        nu_scaling = nu_embedding[:, :self.out_streams, :, :]
        nu_shift = nu_embedding[:, self.out_streams:, :, :]
        
        return F.gelu((out + self.lin(x)) * (1 + torch.tanh(nu_scaling)) + nu_shift + self.forcing)
    
class FNO(nn.Module):
    def __init__(self, streams, grid, sigma_min = 0.001):
        super().__init__()
        #same encoder/decoder logic for FNO, but no neuralode, instead linear transforms
        self.encoder = nn.Conv2d(2 * streams, 16, 1, stride = 1)

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
        
        self.k_embedding = nn.Embedding(5, 8) 
        self.register_buffer("sigmas", sigma_min ** (torch.arange(5) / 4))

        
    def forward(self, u_t, est, nu, k):
        #one dt = 0.1 step of the learned operator, predicting the increment
        is_zero = (k == 0).view(-1, 1, 1, 1)
        k = self.k_embedding(k)
        out = self.encoder(torch.cat([u_t, est], dim = 1))
        for layer in self.fourier:
            out = layer(out, nu, k)
        out = self.decoder(out)
        return torch.where(is_zero, u_t + out, out)
    
    def rollout(self, x0, nu, t_grid, K = 4):
        B, dev = x0.shape[0], x0.device
        traj = [x0]
        for _ in range(1, len(t_grid)):
            u_t = traj[-1]
            k = torch.zeros(B, dtype = torch.long, device = dev)
            est = self.forward(u_t, torch.zeros_like(u_t), nu, k)

            # noise then unnoise now
            for j in range(1, K + 1):
                k = torch.full((B,), j, dtype = torch.long, device = dev)
                noised = est + self.sigmas[j] * torch.randn_like(est)
                eps_hat = self.forward(u_t, noised, nu, k)
                est = noised - self.sigmas[j] * eps_hat

            traj.append(est)
        return torch.stack(traj, dim = 1)


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
    
#build once
# this is a ring in wavenumber space: all the modes whose |k| rounds to the same integer, and bin them
shells = {}

def shell_index(n, device): #for reuse
    key = (n, str(device))
    if key not in shells:
        freq = torch.fft.fftfreq(n, d = 1.0 / n)
        kx, ky = torch.meshgrid(freq, freq, indexing = "ij")
        shell = torch.sqrt(kx ** 2 + ky ** 2).round().long().clamp(max = n // 2)
        shells[key] = shell.flatten().to(device)
    return shells[key]

def radial_spectrum(field):
    #shell averaged enstrophy spectrum, tries to prevent blurring
    n = field.shape[-1]
    hat = torch.fft.fft2(field)
    power = (hat.real ** 2 + hat.imag ** 2) / n ** 4
    out = torch.zeros(*field.shape[:-2], n // 2 + 1, device = field.device, dtype = power.dtype)
    out.index_add_(-1, shell_index(n, field.device), power.flatten(start_dim = -2))
    return out

def traj_loss(prediction, actual):
    n = prediction.shape[-1]
    #drop both ends of the spectrum.
    k_max = int((2 / 3) * (n // 2))
    pred_spectrum = radial_spectrum(prediction)[..., 1:k_max + 1]
    act_spectrum = radial_spectrum(actual)[..., 1:k_max + 1]

    spectral = (torch.log(pred_spectrum + 1e-12) - torch.log(act_spectrum + 1e-12)).abs()

    #ramp across the horizon, so that it is penalized late in the trajectory
    w = (torch.arange(prediction.shape[1], device = prediction.device) / 15).clamp(max = 1.0)
    spectral = spectral * w.view(1, -1, 1, 1)

    return (F.mse_loss(prediction, actual, reduction = 'none').flatten(1).mean(1),
            spectral.flatten(1).mean(1) / w.mean())


if __name__ == "__main__":
    runs = torch.load(os.path.join(HERE, "kolmogorov_train_dataset.pt"))
    dataset = KolmogorovDataset(runs, 30, 5, 0.10, True)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    test_traj = torch.load(os.path.join(HERE, "kolmogorov_test_dataset.pt")) #eval trajectories
    test_dataset = KolmogorovDataset(test_traj, 30, 20, 0.10, False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    num_epochs = 100
    model = FNO(1, 64).to(device)
    print(f"training on {device}")
    t_grid = dataset.t.to(device)
    test_t_grid = test_dataset.t.to(device)
    
    #one embedder and one forcing field per fourier layer now
    no_decay = [p for layer in model.fourier for p in layer.embedding.parameters()]
    no_decay += [layer.forcing for layer in model.fourier]
    no_decay_ids = {id(p) for p in no_decay}
    other_params = [p for p in model.parameters() if id(p) not in no_decay_ids]
    optimizer = torch.optim.Adam([
        {"params": other_params, "weight_decay": 1e-4},
        {"params": no_decay, "weight_decay": 0.0}, #dont decay the viscosity embedder or the learned forcing
    ], lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_test_loss = float("inf")
    best_epoch = None
    n_t = 8
    
    k_weights = torch.tensor([0.65] + [0.35 / 4] * 4, device = device)

    for epoch in range(num_epochs):
        model.train()
        avg_loss = 0
        Lambda = 0.02
        for nu, x0, traj, _, _ in train_loader:
            B, T = traj.shape[:2]
            std = torch.rand(B * n_t, 1, 1, 1, device = device) * 0.2
            #pick random timseteps
            t = torch.randint(0, T - 1, (B, n_t))
            #try to learn to go from t -> t + 1
            idx = torch.arange(B).unsqueeze(1).expand(B, n_t)
            x0 = traj[idx, t].flatten(0, 1).to(device) 
            x0 += torch.randn_like(x0) * std #small noise
            x1 = traj[idx, t + 1].flatten(0, 1).to(device)
            nu_b = nu.repeat_interleave(n_t).to(device)
            #skew toward k = 0
            k = torch.multinomial(k_weights, x0.shape[0], replacement = True)
            #now apply noise if we want? 
            eps = torch.randn_like(x1)
            sig = model.sigmas[k].view(-1, 1, 1, 1)
            is_zero = (k == 0).view(-1, 1, 1, 1)
            conditioning = torch.where(is_zero, torch.zeros_like(x1), x1 + sig * eps)
            target = torch.where(is_zero, x1, eps)

            prediction = model(x0, conditioning, nu_b, k)
            optimizer.zero_grad()
            l = F.mse_loss(prediction, target)
            l.backward()
            optimizer.step()
            avg_loss += l.item()

        scheduler.step()

        if (epoch + 1) % 5 != 0:
            continue
        
        model.eval()
        with torch.no_grad():
            loss_sums = defaultdict(float)
            loss_counts = defaultdict(int)
            test_mse = test_spectral = 0
            for nu, x0, traj, _, _ in test_loader:
                x0, traj, nu = x0.to(device), traj.to(device), nu.to(device)
                #full autoregressive rollout, K+1 refinement passes per dt
                predicted_traj = model.rollout(x0, nu, test_t_grid)
                mse, spectral = traj_loss(predicted_traj, traj)
                test_mse += mse.sum().item()
                test_spectral += spectral.sum().item()
                per_example = mse + Lambda * spectral
                for nu_val, l in zip(nu.tolist(), per_example.tolist()):
                    #unstandardize nu for data reporting
                    key = round(unstandardise_nu(nu_val), 4)
                    loss_sums[key] += l
                    loss_counts[key] += 1

            test_by_nu = {k: loss_sums[k] / loss_counts[k] for k in sorted(loss_sums)}
            avg_test_loss = sum(loss_sums.values()) / sum(loss_counts.values())


        #FOR DISPLAY
        by_nu_str = ", ".join(f"nu={k:g}: {v:.5f}" for k, v in test_by_nu.items())

        is_best = avg_test_loss < best_test_loss
        if is_best:
            best_test_loss = avg_test_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), os.path.join(HERE, "model_best.pth"))
            with open(os.path.join(HERE, "model_best.txt"), "w") as fh:
                fh.write(f"epoch {best_epoch}\ntest loss {best_test_loss:.6f}\n"
                         f"by nu {by_nu_str}\n")

        n_test = sum(loss_counts.values())
        print(f"Epoch {epoch + 1}, lambda = {Lambda:.4f}, "
              f"train 1-step eps = {avg_loss / len(train_loader):.5f}, "
              f"test rollout mse = {test_mse / n_test:.4f}, test spectral = {test_spectral / n_test:.4f}, "
              f"Average test loss = {avg_test_loss:.6f}{'  <- best so far' if is_best else ''}, "
              f"by nu -> {by_nu_str}")

    print(f"\nbest test loss {best_test_loss:.6f} at epoch {best_epoch} "
          f"(saved as model_best.pth); final epoch {num_epochs} was {avg_test_loss:.6f}")

    #brief fine tuning phase 
    model.load_state_dict(torch.load(os.path.join(HERE, "model_best.pth"), map_location = device))
    num_epochs = 25
    optimizer = torch.optim.Adam([
        {"params": other_params, "weight_decay": 5e-5},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    best_test_loss = float("inf")
    best_epoch = None
    L = 8 #how many steps we unroll and backprop through
    Mu = 0.5 #weight on the refiner term, roughly the 35/65 split the pretrain used

    for epoch in range(num_epochs):
        model.train()
        avg_loss = avg_mse = avg_spectral = avg_refine = 0
        Lambda = 0.02
        for nu, x0, traj, _, _ in train_loader:
            T = traj.shape[1]
            #random window of L + 1 frames
            s = torch.randint(0, T - L, (1,)).item()
            sub = traj[:, s:s + L + 1].to(device)
            nu_b = nu.to(device)
            predicted_traj = model.rollout(sub[:, 0], nu_b, t_grid[:L + 1], K = 0)
            optimizer.zero_grad()
            l1, l2 = traj_loss(predicted_traj, sub)

            #keep the denoisers strong
            Bn = traj.shape[0]
            t = torch.randint(0, T - 1, (Bn, n_t))
            idx = torch.arange(Bn).unsqueeze(1).expand(Bn, n_t)
            a = traj[idx, t].flatten(0, 1).to(device)
            b = traj[idx, t + 1].flatten(0, 1).to(device)
            nu_r = nu.repeat_interleave(n_t).to(device)
            #k >= 1 only
            k = torch.randint(1, 5, (a.shape[0],), device = device)
            eps = torch.randn_like(b)
            sig = model.sigmas[k].view(-1, 1, 1, 1)
            refine = F.mse_loss(model(a, b + sig * eps, nu_r, k), eps)

            l = (l1 + Lambda * l2).mean() + Mu * refine
            l.backward()
            optimizer.step()
            avg_loss += l.item()
            avg_mse += l1.mean().item()
            avg_spectral += l2.mean().item()
            avg_refine += refine.item()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            loss_sums = defaultdict(float)
            loss_counts = defaultdict(int)
            test_mse = test_spectral = 0
            for nu, x0, traj, _, _ in test_loader:
                x0, traj, nu = x0.to(device), traj.to(device), nu.to(device)
                predicted_traj = model.rollout(x0, nu, test_t_grid)
                mse, spectral = traj_loss(predicted_traj, traj)
                test_mse += mse.sum().item()
                test_spectral += spectral.sum().item()
                per_example = mse + Lambda * spectral
                for nu_val, l in zip(nu.tolist(), per_example.tolist()):
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
            torch.save(model.state_dict(), os.path.join(HERE, "model_finetuned.pth"))
            with open(os.path.join(HERE, "model_finetuned.txt"), "w") as fh:
                fh.write(f"epoch {best_epoch}\ntest loss {best_test_loss:.6f}\n"
                         f"by nu {by_nu_str}\n")

        n_test = sum(loss_counts.values())
        n_train = len(train_loader)
        print(f"Finetune epoch {epoch + 1}, "
              f"train {L}-step loss = {avg_loss / n_train:.5f} "
              f"(mse {avg_mse / n_train:.4f}, spectral {avg_spectral / n_train:.4f}, refine {avg_refine / n_train:.4f}), "
              f"test 29-step mse = {test_mse / n_test:.4f}, test spectral = {test_spectral / n_test:.4f}, "
              f"Average test loss = {avg_test_loss:.6f}{'  <- best so far' if is_best else ''}, "
              f"by nu -> {by_nu_str}")

    print(f"\nbest finetuned test loss {best_test_loss:.6f} at epoch {best_epoch} "
          f"(saved as model_finetuned.pth)")