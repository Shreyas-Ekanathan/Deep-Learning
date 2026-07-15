import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from kornia.losses import ssim_loss
import wandb
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.dataset import MRI_CT_DATASET
import math
import matplotlib.pyplot as plt


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv1 = nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.down_conv2 = nn.Conv2d(out_channels, out_channels, 3, stride = 1, padding = 1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.project_embedding = nn.Linear(256, out_channels)
        
    def forward(self, x, t_embedding):
        x = self.down_conv1(x)
        x = F.relu(self.norm1(x))
        x = x + self.project_embedding(t_embedding)[:, :, None, None]
        x = self.down_conv2(x)
        x = F.relu(self.norm2(x))
        return self.pool(x), x #for skip layers
    
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2) #double size
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)  # for skip connection
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm2d(out_channels)

        self.project_embedding = nn.Linear(256, out_channels)


    def forward(self, x, skip, t_embedding):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.norm1(self.conv1(x)))
        x = x + self.project_embedding(t_embedding)[:, :, None, None]
        x = F.relu(self.norm2(self.conv2(x)))
        return x

class u_net(nn.Module):
    #the role of the U-net is to predict the noise inejcted into the input image
    def __init__(self):
        super().__init__()
        
        self.time_embedder = nn.Sequential(
            nn.Linear(256, 1024), 
            nn.SiLU(),
            nn.Linear(1024, 256)
        )
        
        in_channels  = [2, 32, 64, 128]
        out_channels = [32, 64, 128, 256]
        self.encoder = nn.ModuleList([EncoderBlock(i, o) for i, o in zip(in_channels, out_channels)])

        self.bottleneck_conv1 = nn.Conv2d(256, 512, 3, padding = 1)
        self.bottleneck_norm1 = nn.InstanceNorm2d(512)
        self.bottleneck_conv2 = nn.Conv2d(512, 512, 3, padding = 1)
        self.bottleneck_norm2 = nn.InstanceNorm2d(512)

        in_channels  = [512, 256, 128, 64]
        out_channels = [256, 128, 64, 32]
        self.decoder = nn.ModuleList([DecoderBlock(i, o) for i, o in zip(in_channels, out_channels)])
        
        self.out_conv = nn.Conv2d(32, 1, 3, padding = 1)
        
    def sinusoidal_embedding(self, t):
        freqs = torch.exp(-math.log(10000) * torch.arange(128) / 128).to(t.device)
        args = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1) 

    def forward(self, x, t, mri):
        sinusoidal = self.sinusoidal_embedding(t)
        embedding = self.time_embedder(sinusoidal)        

        x = torch.cat((x, mri), dim = 1)

        skips = []
        for encoder_layer in self.encoder:
            x, skip = encoder_layer(x, embedding)
            skips.append(skip)
        
        x = F.relu(self.bottleneck_norm1(self.bottleneck_conv1(x)))
        x = F.relu(self.bottleneck_norm2(self.bottleneck_conv2(x)))
        
        for (i, decoder_layer) in enumerate(self.decoder):
            x = decoder_layer(x, skips[-(i + 1)], embedding)
            
        x = self.out_conv(x)
        return x

    def cosine_scheduler(self, t, T = 1000, s = 0.008):
        def f(t):
            return torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
        
        numerator = f(t)
        denom = f(torch.tensor(0.0, device=t.device))
        return numerator / denom
    
def ddpm_sample(model, mri, alpha_bars, T=1000):
    x = torch.randn_like(mri)  # start from pure noise
    for t in reversed(range(T)):
        t_batch = torch.full((mri.shape[0],), t, device=mri.device) #stretch t out
        predicted_noise = model(x, t_batch, mri)
        
        alpha_bar_t = alpha_bars[t]
        alpha_bar_prev = alpha_bars[t - 1] if t > 0 else torch.tensor(1.0).to(mri.device)
        
        # DDPM reverse step
        x_0_pred = ((x - (1 - alpha_bar_t).sqrt() * predicted_noise) / alpha_bar_t.sqrt()).clamp(-1, 1)
        if t == 0:
            x = x_0_pred
            break
        alpha_t = alpha_bar_t / alpha_bar_prev
        beta_t = 1 - alpha_t
        mean = (alpha_bar_prev.sqrt() * beta_t / (1 - alpha_bar_t)) * x_0_pred + (alpha_t.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar_t)) * x
        var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
        x = mean + var.sqrt() * torch.randn_like(x)
    return x

def ddim_sample(model, mri, alpha_bars, T = 1000, num_steps = 50):
    x = torch.randn_like(mri)  # start from pure noise
    tsteps = torch.linspace(T-1, 0, num_steps).long()
    for i, t in enumerate(tsteps):
        t_batch = torch.full((mri.shape[0],), t, device=mri.device) #stretch t out
        predicted_noise = model(x, t_batch, mri)

        alpha_bar_t = alpha_bars[t]
        if i + 1 < len(tsteps):
            alpha_bar_prev = alpha_bars[tsteps[i + 1]]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=mri.device)

        # DDIM reverse step
        x_0_pred = (x - (1 - alpha_bar_t).sqrt() * predicted_noise) / alpha_bar_t.sqrt()
        x_0_pred = x_0_pred.clamp(-1, 1)
        x = alpha_bar_prev.sqrt() * x_0_pred + (1 - alpha_bar_prev).sqrt() * predicted_noise

    return x

def DPM_sample(model, mri, alpha_bars, T = 1000, num_steps = 50):
    #second order solver, more expensive but should be better
    
    x = torch.randn_like(mri)  # start from pure noise
    tsteps = torch.linspace(T-1, 0, num_steps).long()
    l = torch.log(torch.sqrt(alpha_bars.clamp(0, 1 - 1e-6) / (1 - alpha_bars).clamp(1e-6, 1)))
    for i, t in enumerate(tsteps):
        t_batch = torch.full((mri.shape[0],), t, device=mri.device) #stretch t out
        epsilon1 = model(x, t_batch, mri)
        
        alpha_bar_t = alpha_bars[t]
        if i + 1 < len(tsteps):
            alpha_bar_prev = alpha_bars[tsteps[i + 1]]
            t_next = tsteps[i + 1]
        else:
            alpha_bar_prev = torch.tensor(1.0, device=mri.device)
            t_next = 0

        h = l[t_next] - l[t]
        x_guess = alpha_bar_prev.sqrt() / alpha_bar_t.sqrt() * x - (1 - alpha_bar_prev).sqrt() * ((torch.e ** h) - 1) * epsilon1
        x_guess = x_guess.clamp(-1, 1)

        t_next_batch = torch.full((mri.shape[0],), t_next, device=mri.device) #stretch t out
        epsilon2 = model(x_guess, t_next_batch, mri)
        #trapezoidal update
        x = alpha_bar_prev.sqrt() / alpha_bar_t.sqrt() * x - (1 - alpha_bar_prev).sqrt() * ((torch.e ** h) - 1) * (epsilon1 + epsilon2) / 2
        x = x.clamp(-1, 1)
    return x

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') #lets put it on gpu

U_net = u_net().to(device)
state_dict = torch.load("diffusion/diffusion_epoch_140.pth", map_location=device)
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
U_net.load_state_dict(state_dict)
U_net.eval()

data_root = os.path.join(os.path.dirname(__file__), "..", "data")
train_dataset = MRI_CT_DATASET(f"{data_root}/train/mri", f"{data_root}/train/ct")
test_dataset  = MRI_CT_DATASET(f"{data_root}/test/mri",  f"{data_root}/test/ct")
train_loader = DataLoader(train_dataset, batch_size = 8, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False)

sample_mri, sample_ct = next(iter(test_loader))

T = 1000
ts = torch.arange(T).to(device)
alpha_bars = U_net.cosine_scheduler(ts)

with torch.no_grad():
    pred_ddpm = ddpm_sample(U_net, sample_mri.to(device), alpha_bars)
    pred_ddim = ddim_sample(U_net, sample_mri.to(device), alpha_bars)
    pred_dpm  = DPM_sample(U_net, sample_mri.to(device), alpha_bars)

samples = [("DDPM", pred_ddpm), ("DDIM", pred_ddim), ("DPM-Solver-2", pred_dpm)]
for name, pred in samples:
    l1   = F.l1_loss(pred, sample_ct.to(device)).item()
    ssim = ssim_loss(pred, sample_ct.to(device), window_size=11).item()
    print(f"{name}  L1: {l1:.4f}  SSIM: {ssim:.4f}")

fig, axes = plt.subplots(5, 8, figsize=(20, 12))
rows = [("MRI Input", sample_mri), ("Real CT", sample_ct),
        ("DDPM", pred_ddpm), ("DDIM", pred_ddim), ("DPM-Solver-2", pred_dpm)]
for row_idx, (label, imgs) in enumerate(rows):
    for i in range(8):
        axes[row_idx, i].imshow(imgs[i, 0].cpu().clamp(-1, 1), cmap='gray')
        axes[row_idx, i].axis('off')
    axes[row_idx, 0].set_ylabel(label)
plt.tight_layout()
plt.show()
