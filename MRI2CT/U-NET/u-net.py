import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import random
import matplotlib.pyplot as plt
from kornia.losses import ssim_loss
import wandb
from data.dataset import MRI_CT_DATASET

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_conv1 = nn.Conv2d(in_channels, out_channels, 3, stride = 1, padding = 1)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.down_conv2 = nn.Conv2d(out_channels, out_channels, 3, stride = 1, padding = 1)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = self.down_conv1(x)
        x = F.relu(self.norm1(x))
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

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        return x

class u_net(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels  = [1, 32, 64, 128]
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
        
    def forward(self, x):
        skips = []
        for encoder_layer in self.encoder:
            x, skip = encoder_layer(x)
            skips.append(skip)
        
        x = F.relu(self.bottleneck_norm1(self.bottleneck_conv1(x)))
        x = F.relu(self.bottleneck_norm2(self.bottleneck_conv2(x)))
        
        for (i, decoder_layer) in enumerate(self.decoder):
            x = decoder_layer(x, skips[-(i + 1)])
            
        x = self.out_conv(x)
        return torch.tanh(x)
    
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') #lets put it on gpu

if __name__ == "__main__":
    print(device)
    U_net = u_net().to(device)
    U_net = torch.compile(U_net)
    optimizer = torch.optim.Adam(U_net.parameters(), lr=0.001, weight_decay=1e-4)
    num_epochs = 50
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    train_dataset = MRI_CT_DATASET("MRI2CT/data/train/mri", "MRI2CT/data/train/ct")
    test_dataset = MRI_CT_DATASET("MRI2CT/data/test/mri", "MRI2CT/data/test/ct")
    train_loader = DataLoader(train_dataset, batch_size = 8, shuffle = True, num_workers = 4, persistent_workers = True)
    test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False, num_workers = 4, persistent_workers = True)

    wandb.init(project="mri2ct", name="baseline-u-net", config={
        "lr": 0.001,
        "batch_size": 8,
        "epochs": num_epochs,
        "architecture": "unet"
    })

    for epoch in range(num_epochs):
        U_net.train()
        avg_loss = 0
        Lambda = min(0.5, epoch / num_epochs * 0.5)
        for x, real_ct in train_loader:
            x = x.to(device)
            real_ct = real_ct.to(device)
            optimizer.zero_grad()
            predicted_ct  = U_net(x)
            ssim = ssim_loss(predicted_ct, real_ct, window_size=11)
            l1 = F.l1_loss(predicted_ct, real_ct)
            loss = l1  + Lambda * ssim
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        U_net.eval()
        val_loss = 0
        val_l1 = 0
        with torch.no_grad():
            for x, real_ct in test_loader:
                x, real_ct = x.to(device), real_ct.to(device)
                predicted_ct = U_net(x)
                ssim = ssim_loss(predicted_ct, real_ct, window_size=11)
                l1 = F.l1_loss(predicted_ct, real_ct)
                loss = l1  + Lambda * ssim
                val_loss += loss.item()
                val_l1 += l1.item()
                
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_loss / len(train_loader),
            "val_loss": val_loss / len(test_loader),
            "val_l1_loss": val_l1 / len(test_loader),
            "lambda_ssim": Lambda,
        })
                
        if epoch % 10 == 0:
            sample_mri, sample_ct = next(iter(test_loader))
            with torch.no_grad():
                predicted_ct = U_net(sample_mri.to(device))
            wandb.log({
                "mri_input": wandb.Image(sample_mri[0]),
                "ct_predicted": wandb.Image(predicted_ct[0]),
                "ct_real": wandb.Image(sample_ct[0]),
            })
            torch.save(U_net.state_dict(), f"unet_epoch_{epoch}.pth")

        scheduler.step()
