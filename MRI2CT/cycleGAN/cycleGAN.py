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
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
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

class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride = 2, padding = 1)
        self.norm2 = nn.InstanceNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride = 2, padding = 1)
        self.norm3 = nn.InstanceNorm2d(256)
        self.conv4 = nn.Conv2d(256, 1, 3, stride = 2, padding = 1)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.norm2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.norm3(self.conv3(x)), 0.2)
        return self.conv4(x)
  
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') #lets put it on gpu
        
if __name__ == "__main__":
    print(device)
    #forward passes
    forward_U_net = u_net().to(device)
    forward_U_net = torch.compile(forward_U_net)
    forward_discrim = discriminator().to(device)
    forward_discrim_optimizer = torch.optim.Adam(forward_discrim.parameters(), lr=2e-4, weight_decay=1e-4)
    forward_gen_optimizer = torch.optim.Adam(forward_U_net.parameters(), lr=2e-4, weight_decay=1e-4)
    num_epochs = 20
    gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(forward_gen_optimizer, T_max=num_epochs, eta_min=1e-5)
    discrim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(forward_discrim_optimizer, T_max=num_epochs, eta_min=1e-5)
    #backward (i.e. ct -> mri)
    backward_U_net = u_net().to(device)
    backward_U_net = torch.compile(backward_U_net)
    backward_discrim = discriminator().to(device)
    backward_discrim_optimizer = torch.optim.Adam(backward_discrim.parameters(), lr=2e-4, weight_decay=1e-4)
    backward_gen_optimizer = torch.optim.Adam(backward_U_net.parameters(), lr=2e-4, weight_decay=1e-4)
    back_gen_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(backward_gen_optimizer, T_max=num_epochs, eta_min=1e-5)
    back_discrim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(backward_discrim_optimizer, T_max=num_epochs, eta_min=1e-5)

    mri_loader = DataLoader(MRI_CT_DATASET("data/train/mri", "data/train/ct", mode="mri"),  batch_size=8, shuffle=True, drop_last=True)
    ct_loader  = DataLoader(MRI_CT_DATASET("data/train/mri", "data/train/ct", mode="ct"),   batch_size=8, shuffle=True, drop_last=True)    
    test_dataset = MRI_CT_DATASET("data/test/mri", "data/test/ct")
    test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False, num_workers = 4, persistent_workers = True)

    wandb.init(project="mri2ct", name="pix2pix-wgan-gp", dir=os.path.dirname(__file__), config={
        "lr": 2e-4,
        "batch_size": 8,
        "epochs": num_epochs,
        "architecture": "pix2pix"
    })

    for epoch in range(num_epochs):
        forward_U_net.train()
        forward_discrim.train()
        backward_U_net.train()
        backward_discrim.train()
        avg_loss = 0
        avg_fwd_discrim_loss = 0
        avg_bwd_discrim_loss = 0
        avg_fwd_cycle_loss = 0
        avg_bwd_cycle_loss = 0
        Lambda = min(0.5, epoch / num_epochs * 0.5)
        gamma = 10
        lambda_cyc = 10
        count = 0
        for mri, ct in zip(mri_loader, ct_loader):
            mri = mri.to(device)
            ct = ct.to(device)
            
            #get ct from mri
            predicted_ct  = forward_U_net(mri)
            predicted_ct = predicted_ct.detach()
            forward_discrim_optimizer.zero_grad()
            real_score = forward_discrim(ct)
            fake_score = forward_discrim(predicted_ct)
                        
            #forward discrim backprop
            wasserstein = fake_score.mean() - real_score.mean()
            epsilon = torch.rand(mri.shape[0], 1, 1, 1).to(device) #get a random intermediate point for gradients
            interpolated = (epsilon * ct + (1 - epsilon) * predicted_ct).requires_grad_(True)
            interp_score = forward_discrim(interpolated)
            gradients = torch.autograd.grad(
                outputs=interp_score,
                inputs=interpolated,
                grad_outputs=torch.ones_like(interp_score),
                create_graph=True,
                retain_graph=True,
            )[0]
            gp = ((gradients.norm(2, dim=[1,2,3]) - 1) ** 2).mean()
            discrim_loss = wasserstein + gamma * gp
            discrim_loss.backward()
            forward_discrim_optimizer.step()
            avg_fwd_discrim_loss += discrim_loss.item()

            # get mri from ct
            predicted_mri = backward_U_net(ct)
            predicted_mri = predicted_mri.detach()
            backward_discrim_optimizer.zero_grad()
            real_score_back = backward_discrim(mri)
            fake_score_back = backward_discrim(predicted_mri)

            #backward discrim backprop
            wasserstein = fake_score_back.mean() - real_score_back.mean()
            epsilon = torch.rand(mri.shape[0], 1, 1, 1).to(device) #get a random intermediate point for gradients
            interpolated = (epsilon * mri + (1 - epsilon) * predicted_mri).requires_grad_(True)
            interp_score = backward_discrim(interpolated)
            gradients = torch.autograd.grad(
                outputs=interp_score,
                inputs=interpolated,
                grad_outputs=torch.ones_like(interp_score),
                create_graph=True,
                retain_graph=True,
            )[0]
            gp = ((gradients.norm(2, dim=[1,2,3]) - 1) ** 2).mean()
            discrim_loss = wasserstein + gamma * gp
            discrim_loss.backward()
            backward_discrim_optimizer.step()
            avg_bwd_discrim_loss += discrim_loss.item()

            predicted_ct  = forward_U_net(mri)
            recovered_mri = backward_U_net(predicted_ct)
            mri_recovery_loss = F.l1_loss(recovered_mri, mri)

            predicted_mri = backward_U_net(ct)
            recovered_ct = forward_U_net(predicted_mri)
            ct_recovery_loss = F.l1_loss(recovered_ct, ct)
            
            adv_forward  = -forward_discrim(predicted_ct).mean()
            adv_backward = -backward_discrim(predicted_mri).mean()

            gen_loss = adv_forward + adv_backward + lambda_cyc * (ct_recovery_loss + mri_recovery_loss)

            #generators backprop
            forward_gen_optimizer.zero_grad()
            backward_gen_optimizer.zero_grad()
            gen_loss.backward()
            forward_gen_optimizer.step()
            backward_gen_optimizer.step()

            avg_loss += gen_loss.item()
            avg_fwd_cycle_loss += mri_recovery_loss.item()
            avg_bwd_cycle_loss += ct_recovery_loss.item()
            count += 1
            if (count > 0 and count % 100 == 0):
                print("Loss: ", avg_loss / count)

        forward_U_net.eval()
        forward_discrim.eval()
        backward_U_net.eval()
        backward_discrim.eval()
        
        val_l1 = 0
        val_ssim = 0
        sample_mri, sample_ct = next(iter(test_loader))
        with torch.no_grad():
            for x, real_ct in test_loader:
                x, real_ct = x.to(device), real_ct.to(device)
                predicted_ct = forward_U_net(x)
                val_l1 += F.l1_loss(predicted_ct, real_ct).item()
                val_ssim += ssim_loss(predicted_ct, real_ct, window_size=11).item()
            predicted_ct = forward_U_net(sample_mri.to(device))
            predicted_mri = backward_U_net(sample_ct.to(device))

        n = len(mri_loader)
        wandb.log({
            "epoch": epoch,
            "train/gen_loss": avg_loss / n,
            "train/fwd_cycle_loss": avg_fwd_cycle_loss / n,
            "train/bwd_cycle_loss": avg_bwd_cycle_loss / n,
            "train/fwd_discrim_loss": avg_fwd_discrim_loss / n,
            "train/bwd_discrim_loss": avg_bwd_discrim_loss / n,
            "val/l1": val_l1 / len(test_loader),
            "val/ssim": val_ssim / len(test_loader),
            "images/mri_real": wandb.Image((sample_mri[0] + 1) / 2),
            "images/ct_real": wandb.Image((sample_ct[0] + 1) / 2),
            "images/ct_predicted": wandb.Image((predicted_ct[0].cpu() + 1) / 2),
            "images/mri_predicted": wandb.Image((predicted_mri[0].cpu() + 1) / 2),
        })
        torch.save(forward_U_net.state_dict(), os.path.join(os.path.dirname(__file__), f"unet_epoch_{epoch}.pth"))

        gen_scheduler.step()
        discrim_scheduler.step()
        back_gen_scheduler.step()
        back_discrim_scheduler.step()
