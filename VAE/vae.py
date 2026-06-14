#let's make a VAE for CIFAR
# visualize the latent space and test out interpolations, see what interesting stuff happens
#convolutional encoder and decoder

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import DataLoader, Dataset
import random
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision

#input dim is 32x32x3, lets sent to a latent space of dimension 128
#lets do 2 conv layers, 2 pools (2 each), and then flatten
#conv windows can be 3x3 for both

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') #lets put it on gpu

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        #output after this should be like 8x8x64, which becomes 2056x1
        # we flatten this to get our linear layer to give us mu and sigma
        self.mu = nn.Linear(4096, latent_dim)
        self.log_sigma = nn.Linear(4096, latent_dim)
        
        #now decode
        self.decoder_linear = nn.Linear(latent_dim, 4096)
        self.upconv1 = nn.ConvTranspose2d(64, 32, 3, stride = 2, padding = 1, output_padding = 1)
        self.upconv2 = nn.ConvTranspose2d(32, 3, 3, stride = 2, padding = 1, output_padding = 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(-1, 4096)
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        sigma = torch.exp(0.5 * log_sigma)
        
        #now we sample to give something to the decoder
        epsilon = torch.randn_like(sigma)
        decoder_input = mu + sigma * epsilon
        decoder_input = F.relu(self.decoder_linear(decoder_input))
        decoder_input = decoder_input.reshape(-1, 64, 8, 8)
        out = self.upconv1(decoder_input)
        out = self.upconv2(F.relu(out))
        out = torch.sigmoid(out) #mirrors input compression
        return out, mu, log_sigma

    def ELBO(self, out, target, mu, log_sigma, beta):
        mse = F.mse_loss(out, target, reduction='sum')
        kl = -0.5 * torch.sum(1 + log_sigma - mu**2 - torch.exp(log_sigma))
        loss = mse + beta * kl
        return loss

transform = transforms.Compose([
    transforms.ToTensor(),  # scales to [0,1] automatically
])
train_data = torchvision.datasets.CIFAR10(root='./VAE', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

model = VAE(128).to(device)
model = torch.compile(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
num_epochs = 50
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

for epoch in range(num_epochs):
    model.train()
    avg_loss = 0
    for x, target in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        out, mu, log_sigma  = model(x)
        loss = model.ELBO(out, x, mu, log_sigma, beta = min(1.0, epoch / (num_epochs * 0.3)))
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    scheduler.step()
    print("Epoch: ", epoch)
    print("Average loss: ", avg_loss / (len(train_loader)))