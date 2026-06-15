#let's make a VAE for CIFAR
# visualize the latent space and test out interpolations, see what interesting stuff happens
#convolutional encoder and decoder
#add class data to the encodings and decodings 

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
from sklearn.manifold import TSNE

#input dim is 32x32x3, lets sent to a latent space of dimension 128
#lets do 2 conv layers, 2 pools (2 each), and then flatten
#conv windows can be 3x3 for both

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') #lets put it on gpu

class VAE(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        #output after this should be like 4x4x128, which becomes 2048x1
        # we flatten this to get our linear layer to give us mu and sigma
        self.mu = nn.Linear(2048, latent_dim)
        self.log_sigma = nn.Linear(2048, latent_dim)
        
        #now decode
        self.decoder_linear = nn.Linear(latent_dim, 2048)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 3, stride = 2, padding = 1, output_padding = 1)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 3, stride = 2, padding = 1, output_padding = 1)
        self.upconv3 = nn.ConvTranspose2d(32, 3, 3, stride = 2, padding = 1, output_padding = 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(-1, 2048)
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        sigma = torch.exp(0.5 * log_sigma)
        
        #now we sample to give something to the decoder
        epsilon = torch.randn_like(sigma)
        decoder_input = mu + sigma * epsilon
        decoder_input = F.relu(self.decoder_linear(decoder_input))
        decoder_input = decoder_input.reshape(-1, 128, 4, 4)
        out = self.upconv1(decoder_input)
        out = self.upconv2(F.relu(out))
        out = self.upconv3(F.relu(out))
        out = torch.sigmoid(out) #mirrors input compression
        return out, mu, log_sigma

    def ELBO(self, out, target, mu, log_sigma, beta):
        mse = F.binary_cross_entropy(out, target, reduction='sum')
        kl = -0.5 * torch.sum(1 + log_sigma - mu**2 - torch.exp(log_sigma))
        loss = mse + beta * kl
        return loss
    
    def decode(self, decoder_input):
        decoder_input = F.relu(self.decoder_linear(decoder_input))
        decoder_input = decoder_input.reshape(-1, 128, 4, 4)
        out = self.upconv1(decoder_input)
        out = self.upconv2(F.relu(out))
        out = self.upconv3(F.relu(out))
        out = torch.sigmoid(out)
        return out
    
    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(-1, 2048)
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        sigma = torch.exp(0.5 * log_sigma)
        
        #now we sample to give something to the decoder
        epsilon = torch.randn_like(sigma)
        encoding = mu + sigma * epsilon
        return encoding, mu


transform = transforms.Compose([
    transforms.ToTensor(),  # scales to [0,1] automatically
])
train_data = torchvision.datasets.CIFAR10(root='./VAE', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

latent_dim = 256
model = VAE(latent_dim).to(device)
model = torch.compile(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
num_epochs = 75
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

for epoch in range(num_epochs):
    model.train()
    avg_loss = 0
    for x, target in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        out, mu, log_sigma  = model(x)
        loss = model.ELBO(out, x, mu, log_sigma, beta = min(0.3, epoch / (num_epochs * 0.5)))
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    scheduler.step()
    print("Epoch: ", epoch)
    print("Average loss: ", avg_loss / (len(train_loader)))

#post data analysis: let's check how good our reconstructions are
model.eval()
fig, axes = plt.subplots(2, 10, figsize=(20, 4))

with torch.no_grad():
    images, _ = next(iter(train_loader))
    images = images[:10].to(device)
    reconstructions = model(images)[0].cpu()

for i in range(10):
    axes[0, i].imshow(images[i].cpu().permute(1, 2, 0))
    axes[0, i].axis('off')
    axes[1, i].imshow(reconstructions[i].permute(1, 2, 0).clip(0, 1))
    axes[1, i].axis('off')

axes[0, 0].set_ylabel('Original')
axes[1, 0].set_ylabel('Reconstructed')
plt.tight_layout()
plt.savefig('VAE/data_analysis/reconstructions2.png', dpi=150, bbox_inches='tight')
#reconstructions is with 2 conv layers, latent dim 128, 
# reconstructions2 is with 3, latent dim 256
plt.close()

for n in range(3):
    images, _ = next(iter(train_loader))
    img1 = images[0:1].to(device)
    img2 = images[1:2].to(device)

    with torch.no_grad():
        _, mu1 = model._orig_mod.encode(img1)
        _, mu2 = model._orig_mod.encode(img2)

        steps = 10
        fig, axes = plt.subplots(1, steps, figsize=(20, 2))
        for i, t in enumerate(torch.linspace(0, 1, steps)):
            z = (1 - t) * mu1 + t * mu2
            out = model._orig_mod.decode(z).cpu()
            axes[i].imshow(out[0].permute(1, 2, 0).clip(0, 1))
            axes[i].axis('off')

    plt.savefig(f'VAE/data_analysis/interpolation{n}.png', dpi=150, bbox_inches='tight')
    plt.close()


all_mu = []
all_labels = []
with torch.no_grad():
    for x, labels in train_loader:
        x = x.to(device)
        _, mu = model._orig_mod.encode(x)
        all_mu.append(mu.cpu().numpy())
        all_labels.append(labels.numpy())

all_mu = np.concatenate(all_mu)
all_labels = np.concatenate(all_labels)

#sample some points
idx = np.random.choice(len(all_mu), 5000, replace=False)
embedded = TSNE(n_components=2, perplexity=30).fit_transform(all_mu[idx])

classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']
plt.figure(figsize=(10, 8))
for c in range(10):
    mask = all_labels[idx] == c
    plt.scatter(embedded[mask, 0], embedded[mask, 1], s=2, label=classes[c], alpha=0.5)
plt.legend(markerscale=5)
plt.savefig('VAE/data_analysis/tsne.png', dpi=150, bbox_inches='tight')
plt.close()
