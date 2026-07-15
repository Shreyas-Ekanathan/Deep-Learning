import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from kornia.losses import ssim_loss
import wandb
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.dataset import MRI_CT_DATASET

class VAE(nn.Module): #borrowed from prev work
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, stride = 2, padding = 1)
        self.conv4 = nn.Conv2d(128, 256, 3, stride = 2, padding = 1)
        #output after this should be like 8x8x256, which becomes 16384x1
        # we flatten this to get our linear layer to give us mu and sigma
        self.mu = nn.Linear(16384, latent_dim)
        self.log_sigma = nn.Linear(16384, latent_dim)
        
        #now decode
        self.decoder_linear = nn.Linear(latent_dim, 16384)
        self.upconv1 = nn.ConvTranspose2d(256, 128, 3, stride = 2, padding = 1, output_padding = 1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 3, stride = 2, padding = 1, output_padding = 1)
        self.upconv3 = nn.ConvTranspose2d(64, 32, 3, stride = 2, padding = 1, output_padding = 1)
        self.upconv4 = nn.ConvTranspose2d(32, 1, 3, stride = 2, padding = 1, output_padding = 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(-1, 16384)
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        sigma = torch.exp(0.5 * log_sigma)
        
        #now we sample to give something to the decoder
        epsilon = torch.randn_like(sigma)
        decoder_input = mu + sigma * epsilon
        decoder_input = F.relu(self.decoder_linear(decoder_input))
        decoder_input = decoder_input.reshape(-1, 256, 8, 8)
        out = self.upconv1(decoder_input)
        out = self.upconv2(F.relu(out))
        out = self.upconv3(F.relu(out))
        out = self.upconv4(F.relu(out))
        out = torch.tanh(out) #mirrors input compression
        return out, mu, log_sigma

    def ELBO(self, out, target, mu, log_sigma, beta):
        mse = F.l1_loss(out, target, reduction='mean')
        kl = -0.5 * torch.mean(1 + log_sigma - mu**2 - torch.exp(log_sigma))
        loss = mse + beta * kl
        return loss
    
    def decode(self, decoder_input):
        decoder_input = F.relu(self.decoder_linear(decoder_input))
        decoder_input = decoder_input.reshape(-1, 256, 8, 8)
        out = self.upconv1(decoder_input)
        out = self.upconv2(F.relu(out))
        out = self.upconv3(F.relu(out))
        out = self.upconv4(F.relu(out))
        out = torch.tanh(out)
        return out
    
    def encode(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(-1, 16384)
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        sigma = torch.exp(0.5 * log_sigma)
        
        #now we sample to give something to the decoder
        epsilon = torch.randn_like(sigma)
        encoding = mu + sigma * epsilon
        return encoding, mu

class MLP(nn.Module): #this will transform from the MRI latent space to the CT latent space:
    def __init__(self, latent_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim),
        )
    
    def forward(self, x):
        return self.layers(x)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu') #lets put it on gpu

if __name__ == "__main__":
    print(device)
    latent_dim = 256
    mri_encoder = VAE(latent_dim).to(device)
    CT_encoder = VAE(latent_dim).to(device)
    mri_optimizer = torch.optim.Adam(mri_encoder.parameters(), lr=2e-4, weight_decay=1e-4)
    CT_optimizer = torch.optim.Adam(CT_encoder.parameters(), lr=2e-4, weight_decay=1e-4)
    num_epochs = 20
    mri_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(mri_optimizer, T_max=num_epochs, eta_min=1e-5)
    CT_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(CT_optimizer, T_max=num_epochs, eta_min=1e-5)

    data_root = os.path.join(os.path.dirname(__file__), "..", "data")
    train_dataset = MRI_CT_DATASET(f"{data_root}/train/mri", f"{data_root}/train/ct")
    test_dataset = MRI_CT_DATASET(f"{data_root}/test/mri", f"{data_root}/test/ct")
    train_loader = DataLoader(train_dataset, batch_size = 8, shuffle = True, num_workers = 4, persistent_workers = True)
    test_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False, num_workers = 4, persistent_workers = True)

    wandb.init(project="mri2ct", name="disjoint-latent-spaces", dir=os.path.dirname(__file__), config={
        "lr": 2e-4,
        "batch_size": 8,
        "epochs": num_epochs,
        "architecture": "disjoint-latent-spaces"
    })

    for epoch in range(num_epochs):
        mri_encoder.train()
        CT_encoder.train()
        mri_avg_loss = 0
        ct_avg_loss = 0
        count = 0
        for mri, ct in train_loader:
            #encode mri
            mri = mri.to(device)
            mri_optimizer.zero_grad()
            mri_recovered, mu, log_sigma  = mri_encoder(mri)
            loss = mri_encoder.ELBO(mri_recovered, mri, mu, log_sigma, beta = min(0.3, epoch / (num_epochs * 0.5)))
            loss.backward()
            mri_optimizer.step()
            mri_avg_loss += loss.item()
            #encode ct
            ct = ct.to(device)
            CT_optimizer.zero_grad()
            CT_recovered, mu, log_sigma  = CT_encoder(ct)
            loss = CT_encoder.ELBO(CT_recovered, ct, mu, log_sigma, beta = min(0.3, epoch / (num_epochs * 0.5)))
            loss.backward()
            CT_optimizer.step()
            ct_avg_loss += loss.item()
            if (count > 0 and count % 500 == 0):
                print("MRI Average loss: ", mri_avg_loss / count)
                print("CT Average loss: ", ct_avg_loss / count)
            count += 1

        mri_scheduler.step()
        CT_scheduler.step()

        sample_mri, sample_ct = next(iter(test_loader))
        mri_encoder.eval()
        CT_encoder.eval()
        with torch.no_grad():
            mri_recon, _, _ = mri_encoder(sample_mri.to(device))
            ct_recon, _, _  = CT_encoder(sample_ct.to(device))
        mri_encoder.train()
        CT_encoder.train()

        wandb.log({
            "vae_epoch": epoch,
            "vae/mri_loss": mri_avg_loss / len(train_loader),
            "vae/ct_loss": ct_avg_loss / len(train_loader),
            "vae/mri_real":  wandb.Image((sample_mri[0] + 1) / 2),
            "vae/mri_recon": wandb.Image((mri_recon[0].cpu() + 1) / 2),
            "vae/ct_real":   wandb.Image((sample_ct[0] + 1) / 2),
            "vae/ct_recon":  wandb.Image((ct_recon[0].cpu() + 1) / 2),
        })
        torch.save(mri_encoder.state_dict(), os.path.join(os.path.dirname(__file__), f"mri_encoder_epoch_{epoch}.pth"))
        torch.save(CT_encoder.state_dict(),  os.path.join(os.path.dirname(__file__), f"ct_encoder_epoch_{epoch}.pth"))

    converter = MLP(latent_dim).to(device)
    converter_optimizer = torch.optim.Adam(converter.parameters(), lr=2e-4, weight_decay=1e-4)
    converter_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(converter_optimizer, T_max=num_epochs, eta_min=1e-5)
    loss_fn = nn.MSELoss()

    mri_encoder.eval() #free encoders
    CT_encoder.eval()
    for p in mri_encoder.parameters(): p.requires_grad_(False)
    for p in CT_encoder.parameters(): p.requires_grad_(False)

    for epoch in range(num_epochs):
        converter.train()
        avg_loss = 0
        count = 0
        for mri, ct in train_loader:
            #encode mri and CT, then transform
            
            mri = mri.to(device)
            ct = ct.to(device)
            converter_optimizer.zero_grad()
            
            mri = mri_encoder.encode(mri)[0]
            ct = CT_encoder.encode(ct)[0]
            predicted_ct_latent = converter(mri)
            loss = loss_fn(predicted_ct_latent, ct)
            loss.backward()
            converter_optimizer.step()
            avg_loss += loss.item()
            if (count > 0 and count % 500 == 0):
                print("Converter Average loss: ", avg_loss / (len(train_loader)))
            count += 1
            
        converter_scheduler.step()

        converter.eval()
        val_loss = 0
        val_l1 = 0
        sample_mri, sample_ct = next(iter(test_loader))
        with torch.no_grad():
            for mri, real_ct in test_loader:
                mri, real_ct = mri.to(device), real_ct.to(device)
                encoding = mri_encoder.encode(mri)[0]
                ct_latent = converter(encoding)
                ct_predicted = CT_encoder.decode(ct_latent)
                val_l1 += F.l1_loss(ct_predicted, real_ct).item()
                val_loss += ssim_loss(ct_predicted, real_ct, window_size=11).item()
            encoding = mri_encoder.encode(sample_mri.to(device))[0]
            ct_predicted = CT_encoder.decode(converter(encoding))

        wandb.log({
            "epoch": epoch,
            "train/converter_loss": avg_loss / len(train_loader),
            "val/l1": val_l1 / len(test_loader),
            "val/ssim": val_loss / len(test_loader),
            "images/mri_input":    wandb.Image((sample_mri[0] + 1) / 2),
            "images/ct_predicted": wandb.Image((ct_predicted[0].cpu() + 1) / 2),
            "images/ct_real":      wandb.Image((sample_ct[0] + 1) / 2),
        })
        torch.save(converter.state_dict(), os.path.join(os.path.dirname(__file__), f"disjoint_converter_epoch_{epoch}.pth"))

