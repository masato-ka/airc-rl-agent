import torch
from torch import nn
from torch.nn import functional as F
import torchvision

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=256):
        return input.view(input.size(0), size, 3, 8)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=6144, z_dim=32):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),
        )

        self.out1 = nn.Sequential(nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2),
                                  nn.Sigmoid(),
                                  )
        self.out2 = nn.Sequential(nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2),
                                  nn.Sigmoid(),
                                  )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)  # F.softplus(self.fc2(h))
        if self.training:
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        z = mu
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        x = self.decoder(z)
        mu_y = self.out1(x)
        sigma_y = self.out1(x)
        return mu_y, sigma_y

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        mu_y, sigma_y = self.decode(z)
        return mu_y, sigma_y, mu, logvar

    def loss_fn(self, image, mu_y, sigma_y, mean, logvar):
        m_vae_loss = (image - mu_y) ** 2 / sigma_y
        m_vae_loss = 0.5 * torch.sum(m_vae_loss)
        a_vae_loss = torch.log(2 * 3.14 * sigma_y)
        a_vae_loss = 0.5 * torch.sum(a_vae_loss)
        KL = -0.5 * torch.sum((1 + logvar - mean.pow(2) - logvar.exp()), dim=0)
        KL = torch.mean(KL)
        return KL + m_vae_loss + a_vae_loss
