import torch 
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
from config import EMBEDDING_DIM, CONVOLVED_SHAPE

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(4, 8, kernel_size=(3, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1),
            nn.Flatten()
        )
        c, h, w = CONVOLVED_SHAPE
        self.fc_mean = nn.Linear(128 * h * w, EMBEDDING_DIM)
        self.fc_log_var = nn.Linear(128 * h * w, EMBEDDING_DIM)

    def forward(self, x):
        x = self.cnn(x)
        return self.fc_mean(x), self.fc_log_var(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        c, h, w = CONVOLVED_SHAPE
        self.fc = nn.Linear(EMBEDDING_DIM, c * h * w)
        self.reshape = lambda x: x.reshape(-1, c, h, w)
        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=(3, 3), stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(4, 1, kernel_size=(3, 3), stride=1, padding=1),
            nn.Conv2d(1, 1, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.reshape(x)
        x = self.cnn(x)
        return x


class VAE(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.to(device)

    def sample_latent(self, mean, log_var):
        dist = Normal(loc=0, scale=1)
        epsilon_star = dist.sample(mean.shape).to(self.device)
        latent = epsilon_star * torch.exp(log_var / 2) + mean
        return latent
    
    def kl_loss(self, mean, log_var):
        loss = torch.sum(torch.exp(log_var) + mean ** 2 + 1 - log_var, dim=1) / 2
        return torch.mean(loss)

    def forward(self, x):
        mean, log_var = self.encoder.forward(x)
        latent = self.sample_latent(mean, log_var)
        pred = self.decoder(latent)
        return mean, log_var, pred
    
    def to(self, device):
        super().to(device)
        self.device = device

    
