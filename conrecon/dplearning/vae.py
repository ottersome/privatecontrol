import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.distributions.utils import broadcast_all
from torch.nn.modules.loss import _Loss

class VAE(nn.Module):
    def __init__(self, input_size, latent_size, hidden_size):
        super(VAE, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

        self.N = Normal(torch.zeros(latent_size), torch.ones(latent_size))
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, sigma):
        z = mu + sigma * self.N.sample(mu.shape)
        # CHECK: this to be correct.
        # self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        self.kl = 0.5 * torch.sum(sigma**2 + mu**2 - torch.log(sigma**2) - 1)
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        return self.decode(z)

# Recurrent Variational Autoencoder for Time Series
class RecurrentVAE(nn.Module):
    def __init__(
        self, input_size: int, latent_size: int, hidden_size: int, sequence_length: int
    ):
        super(RecurrentVAE, self).__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        # Encoder LSTM
        self.lstm_encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc21 = nn.Linear(hidden_size * sequence_length, latent_size)
        self.fc22 = nn.Linear(hidden_size * sequence_length, latent_size)

        # Decoder LSTM
        self.fc3 = nn.Linear(latent_size, hidden_size * sequence_length)
        self.lstm_decoder = nn.LSTM(hidden_size, input_size, batch_first=True)

        self.N = Normal(torch.zeros(latent_size), torch.ones(latent_size))
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def encode(self, x):
        # Pass input through LSTM encoder
        h1, _ = self.lstm_encoder(x)
        h1 = h1.contiguous().view(h1.size(0), -1)  # Flatten LSTM output
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, sigma):
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = 0.5 * torch.sum(sigma**2 + mu**2 - torch.log(sigma**2) - 1)
        return z

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = h3.view(h3.size(0), self.sequence_length, -1)  # Reshape to sequence format
        h3, _ = self.lstm_decoder(h3)
        return torch.sigmoid(h3)

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        return self.decode(z)
    
