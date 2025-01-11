import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(42)

# A simple version of VAE model
class VAE(nn.Module):
    def __init__(self, latent_dim = 20, hidden_dim = 500):
        super(VAE, self).__init__()
        self.size = 784
        # Define some linear layers
        self.func_e = nn.Linear(self.size, hidden_dim)
        self.func_mean = nn.Linear(hidden_dim, latent_dim)
        self.func_logvar = nn.Linear(hidden_dim, latent_dim)
        self.func_d1 = nn.Linear(latent_dim, hidden_dim)
        self.func_d2 = nn.Linear(hidden_dim, self.size)
        self.relu = F.relu
        self.sigmoid = F.sigmoid
    
    # Define Encoder
    def encoder(self, x_input):
        x = self.relu(self.func_e(x_input.view(-1, self.size)))
        mean = self.func_mean(x)
        logvar = self.func_logvar(x)
        return mean, logvar
    
    # Define Decoder
    def decoder(self, z):
        z = self.relu(self.func_d1(z))
        x_out = self.sigmoid(self.func_d2(z))
        x_out = x_out.view(-1, 1, 28, 28)
        return x_out
    
    # Define sample
    def sample_normal(self, mean, logvar):
        sd = torch.exp(logvar * 0.5)
        e = Variable(torch.randn(sd.size())) # Sample from standard normal
        z = e.mul(sd).add_(mean)
        return z
    
    # Define forward function
    def forward(self, x_in):
        z_mean, z_logvar = self.encoder(x_in)
        z = self.sample_normal(z_mean, z_logvar)
        x_out = self.decoder(z)
        return x_out, z_mean, z_logvar
