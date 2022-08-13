import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class AutoEncoder(nn.Module):

    def __init__(self, input_dims, max_dims, latent_dims, device='cpu'):
        super(AutoEncoder, self).__init__()
        self.device = device
        encoder_layers = []
        last_idx = len(list(range(int(np.log2(max_dims)), int(np.log2(latent_dims))-1, -1))) -1
        for i, log_dim in enumerate(range(int(np.log2(max_dims)), int(np.log2(latent_dims))-1, -1)):
            if i == 0:
                encoder_layers.append(nn.Linear(input_dims, 2**log_dim))
                encoder_layers.append(nn.LeakyReLU())
            elif i == last_idx:
                encoder_layers.append(nn.Linear(2**(log_dim+1), 2**log_dim))
            else:
                encoder_layers.append(nn.Linear(2**(log_dim+1), 2**log_dim))
                encoder_layers.append(nn.LeakyReLU())

        decoder_layers = []
        for i, log_dim in enumerate(range(int(np.log2(latent_dims)), int(np.log2(max_dims))+1)):
            if i == 0:
                decoder_layers.append(nn.Linear(2 ** log_dim, 2 ** (log_dim + 1)))
                decoder_layers.append(nn.LeakyReLU())
            elif i == last_idx:
                decoder_layers.append(nn.Linear(2 ** log_dim, input_dims))
                decoder_layers.append(nn.Sigmoid())
            else:
                decoder_layers.append(nn.Linear(2 ** log_dim, 2 ** (log_dim + 1)))
                decoder_layers.append(nn.LeakyReLU())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        z = self.encoder(x)
        recons = self.decoder(z)
        return recons

    def mse(self, x, recons, reduction='mean'):
        x = x.to(self.device)
        mse = F.mse_loss(x, recons, reduction=reduction)
        if reduction == 'none':
            mse = torch.mean(mse, dim=1)
        mse = mse
        return mse
