import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from types_ import *
from pytorch_lightning import LightningModule


class VAE(LightningModule):
    def __init__(self,
                 num_channels: int,
                 sample_size: int,
                 hyperparams: Dict,
                 categorical: bool,
                 **kwargs):
        super().__init__()

        self.post_epoch_fn = None

        d = hyperparams['d']
        hidden_dims = hyperparams['hidden_dims']
        self.kld_weight = hyperparams['kld_weight']
        self.kld_annealing = hyperparams['kld_annealing']

        self.num_channels = num_channels
        self.sample_size = sample_size
        self.latent_dim = d
        self.categorical = categorical

        in_channels = num_channels * sample_size

        modules = [nn.Flatten()]
        if hidden_dims is None:
            hidden_dims = [128, 64, 32, 16, 8, 4]

        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.ELU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], d)
        self.fc_var = nn.Linear(hidden_dims[-1], d)

        modules = []

        hidden_dims.reverse()

        modules.append(nn.Sequential(
                    nn.Linear(d, hidden_dims[0]),
                    nn.BatchNorm1d(hidden_dims[0]),
                    nn.ELU())
                    )

        for i in range(1, len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                    nn.BatchNorm1d(hidden_dims[i]),
                    nn.ELU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-1], sample_size * num_channels)
                            )

    def encode(self, input):
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        result = self.decoder(z)
        result = self.final_layer(result)

        if self.categorical:
            result = torch.reshape(result, (-1, self.num_channels, self.sample_size))

        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def get_all_embeddings(self, dataloader):
        self.cuda()
        self.eval()

        ret = []

        for batch in dataloader:
            mu, log_var = self.encode(batch.cuda())
            z = self.reparameterize(mu, log_var)
            ret.append(z)

        ret = torch.cat(ret, dim=0)

        return ret.detach().cpu().numpy()

    def get_monotonic_coefficient(self):
        return np.maximum(((1 / (1 + np.exp(-((self.current_epoch - 10) * 0.1)))) - 0.5) * 2., 0.)

    def loss_function(self,
                      *args,
                      **kwargs):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        if self.categorical:
            recons_loss = F.cross_entropy(recons, torch.argmax(input, 1))
        else:
            recons_loss = F.binary_cross_entropy_with_logits(recons, input, reduction='sum')

        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())

        if self.kld_annealing:
            weight = self.get_monotonic_coefficient()
        else:
            weight = self.kld_weight

        loss = recons_loss + (weight * kld_loss)

        return loss, recons_loss, kld_loss

    def training_step(self, batch, batch_idx):
        x = batch

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)

        loss, recons_loss, kld_loss = self.loss_function(recons, x, mu, log_var)
        self.log_dict({'loss': loss, 'reconstruction_loss': recons_loss,
                       'KLD_loss': kld_loss})

        return loss

    def training_epoch_end(self, outputs):
        if self.post_epoch_fn is not None:
            self.post_epoch_fn()

    def test_step(self, batch, batch_idx):
        x = batch

        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recons = self.decode(z)

        loss, recons_loss, kld_loss = self.loss_function(recons, x, mu, log_var)
        return {'loss': loss, 'reconstruction_loss': recons_loss, 'KLD_loss': kld_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
