import torch
from torch import nn
from torch.nn import functional as F
from types_ import *
from pytorch_lightning import LightningModule


class autoencoder(LightningModule):
    def __init__(self,
                 num_channels: int,
                 sample_size: int,
                 hyperparams: Dict,
                 categorical: bool,
                 **kwargs):
        super().__init__()

        d = hyperparams['d']
        hidden_dims = hyperparams['hidden_dims']

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
                    # nn.BatchNorm1d(h_dim),
                    nn.CELU())
            )
            in_channels = h_dim

        modules.append(nn.Linear(hidden_dims[-1], d))

        self.encoder = nn.Sequential(*modules)

        modules = []

        hidden_dims.reverse()

        modules.append(nn.Sequential(
                    nn.Linear(d, hidden_dims[0]),
                    # nn.BatchNorm1d(hidden_dims[0]),
                    nn.CELU()))

        for i in range(1, len(hidden_dims)):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                    # nn.BatchNorm1d(hidden_dims[i]),
                    nn.CELU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.Linear(hidden_dims[-1], sample_size * num_channels)
                            )

    def encode(self, input: Tensor) -> Tensor:
        z = self.encoder(input)

        return z

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder(z)
        result = self.final_layer(result)

        if self.categorical:
            result = torch.reshape(result, (-1, self.num_channels, self.sample_size))

        return result

    def get_all_embeddings(self, dataloader):
        self.cuda()
        self.eval()

        ret = []

        for batch in dataloader:
            z = self.encode(batch.cuda())
            ret.append(z)

        ret = torch.cat(ret, dim=0)

        return ret.detach().cpu().numpy()

    def loss_function(self, *args, **kwargs):
        recons = args[0]
        input = args[1]

        if self.categorical:
            recons_loss = F.cross_entropy(recons, torch.argmax(input, 1))
        else:
            recons_loss = F.mse_loss(recons, input)

        return recons_loss

    def training_step(self, batch, batch_idx):
        x = batch

        z = self.encode(x)
        recons = self.decode(z)

        loss = self.loss_function(recons, x)
        self.log_dict({'reconstruction_loss': loss})

        return loss

    def test_step(self, batch, batch_idx):
        x = batch

        z = self.encode(x)
        recons = self.decode(z)

        loss = self.loss_function(recons, x)
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
