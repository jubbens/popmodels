import torch
from torch import nn
from types_ import *
from pytorch_lightning import LightningModule


class ContrastiveEmbeddingModel(LightningModule):
    def __init__(self,
                 num_channels: int,
                 sample_size: int,
                 hyperparams: Dict,
                 categorical: bool,
                 **kwargs):
        super().__init__()

        d = hyperparams['d']
        hidden_dims = hyperparams['hidden_dims']
        self.flip_prob = hyperparams['flip_prob']

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
                    nn.ELU())
            )
            in_channels = h_dim

        modules.append(nn.Linear(hidden_dims[-1], d))

        self.encoder = nn.Sequential(*modules)

    def encode(self, input):
        z = self.encoder(input)

        return z

    def get_all_embeddings(self, dataloader):
        self.cuda()
        self.eval()

        ret = []

        for batch in dataloader:
            z = self.encode(batch.cuda())
            ret.append(z)

        ret = torch.cat(ret, dim=0)

        return ret.detach().cpu().numpy()

    def augment(self, x):
        if self.categorical:
            rand_mask = torch.rand((x.shape[0], x.shape[2]), device=torch.device("cuda")) >= self.flip_prob
            x[:, 0, :] = torch.squeeze(x[:, 0, :], -1) * rand_mask
            temp = torch.squeeze(x[:, 1, :], -1)
            temp[~rand_mask] = 1
            x[:, 1, :] = temp
            x[:, 2, :] = torch.squeeze(x[:, 2, :], -1) * rand_mask
        else:
            rand_mask = torch.rand_like(x, device=torch.device("cuda")) <= self.flip_prob
            rand_ints = torch.round(torch.rand_like(x, device=torch.device("cuda")) * 3.)
            x[rand_mask] = rand_ints[rand_mask]

        return x

    def loss_function(self, *args, **kwargs):
        similarity = torch.cosine_similarity

        x = args[0]
        xplus = args[1]
        xminus = args[2]

        x_xplus = torch.exp(similarity(x, xplus))
        x_xminus = torch.exp(similarity(x, xminus))

        return torch.mean(-torch.log(x_xplus / (x_xplus + x_xminus)))

    def training_step(self, batch, batch_idx):
        x = batch

        x_augmented = self.augment(x.clone())

        if self.categorical:
            x_diff = torch.cat((x[1:, :, :].clone(), torch.unsqueeze(x[0, :, :].clone(), 0)), 0)
        else:
            x_diff = torch.cat((x[1:, :].clone(), torch.unsqueeze(x[0, :].clone(), 0)), 0)

        z = self.encode(x)
        z_augmented = self.encode(x_augmented)
        z_diff = self.encode(x_diff)

        loss = self.loss_function(z, z_augmented, z_diff)
        self.log_dict({'contrastive_loss': loss})

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
