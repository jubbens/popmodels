import torch
from torch import nn
from types_ import *


class RandomProjection(object):
    def __init__(self,
                 num_channels: int,
                 sample_size: int,
                 hyperparams: Dict,
                 categorical: bool,
                 **kwargs):
        super().__init__()

        num_hidden = hyperparams['num_hidden']
        d = hyperparams['d']

        self.num_channels = num_channels
        self.sample_size = sample_size
        self.categorical = categorical

        in_channels = num_channels * sample_size

        self.encoder = nn.Sequential(
            nn.Linear(in_channels, num_hidden),
            nn.Tanh(),
            nn.Linear(num_hidden, d))

    def encode(self, input):
        return self.encoder(input)

    def get_all_embeddings(self, dataloader):
        ret = []

        for batch in dataloader:
            z = self.encode(batch)
            ret.append(z)

        ret = torch.cat(ret, dim=0)

        return ret.detach().cpu().numpy()
