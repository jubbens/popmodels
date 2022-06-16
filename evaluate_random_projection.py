import torch
from torch.nn import functional as F
from joblib import load, dump
import numpy as np
from data_handlers import build_torch_dataloader
from rp_model import RandomProjection
from scipy.spatial.distance import cdist
import sys

data_file = sys.argv[1]
labels = None

model_name = 'rp'
categorical = False

hp_dict = {'d': 2, 'num_hidden': 128}

data = load(data_file)
x = np.array([d[1] for d in data])
print('Data shape: {0}'.format(x.shape))

if categorical:
    x = F.one_hot(torch.Tensor(x).to(torch.int64))
    x = x.to(torch.float32)
    x = x.permute(0, 2, 1)

model = RandomProjection(num_channels=1, sample_size=x.shape[1], categorical=categorical, hyperparams=hp_dict)

dataloader = build_torch_dataloader(x, shuffle=False)

embs = model.get_all_embeddings(dataloader)

dm = cdist(embs, embs, 'euclidean')
dump(dm, 'rp_distance_matrix.bin')
