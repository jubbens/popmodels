import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from joblib import load, dump
import numpy as np
import wandb
from data_handlers import build_torch_dataloader
from vae_model import VAE
from write_plots import plot_scatter
from scipy.spatial.distance import cdist
import sys
import os
from csv import writer, reader

data_file = sys.argv[1]
labels = None

run_name = 'VAE'
load_model = False
do_log = False

if load_model:
    run_name = run_name + ' (loaded)'

batch_size = 32
categorical = False
num_epochs = 100

hp_dict = {'d': 2, 'kld_weight': 1., 'hidden_dims': [128, 64, 32, 16, 8, 4], 'kld_annealing': False}

if do_log:
    wandb_logger = WandbLogger(name=run_name, project='pop-model')
    wandb.init(name=run_name, project='pop-model')

data = load(data_file)
x = np.array([d[1] for d in data])
print('Data shape: {0}'.format(x.shape))

if categorical:
    x = F.one_hot(torch.Tensor(x).to(torch.int64))
    x = x.to(torch.float32)
    x = x.permute(0, 2, 1)
else:
    if np.max(x) != 1. or np.min(x) != 0.:
        print('Data is not normalized to [0, 1], will do that now on account of we want to use BCE.')
        x = x - np.min(x)
        x = x / np.max(x)

dataloader = build_torch_dataloader(x, batch_size=batch_size, drop_last=True)

if categorical:
    model = VAE(num_channels=x.shape[1], sample_size=x.shape[2], categorical=categorical, hyperparams=hp_dict)
else:
    model = VAE(num_channels=1, sample_size=x.shape[1], categorical=categorical, hyperparams=hp_dict)

# Fit the model
if load_model is False:
    log = wandb_logger if do_log else False
    trainer = pl.Trainer(gpus=2, max_epochs=num_epochs, logger=log, accelerator="dp", checkpoint_callback=False)
    trainer.fit(model, dataloader)

dataloader = build_torch_dataloader(x, batch_size=batch_size, shuffle=False)

embs = model.get_all_embeddings(dataloader)
print(embs.shape)

groups_file = os.path.join(os.path.dirname(data_file), 'subpops.csv')

if os.path.isfile(groups_file):
    labels = []

    with open(groups_file, 'r') as file:
        for i, l in enumerate(file):
            labels.extend([i] * len(l.split(',')))

    if len(labels) != embs.shape[0]:
        print('Something is wrong with the subpop labels file, will not output the embedding plot.')
    else:
        plot_scatter(embs[:, 0], embs[:, 1], labels, title='VAE', legend=False)
        plt.savefig('vae.png')
        plt.savefig('vae.pdf')
else:
    plot_scatter(embs[:, 0], embs[:, 1], title='beta-VAE')
    plt.savefig('vae.png')
    plt.savefig('vae.pdf')

dm = cdist(embs, embs, 'euclidean')
dump(dm, 'vae_distance_matrix.bin')

if do_log:
    wandb.log({"plot": wandb.Image("vae.png")})

all_ids = []

if os.path.isfile(groups_file):
    with open(groups_file) as file:
        for row in reader(file, delimiter=','):
            all_ids.extend(row)

all_ids = np.array(all_ids)

tosave = np.vstack((all_ids, embs[:, 0], embs[:, 1])).T
writer(open(os.path.join(os.path.dirname(data_file), 'vae_embs.csv'), 'w+', newline='')).writerows(tosave)
