import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from joblib import load, dump
import numpy as np
import wandb
from data_handlers import build_torch_dataloader
from ae_model import autoencoder
from write_plots import plot_scatter
from scipy.spatial.distance import cdist
import sys
import os
from csv import writer, reader

data_file = sys.argv[1]
labels = None

run_name = 'AE'
load_model = False
do_log = False

if load_model:
    run_name = run_name + ' (loaded)'

batch_size = 32
lr = 0.0001
categorical = True
num_epochs = 100

hp_dict = {'d': 2, 'hidden_dims': [256, 128]}

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

dataloader = build_torch_dataloader(x, batch_size=batch_size)

if categorical:
    model = autoencoder(num_channels=x.shape[1], sample_size=x.shape[2], categorical=categorical, hyperparams=hp_dict)
else:
    model = autoencoder(num_channels=1, sample_size=x.shape[1], categorical=categorical, hyperparams=hp_dict)

if load_model is False:
    log = wandb_logger if do_log else False
    trainer = pl.Trainer(gpus=2, max_epochs=num_epochs, logger=log, accelerator="dp", checkpoint_callback=False)
    trainer.fit(model, dataloader)
    trainer.save_checkpoint('saved.ckpt')

dataloader = build_torch_dataloader(x, batch_size=batch_size, shuffle=False)

embs = model.get_all_embeddings(dataloader)
print(embs.shape)

groups_file = os.path.join(os.path.dirname(data_file), 'subpops.csv')

if os.path.isfile(groups_file):
    labels = []

    with open(groups_file, 'r') as file:
        for i, l in enumerate(file):
            labels.extend([i] * len(l.split(',')))

    plot_scatter(embs[:, 0], embs[:, 1], labels, title='Autoencoder', legend=False)
else:
    plot_scatter(embs[:, 0], embs[:, 1], title='Autoencoder')

plt.savefig('autoencoder.pdf')
plt.savefig('autoencoder.png')

dm = cdist(embs, embs, 'euclidean')
dump(dm, 'autoencoder_distance_matrix.bin')

if do_log:
    wandb.log({"plot": wandb.Image("autoencoder.png")})

all_ids = []

with open(groups_file) as file:
    for row in reader(file, delimiter=','):
        all_ids.extend(row)

all_ids = np.array(all_ids)

tosave = np.vstack((all_ids, embs[:, 0], embs[:, 1])).T
writer(open(os.path.join(os.path.dirname(data_file), 'autoencoder_embs.csv'), 'w+', newline='')).writerows(tosave)
