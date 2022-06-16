from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from write_plots import plot_scatter
import matplotlib
matplotlib.use('qt5agg')
from matplotlib import pyplot as plt
from joblib import load
import numpy as np
import umap
import sys
import os
from csv import reader, writer

data_file = sys.argv[1]
groups_file = os.path.join(os.path.dirname(data_file), 'subpops.csv')

all_ids = []

with open(groups_file) as file:
    for row in reader(file, delimiter=','):
        all_ids.extend(row)

all_ids = np.array(all_ids)

data = load(data_file)
x = np.array([d[1] for d in data])
print('Data shape: {0}'.format(x.shape))

# Do PCA and t-SNE on raw data
print('Doing PCA...')
pca_embs = PCA(n_components=2).fit_transform(x)
if os.path.isfile(groups_file):
    labels = []

    with open(groups_file, 'r') as file:
        for i, l in enumerate(file):
            labels.extend([i] * len(l.split(',')))

    plot_scatter(pca_embs[:, 0], pca_embs[:, 1], labels, title='PCA', legend=False)
else:
    plot_scatter(pca_embs[:, 0], pca_embs[:, 1], title='PCA')

plt.savefig('pca.pdf')
plt.close()

# Save embeddings
tosave = np.vstack((all_ids, pca_embs[:, 0], pca_embs[:, 1])).T
writer(open(os.path.join(os.path.dirname(data_file), 'pca_embs.csv'), 'w+', newline='')).writerows(tosave)


print('Doing T-SNE...')
tsne_embs = TSNE(n_components=2).fit_transform(x)
if os.path.isfile(groups_file):
    labels = []

    with open(groups_file, 'r') as file:
        for i, l in enumerate(file):
            labels.extend([i] * len(l.split(',')))

    plot_scatter(tsne_embs[:, 0], tsne_embs[:, 1], labels, title='t-SNE', legend=False)
else:
    plot_scatter(tsne_embs[:, 0], tsne_embs[:, 1], title='t-SNE')

plt.savefig('tsne.pdf')
plt.close()

# Save embeddings
tosave = np.vstack((all_ids, tsne_embs[:, 0], tsne_embs[:, 1])).T
writer(open(os.path.join(os.path.dirname(data_file), 'tsne_embs.csv'), 'w+', newline='')).writerows(tosave)

print('Doing UMAP...')
umap_embs = umap.UMAP(n_components=2).fit_transform(x)
if os.path.isfile(groups_file):
    labels = []

    with open(groups_file, 'r') as file:
        for i, l in enumerate(file):
            labels.extend([i] * len(l.split(',')))

    plot_scatter(umap_embs[:, 0], umap_embs[:, 1], labels, title='UMAP', legend=False)
else:
    plot_scatter(umap_embs[:, 0], umap_embs[:, 1], title='UMAP')

plt.savefig('umap.pdf')
plt.close()

# Save embeddings
tosave = np.vstack((all_ids, umap_embs[:, 0], umap_embs[:, 1])).T
writer(open(os.path.join(os.path.dirname(data_file), 'umap_embs.csv'), 'w+', newline='')).writerows(tosave)

print('Doing MDS...')
mds_embs = MDS(n_components=2).fit_transform(x)
if os.path.isfile(groups_file):
    labels = []

    with open(groups_file, 'r') as file:
        for i, l in enumerate(file):
            labels.extend([i] * len(l.split(',')))

    plot_scatter(mds_embs[:, 0], mds_embs[:, 1], labels, title='MDS', legend=False)
else:
    plot_scatter(mds_embs[:, 0], mds_embs[:, 1], title='MDS')

plt.savefig('mds.pdf')
plt.close()

# Save embeddings
tosave = np.vstack((all_ids, mds_embs[:, 0], mds_embs[:, 1])).T
writer(open(os.path.join(os.path.dirname(data_file), 'mds_embs.csv'), 'w+', newline='')).writerows(tosave)
