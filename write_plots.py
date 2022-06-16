import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.manifold import MDS
import pandas as pd


def plot_scatter(x, y, labels=None, title=None, legend=True):
    plt.figure()

    if labels is not None:
        data = pd.DataFrame({'x': x, 'y': y, 'label': labels})
        groups = data.groupby('label')

        for name, group in groups:
            plt.scatter(group['x'], group['y'], label=name, alpha=0.5)

        if legend:
            plt.legend()
    else:
        plt.scatter(x, y, alpha=0.5)

    if title is not None:
        plt.title(title)

    plt.grid()


def plot_scatter_3d(x, y, z, labels=None, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    if labels is not None:
        data = pd.DataFrame({'x': x, 'y': y, 'z': z, 'label': labels})
        groups = data.groupby('label')

        for name, group in groups:
            ax.scatter(group['x'], group['y'], group['z'], label=name, alpha=0.5)

        ax.legend()
    else:
        ax.scatter(x, y, alpha=0.5)

    if title is not None:
        ax.title(title)

    plt.grid()


def write_mds_plot(distance_matrix, labels=None, title=None, legend=False):
    plt.figure()
    mds_coords = MDS(n_components=2, dissimilarity='precomputed').fit_transform(distance_matrix)
    plot_scatter(mds_coords[:, 0], mds_coords[:, 1], labels, legend=legend)
    plt.title(title)
    plt.savefig('plot.pdf')


def write_distance_matrix(distance_matrix):
    plt.figure()
    plt.matshow(distance_matrix)
    plt.savefig('distance_matrix.png')
