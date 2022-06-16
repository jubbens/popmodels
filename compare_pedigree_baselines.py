from joblib import load
import numpy as np
from scipy.stats import pearsonr
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import umap
import sys


def get_results_baselines(data_file, pm_file):
    ped = load(pm_file)
    data = load(data_file)
    x = np.array([d[1] for d in data])
    pm = np.array(ped[1])

    print('Doing PCA...')
    pca_embs = PCA(n_components=2).fit_transform(x)
    pca_dm = cdist(pca_embs, pca_embs, 'euclidean')
    # mask = pm > 0.
    mask = np.where(~np.eye(pm.shape[0], dtype=np.bool))
    ret_d = pca_dm[mask].flatten()
    ret_p = pm[mask].flatten()

    pca_corr, _ = pearsonr(ret_d, ret_p)
    print('PCA:')
    print(pca_corr)
    ret_d[ret_d > 0.] = np.log2(ret_d[ret_d > 0.])
    log_pca_corr, _ = pearsonr(ret_d, ret_p)
    print('PCA (log2):')
    print(log_pca_corr)

    print('Doing t-SNE...')
    tsne_embs = TSNE(n_components=2).fit_transform(x)
    tsne_dm = cdist(tsne_embs, tsne_embs, 'euclidean')
    # mask = pm > 0.
    mask = np.where(~np.eye(pm.shape[0], dtype=np.bool))
    ret_d = tsne_dm[mask].flatten()
    ret_p = pm[mask].flatten()

    tsne_corr, _ = pearsonr(ret_d, ret_p)
    print('t-SNE:')
    print(tsne_corr)
    ret_d[ret_d > 0.] = np.log2(ret_d[ret_d > 0.])
    log_tsne_corr, _ = pearsonr(ret_d, ret_p)
    print('t-SNE (log2):')
    print(log_tsne_corr)

    print('Doing UMAP...')
    umap_embs = umap.UMAP(n_components=2).fit_transform(x)
    umap_dm = cdist(umap_embs, umap_embs, 'euclidean')
    # mask = pm > 0.
    mask = np.where(~np.eye(pm.shape[0], dtype=np.bool))
    ret_d = umap_dm[mask].flatten()
    ret_p = pm[mask].flatten()

    umap_corr, _ = pearsonr(ret_d, ret_p)
    print('UMAP:')
    print(umap_corr)
    ret_d[ret_d > 0.] = np.log2(ret_d[ret_d > 0.])
    log_umap_corr, _ = pearsonr(ret_d, ret_p)
    print('UMAP (log2):')
    print(log_umap_corr)

    print('Doing MDS...')
    mds_embs = MDS(n_components=2).fit_transform(x)
    mds_dm = cdist(mds_embs, mds_embs, 'euclidean')
    # mask = pm > 0.
    mask = np.where(~np.eye(pm.shape[0], dtype=np.bool))
    ret_d = mds_dm[mask].flatten()
    ret_p = pm[mask].flatten()

    mds_corr, _ = pearsonr(ret_d, ret_p)
    print('MDS:')
    print(mds_corr)
    ret_d[ret_d > 0.] = np.log2(ret_d[ret_d > 0.])
    log_mds_corr, _ = pearsonr(ret_d, ret_p)
    print('MDS (log2):')
    print(log_mds_corr)

    print('Doing ambient...')
    ambient_dm = cdist(x, x, 'euclidean')
    # mask = pm > 0.
    mask = np.where(~np.eye(pm.shape[0], dtype=np.bool))
    ret_d = ambient_dm[mask].flatten()
    ret_p = pm[mask].flatten()

    ambient_corr, _ = pearsonr(ret_d, ret_p)
    print('ambient:')
    print(ambient_corr)
    ret_d[ret_d > 0.] = np.log2(ret_d[ret_d > 0.])
    log_ambient_corr, _ = pearsonr(ret_d, ret_p)
    print('ambient (log2):')
    print(log_ambient_corr)

    return pca_corr, log_pca_corr, tsne_corr, log_tsne_corr, umap_corr, log_umap_corr, mds_corr, log_mds_corr


if __name__ == "__main__":
    data_file = sys.argv[1]
    pm_file = sys.argv[2]

    get_results_baselines(data_file, pm_file)
