from simpop_random import generate_random_population
from compare_pedigree_baselines import get_results_baselines
from compare_pedigree_custom import get_results_custom
from shutil import rmtree
import os
import numpy as np
from joblib import dump, load
import pandas as pd

n_iter = 100
output_dir = 'temp'
do_keep_populations = False
cache_dir = 'data/simpop-cache/migration-random'
calculate_ibd = True

do_splitting = True
do_selection = False

all_r = []
log_all_r = []

for i in range(n_iter):
    print('-------- ITER {0} --------'.format(i))

    if os.path.isdir(output_dir):
        rmtree(output_dir)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # Make a new generation
    # lmao nevermind
    # if os.path.isfile(os.path.join(cache_dir, 'simpop_random_ped-{0}.bin'.format(i))):
    #     print('Using cached simulation')
    #     data_file = os.path.join(cache_dir, 'simpop_random-{0}.bin'.format(i))
    #     pm_file = os.path.join(cache_dir, 'simpop_random_ped-{0}.bin'.format(i))
    # else:
    generate_random_population(output_dir, do_splitting=do_splitting, do_selection=do_selection)

    data_file = os.path.join(output_dir, 'simpop_random.bin')
    pm_file = os.path.join(output_dir, 'simpop_random_ped.bin')

    # Replace the pedigree matrix with an IBD matrix calculated from parents file
    if calculate_ibd:
        os.system('Rscript {0}'.format('Calculate_IBD.R'))
        ibd_matrix = pd.read_csv(os.path.join(output_dir, 'sim_Amat.csv')).to_numpy()
        all_inds = load(pm_file)[0]
        # Filter out individuals not in the returned list
        ibd_matrix = ibd_matrix[np.array(all_inds)-1, :][:, np.array(all_inds)-1]

        # Just to be sure
        if os.path.isfile(pm_file):
            os.remove(pm_file)

        dump([all_inds, ibd_matrix], pm_file)

    # Run all models
    pca_corr, log_pca_corr, tsne_corr, log_tsne_corr, umap_corr, log_umap_corr, mds_corr, log_mds_corr = get_results_baselines(data_file, pm_file)

    # autoencoder
    print('Doing autoencoder...')
    os.system('python evaluate_autoencoder.py {0}'.format(data_file))
    ae_corr, log_ae_corr = get_results_custom(data_file, pm_file, 'autoencoder_distance_matrix.bin')

    # VAE
    print('Doing VAE...')
    os.system('python evaluate_vae.py {0}'.format(data_file))
    vae_corr, log_vae_corr = get_results_custom(data_file, pm_file, 'vae_distance_matrix.bin')

    # contrastive
    print('Doing contrastive...')
    os.system('python evaluate_contrastive.py {0}'.format(data_file))
    contrastive_corr, log_contrastive_corr = get_results_custom(data_file, pm_file, 'contrastive_distance_matrix.bin')

    # random projection
    print('Doing random projection...')
    os.system('python evaluate_random_projection.py {0}'.format(data_file))
    rp_corr, log_rp_corr = get_results_custom(data_file, pm_file, 'rp_distance_matrix.bin')

    all_r.append([pca_corr, tsne_corr, umap_corr, mds_corr, ae_corr, vae_corr, contrastive_corr, rp_corr])
    log_all_r.append([log_pca_corr, log_tsne_corr, log_umap_corr, log_mds_corr, log_ae_corr, log_vae_corr, log_contrastive_corr, log_rp_corr])

    # Save all populations for later
    if do_keep_populations:
        os.rename(os.path.join(output_dir, 'simpop_random.bin'), os.path.join(output_dir, 'simpop_random-{0}.bin'.format(i)))
        os.rename(os.path.join(output_dir, 'simpop_random_ped.bin'), os.path.join(output_dir, 'simpop_random_ped-{0}.bin'.format(i)))


# Aggregate results
print('For normal:')
# print('{0} ({1})'.format(np.mean(all_r, axis=0), np.std(all_r, axis=0)))
print([(a, b) for a, b in zip(list(np.mean(all_r, axis=0)), list(np.std(all_r, axis=0)))])

print('For log2:')
# print('{0} ({1})'.format(np.mean(log_all_r, axis=0), np.std(log_all_r, axis=0)))
print([(a, b) for a, b in zip(list(np.mean(log_all_r, axis=0)), list(np.std(log_all_r, axis=0)))])

dump(all_r, 'all_r.bin')
dump(log_all_r, 'log_all_r.bin')
