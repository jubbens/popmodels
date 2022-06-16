from joblib import load
import numpy as np
from scipy.stats import pearsonr
import sys


# def get_pairs(dm, pm, dm_labels, pm_labels):
#     ret_d = []
#     ret_p = []
#
#     for i, l1 in enumerate(dm_labels):
#         for j, l2 in enumerate(dm_labels):
#             if l1 in pm_labels and l2 in pm_labels:
#                 ind_1 = pm_labels.index(l1)
#                 ind_2 = pm_labels.index(l2)
#
#                 p_dist = pm[ind_1][ind_2]
#
#                 if not np.isnan(p_dist) and p_dist > 0.:
#                     d_dist = dm[i][j]
#
#                     # print('{0} - {1}'.format(p_dist, d_dist))
#                     ret_d.append(d_dist)
#                     ret_p.append(p_dist)
#
#     return np.array(ret_d), np.array(ret_p)

def get_results_custom(data_file, pm_file, dm_file):
    ped = load(pm_file)
    data = load(data_file)
    x = np.array([d[1] for d in data])
    dm_labels = [d[0] for d in data]

    pm_labels = ped[0]
    pm = np.array(ped[1])

    print('Doing provided distance matrix...')
    dm = load(dm_file)
    # ret_d, ret_p = get_pairs(dm, pm, dm_labels, pm_labels)
    # mask = pm > 0.
    mask = np.where(~np.eye(pm.shape[0], dtype=np.bool))
    ret_d = dm[mask].flatten()
    ret_p = pm[mask].flatten()

    corr, _ = pearsonr(ret_d, ret_p)
    print('Provided distance matrix:')
    print(corr)

    ret_d[ret_d > 0.] = np.log2(ret_d[ret_d > 0.])
    log_corr, _ = pearsonr(ret_d, ret_p)
    print('Provided distance matrix (log2):')
    print(log_corr)

    return corr, log_corr


if __name__ == "__main__":
    data_file = sys.argv[1]
    pm_file = sys.argv[2]
    dm_file = sys.argv[3]

    get_results_custom(data_file, pm_file, dm_file)
