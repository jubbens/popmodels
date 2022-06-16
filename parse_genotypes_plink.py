from plinkio import plinkfile
import numpy as np
from joblib import dump
from tqdm import tqdm
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

plink_file = plinkfile.open(input_file)

sample_list = [s.iid for s in plink_file.get_samples()]
print(len(sample_list))
locus_list = plink_file.get_loci( )
print(len(plink_file.get_loci()))

all_labels = [s.iid for s in plink_file.get_samples()]
print(len(all_labels))

# Combine
all_genos = []
ret = []

for i, row in enumerate(tqdm(plink_file)):
    all_genos.append(np.expand_dims(np.array(row), axis=1))

all_genos = np.concatenate(all_genos, axis=1)
print(all_genos.shape)

for i, label in enumerate(all_labels):
    ret.append([label, np.squeeze(all_genos[i])])

dump(ret, output_file)
