import pandas as pd
import numpy as np
from joblib import dump
from tqdm import tqdm
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

raw = pd.read_csv(input_file)
all_labels = list(raw['1'])
ret = []

print(len(all_labels))

for i, label in tqdm(enumerate(all_labels)):
    col = raw.iloc[i].tolist()

    tag = col.pop(0)
    assert tag == label

    col = np.array(col)
    col[col == -1.] = 3

    ret.append([label, col])

dump(ret, output_file)
