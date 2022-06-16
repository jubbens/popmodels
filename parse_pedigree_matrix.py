import pandas as pd
from joblib import dump, load
import numpy as np
from tqdm import tqdm
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]
data_path = sys.argv[3]

id_key = -999
# id_key = 'Unnamed: 0'
data = load(data_path)
dm_labels = [d[0] for d in data]
raw = pd.read_csv(input_path, low_memory=False)

pm_labels = list(raw['Unnamed: 0'])

# Convert columns to integers
raw.columns.values[0] = id_key
raw.columns = raw.columns.astype(int)

# Prune rows
raw = raw[raw[id_key].isin(dm_labels)]

# Prune columns
raw = raw[raw.columns.intersection(dm_labels + [id_key])]

print(len(dm_labels))
print(np.array(raw).shape)

# Find all data points absent from the ped matrix:
absent = [x for x in dm_labels if x not in pm_labels]
print('Absent:')
print(absent)

all_rows = []

# Yes this is inefficient as sin, but it provides the fewest opportunities for silent errors
for l1 in tqdm(dm_labels):
        row = []

        for l2 in dm_labels:
            if l1 in absent or l2 in absent:
                row.append(np.nan)
            else:
                entry = raw.loc[raw[id_key] == l1][l2]
                row.append(np.array(entry)[0])

        all_rows.append(row)

ret = np.array(all_rows)
print(ret.shape)

n = [dm_labels, ret]
dump(n, output_path)
