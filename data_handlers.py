import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing


def build_torch_dataloader(samples, labels=None, batch_size=16, drop_last=False, shuffle=True):
    if samples is None or len(samples) == 0:
        return None

    if torch.is_tensor(samples):
        data = samples
    else:
        data = torch.tensor(samples, dtype=torch.float32)

    if labels is not None:
        le = preprocessing.LabelEncoder()
        transformed_labels = torch.tensor(le.fit_transform(labels), dtype=torch.int32)
        dataset = TensorDataset(data, transformed_labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, num_workers=8)
    else:
        data = torch.tensor(samples, dtype=torch.float32)
        dataloader = DataLoader(data, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, num_workers=8)

    return dataloader
