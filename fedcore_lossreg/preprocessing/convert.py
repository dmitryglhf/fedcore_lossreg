import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset


def convert_df_to_dl(X_train, y_train: pd.DataFrame,
                     batch_size: int = 128,
                     shuffle: bool = True):
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    train_ds = TensorDataset(X_train, y_train)
    return DataLoader(train_ds, batch_size, shuffle=shuffle)
