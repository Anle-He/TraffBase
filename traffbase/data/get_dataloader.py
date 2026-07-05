import numpy as np
from typing import Any
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from traffbase.utils import StandardScaler
from traffbase.utils import print_log


FEATURE_MAIN = 0
FEATURE_TOD = 1
FEATURE_DOW = 2


def _build_features(tod: bool, dow: bool) -> list[int]:
    features = [FEATURE_MAIN]
    if tod:
        features.append(FEATURE_TOD)
    if dow:
        features.append(FEATURE_DOW)
    return features


class SlidingWindowDataset(Dataset):
    '''Slices (x, y) windows out of the base series on access.

    Materializing every window up front copies the base series roughly
    ``in_steps`` times over (adjacent windows share almost all their data);
    slicing in ``__getitem__`` keeps a single float32 copy of the series in
    memory regardless of the number of samples.
    '''

    def __init__(
        self,
        data: torch.Tensor,
        indices: np.ndarray,
        x_features: list[int],
        y_features: list[int],
    ) -> None:
        self.data = data  # [timesteps, num_nodes, num_features]
        self.indices = indices  # [num_samples, 3] of (start, mid, end)
        self.x_features = torch.tensor(x_features)
        self.y_features = torch.tensor(y_features)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        start, mid, end = (int(v) for v in self.indices[i])
        x = self.data[start:mid].index_select(-1, self.x_features)
        y = self.data[mid:end].index_select(-1, self.y_features)
        return x, y


def _create_dataloaders(
    trainset: Dataset,
    valset: Dataset,
    testset: Dataset,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    # The datasets are in-memory tensor slices; worker processes would only add
    # IPC overhead (and copy the series per worker under spawn), so load in the
    # main process.
    train_loader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        valset, batch_size=batch_size, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def _log_dataset_shapes(
    trainset: SlidingWindowDataset,
    valset: SlidingWindowDataset,
    testset: SlidingWindowDataset,
    log: Any = None,
) -> None:
    for name, dataset in (
        ('Trainset:', trainset),
        ('Valset:', valset),
        ('Testset:', testset),
    ):
        x, y = dataset[0]
        x_shape = (len(dataset), *x.shape)
        y_shape = (len(dataset), *y.shape)
        print_log(f'{name:<10}x-{str(x_shape):<22}y-{y_shape}', log=log)


def build_LTSF_dataloader(
    data_dir: str,
    batch_size: int = 32,
    in_steps: int = 96,
    out_steps: int = 96,
    x_tod: bool = False,
    x_dow: bool = False,
    y_tod: bool = False,
    y_dow: bool = False,
    log: Any = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_path = Path(data_dir)
    data_file = data_path / 'processed_data.npz'
    index_file = data_path / f'index_in{in_steps}_out{out_steps}.npz'

    missing_files = [path for path in (data_file, index_file) if not path.is_file()]
    if missing_files:
        missing = ', '.join(str(path) for path in missing_files)
        raise FileNotFoundError(f'Required dataset files not found: {missing}')

    data = np.load(data_file)['data'].astype(np.float32)
    index = np.load(index_file)

    x_features = _build_features(x_tod, x_dow)
    y_features = _build_features(y_tod, y_dow)

    train_index = index['train']  # [num_samples, 3]
    val_index = index['val']
    test_index = index['test']

    len_train = train_index[-1][1]
    scaler = StandardScaler(
        mean=data[:len_train, :, FEATURE_MAIN].mean(axis=0),
        std=data[:len_train, :, FEATURE_MAIN].std(axis=0),
    )

    data[..., FEATURE_MAIN] = scaler.transform(data[..., FEATURE_MAIN])
    series = torch.from_numpy(data)

    trainset = SlidingWindowDataset(series, train_index, x_features, y_features)
    valset = SlidingWindowDataset(series, val_index, x_features, y_features)
    testset = SlidingWindowDataset(series, test_index, x_features, y_features)

    _log_dataset_shapes(trainset, valset, testset, log)
    print_log('INFO: Using scaled X and Y for LTSF task', log=log)

    return _create_dataloaders(trainset, valset, testset, batch_size)
