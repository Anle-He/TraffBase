"""Generate LTSF processed data and sliding-window indices for a dataset.

Run from the repository root, e.g.::

    python traffbase/data/generate_LTSF_data.py --dataset PEMS08

This reads the raw series from ``traffbase/data/raw_data/<DATASET>/`` and writes
``processed_data.npz`` plus one ``index_in{IN}_out{OUT}.npz`` per horizon into
``traffbase/data/datasets/<DATASET>/`` (the directory consumed by the dataloader).
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATA_ROOT = Path('traffbase/data')
RAW_DIR = DATA_ROOT / 'raw_data'
OUT_DIR = DATA_ROOT / 'datasets'

TARGET_CHANNEL = 0  # Target traffic channel
STEPS_PER_DAY = 288
DAYS_PER_WEEK = 7

# Per-dataset settings. The key is the output directory name (matches datasets/).
DATASETS: dict[str, dict[str, Any]] = {
    'BJ500': {'raw': 'BJ500/BJ500.csv', 'ratio': (0.7, 0.1, 0.2), 'domain': 'traffic speed'},
    'PEMS03': {'raw': 'PeMS03/PeMS03.npz', 'ratio': (0.6, 0.2, 0.2), 'domain': 'traffic flow'},
    'PEMS04': {'raw': 'PeMS04/PeMS04.npz', 'ratio': (0.6, 0.2, 0.2), 'domain': 'traffic flow'},
    'PEMS07': {'raw': 'PeMS07/PeMS07.npz', 'ratio': (0.6, 0.2, 0.2), 'domain': 'traffic flow'},
    'PEMS08': {'raw': 'PeMS08/PeMS08.npz', 'ratio': (0.6, 0.2, 0.2), 'domain': 'traffic flow'},
}


def load_raw(raw_path: Path) -> tuple[np.ndarray, pd.DataFrame | None, str]:
    """Load the raw series as ``[all_steps, num_nodes, 1]`` plus its timestamp frame."""
    suffix = raw_path.suffix
    if suffix == '.h5':
        df = pd.read_hdf(raw_path)
        return np.expand_dims(df.values, axis=-1), df, 'hdf'
    if suffix == '.npz':
        return np.load(raw_path)['data'], None, 'npz'
    if suffix == '.csv':
        df = pd.read_csv(raw_path)
        df.index = pd.to_datetime(df['date'].values, format='%Y-%m-%d %H:%M:%S')
        df = df[df.columns[1:]]
        return np.expand_dims(df.values, axis=-1), df, 'csv'
    raise ValueError(f'Unsupported raw data file type: {suffix}')


def build_time_features(
    num_steps: int, num_nodes: int, df: pd.DataFrame | None, file_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return time-of-day and day-of-week features, each shaped ``[all_steps, num_nodes, 1]``."""
    if file_type in ('hdf', 'csv'):
        assert df is not None
        tod = (df.index.values - df.index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        dow = df.index.dayofweek.to_numpy()
    else:
        tod = np.array([i % STEPS_PER_DAY / STEPS_PER_DAY for i in range(num_steps)])
        dow = np.array([(i // STEPS_PER_DAY) % DAYS_PER_WEEK for i in range(num_steps)])

    tod_tiled = np.tile(tod, [1, num_nodes, 1]).transpose((2, 1, 0))
    dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
    return tod_tiled, dow_tiled


def build_indices(
    split: str, num_steps: int, in_steps: int, out_steps: int, ratio: tuple[float, float, float]
) -> tuple[list[tuple[int, int, int]], ...]:
    """Build (start, mid, end) sliding-window indices for train/val/test under ``split``."""
    train_ratio, val_ratio, _ = ratio
    L = num_steps

    if split in ('DEFAULT', 'STF'):
        # First sliding window, then split: crosses set boundaries, yields most samples.
        num_samples = L - (in_steps + out_steps) + 1
        train_num = round(num_samples * train_ratio)
        val_num = round(num_samples * val_ratio)
        test_num = num_samples - train_num - val_num

        index_list = [
            (t - in_steps, t, t + out_steps) for t in range(in_steps, num_samples + in_steps)
        ]
        train_index = index_list[:train_num]
        val_index = index_list[train_num:train_num + val_num]
        test_index = index_list[train_num + val_num:train_num + val_num + test_num]

    elif split == 'STRICT':
        # First split, then slide within each set: no overlap, yields fewest samples.
        split1 = round(L * train_ratio)
        split2 = round(L * (train_ratio + val_ratio))
        train_index = [
            (t - in_steps, t, t + out_steps) for t in range(in_steps, split1 - out_steps + 1)
        ]
        val_index = [
            (t - in_steps, t, t + out_steps) for t in range(split1 + in_steps, split2 - out_steps + 1)
        ]
        test_index = [
            (t - in_steps, t, t + out_steps) for t in range(split2 + in_steps, L - out_steps + 1)
        ]

    elif split == 'LTSF':
        # LTSF-Linear convention: strict train, val overlaps train, test overlaps val.
        # See https://github.com/cure-lab/LTSF-Linear/blob/main/data_provider/data_loader.py#L238
        # Changing in_steps does not affect the number of val/test samples.
        test_ratio = 1 - train_ratio - val_ratio
        split1 = int(L * train_ratio)
        split2 = L - int(L * test_ratio)
        train_index = [
            (t - in_steps, t, t + out_steps) for t in range(in_steps, split1 - out_steps + 1)
        ]
        val_index = [
            (t - in_steps, t, t + out_steps) for t in range(split1, split2 - out_steps + 1)
        ]
        test_index = [
            (t - in_steps, t, t + out_steps) for t in range(split2, L - out_steps + 1)
        ]

    else:
        raise ValueError(f'Unknown split: {split}')

    return train_index, val_index, test_index


def generate(
    dataset: str,
    in_steps: int,
    out_steps_list: list[int],
    split: str,
    force: bool,
) -> None:
    settings = DATASETS[dataset]
    raw_path = RAW_DIR / settings['raw']
    out_dir = OUT_DIR / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    data, df, file_type = load_raw(raw_path)
    data = data[..., [TARGET_CHANNEL]]  # [all_steps, num_nodes, 1]
    print(f'Raw {settings["domain"]} series shape: {data.shape}')

    num_steps, num_nodes, _ = data.shape
    tod, dow = build_time_features(num_steps, num_nodes, df, file_type)
    processed_data = np.concatenate([data, tod, dow], axis=-1)
    print(f'Processed data shape (series + tod + dow): {processed_data.shape}')

    data_file = out_dir / 'processed_data.npz'
    if data_file.is_file() and not force:
        print(f'{data_file} exists; skipping (use --force to overwrite).')
    else:
        np.savez_compressed(data_file, data=processed_data)
        print(f'Wrote {data_file}')

    for out_steps in out_steps_list:
        index_file = out_dir / f'index_in{in_steps}_out{out_steps}.npz'
        if index_file.is_file() and not force:
            print(f'{index_file} exists; skipping (use --force to overwrite).')
            continue

        train_index, val_index, test_index = build_indices(
            split, num_steps, in_steps, out_steps, settings['ratio']
        )
        print(
            f'  out={out_steps}: '
            f'train={len(train_index)} val={len(val_index)} test={len(test_index)}'
        )
        np.savez_compressed(
            index_file, train=train_index, val=val_index, test=test_index
        )
        print(f'Wrote {index_file}')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dataset', required=True, choices=sorted(DATASETS), help='Dataset to process.'
    )
    parser.add_argument('--in-steps', type=int, default=96, help='Input window length.')
    parser.add_argument(
        '--out-steps', type=int, nargs='+', default=[12, 24, 48, 96],
        help='Forecast horizons; one index file is written per value.',
    )
    parser.add_argument(
        '--split', default='LTSF', choices=['LTSF', 'STRICT', 'DEFAULT', 'STF'],
        help='Train/val/test windowing strategy.',
    )
    parser.add_argument(
        '--force', action='store_true', help='Overwrite existing output files.'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f'Generating LTSF data for {args.dataset}...')
    generate(args.dataset, args.in_steps, args.out_steps, args.split, args.force)


if __name__ == '__main__':
    main()
