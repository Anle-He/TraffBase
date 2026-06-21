"""Aggregate the machine-readable ``RESULT`` lines emitted into ``logs/``.

Each run appends one line of the form::

    RESULT | model=SMamba dataset=PEMS08 horizon=96 seed=2024 params=1234567 mse=0.20000 mae=0.30000

This script scans a log directory for those lines, groups them by
(model, dataset, horizon), and reports the mean +/- std of each metric across
seeds. Optionally writes the aggregated table to CSV.

Usage (from the repository root)::

    python analysis/aggregate_results.py
    python analysis/aggregate_results.py --logs-dir logs --csv results.csv
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

RESULT_PREFIX = 'RESULT |'
GROUP_KEYS = ('model', 'dataset', 'horizon')
METRICS = ('mse', 'mae')


def parse_result_line(line: str) -> dict[str, str] | None:
    line = line.strip()
    if not line.startswith(RESULT_PREFIX):
        return None

    payload = line[len(RESULT_PREFIX):].strip()
    fields: dict[str, str] = {}
    for token in payload.split():
        if '=' not in token:
            continue
        key, value = token.split('=', 1)
        fields[key] = value

    return fields


def collect_results(logs_dir: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for log_file in sorted(logs_dir.rglob('*.log')):
        with log_file.open('r', encoding='utf-8') as f:
            for line in f:
                parsed = parse_result_line(line)
                if parsed is not None:
                    records.append(parsed)

    return records


def aggregate(records: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        key = tuple(record.get(k, '') for k in GROUP_KEYS)
        groups[key].append(record)

    rows: list[dict[str, Any]] = []
    for key, group in groups.items():
        row: dict[str, Any] = dict(zip(GROUP_KEYS, key))
        row['seeds'] = len(group)

        for metric in METRICS:
            values = [float(r[metric]) for r in group if metric in r]
            row[f'{metric}_mean'] = statistics.fmean(values) if values else float('nan')
            row[f'{metric}_std'] = (
                statistics.pstdev(values) if len(values) > 1 else 0.0
            )

        rows.append(row)

    # Sort by model, dataset, then numeric horizon when possible.
    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        horizon = row['horizon']
        horizon_num = int(horizon) if horizon.isdigit() else 0
        return (row['model'], row['dataset'], horizon_num)

    return sorted(rows, key=sort_key)


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print('No RESULT lines found.')
        return

    header = (
        f'{"model":<12}{"dataset":<10}{"horizon":>8}{"seeds":>7}'
        f'{"MSE (mean+/-std)":>24}{"MAE (mean+/-std)":>24}'
    )
    print(header)
    print('-' * len(header))
    for row in rows:
        mse = f'{row["mse_mean"]:.5f} +/- {row["mse_std"]:.5f}'
        mae = f'{row["mae_mean"]:.5f} +/- {row["mae_std"]:.5f}'
        print(
            f'{row["model"]:<12}{row["dataset"]:<10}{row["horizon"]:>8}'
            f'{row["seeds"]:>7}{mse:>24}{mae:>24}'
        )


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        'model',
        'dataset',
        'horizon',
        'seeds',
        'mse_mean',
        'mse_std',
        'mae_mean',
        'mae_std',
    ]
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--logs-dir',
        type=str,
        default='logs',
        help='Directory to scan recursively for *.log files (default: logs).',
    )
    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Optional path to write the aggregated table as CSV.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        raise SystemExit(f'Logs directory not found: {logs_dir}')

    records = collect_results(logs_dir)
    rows = aggregate(records)
    print_table(rows)

    if args.csv is not None:
        csv_path = Path(args.csv)
        write_csv(rows, csv_path)
        print(f'\nWrote {len(rows)} rows to {csv_path}')


if __name__ == '__main__':
    main()
