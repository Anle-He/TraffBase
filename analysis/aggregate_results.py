"""Aggregate the machine-readable ``RESULT`` lines emitted into ``logs/``.

Each run appends one line of the form::

    RESULT | model=SMamba dataset=PEMS08 horizon=96 seed=2024 config_id=1a2b3c4d5e6f params=1234567 epoch_time=12.345 infer_time=1.234 mse=0.20000 mae=0.30000

This script scans a log directory for those lines, groups them by
(model, dataset, horizon, config_id), keeps only the latest record for a repeated
seed, and reports the mean +/- std of each metric across seeds. Optionally writes
the aggregated table to CSV.

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
GROUP_KEYS = ('model', 'dataset', 'horizon', 'config_id')
# Error metrics (5 decimals) and wall-clock timings (seconds, 3 decimals). Both
# are averaged with mean +/- std across seeds; they differ only in formatting.
ERROR_METRICS = ('mse', 'mae')
TIME_METRICS = ('epoch_time', 'infer_time')
METRICS = ERROR_METRICS + TIME_METRICS


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
    groups: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for record_index, record in enumerate(records):
        key = tuple(record.get(k, '') for k in GROUP_KEYS)
        # Logs are collected in timestamp-sorted path order, so assigning by seed
        # makes a later rerun replace an earlier result for the same configuration.
        seed_key = record.get('seed') or f'__record_{record_index}'
        groups[key][seed_key] = record

    rows: list[dict[str, Any]] = []
    for key, records_by_seed in groups.items():
        group = list(records_by_seed.values())
        row: dict[str, Any] = dict(zip(GROUP_KEYS, key))
        row['seeds'] = len(group)

        # Parameter count is fixed by the model/config, so it is constant across
        # seeds; report the single value (blank if any seed disagrees).
        param_values = {r['params'] for r in group if 'params' in r}
        row['params'] = param_values.pop() if len(param_values) == 1 else ''

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
        return (row['model'], row['dataset'], horizon_num, row['config_id'])

    return sorted(rows, key=sort_key)


def print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print('No RESULT lines found.')
        return

    # (title, width) for each column. The four aggregated columns share a width
    # so the mean +/- std values line up under their headers.
    AGG_W = 22
    columns = [
        ('model', 12, '<'),
        ('dataset', 10, '<'),
        ('horizon', 8, '>'),
        ('config_id', 14, '>'),
        ('seeds', 7, '>'),
        ('params', 14, '>'),
        ('MSE (mean+/-std)', AGG_W, '>'),
        ('MAE (mean+/-std)', AGG_W, '>'),
        ('Train s/epoch', AGG_W, '>'),
        ('Infer s', AGG_W, '>'),
    ]

    header = ''.join(f'{title:{align}{width}}' for title, width, align in columns)
    print(header)
    print('-' * len(header))

    def fmt_agg(mean: float, std: float, decimals: int) -> str:
        return f'{mean:.{decimals}f} +/- {std:.{decimals}f}'

    for row in rows:
        params = f'{int(row["params"]):,}' if row['params'] != '' else '-'
        values = [
            row['model'],
            row['dataset'],
            row['horizon'],
            row['config_id'] or '-',
            row['seeds'],
            params,
            fmt_agg(row['mse_mean'], row['mse_std'], 5),
            fmt_agg(row['mae_mean'], row['mae_std'], 5),
            fmt_agg(row['epoch_time_mean'], row['epoch_time_std'], 3),
            fmt_agg(row['infer_time_mean'], row['infer_time_std'], 3),
        ]
        print(''.join(
            f'{value:{align}{width}}'
            for value, (_, width, align) in zip(values, columns)
        ))


def write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    fieldnames = [
        'model',
        'dataset',
        'horizon',
        'config_id',
        'seeds',
        'params',
        'mse_mean',
        'mse_std',
        'mae_mean',
        'mae_std',
        'epoch_time_mean',
        'epoch_time_std',
        'infer_time_mean',
        'infer_time_std',
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
