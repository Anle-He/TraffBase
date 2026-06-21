# analysis

Post-hoc analysis scripts for experiment runs. These are standalone tools (not
part of the `traffbase` package) and read the artifacts produced under `logs/`.

Run them from the repository root.

## aggregate_results.py

Scans `logs/` for the `RESULT | ...` lines emitted at the end of each run,
groups by (model, dataset, horizon), and reports mean +/- std of MSE/MAE across
seeds.

```bash
# Print the aggregated table
python analysis/aggregate_results.py

# Use a different log directory and also dump a CSV
python analysis/aggregate_results.py --logs-dir logs --csv results.csv
```

Depends only on the Python standard library.
