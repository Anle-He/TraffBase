# analysis

Post-hoc analysis scripts for experiment runs. These are standalone tools (not
part of the `traffbase` package) and read the artifacts produced under `logs/`.

Run them from the repository root.

## aggregate_results.py

Scans `logs/` for the `RESULT | ...` lines emitted at the end of each run,
groups by (model, dataset, horizon), and reports across seeds:

- the parameter count (constant per model/config, shown as a single value);
- mean +/- std of MSE/MAE;
- mean +/- std of the training time (seconds per epoch) and inference time
  (seconds for the test pass).

Older `RESULT` lines without the `params`/`epoch_time`/`infer_time` fields are
still parsed; the missing columns show `-` or `nan`.

```bash
# Print the aggregated table
python analysis/aggregate_results.py

# Use a different log directory and also dump a CSV
python analysis/aggregate_results.py --logs-dir logs --csv results.csv
```

Depends only on the Python standard library.
