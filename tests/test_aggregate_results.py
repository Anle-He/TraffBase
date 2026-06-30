import unittest

from analysis.aggregate_results import aggregate


def _record(config_id: str, seed: str, mse: str) -> dict[str, str]:
    return {
        'model': 'SMamba',
        'dataset': 'BJ500',
        'horizon': '96',
        'config_id': config_id,
        'seed': seed,
        'params': '100',
        'mse': mse,
        'mae': mse,
        'epoch_time': '1',
        'infer_time': '1',
    }


class AggregateResultsTests(unittest.TestCase):
    def test_different_configs_are_not_averaged_together(self) -> None:
        rows = aggregate([
            _record('config-a', '2024', '1.0'),
            _record('config-b', '2024', '3.0'),
        ])

        self.assertEqual(len(rows), 2)
        self.assertEqual([row['config_id'] for row in rows], ['config-a', 'config-b'])
        self.assertEqual([row['mse_mean'] for row in rows], [1.0, 3.0])

    def test_latest_duplicate_seed_replaces_earlier_result(self) -> None:
        rows = aggregate([
            _record('config-a', '2024', '1.0'),
            _record('config-a', '2024', '3.0'),
            _record('config-a', '2025', '5.0'),
        ])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['seeds'], 2)
        self.assertEqual(rows[0]['mse_mean'], 4.0)


if __name__ == '__main__':
    unittest.main()
