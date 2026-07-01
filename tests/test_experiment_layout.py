import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / 'traffbase' / 'models'
SCRIPTS_ROOT = ROOT / 'scripts'


class ExperimentConfigLayoutTests(unittest.TestCase):
    def test_one_base_config_per_model_dataset(self) -> None:
        configs = sorted(CONFIG_ROOT.glob('*/configs/*.yaml'))

        self.assertTrue(configs)
        self.assertFalse(
            any('_IN' in path.stem or '_OUT' in path.stem for path in configs)
        )

        for path in configs:
            with self.subTest(config=path.relative_to(ROOT)):
                cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
                self.assertEqual(cfg['DATA']['out_steps'], 12)
                self.assertEqual(cfg['GENERAL']['runner'], 'LTSFTrainer')
                self.assertEqual(
                    cfg['OPTIM']['lr_scheduler_type'],
                    'ExponentialLR',
                )

    def test_dataset_launchers_resolve_existing_base_configs(self) -> None:
        launchers = sorted(
            path
            for dataset in ('BJ500', 'PEMS08')
            for path in (SCRIPTS_ROOT / dataset).glob('*.sh')
        )
        invocation = re.compile(
            r'run_grid\.sh"\s+\'(?P<model>[^\']+)\'\s+\'(?P<dataset>[^\']+)\''
        )

        self.assertTrue(launchers)
        for path in launchers:
            with self.subTest(launcher=path.relative_to(ROOT)):
                text = path.read_text(encoding='utf-8')
                match = invocation.search(text)
                self.assertIsNotNone(match)
                assert match is not None
                dataset = match.group('dataset')
                config = (
                    CONFIG_ROOT
                    / match.group('model')
                    / 'configs'
                    / f'{dataset}.yaml'
                )
                self.assertTrue(config.is_file(), config)

    def test_grid_runner_owns_horizon_override(self) -> None:
        text = (SCRIPTS_ROOT / 'run_grid.sh').read_text(encoding='utf-8')
        smamba = (SCRIPTS_ROOT / 'BJ500' / 'smamba.sh').read_text(
            encoding='utf-8'
        )

        self.assertIn('python -u -m traffbase.main', text)
        self.assertIn('-o "DATA.out_steps=$HORIZON"', text)
        self.assertIn("HORIZONS='48 96'", smamba)
