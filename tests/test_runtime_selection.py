import unittest

from traffbase.main import _validate_runtime_selection


class RuntimeSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = {
            'GENERAL': {'runner': 'LTSFTrainer'},
            'OPTIM': {'lr_scheduler_type': 'ExponentialLR'},
        }

    def test_ltsf_compatibility_values_are_accepted(self) -> None:
        _validate_runtime_selection('ltsf', self.cfg)

    def test_unknown_task_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, 'supports only LTSF'):
            _validate_runtime_selection('STF', self.cfg)

    def test_unknown_runner_is_rejected(self) -> None:
        self.cfg['GENERAL']['runner'] = 'OtherTrainer'
        with self.assertRaisesRegex(ValueError, 'supports only LTSFTrainer'):
            _validate_runtime_selection('LTSF', self.cfg)

    def test_unknown_scheduler_is_rejected(self) -> None:
        self.cfg['OPTIM']['lr_scheduler_type'] = 'OneCycleLR'
        with self.assertRaisesRegex(ValueError, 'supports only ExponentialLR'):
            _validate_runtime_selection('LTSF', self.cfg)
