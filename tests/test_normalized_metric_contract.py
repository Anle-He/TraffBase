import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NormalizedMetricContractTests(unittest.TestCase):
    def test_standard_scaler_has_no_inverse_transform_entrypoint(self) -> None:
        utils_path = ROOT / 'traffbase' / 'utils.py'
        module = ast.parse(utils_path.read_text(encoding='utf-8'))
        scaler = next(
            node
            for node in module.body
            if isinstance(node, ast.ClassDef) and node.name == 'StandardScaler'
        )
        method_names = {
            node.name
            for node in scaler.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        self.assertNotIn('inverse_transform', method_names)


if __name__ == '__main__':
    unittest.main()
