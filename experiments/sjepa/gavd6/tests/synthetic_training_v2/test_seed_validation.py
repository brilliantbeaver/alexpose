"""Reject unusable random seeds before preparing data or scheduling a fit."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig


class SeedValidationTests(unittest.TestCase):
    @staticmethod
    def config(**values):
        return RunConfig(run_id='seed-check', mode='source', bundle='future-bundle', device='cuda', **values)

    def test_primary_and_training_seeds_reject_invalid_numbers_and_types(self):
        for seed in (-1, 2 ** 32, True, False, 17., '17', None, float('nan'), float('inf')):
            for field in ('seed', 'seeds'):
                with self.subTest(field=field, seed=seed), self.assertRaisesRegex(ValueError, '[Ss]eed'):
                    self.config(**{field: seed if field == 'seed' else [seed]})

    def test_training_seeds_require_nonempty_unique_ordered_integers(self):
        for seeds in ([], (), [17, 17], '17', 17, {17}, [[17]], None):
            with self.subTest(seeds=seeds), self.assertRaisesRegex(ValueError, '[Ss]eed'):
                self.config(seeds=seeds)

    def test_supported_boundaries_and_json_roundtrip_preserve_integer_values(self):
        for seed in (0, 17, 2 ** 32 - 1):
            original = self.config(seed=seed, seeds=[seed])
            with self.subTest(seed=seed), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'config.json'
                path.write_text(json.dumps(original.as_dict()))
                loaded = RunConfig.load(path)
                self.assertEqual(loaded.seed, seed)
                self.assertEqual(loaded.seeds, [seed])
                self.assertIs(type(loaded.seed), int)

    def test_managed_initialization_rejects_negative_seed_before_files_or_assets(self):
        scripts = Path(__file__).resolve().parents[2] / 'scripts/research_directions/synthetic_training_v2'
        with patch.object(sys, 'path', [str(scripts), *sys.path]):
            spec = importlib.util.spec_from_file_location('stv2_seed_validation_manager', scripts / 'haic.py')
            manager = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(manager)
        arguments = SimpleNamespace(name='negative-seed', gpu_hours=4., prior_gpu_hours=0.,
                                    prior_ledger=None, updates=200, seeds=[-1])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(manager, 'ROOT', root), \
                 patch.object(manager, 'environment_ok', return_value=True), \
                 patch.object(manager, 'preparation_config', return_value={}) as assets, \
                 patch.object(manager, 'create_drafts') as drafts:
                with self.assertRaisesRegex(ValueError, '[Ss]eed'):
                    manager.initialize(arguments, {'USER': 'fixture'})
                assets.assert_not_called()
                drafts.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
