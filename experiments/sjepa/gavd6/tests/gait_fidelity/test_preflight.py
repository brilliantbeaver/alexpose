"""Login-node preflight must not initialize a desktop OpenGL dependency."""
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from gavd6_sjepa.research_directions.gait_fidelity.config import preflight


class PreflightTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        self.cfg = dict(fixture=False, data=dict(source_selection='legacy_roster'),
            preparation={key: str(root) for key in (
                'manifest_dir', 'amass_root', 'body_model_root', 'dmpl_root',
                'uv_path', 'texture_dir', 'background_dir', 'locomotion_audit',
                'reservation_csv')})
        self.cfg['preparation']['estimators'] = []
        self.imported = []
        stack = ExitStack()
        self.addCleanup(stack.close)
        prefix = 'gavd6_sjepa.research_directions.gait_fidelity.config.'
        stack.enter_context(patch(prefix + 'shutil.which', return_value='/tools/ffmpeg'))
        self.spec = Mock(return_value=object())
        self.version = Mock(return_value='0.1.45')
        stack.enter_context(patch(prefix + 'importlib', SimpleNamespace(
            import_module=self.import_module, util=SimpleNamespace(find_spec=self.spec),
            metadata=SimpleNamespace(version=self.version))))

    def import_module(self, name):
        self.imported.append(name)
        if name == 'pyrender':
            raise ImportError('Library "GLU" not found.')
        version = {'torch': '2.6.0+cu124', 'torchvision': '0.21.0+cu124'}.get(name, 'test')
        return SimpleNamespace(__version__=version)

    def test_cpu_records_installed_renderer_without_importing_opengl(self):
        result = preflight(self.cfg)
        self.assertEqual(result['status'], 'CPU_PREFLIGHT_PASSED')
        self.assertEqual(result['packages']['pyrender'], '0.1.45')
        self.assertNotIn('pyrender', self.imported)
        self.assertEqual(result['deferred_checks'], ['pyrender_import', 'egl_render'])
        self.spec.assert_called_once_with('pyrender')
        self.version.assert_called_once_with('pyrender')

    def test_missing_renderer_still_fails_cpu_preflight(self):
        self.spec.return_value = None
        with self.assertRaisesRegex(ModuleNotFoundError, 'pyrender'):
            preflight(self.cfg)
        self.version.assert_not_called()

    def test_gpu_preflight_retains_native_library_failure(self):
        with self.assertRaisesRegex(ImportError, 'Library "GLU" not found'):
            preflight(self.cfg, gpu=True)
        self.assertIn('pyrender', self.imported)

    def test_cpu_still_checks_assets_after_deferring_renderer(self):
        self.cfg['preparation']['amass_root'] += '/missing'
        with self.assertRaisesRegex(FileNotFoundError, 'amass_root'):
            preflight(self.cfg)

    def test_fixture_does_not_require_renderer(self):
        result = preflight(dict(fixture=True))
        self.assertEqual(result['status'], 'CPU_PREFLIGHT_PASSED')
        self.assertNotIn('pyrender', self.imported)
        self.spec.assert_not_called()
        self.version.assert_not_called()


if __name__ == '__main__':
    unittest.main()
