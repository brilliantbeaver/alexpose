"""Environment policy tests; no study packages, CUDA or network are required."""
from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/research_directions/synthetic_training/check_environment.py"
SPEC = importlib.util.spec_from_file_location("synthetic_training_environment_check", SCRIPT)
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class FakeDistribution:
    def __init__(self, direct_url=None, root=Path("/opt/study/site-packages")):
        self.direct_url, self.root = direct_url, root

    def read_text(self, name):
        return None if self.direct_url is None else json.dumps(self.direct_url)

    def locate_file(self, name):
        return self.root / name


class EnvironmentCheckTests(unittest.TestCase):
    def setUp(self):
        self.contract = checker.read_contract()

    def good_checks(self):
        """Replace native probes only; retain version, provenance and flow rules."""
        stack = ExitStack()
        stack.enter_context(patch.object(checker.sys, "version_info", (3, 11, 12)))
        stack.enter_context(patch.object(checker.sys, "prefix", "/opt/study"))
        stack.enter_context(patch.object(checker.platform, "system", return_value="Linux"))
        stack.enter_context(patch.object(checker.platform, "machine", return_value="x86_64"))
        stack.enter_context(patch.object(checker.metadata, "version", side_effect=self.contract["versions"].__getitem__))

        def distribution(name):
            source = self.contract["git_sources"].get(name)
            direct = None if source is None else {"url": source["git"], "vcs_info": {"vcs": "git", "commit_id": source["rev"]}}
            return FakeDistribution(direct)

        stack.enter_context(patch.object(checker.metadata, "distribution", side_effect=distribution))
        stack.enter_context(patch.object(checker.importlib, "import_module", side_effect=lambda name:
                                        SimpleNamespace(__file__=f"/opt/study/site-packages/{name}/__init__.py")))
        for name in ("_check_torch_cpu", "_check_pose_imports", "_check_chumpy", "_check_body_model", "_check_notebook_imports"):
            stack.enter_context(patch.object(checker, name, return_value="fixture probe passed"))
        stack.enter_context(patch.object(checker, "_check_cuda", return_value={"available": False, "exercised": False, "detail": "login node"}))
        return stack

    def test_contract_records_hashes_and_exact_cuda_and_git_pins(self):
        self.assertEqual(self.contract["versions"]["torch"], "2.1.0+cu121")
        self.assertEqual(self.contract["versions"]["torchvision"], "0.16.0+cu121")
        self.assertEqual(self.contract["versions"]["chumpy"], "0.71")
        self.assertEqual(set(self.contract["git_sources"]), {"mmpose", "chumpy", "human-body-prior"})
        for key in ("manifest_sha256", "lock_sha256"):
            self.assertRegex(self.contract[key], r"^[0-9a-f]{64}$")

    def test_wrong_platform_or_version_prevents_native_imports(self):
        with self.good_checks(), patch.object(checker.platform, "system", return_value="Darwin"), \
                patch.object(checker.importlib, "import_module") as importer:
            result = checker.run_checks()
        self.assertFalse(result["passed"])
        importer.assert_not_called()
        self.assertEqual(result["cuda"], {"available": None, "exercised": False})
        self.assertTrue(any(c["name"] == "platform" and c["status"] == "failed" for c in result["checks"]))

    def test_cpu_torch_build_cannot_pass_cuda_wheel_contract(self):
        versions = dict(self.contract["versions"], torch="2.1.0")
        with self.good_checks(), patch.object(checker.metadata, "version", side_effect=versions.__getitem__):
            result = checker.run_checks()
        failure = next(c for c in result["checks"] if c["name"] == "version:torch")
        self.assertEqual(failure["status"], "failed")
        self.assertIn("required 2.1.0+cu121", failure["detail"])

    def test_login_node_without_gpu_can_pass_but_gpu_is_not_certified(self):
        with self.good_checks():
            result = checker.run_checks()
        self.assertTrue(result["passed"])
        self.assertFalse(result["cuda"]["available"])
        self.assertFalse(result["cuda"]["exercised"])
        self.assertIn("full experiment execution", result["not_checked"])

    def test_native_failure_is_retained_while_other_probes_continue(self):
        with self.good_checks(), patch.object(checker, "_check_torch_cpu", side_effect=OSError("libcudart missing")), \
                patch.object(checker, "_check_chumpy", return_value="passed later") as later:
            result = checker.run_checks()
        self.assertFalse(result["passed"])
        later.assert_called_once()
        self.assertTrue(any("libcudart missing" in c["detail"] for c in result["checks"] if isinstance(c["detail"], str)))

    def test_cuda_requirement_fails_when_driver_has_no_device(self):
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.object(checker.importlib, "import_module", return_value=fake_torch):
            self.assertEqual(checker._check_cuda(False)["available"], False)
            with self.assertRaisesRegex(RuntimeError, "inside a GPU allocation"):
                checker._check_cuda(True)
        with self.good_checks(), patch.object(checker, "_check_cuda", side_effect=RuntimeError("CUDA is unavailable")) as cuda:
            result = checker.run_checks(require_cuda=True)
        cuda.assert_called_once_with(True)
        self.assertFalse(result["passed"])
        self.assertTrue(result["require_cuda"])

    def test_git_commit_url_and_editable_mismatches_are_rejected(self):
        source = self.contract["git_sources"]["chumpy"]
        good = {"url": source["git"], "vcs_info": {"vcs": "git", "commit_id": source["rev"]}}
        self.assertIn(source["rev"], checker.check_git_source(FakeDistribution(good), source))
        bad_cases = [None, {**good, "url": "https://example.org/chumpy.git"},
                     {**good, "vcs_info": {"vcs": "git", "commit_id": "0" * 40}},
                     {**good, "dir_info": {"editable": True}}]
        for direct in bad_cases:
            with self.subTest(direct=direct), self.assertRaises(ValueError):
                checker.check_git_source(FakeDistribution(direct), source)

    def test_shadowed_runtime_package_is_rejected(self):
        distribution = FakeDistribution()
        installed = SimpleNamespace(__file__="/opt/study/site-packages/chumpy/__init__.py")
        with patch.object(checker.sys, "prefix", "/opt/study"):
            checker.check_import_path(installed, distribution, "chumpy")
            for path in ("/tmp/chumpy/__init__.py", "/opt/study/site-packages/chumpy_extra/__init__.py"):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, "import shadowed"):
                    checker.check_import_path(SimpleNamespace(__file__=path), distribution, "chumpy")

    def test_shadowing_package_and_its_metadata_cannot_escape_interpreter_environment(self):
        external = FakeDistribution(root=Path("/opt/other/site-packages"))
        module = SimpleNamespace(__file__="/opt/other/site-packages/chumpy/__init__.py")
        with patch.object(checker.sys, "prefix", "/opt/study"), self.assertRaisesRegex(ValueError, "outside this environment"):
            checker.check_import_path(module, external, "chumpy")

    def test_body_model_api_requires_explicit_dmpl_support(self):
        def supported(bm_fname, dmpl_fname, num_betas=16, num_dmpls=8):
            pass

        def unsupported(model_path, **kwargs):
            pass

        self.assertIn("licensed assets were not loaded", checker.check_body_model_api(supported))
        with self.assertRaisesRegex(ValueError, "dmpl_fname"):
            checker.check_body_model_api(unsupported)

    def test_cli_exit_and_atomic_json_preserve_failed_result(self):
        report = {"passed": False, "checks": [{"name": "probe", "status": "failed", "detail": "native error"}],
                  "not_checked": ["EGL"], "cuda": {"exercised": False}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checks" / "environment.json"
            with patch.object(checker, "run_checks", return_value=report), redirect_stdout(io.StringIO()):
                self.assertEqual(checker.main(["--json-output", str(path)]), 1)
            self.assertEqual(json.loads(path.read_text()), report)
            self.assertEqual(list(path.parent.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
