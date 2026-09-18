"""Exercise the real setup shell with isolated fake package-manager boundaries.

No real packages are installed. The wrapper runs against a temporary checkout;
fake executables record its requests and emulate installed Python metadata.
The lock helper uses the host's real advisory locks, including contention.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "slurm/synthetic-training"

# Each fake starts in isolated Python mode so an intentionally poisoned
# PYTHONHOME/PYTHONPATH cannot break the test harness before the wrapper clears it.
FAKE_BODY = r'''
import fcntl, json, os, pathlib, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
log = pathlib.Path(os.environ["ST_MOCK_LOG"])

def record(kind, argv):
    with log.open("a") as handle:
        handle.write(json.dumps({"kind": kind, "argv": argv, "env": {
            key: value for key, value in os.environ.items()
            if key.startswith("UV_") or key in {
                "ST_PYTHON", "VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH",
                "PYTHONNOUSERSITE", "HTTPS_PROXY"
            }
        }}) + "\n")

def create_environment():
    root = pathlib.Path(os.environ["UV_PROJECT_ENVIRONMENT"])
    (root / "bin").mkdir(parents=True, exist_ok=True)
    (root / "pyvenv.cfg").write_text("home = synthetic-test-only\n")
    python = pathlib.Path(os.environ["ST_PYTHON"])
    python.write_text(pathlib.Path(os.environ["ST_MOCK_TEMPLATE"]).read_text())
    python.chmod(0o755)
    return root

if name == "uname":
    print(os.environ.get("ST_MOCK_OS", "Linux") if args == ["-s"]
          else os.environ.get("ST_MOCK_ARCH", "x86_64"))
elif name == "realpath":
    print(pathlib.Path(args[-1]).resolve())
elif name == "flock":
    try:
        fcntl.flock(int(args[-1]), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(1)
elif name == "nvcc":
    print(os.environ.get("ST_MOCK_NVCC", "Cuda compilation tools, release 12.4, V12.4.131"))
elif name == "uv":
    record("uv", args)
    if args == ["--version"]:
        print(os.environ.get("ST_MOCK_UV_VERSION", "uv 0.12.15 (test)"))
    elif args[0] == "sync":
        if "--check" in args:
            raise SystemExit(int(os.environ.get("ST_MOCK_SYNC_CHECK_EXIT", "0")))
        root = create_environment()
        if os.environ.get("ST_MOCK_SYNC_EXIT"):
            # An interrupted first sync may leave a usable venv to resume.
            raise SystemExit(int(os.environ["ST_MOCK_SYNC_EXIT"]))
        (root / "installed-packages.json").write_text('{"study": "locked"}\n')
    elif args[:2] == ["pip", "check"]:
        raise SystemExit(int(os.environ.get("ST_MOCK_PIP_CHECK_EXIT", "0")))
    else:
        raise SystemExit("Unexpected uv request: " + repr(args))
elif name in {"python", "python3", "python3.11"}:
    record("python", args)
    if args[0] == "-c":
        # Run the wrapper's actual preflight expressions, substituting only
        # the version/prefix metadata that a real study interpreter supplies.
        sys.version_info = tuple(map(int, os.environ.get("ST_MOCK_PY_VERSION", "3.11.12").split(".")))
        sys.prefix = os.environ.get("ST_MOCK_PREFIX", str(pathlib.Path(sys.argv[0]).parents[1]))
        sys.base_prefix = "/synthetic-test-base-python"
        sys.argv = ["-c", *args[2:]]
        exec(compile(args[1], "<setup-preflight>", "exec"))
    else:
        assert pathlib.Path(args[0]).name == "check_environment.py", args
        if os.environ.get("ST_MOCK_CHECKER_EXIT"):
            raise SystemExit(int(os.environ["ST_MOCK_CHECKER_EXIT"]))
        if "--json-output" in args:
            pathlib.Path(args[args.index("--json-output") + 1]).write_text('{"passed": true}\n')
elif name == "git":
    record("git", args)
    if "rev-parse" in args:
        print(os.environ["ST_MOCK_POSE_REVISION"])
    elif "status" in args:
        print(os.environ.get("ST_MOCK_GIT_STATUS", ""))
    elif args[0] == "clone":
        (pathlib.Path(args[-1]) / ".git").mkdir(parents=True)
    else:
        raise SystemExit("Unexpected git request: " + repr(args))
elif name == "curl":
    record("curl", args)
    pathlib.Path(args[args.index("--output") + 1]).write_bytes(b"test checkpoint\n")
else:
    raise SystemExit("Unexpected fake executable: " + name)
'''


class EnvironmentSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="st-setup-test-")
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name).resolve()
        self.checkout = self.folder / "checkout"
        self.project = self.checkout / "slurm/synthetic-training"
        self.project.mkdir(parents=True)
        for name in ("setup-environment.sh", "download-students.sh", "common.sh", "pyproject.toml", "uv.lock"):
            shutil.copyfile(STUDY / name, self.project / name)
        (self.checkout / "src/gavd6_sjepa").mkdir(parents=True)
        self.root_venv = self.checkout / ".venv"
        self.root_venv.mkdir()
        (self.root_venv / "keep.txt").write_text("root environment remains unchanged\n")
        self.env_root = self.folder / "envs/study"
        self.python = self.env_root / "bin/python"
        self.bin = self.folder / "fake-bin"
        self.bin.mkdir()
        self.log = self.folder / "calls.jsonl"
        self.fake = "#!" + sys.executable + " -I\n" + FAKE_BODY
        self.template = self.folder / "fake-template"
        self.template.write_text(self.fake)
        for name in ("uv", "uname", "realpath", "flock", "git", "curl", "nvcc"):
            path = self.bin / name
            path.write_text(self.fake)
            path.chmod(0o755)
        self.environment = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("UV_", "ST_"))
            and key not in {"PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "GAVD6_ROOT"}
        }
        self.environment.update(
            PATH=str(self.bin) + os.pathsep + os.environ["PATH"],
            ST_PYTHON=str(self.python), GAVD6_ROOT=str(self.checkout),
            CUDA_HOME=str(self.folder), SLURM_JOB_ID="fixture-allocation",
            ST_MOCK_LOG=str(self.log), ST_MOCK_TEMPLATE=str(self.template),
        )
        # Match CUDA_HOME/bin/nvcc without conflating it with the executable PATH.
        (self.folder / "bin").symlink_to(self.bin, target_is_directory=True)

    def calls(self, kind=None):
        rows = [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []
        return [row for row in rows if kind is None or row["kind"] == kind]

    @staticmethod
    def snapshot(folder):
        return {str(path.relative_to(folder)): (
            hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns
        ) for path in folder.rglob("*") if path.is_file()}

    def run_setup(self, *options, overrides=None):
        before = self.snapshot(self.checkout)
        result = subprocess.run(
            ["bash", str(self.project / "setup-environment.sh"), *options],
            cwd=self.folder, env={**self.environment, **(overrides or {})},
            text=True, capture_output=True, timeout=20,
        )
        self.assertEqual(before, self.snapshot(self.checkout), "setup modified the checkout or root environment")
        return result

    def make_existing_environment(self):
        self.python.parent.mkdir(parents=True)
        self.python.write_text(self.fake)
        self.python.chmod(0o755)
        (self.env_root / "pyvenv.cfg").write_text("home = synthetic-test-only\n")
        (self.env_root / "installed-packages.json").write_text('{"study": "locked"}\n')

    def sync_calls(self):
        return [row for row in self.calls("uv") if row["argv"][0] == "sync"]

    def test_new_environment_targets_only_declared_python_and_study_lock(self):
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        sync, = self.sync_calls()
        self.assertEqual(sync["argv"], ["sync", "--project", str(self.project), "--locked", "--python", "3.11"])
        self.assertEqual(sync["env"]["UV_PROJECT_ENVIRONMENT"], str(self.env_root))
        self.assertEqual(sync["env"]["ST_PYTHON"], str(self.python))
        self.assertTrue(self.python.is_file())
        self.assertTrue((self.env_root / "synthetic-training-environment.json").is_file())

    def test_existing_environment_reuses_exact_interpreter(self):
        self.make_existing_environment()
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.sync_calls()[0]["argv"][-2:], ["--python", str(self.python)])
        self.assertEqual(self.calls("python")[0]["argv"][0], "-c")

    def test_build_requires_compute_allocation_and_matching_toolkit(self):
        for overrides, message in (
            ({"SLURM_JOB_ID": ""}, "compute node"),
            ({"CUDA_HOME": "/missing-toolkit"}, "compiler unavailable"),
            ({"ST_MOCK_NVCC": "Cuda compilation tools, release 12.1, V12.1.0"}, "toolkit 12.4"),
        ):
            with self.subTest(overrides=overrides):
                result = self.run_setup(overrides=overrides)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertFalse(self.sync_calls())
                self.assertFalse(self.env_root.exists())

    def test_check_does_not_require_toolkit_or_compute_allocation(self):
        self.make_existing_environment()
        result = self.run_setup("--check", overrides={"CUDA_HOME": "", "SLURM_JOB_ID": ""})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_check_is_offline_and_does_not_change_environment_files(self):
        self.make_existing_environment()
        before = self.snapshot(self.env_root)
        result = self.run_setup("--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(before, self.snapshot(self.env_root))
        sync, = self.sync_calls()
        self.assertIn("--check", sync["argv"])
        self.assertIn("--offline", sync["argv"])
        self.assertNotIn("--json-output", self.calls("python")[-1]["argv"])

    def test_cuda_requirement_is_forwarded_to_runtime_checker(self):
        self.make_existing_environment()
        result = self.run_setup("--check", "--require-cuda")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.calls("python")[-1]["argv"][-1], "--require-cuda")

    def test_old_overrides_are_scrubbed_and_transport_preferences_retained(self):
        poison = {
            name: "poison" for name in (
                "UV_CONSTRAINT", "UV_OVERRIDE", "UV_BUILD_CONSTRAINT", "UV_NO_CONFIG",
                "UV_CONFIG_FILE", "UV_PROJECT", "UV_WORKING_DIR", "UV_PYTHON",
                "UV_SYSTEM_PYTHON", "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_INDEX",
                "UV_DEFAULT_INDEX", "UV_FIND_LINKS", "UV_NO_INDEX", "UV_NO_SOURCES",
                "UV_NO_SOURCES_PACKAGE", "UV_TORCH_BACKEND", "UV_EXCLUDE_NEWER",
                "UV_NO_BUILD_ISOLATION", "UV_NO_BUILD", "UV_NO_BINARY", "UV_FROZEN",
                "UV_ISOLATED", "UV_NO_PROJECT", "UV_NO_SYNC", "UV_NO_INSTALL_PACKAGE",
                "UV_NO_INSTALL_LOCAL", "UV_NO_INSTALL_PROJECT", "UV_NO_INSTALL_WORKSPACE",
                "UV_NO_BINARY_PACKAGE", "UV_NO_BUILD_PACKAGE", "PYTHONHOME", "PYTHONPATH",
                "VIRTUAL_ENV",
            )
        }
        poison.update(UV_PROJECT_ENVIRONMENT=str(self.root_venv), UV_CACHE_DIR="/test/cache",
                      UV_HTTP_TIMEOUT="90", HTTPS_PROXY="http://proxy.example.invalid")
        result = self.run_setup(overrides=poison)
        self.assertEqual(result.returncode, 0, result.stderr)
        actual = self.sync_calls()[0]["env"]
        for name in poison.keys() - {"UV_PROJECT_ENVIRONMENT", "UV_CACHE_DIR", "UV_HTTP_TIMEOUT", "HTTPS_PROXY"}:
            self.assertNotIn(name, actual)
        self.assertEqual(actual["UV_PROJECT_ENVIRONMENT"], str(self.env_root))
        for name in ("UV_CACHE_DIR", "UV_HTTP_TIMEOUT", "HTTPS_PROXY"):
            self.assertEqual(actual[name], poison[name])
        self.assertEqual(actual["PYTHONNOUSERSITE"], "1")

    def test_directory_relative_and_root_environment_paths_are_rejected(self):
        for value in (str(self.env_root), "relative/bin/python", str(self.root_venv / "bin/python")):
            with self.subTest(value=value):
                result = self.run_setup(overrides={"ST_PYTHON": value})
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.sync_calls())
        self.assertFalse(self.env_root.exists())

    def test_symlink_to_root_environment_is_rejected(self):
        alias = self.folder / "environment-alias"
        alias.symlink_to(self.root_venv, target_is_directory=True)
        result = self.run_setup(overrides={"ST_PYTHON": str(alias / "bin/python")})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to synchronize", result.stderr)
        self.assertFalse(self.sync_calls())

    def test_existing_invalid_directory_is_preserved(self):
        self.env_root.mkdir(parents=True)
        marker = self.env_root / "unrelated-data"
        marker.write_text("preserve")
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(marker.read_text(), "preserve")
        self.assertFalse(self.sync_calls())

    def test_wrong_python_version_and_prefix_are_rejected_before_sync(self):
        self.make_existing_environment()
        before = self.snapshot(self.env_root)
        for override in ({"ST_MOCK_PY_VERSION": "3.12.1"}, {"ST_MOCK_PREFIX": str(self.root_venv)}):
            with self.subTest(override=override):
                result = self.run_setup(overrides=override)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.sync_calls())
                self.assertEqual(before, self.snapshot(self.env_root))

    def test_wrong_host_and_checkout_are_rejected_without_environment_creation(self):
        for override in ({"ST_MOCK_OS": "Darwin"}, {"ST_MOCK_ARCH": "aarch64"}, {"GAVD6_ROOT": str(self.folder)}):
            with self.subTest(override=override):
                result = self.run_setup(overrides=override)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.env_root.exists())
                self.assertFalse(self.sync_calls())

    def test_missing_environment_check_and_unsupported_uv_fail_before_sync(self):
        result = self.run_setup("--check")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Environment does not exist", result.stderr)
        for version in ("uv 0.6.14", "uv 0.12.14", "uv 0.13.0"):
            with self.subTest(version=version):
                result = self.run_setup(overrides={"ST_MOCK_UV_VERSION": version})
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.sync_calls())
        self.assertFalse(self.env_root.exists())

    def test_interrupted_sync_can_resume_and_does_not_run_checker_early(self):
        result = self.run_setup(overrides={"ST_MOCK_SYNC_EXIT": "17"})
        self.assertEqual(result.returncode, 17)
        self.assertTrue((self.env_root / "pyvenv.cfg").is_file())
        self.assertFalse(self.calls("python"))
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.sync_calls()[-1]["argv"][-1], str(self.python))
        self.assertTrue((self.env_root / "synthetic-training-environment.json").is_file())

    def test_metadata_and_runtime_failure_codes_propagate(self):
        result = self.run_setup(overrides={"ST_MOCK_PIP_CHECK_EXIT": "23"})
        self.assertEqual(result.returncode, 23)
        self.assertFalse(self.calls("python"))
        result = self.run_setup(overrides={"ST_MOCK_CHECKER_EXIT": "29"})
        self.assertEqual(result.returncode, 29)
        self.assertNotIn("Study package checks passed", result.stdout)

    def test_failed_lock_check_does_not_run_runtime_checker(self):
        self.make_existing_environment()
        result = self.run_setup("--check", overrides={"ST_MOCK_SYNC_CHECK_EXIT": "31"})
        self.assertEqual(result.returncode, 31)
        self.assertEqual(len(self.calls("python")), 1)  # interpreter preflight only

    def test_concurrent_writer_lock_rejects_then_releases_for_retry(self):
        self.env_root.parent.mkdir(parents=True)
        lock_path = Path(str(self.env_root) + ".synthetic-training-setup.lock")
        with lock_path.open("w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_setup()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Another setup/check", result.stderr)
            self.assertFalse(self.sync_calls())
            self.assertFalse(self.env_root.exists())
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_downloader_rejects_mismatched_or_dirty_checkout_before_weights(self):
        import tomllib

        self.make_existing_environment()
        models = self.folder / "models"
        (models / "mmpose/.git").mkdir(parents=True)
        marker = models / "mmpose/configs/local.py"
        marker.parent.mkdir()
        marker.write_text("existing user configuration")
        revision = tomllib.loads((self.project / "pyproject.toml").read_text())["tool"]["uv"]["sources"]["mmpose"]["rev"]
        for expected, status in (("0" * 40, ""), (revision, " M configs/local.py")):
            with self.subTest(revision=expected, status=status):
                result = subprocess.run(
                    ["bash", str(self.project / "download-students.sh")], cwd=self.folder,
                    env={**self.environment, "ST_MODEL_ROOT": str(models),
                         "ST_MOCK_POSE_REVISION": expected, "ST_MOCK_GIT_STATUS": status},
                    text=True, capture_output=True, timeout=20,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.calls("curl"))
                self.assertEqual(marker.read_text(), "existing user configuration")

    def test_common_launches_declared_python_with_clean_import_environment(self):
        self.make_existing_environment()
        code = (
            'import os; assert "PYTHONHOME" not in os.environ; '
            'assert "PYTHONPATH" not in os.environ; '
            'assert os.environ["PYTHONNOUSERSITE"] == "1"; '
            'print("clean study launch")'
        )
        result = subprocess.run(
            ["bash", "-c", 'source "$1"; st_python -c "$2"', "test-common",
             str(self.project / "common.sh"), code], cwd=self.folder,
            env={**self.environment, "ST_RUN_ROOT": str(self.folder / "run"),
                 "PYTHONHOME": "/missing/other-python", "PYTHONPATH": "/other/project/src",
                 "PYTHONNOUSERSITE": "0"},
            text=True, capture_output=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("clean study launch", result.stdout)
        invocation, = self.calls("python")
        self.assertEqual(invocation["env"]["ST_PYTHON"], str(self.python))
        self.assertNotIn("PYTHONHOME", invocation["env"])
        self.assertNotIn("PYTHONPATH", invocation["env"])
        self.assertEqual(invocation["env"]["PYTHONNOUSERSITE"], "1")

    def test_common_requires_explicit_absolute_python(self):
        self.make_existing_environment()
        for value in (None, "envs/study/bin/python"):
            with self.subTest(value=value):
                environment = {**self.environment, "ST_RUN_ROOT": str(self.folder / "run")}
                if value is None:
                    environment.pop("ST_PYTHON")
                else:
                    environment["ST_PYTHON"] = value
                result = subprocess.run(
                    ["bash", "-c", 'source "$1"; st_python -c "print(1)"',
                     "test-common", str(self.project / "common.sh")], cwd=self.folder,
                    env=environment, text=True, capture_output=True, timeout=20,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.calls("python"))

    def test_downloader_clears_import_overrides_before_reading_manifest(self):
        import tomllib

        self.make_existing_environment()
        models = self.folder / "models"
        (models / "mmpose/.git").mkdir(parents=True)
        revision = tomllib.loads((self.project / "pyproject.toml").read_text())["tool"]["uv"]["sources"]["mmpose"]["rev"]
        result = subprocess.run(
            ["bash", str(self.project / "download-students.sh")], cwd=self.folder,
            env={**self.environment, "ST_MODEL_ROOT": str(models),
                 "ST_MOCK_POSE_REVISION": revision,
                 "PYTHONHOME": "/missing/other-python", "PYTHONPATH": "/other/project/src",
                 "PYTHONNOUSERSITE": "0"},
            text=True, capture_output=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        invocation, = self.calls("python")
        self.assertNotIn("PYTHONHOME", invocation["env"])
        self.assertNotIn("PYTHONPATH", invocation["env"])
        self.assertEqual(invocation["env"]["PYTHONNOUSERSITE"], "1")
        self.assertEqual(len(self.calls("curl")), 5)
        self.assertEqual(len(list((models / "pose").glob("*.pth"))), 5)


if __name__ == "__main__":
    unittest.main()
