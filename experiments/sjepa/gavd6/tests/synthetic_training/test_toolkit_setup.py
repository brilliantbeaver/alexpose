"""Test rootless toolkit setup at the download/package-manager boundaries.

The real shell runs with fake network and CUDA programs; no toolkit is installed.
The separate development check used the real micromamba Linux dependency solver.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "slurm/synthetic-training/setup-cuda-toolkit.sh"
HEADERS = ("cuda.h", "cuda_runtime.h", "cuda_fp16.h", "cublas_v2.h", "cusparse.h")
FAKE = r'''
import fcntl, json, os, pathlib, shutil, sys
name = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
if name == "uname":
    print("Linux" if args == ["-s"] else "x86_64")
elif name == "flock":
    try:
        fcntl.flock(int(args[-1]), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(1)
elif name == "sha256sum":
    line = sys.stdin.read()
    assert line.startswith("9689782d863c05a1bf5d2d371ba527104e7a4eb4310c1637d8653b751aed9c82  "), line
    raise SystemExit(int(os.environ.get("ST_MOCK_HASH_FAIL", "0")))
elif name == "curl":
    shutil.copyfile(os.environ["ST_MOCK_PROGRAM"], args[args.index("--output") + 1])
elif name == "micromamba-2.8.1":
    pathlib.Path(os.environ["ST_MOCK_CALL"]).write_text(json.dumps(args))
    root = pathlib.Path(args[args.index("--prefix") + 1])
    (root / "conda-meta").mkdir(parents=True, exist_ok=True)
    (root / "conda-meta/history").touch()
    if os.environ.get("ST_MOCK_INSTALL_FAIL"):
        raise SystemExit(7)
    (root / "bin").mkdir(exist_ok=True)
    compiler = root / "bin/nvcc"
    compiler.write_text('#!/bin/sh\nprintf "Cuda compilation tools, release 12.4, V12.4.131\\n"\n')
    compiler.chmod(0o755)
    (root / "include").mkdir(exist_ok=True)
    for name in ("cuda.h", "cuda_runtime.h", "cuda_fp16.h", "cublas_v2.h", "cusparse.h"):
        (root / "include" / name).touch()
    (root / "lib").mkdir(exist_ok=True)
    (root / "lib/libcudart.so").touch()
else:
    raise SystemExit("Unexpected mock: " + name)
'''


class ToolkitSetupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="st-toolkit-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.toolchains = self.root / "toolchains"
        self.fake_bin = self.root / "bin"
        self.fake_bin.mkdir()
        program = "#!" + sys.executable + " -I\n" + FAKE
        self.fake_program = self.root / "fake-program"
        self.fake_program.write_text(program)
        for name in ("uname", "flock", "curl", "sha256sum"):
            path = self.fake_bin / name
            path.write_text(program)
            path.chmod(0o755)
        self.env = {
            key: value for key, value in os.environ.items()
            if not key.startswith("ST_") and key not in {"CUDA_HOME", "PYTHONHOME", "PYTHONPATH"}
        }
        self.env.update(
            PATH=str(self.fake_bin) + os.pathsep + os.environ["PATH"],
            ST_TOOLCHAIN_ROOT=str(self.toolchains),
            ST_MOCK_PROGRAM=str(self.fake_program),
            ST_MOCK_CALL=str(self.root / "micromamba-call.json"),
        )

    def run_setup(self, **overrides):
        return subprocess.run(
            ["bash", str(SCRIPT)], env={**self.env, **overrides},
            cwd=self.root, capture_output=True, text=True, timeout=20,
        )

    def make_toolkit(self, root, version="12.4"):
        (root / "bin").mkdir(parents=True)
        compiler = root / "bin/nvcc"
        compiler.write_text(f'#!/bin/sh\nprintf "Cuda compilation tools, release {version}, V{version}.131\\n"\n')
        compiler.chmod(0o755)
        (root / "include").mkdir()
        for header in HEADERS:
            (root / "include" / header).touch()
        (root / "lib64").mkdir()
        (root / "lib64/libcudart.so").touch()

    def test_existing_toolkit_publishes_safely_quoted_shell_configuration(self):
        toolkit = self.root / "CUDA ' $(touch unexpected) toolkit"
        self.make_toolkit(toolkit)
        result = self.run_setup(CUDA_HOME=str(toolkit))
        self.assertEqual(result.returncode, 0, result.stderr)
        sourced = subprocess.run(
            ["bash", "-c", 'source "$ST_TOOLCHAIN_ROOT/cuda.env"; printf "%s" "$CUDA_HOME"'],
            env=self.env, cwd=self.root, capture_output=True, text=True, check=True,
        )
        self.assertEqual(sourced.stdout, str(toolkit))
        self.assertFalse((self.root / "unexpected").exists())
        self.assertFalse((self.root / "micromamba-call.json").exists())

    def test_obsolete_override_uses_existing_managed_toolkit(self):
        self.make_toolkit(self.toolchains / "cuda-12.4.1")
        result = self.run_setup(CUDA_HOME="/nonexistent/cuda-12.1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("not a complete CUDA 12.4 toolkit", result.stdout)
        self.assertIn("CUDA_TOOLKIT_PASSED", result.stdout)
        self.assertFalse((self.root / "micromamba-call.json").exists())

    def test_incomplete_directory_is_preserved_without_publication(self):
        toolkit = self.toolchains / "cuda-12.4.1"
        toolkit.mkdir(parents=True)
        keep = toolkit / "partial-download"
        keep.write_text("keep for inspection")
        result = self.run_setup()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fresh directory", result.stderr)
        self.assertEqual(keep.read_text(), "keep for inspection")
        self.assertFalse((self.toolchains / "cuda.env").exists())

    def test_install_uses_only_pinned_nvidia_channel_and_managed_prefixes(self):
        result = self.run_setup(CUDA_HOME="/nonexistent/cuda-12.1")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads((self.root / "micromamba-call.json").read_text())
        self.assertEqual(args[:3], ["--no-rc", "--no-env", "create"])
        self.assertIn("--override-channels", args)
        self.assertEqual(args[args.index("--channel") + 1], "https://conda.anaconda.org/nvidia/label/cuda-12.4.1")
        self.assertEqual(args[-1], "cuda-toolkit=12.4.1")
        self.assertEqual(args[args.index("--prefix") + 1], str(self.toolchains / "cuda-12.4.1"))
        self.assertTrue((self.toolchains / "cuda.env").exists())

    def test_failed_install_can_resume_from_package_metadata(self):
        failed = self.run_setup(ST_MOCK_INSTALL_FAIL="1")
        self.assertNotEqual(failed.returncode, 0)
        self.assertFalse((self.toolchains / "cuda.env").exists())
        resumed = self.run_setup()
        self.assertEqual(resumed.returncode, 0, resumed.stderr)
        args = json.loads((self.root / "micromamba-call.json").read_text())
        self.assertEqual(args[2], "install")
        self.assertTrue((self.toolchains / "cuda.env").exists())

    def test_bad_download_is_never_executed_or_published(self):
        result = self.run_setup(ST_MOCK_HASH_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("checksum did not match", result.stderr)
        self.assertFalse((self.root / "micromamba-call.json").exists())
        self.assertFalse((self.toolchains / "cuda.env").exists())

    def test_old_compiler_version_is_not_reused(self):
        obsolete = self.root / "cuda-12.1"
        self.make_toolkit(obsolete, version="12.1")
        result = self.run_setup(CUDA_HOME=str(obsolete))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "micromamba-call.json").exists())
        self.assertIn("not a complete CUDA 12.4 toolkit", result.stdout)


if __name__ == "__main__":
    unittest.main()
