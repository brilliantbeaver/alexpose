"""Local, content-addressed CV caches; never share fitted models across folds.

Only use a cache directory you trust: checkpoints contain PyTorch training state.
Manifests are committed last, after atomic writes, so interrupted writes are misses.
"""
from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import tempfile
from zipfile import BadZipFile

import numpy as np

from .splits import digest, file_sha256


@contextmanager
def atomic_path(path):
    """Write beside the destination, then atomically replace it on success."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic_path(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_arrays(path, **arrays):
    with atomic_path(path) as temporary:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)


def read_arrays(path):
    try:
        with np.load(path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
        if all(np.isfinite(a).all() for a in arrays.values()):
            return arrays
    except (OSError, ValueError, EOFError, BadZipFile):
        pass
    return None


def read_manifest(path, context):
    """A complete manifest must match provenance and every artifact's bytes."""
    path = Path(path)
    try:
        saved = json.loads(path.read_text())
        if saved["context"] != context or digest(saved["value"]) != saved["value_sha256"]:
            return None
        for name, checksum in saved["files"].items():
            if Path(name).name != name or file_sha256(path.parent / name) != checksum:
                return None
        return saved["value"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def write_manifest(path, context, value, files=()):
    path = Path(path)
    write_json(path, dict(context=context, value=value, value_sha256=digest(value),
                         files={name: file_sha256(path.parent / name) for name in files}))


def cache_identity(records, registry, cfg, device, updates, more_updates, cpu_threads):
    """Hash actual inputs again: an in-memory registry can outlive changed files."""
    import ambient
    import torch

    inputs = {r.clip_name: file_sha256(r.path) for r in records}
    expected = {r["clip"]: r["sha256"] for r in registry["inventory"].get("cache", [])}
    if expected and inputs != expected:
        raise ValueError("Pose cache changed since the registry was loaded; reload/rebuild the registry")
    roots = {"sjepa": Path(__file__).parent, "ambient": Path(ambient.__file__).parent}
    code = {f"{name}/{p.relative_to(root)}": file_sha256(p)
            for name, root in roots.items() for p in sorted(root.rglob("*.py"))
            if "tests" not in p.relative_to(root).parts}
    environment = {name: version(name) for name in ("numpy", "torch", "scikit-learn", "scipy", "joblib")}
    environment.update(python=platform.python_version(), platform=platform.platform(),
                       machine=platform.machine(), cpu_threads=cpu_threads,
                       dtype=str(torch.get_default_dtype()),
                       matmul_precision=torch.get_float32_matmul_precision(),
                       deterministic=torch.are_deterministic_algorithms_enabled(),
                       cudnn_benchmark=torch.backends.cudnn.benchmark,
                       cudnn_deterministic=torch.backends.cudnn.deterministic)
    # Device type alone is insufficient when the same cache is used on two GPUs.
    hardware = str(device)
    if torch.device(device).type == "cuda":
        hardware += ":" + str(torch.cuda.get_device_properties(device))
    common = dict(schema=1, code=digest(code), environment=environment, inputs=inputs)
    run = dict(**common, registry=registry["registry_sha256"], config=cfg.to_dict(),
               device=hardware, updates=updates, more_updates=more_updates)
    features = dict(**common, fps=cfg.target_fps)
    return run, features


class FeatureCache:
    """Cache only deterministic per-clip features, not scalers or classifiers."""

    def __init__(self, directory, fps, jobs=1):
        self.directory, self.fps, self.jobs = Path(directory), fps, jobs

    def _one(self, record):
        from .classical import sequence_to_feature_vector
        from .full_experiment import nuisance_features

        # The parent directory is already keyed by all input bytes and code.
        path = self.directory / f"{digest(record.clip_name)}.npz"
        arrays = read_arrays(path)
        if (arrays is not None and set(arrays) == {"rf", "visibility", "mean_pose"}
                and all(a.ndim == 1 for a in arrays.values())
                and arrays["visibility"].shape == arrays["mean_pose"].shape == (66,)):
            return arrays
        arrays = dict(rf=sequence_to_feature_vector(record, self.fps).to_array(),
                      visibility=nuisance_features([record], "visibility")[0],
                      mean_pose=nuisance_features([record], "mean_pose")[0])
        if not all(np.isfinite(a).all() for a in arrays.values()):
            raise ValueError(f"Non-finite features for {record.clip_name}")
        write_arrays(path, **arrays)
        return arrays

    def matrix(self, records):
        from joblib import Parallel, delayed, parallel_config

        if self.jobs == 1:
            rows = [self._one(r) for r in records]
        else:
            # Processes bypass Python's GIL for angle calculation. Each gets one
            # BLAS thread; fold-level parallelism disables this inner pool.
            with parallel_config(backend="loky", inner_max_num_threads=1):
                rows = Parallel(n_jobs=self.jobs)(delayed(self._one)(r) for r in records)
        return {name: np.stack([row[name] for row in rows])
                for name in ("rf", "visibility", "mean_pose")}
