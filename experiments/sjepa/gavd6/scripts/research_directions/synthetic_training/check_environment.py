#!/usr/bin/env python3
"""Check the pinned study packages without loading weights or creating a renderer.

The default check permits a login node without CUDA. ``--require-cuda`` also
allocates CUDA tensors and runs both native NMS implementations on the GPU.
Only standard-library modules are imported until the runtime checks begin.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata as metadata
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import warnings


ROOT = Path(__file__).resolve().parents[3]
STUDY_PROJECT = ROOT / "slurm" / "synthetic-training"
EXCLUDED_CHECKS = [
    "EGL/offscreen rendering", "licensed body and motion assets", "datasets",
    "downloaded checkpoint compatibility", "full experiment execution",
]
MODULES = {
    "torch": "torch", "torchvision": "torchvision", "numpy": "numpy",
    "scipy": "scipy", "mmcv": "mmcv", "mmengine": "mmengine",
    "mmdet": "mmdet", "mmpretrain": "mmpretrain", "mmpose": "mmpose",
    "opencv-python": "cv2", "chumpy": "chumpy",
    "human-body-prior": "human_body_prior",
}


def read_contract(project: Path = STUDY_PROJECT) -> dict:
    import tomllib

    manifest_path, lock_path = project / "pyproject.toml", project / "uv.lock"
    manifest_bytes, lock_bytes = manifest_path.read_bytes(), lock_path.read_bytes()
    manifest = tomllib.loads(manifest_bytes.decode())
    # Direct exact pins are the version-sensitive ABI/API contract. uv sync
    # --locked and uv pip check separately enforce the complete dependency set.
    versions = {}
    for requirement in manifest["project"]["dependencies"]:
        if "==" in requirement:
            name, version = requirement.split("==", 1)
            versions[name] = version
    sources = manifest["tool"]["uv"]["sources"]
    return {
        "versions": versions,
        "git_sources": {name: source for name, source in sources.items() if "git" in source},
        "manifest": str(manifest_path), "lock": str(lock_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
    }


def check_git_source(distribution, expected: dict) -> str:
    raw = distribution.read_text("direct_url.json")
    if raw is None:
        raise ValueError("missing direct_url.json; expected the pinned Git installation")
    direct = json.loads(raw)
    vcs = direct.get("vcs_info", {})
    if (direct.get("url") != expected["git"] or vcs.get("vcs") != "git"
            or vcs.get("commit_id") != expected["rev"] or direct.get("dir_info")):
        raise ValueError(f"Git provenance mismatch: expected {expected['git']}@{expected['rev']}; got {direct}")
    return f"{direct['url']}@{vcs['commit_id']}"


def check_import_path(module, distribution, name: str) -> str:
    actual = Path(module.__file__).resolve()
    expected = Path(distribution.locate_file(name.replace(".", "/"))).resolve()
    environment = Path(sys.prefix).resolve()
    if not expected.is_relative_to(environment):
        raise ValueError(f"package metadata resolves outside this environment: {expected}; interpreter prefix is {environment}")
    if not actual.is_relative_to(expected):
        raise ValueError(f"import shadowed: {actual}; installed package is {expected}")
    return str(actual)


def _nms_check(torch, tv_nms, mmcv_nms, device: str) -> None:
    boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.], [4., 4., 6., 6.]], device=device)
    scores = torch.tensor([0.9, 0.8, 0.7], device=device)
    tv_keep = tv_nms(boxes, scores, 0.5)
    _, mmcv_keep = mmcv_nms(boxes, scores, 0.5)
    if tv_keep.cpu().tolist() != [0, 2] or mmcv_keep.cpu().tolist() != [0, 2]:
        raise ValueError("NMS did not suppress the duplicate box and retain the separate box")


def _check_torch_cpu() -> str:
    torch = importlib.import_module("torch")
    if torch.version.cuda != "12.1":
        raise ValueError(f"Torch was compiled for CUDA {torch.version.cuda}; expected 12.1")
    x = torch.tensor([[1., 2.], [3., 4.]])
    torch.testing.assert_close(x @ x, torch.tensor([[7., 10.], [15., 22.]]), rtol=0, atol=0)
    _nms_check(torch, importlib.import_module("torchvision.ops").nms,
               importlib.import_module("mmcv.ops").nms, "cpu")
    return "CPU matrix arithmetic and Torchvision/MMCV NMS passed; Torch compiled for CUDA 12.1"


def _check_pose_imports() -> str:
    apis = importlib.import_module("mmpose.apis")
    # ViTPose uses MMPreTrain backbone registrations; a top-level package
    # import alone would not exercise their Torch/transform dependencies.
    importlib.import_module("mmpretrain.models")
    dataset = importlib.import_module("mmengine.dataset")
    runner = importlib.import_module("mmengine.runner")
    for function in (apis.init_model, dataset.Compose, dataset.pseudo_collate, runner.load_checkpoint):
        if not callable(function):
            raise TypeError(f"Required pose API is not callable: {function}")
    utils = importlib.import_module("mmengine.utils")
    installed = Path(utils.get_installed_path("mmpose"))
    configs = installed / ".mim" / "configs"
    config = configs / "body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py"
    if not config.is_file() or not (configs / "_base_/datasets/coco.py").is_file():
        raise FileNotFoundError(f"Installed MMPose is missing bundled .mim pilot/COCO configs under {configs}")
    # Exercise MMPose's non-editable-install fallback, independently of cwd.
    parser = importlib.import_module("mmpose.datasets.datasets.utils").parse_pose_metainfo
    with tempfile.TemporaryDirectory(prefix="st-mmpose-meta-") as temporary:
        with warnings.catch_warnings(record=True):
            metainfo = parser({"from_file": str(Path(temporary) / "absent" / "coco.py")})
    if metainfo["dataset_name"] != "coco" or metainfo["num_keypoints"] != 17:
        raise ValueError("Bundled MMPose metadata does not describe the required COCO17 dataset")
    return f"Pose APIs, MMEngine package lookup and COCO17 metadata fallback passed; configs={configs}"


def _check_chumpy() -> str:
    np, ch = importlib.import_module("numpy"), importlib.import_module("chumpy")
    x = ch.array([1., 2., 3.])
    y = x * x + 2 * x
    np.testing.assert_allclose(y.r, [3., 8., 15.], rtol=0, atol=1e-12)
    np.testing.assert_allclose(y.dr_wrt(x).toarray(), np.diag([4., 6., 8.]), rtol=0, atol=1e-12)
    function = ch.Ch(lambda a: a * a, initial_args={"a": np.array([2., 3.])})
    np.testing.assert_allclose(function.r, [4., 9.], rtol=0, atol=1e-12)
    return "Chumpy values, derivatives and callback argument inspection passed"


def check_body_model_api(body_model) -> str:
    required = {"bm_fname", "dmpl_fname", "num_betas", "num_dmpls"}
    missing = required - set(inspect.signature(body_model).parameters)
    if missing:
        raise ValueError(f"BodyModel constructor lacks required parameters: {sorted(missing)}")
    return "BodyModel constructor supports SMPL-H/DMPL arguments; licensed assets were not loaded"


def _check_body_model() -> str:
    return check_body_model_api(importlib.import_module("human_body_prior.body_model.body_model").BodyModel)


def _check_notebook_imports() -> str:
    for name in ("nbclient", "nbformat", "ipykernel", "jupyter_client", "pandas", "sklearn",
                 "matplotlib", "joblib", "PIL.Image", "trimesh", "einops", "timm", "iopath", "yaml"):
        importlib.import_module(name)
    return "Notebook, data and encoder support imports passed; no renderer or model weights loaded"


def _check_cuda(require_cuda: bool) -> dict:
    torch = importlib.import_module("torch")
    available = bool(torch.cuda.is_available())
    if not require_cuda:
        return {"available": available, "exercised": False,
                "detail": f"CUDA available={available}; GPU execution was not requested (login-node check)"}
    if not available:
        raise RuntimeError("CUDA is unavailable; rerun --require-cuda inside a GPU allocation")
    # is_available() alone is insufficient: exercise the actual driver and both
    # compiled kernels with a tiny allocation, then synchronize device errors.
    value = torch.tensor([2., 3.], device="cuda")
    torch.testing.assert_close((value * value).cpu(), torch.tensor([4., 9.]), rtol=0, atol=0)
    _nms_check(torch, importlib.import_module("torchvision.ops").nms,
               importlib.import_module("mmcv.ops").nms, "cuda")
    torch.cuda.synchronize()
    return {"available": True, "exercised": True,
            "detail": f"CUDA arithmetic and Torchvision/MMCV NMS passed on {torch.cuda.get_device_name(0)}"}


def run_checks(require_cuda: bool = False) -> dict:
    checks = []

    def record(name, operation):
        try:
            detail = operation()
        except Exception as error:
            checks.append({"name": name, "status": "failed", "detail": f"{type(error).__name__}: {error}"})
            return None
        checks.append({"name": name, "status": "passed", "detail": detail})
        return detail

    def require(condition: bool, detail: str) -> str:
        if not condition:
            raise ValueError(detail)
        return detail

    report = {"schema_version": 1, "checked_at": datetime.now(timezone.utc).isoformat(),
              "python": sys.executable, "python_version": platform.python_version(),
              "platform": platform.platform(), "require_cuda": require_cuda,
              "checks": checks, "not_checked": EXCLUDED_CHECKS,
              "cuda": {"available": None, "exercised": False}}
    record("python", lambda: require(sys.version_info[:2] == (3, 11),
                                      f"Python {platform.python_version()}; required 3.11"))
    record("platform", lambda: require(platform.system() == "Linux" and platform.machine() == "x86_64",
                                        f"{platform.system()} {platform.machine()}; required Linux x86_64"))
    contract = record("contract", read_contract)
    if contract is not None:
        report["contract"] = contract
        for package, expected in contract["versions"].items():
            def version_check(package=package, expected=expected):
                actual = metadata.version(package)
                return require(actual == expected, f"{package}={actual}; required {expected}")
            record(f"version:{package}", version_check)
        for package, expected in contract["git_sources"].items():
            record(f"source:{package}", lambda package=package, expected=expected:
                   check_git_source(metadata.distribution(package), expected))
    prerequisites_passed = all(check["status"] == "passed" for check in checks)
    runtime = (("torch_cpu", _check_torch_cpu), ("pose_api_and_configs", _check_pose_imports),
               ("chumpy", _check_chumpy), ("body_model_api", _check_body_model),
               ("notebook_imports", _check_notebook_imports))
    if prerequisites_passed:
        for package, module_name in MODULES.items():
            record(f"import:{package}", lambda package=package, module_name=module_name:
                   check_import_path(importlib.import_module(module_name), metadata.distribution(package), module_name))
        # Continue independent probes after an import failure to expose other
        # concrete problems, while retaining the failed overall result.
        for name, probe in runtime:
            record(name, probe)
        cuda = record("cuda", lambda: _check_cuda(require_cuda))
        if cuda is not None:
            report["cuda"] = cuda
    else:
        for name, _ in runtime:
            checks.append({"name": name, "status": "skipped", "detail": "Fix platform/version/provenance failures before importing native packages"})
        checks.append({"name": "cuda", "status": "skipped", "detail": "Prerequisite checks failed"})
    report["passed"] = all(check["status"] == "passed" for check in checks)
    return report


def write_report(path: Path, report: dict) -> None:
    """Atomically replace one explicitly requested environment-check report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true", help="Also execute small CUDA arithmetic/NMS checks inside a GPU allocation")
    parser.add_argument("--json-output", type=Path, help="Atomically save the check report to this path")
    args = parser.parse_args(argv)
    report = run_checks(args.require_cuda)
    for check in report["checks"]:
        detail = check["detail"]
        if isinstance(detail, dict):
            detail = detail.get("detail", f"manifest_sha256={detail.get('manifest_sha256')}; lock_sha256={detail.get('lock_sha256')}")
        print(f"{check['status'].upper()}: {check['name']}: {detail}")
    if args.json_output:
        write_report(args.json_output, report)
        print(f"Report: {args.json_output}")
    print("Environment checks passed." if report["passed"] else "Environment checks failed; resolve the failures above.")
    print("Not checked: " + "; ".join(report["not_checked"]) + ".")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
