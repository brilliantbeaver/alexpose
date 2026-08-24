"""Regression checks for IDE-independent GaitParity notebook startup."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = (
    PROJECT_DIR / "notebooks" / "experiments" / "idea09_reflection_equivariance"
)
REPO_DIR = PROJECT_DIR.parents[2]
EXPECTED_NOTEBOOKS = {
    "03_gavd_contract.ipynb",
    "04_gavd_training.ipynb",
    "05_gavd_audit.ipynb",
    "06_cpu_replication.ipynb",
    "07_gpu_replication.ipynb",
    "08_amass_core11_training.ipynb",
}


def startup_source(path: Path) -> str:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if "def find_notebook_root" in source:
            return source.split("\nif str(PROJECT_DIR)", maxsplit=1)[0]
    raise AssertionError(f"No project locator found in {path}")


def assert_code_cells_compile(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cell_ids = [cell.get("id") for cell in notebook["cells"]]
    assert all(cell_ids), f"Missing cell ID in {path}"
    assert len(cell_ids) == len(set(cell_ids)), f"Duplicate cell ID in {path}"
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell.get("source", [])), f"{path}:cell-{index}", "exec")


def assert_startup_from(path: Path, working_directory: Path) -> None:
    previous_directory = Path.cwd()
    namespace: dict[str, object] = {}
    try:
        os.chdir(working_directory)
        exec(compile(startup_source(path), str(path), "exec"), namespace)
    finally:
        os.chdir(previous_directory)
    assert namespace["PROJECT_DIR"] == PROJECT_DIR, (
        path.name,
        working_directory,
        namespace["PROJECT_DIR"],
    )


def main() -> None:
    paths = [NOTEBOOK_DIR / name for name in sorted(EXPECTED_NOTEBOOKS)]
    for path in paths:
        assert_startup_from(path, REPO_DIR)
        assert_startup_from(path, PROJECT_DIR)
        assert_startup_from(path, NOTEBOOK_DIR)
        assert_code_cells_compile(path)

    cpu_source = (NOTEBOOK_DIR / "06_cpu_replication.ipynb").read_text(encoding="utf-8")
    assert 'STARTUP_REVISION = \\"ide-safe-v2\\"' in cpu_source

    previous_override = os.environ.get("GAIT_PARITY_PROJECT_DIR")
    try:
        os.environ["GAIT_PARITY_PROJECT_DIR"] = str(PROJECT_DIR)
        with tempfile.TemporaryDirectory() as temporary_directory:
            for path in paths:
                assert_startup_from(path, Path(temporary_directory))
    finally:
        if previous_override is None:
            os.environ.pop("GAIT_PARITY_PROJECT_DIR", None)
        else:
            os.environ["GAIT_PARITY_PROJECT_DIR"] = previous_override

    for name in ("06_cpu_replication.ipynb", "07_gpu_replication.ipynb"):
        source = (NOTEBOOK_DIR / name).read_text(encoding="utf-8")
        assert 'PROJECT_DIR / \\"notebooks\\" / \\"experiments\\" /' in source
        assert '\\"idea09_reflection_equivariance\\" / name' in source

    generator_dir = PROJECT_DIR / "scripts" / "notebook_builders" / "idea09"
    for name in (
        "build_gavd_sequence.py",
        "build_replications.py",
        "build_amass_training.py",
    ):
        path = generator_dir / name
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        assert '"notebooks" / "experiments" / "idea09_reflection_equivariance"' in source
        if name == "build_amass_training.py":
            assert '"08_amass_core11_training.ipynb"' in source
        else:
            assert '/ "01_encoder_contract.ipynb"' in source

    print(f"PASS: {len(paths)} notebooks start from repo, project, notebook, and override directories")


if __name__ == "__main__":
    main()
