"""Regression checks for IDE-independent GaitParity notebook startup."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_DIR / "notebooks"
REPO_DIR = PROJECT_DIR.parents[2]
EXPECTED_NOTEBOOKS = {
    "nb_09c_gavd_matched_jepa_contract.ipynb",
    "nb_09d_gavd_matched_jepa_training.ipynb",
    "nb_09e_gavd_matched_jepa_audit.ipynb",
    "nb_09f_full_gavd_cpu_replication.ipynb",
    "nb_09g_full_gavd_gpu_replication.ipynb",
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

    cpu_source = (NOTEBOOK_DIR / "nb_09f_full_gavd_cpu_replication.ipynb").read_text(encoding="utf-8")
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

    for name in ("nb_09f_full_gavd_cpu_replication.ipynb", "nb_09g_full_gavd_gpu_replication.ipynb"):
        source = (NOTEBOOK_DIR / name).read_text(encoding="utf-8")
        assert 'PROJECT_DIR / \\"notebooks\\" / name' in source

    generator_dir = PROJECT_DIR / "notes" / "ideas-claude" / "09-reflection-equivariant-symmetry-axis"
    for name in ("_build_nb_09cde.py", "_build_nb_09fg.py"):
        path = generator_dir / name
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        assert 'ROOT / "notebooks"' in source
        assert 'candidate / "notebooks" / "nb_09a_equivariant_encoder_contract.ipynb"' in source

    print(f"PASS: {len(paths)} notebooks start from repo, project, notebook, and override directories")


if __name__ == "__main__":
    main()
