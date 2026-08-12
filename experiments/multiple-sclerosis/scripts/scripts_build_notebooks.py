"""Generate the seven tutorial notebooks as .ipynb files.

Writing notebooks from Python keeps the shared header identical across all seven
and lets us reuse the verified sjepa package APIs exactly. Each notebook gets an
"Open in Colab" badge, a Colab install cell, an import-or-vendor bootstrap so
`import sjepa` and `import ambient` work locally and in Colab, and a paths cell.

Run:  python scripts_build_notebooks.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent  # experiment dir (scripts/ is one level down)
REPO_SLUG = "your-org/alexpose"  # users edit this to their fork for Colab
COLAB_BASE = f"https://colab.research.google.com/github/{REPO_SLUG}/blob/main/experiments/multiple-sclerosis"

# Where notebooks are written. Overridable via --output-dir so the generator can
# emit into a temporary directory for an idempotence / diff check without
# touching the canonical .ipynb files.
_OUT_DIR = EXP
_CHECK_ONLY = False


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}


def _src(lines):
    # Accept a single multi-line string or a list of lines.
    if len(lines) == 1 and "\n" in lines[0]:
        text = lines[0]
    else:
        text = "\n".join(lines)
    out = text.split("\n")
    return [l + "\n" for l in out[:-1]] + [out[-1]]


def colab_badge(nb_name: str):
    return md(
        f'<a href="{COLAB_BASE}/{nb_name}" target="_parent">'
        f'<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>'
    )


def bootstrap_cells(need_torch=True):
    """The shared setup cells: install (Colab), import-or-vendor, paths."""
    install_lines = [
        "# --- Setup: install dependencies (Colab installs; local usually already has them) ---",
        "import importlib, importlib.util, subprocess, sys, os",
        "",
        "IN_COLAB = 'google.colab' in sys.modules",
        "",
        "def _need(mod):",
        "    return importlib.util.find_spec(mod) is None",
        "",
        "# Light deps used by every notebook.",
        "_pkgs = []",
        "for mod, pip_name in [('cv2','opencv-python'), ('mediapipe','mediapipe'),",
        "                      ('sklearn','scikit-learn'), ('pandas','pandas'),",
        "                      ('matplotlib','matplotlib'), ('tqdm','tqdm')]:",
        "    if _need(mod):",
        "        _pkgs.append(pip_name)",
    ]
    if need_torch:
        install_lines += [
            "# torch is guarded so Colab's preinstalled GPU torch is never downgraded.",
            "if _need('torch'):",
            "    _pkgs.append('torch')",
        ]
    install_lines += [
        "if _pkgs:",
        "    print('installing:', _pkgs)",
        "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *_pkgs])",
        "else:",
        "    print('all light dependencies already present')",
    ]
    install = code("\n".join(install_lines))
    vendor = code(
        "# --- Make `sjepa` and `ambient` importable, locally and in Colab ---",
        "from pathlib import Path",
        "import sys, subprocess",
        "",
        "def _find_exp_dir():",
        "    # Local run: this notebook sits in experiments/multiple-sclerosis.",
        "    here = Path.cwd()",
        "    for p in [here, *here.parents]:",
        "        if (p / 'sjepa' / '__init__.py').exists():",
        "            return p",
        "    return None",
        "",
        "EXP_DIR = _find_exp_dir()",
        "if EXP_DIR is None:",
        "    # Colab: clone the repo, then point at the experiment folder.",
        "    REPO = 'https://github.com/your-org/alexpose.git'  # <-- edit to your fork",
        "    if not Path('alexpose').exists():",
        "        subprocess.check_call(['git', 'clone', '--depth', '1', REPO])",
        "    EXP_DIR = Path('alexpose') / 'experiments' / 'multiple-sclerosis'",
        "",
        "REPO_ROOT = EXP_DIR.parents[1]",
        "for p in (str(EXP_DIR), str(REPO_ROOT)):",
        "    if p not in sys.path:",
        "        sys.path.insert(0, p)",
        "print('experiment dir:', EXP_DIR)",
        "print('repo root     :', REPO_ROOT)",
    )
    paths = code(
        "# --- Paths and profile (reads the root .env if python-dotenv is present) ---",
        "import os",
        "try:",
        "    from dotenv import load_dotenv",
        "    load_dotenv(REPO_ROOT / '.env')",
        "except Exception:",
        "    pass",
        "",
        "VIDEO_DIR = EXP_DIR / 'video-data'",
        "ARTIFACT_DIR = EXP_DIR / 'artifacts'",
        "KEYPOINTS_DIR = ARTIFACT_DIR / 'keypoints'",
        "IMAGES_DIR = EXP_DIR / 'images'",
        "ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)",
        "",
        "# Pick the model size profile. 'laptop' is the fast default; set SJEPA_PROFILE=gpu",
        "# in your .env for a larger model, or SJEPA_SMOKE=1 for a near-instant test run.",
        "os.environ.setdefault('SJEPA_PROFILE', 'laptop')",
        "print('SJEPA_PROFILE =', os.environ['SJEPA_PROFILE'],",
        "      '| SJEPA_SMOKE =', os.environ.get('SJEPA_SMOKE', '0'))",
    )
    return [install, vendor, paths]


def _render_nb(cells) -> str:
    # Give each cell a stable id to satisfy nbformat 4.5+ (id must be set before
    # rendering; keyed off position, matching the original behaviour).
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(nb, indent=1)


def write_nb(name: str, cells):
    for i, cell in enumerate(cells):
        cell.setdefault("id", f"{Path(name).stem}-{i:02d}")
    content = _render_nb(cells)
    target = _OUT_DIR / name
    if _CHECK_ONLY:
        existing = target.read_text() if target.exists() else ""
        status = "MATCH" if existing == content else "DIFFERS"
        print(f"check {name}: {status}")
        return status == "MATCH"
    target.write_text(content)
    print("wrote", target)
    return True


# The per-notebook content is defined in notebook_content.py to keep this file short.
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=None,
                    help="write notebooks here instead of the experiment dir")
    ap.add_argument("--check", action="store_true",
                    help="do not write; report whether each notebook matches on disk")
    args = ap.parse_args()
    if args.output_dir:
        _OUT_DIR = Path(args.output_dir)
        _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _CHECK_ONLY = args.check

    import notebook_content as nc
    _results = []

    def _write_nb_collect(name, cells):
        _results.append(write_nb(name, cells))

    nc.build(md, code, colab_badge, bootstrap_cells, _write_nb_collect)
    if _CHECK_ONLY:
        ok = all(_results)
        print("\nIDEMPOTENCE:", "all notebooks match" if ok else "SOME DIFFER")
        sys.exit(0 if ok else 1)
    print("\nAll notebooks written.")
