"""Presentation helpers for independently executed, real-data iteration02 notebooks.

This module does not fit a predictor, change an observation rule, or choose a run.
It loads an explicit verified run and turns its saved evidence into teaching views.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments/aaai"
COLORS = ("#087f84", "#315cb5", "#bd5b14", "#6a51a3")
JOINT_NAMES = {
    11: "Left shoulder", 12: "Right shoulder", 13: "Left elbow",
    14: "Right elbow", 15: "Left wrist", 16: "Right wrist",
    23: "Left hip", 24: "Right hip", 25: "Left knee", 26: "Right knee",
    27: "Left ankle", 28: "Right ankle", 29: "Left heel", 30: "Right heel",
    31: "Left foot index", 32: "Right foot index",
}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_versions() -> pd.DataFrame:
    records = [{"component": "Python", "version": platform.python_version()},
               {"component": "interpreter", "version": sys.executable}]
    for name in ("numpy", "pandas", "scikit-learn", "torch", "matplotlib",
                 "python-dotenv", "nbformat", "nbclient", "opencv-python"):
        try:
            value = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            value = "not installed"
        records.append({"component": name, "version": value})
    return pd.DataFrame(records)


class EvidenceRun:
    """A deliberately explicit run, checked before any numerical result is shown."""

    def __init__(self, run_dir: str | Path):
        self.path = Path(run_dir).expanduser().resolve()
        parent = (BASE / "artifacts/iteration02").resolve()
        if not self.path.is_relative_to(parent):
            raise ValueError(f"Use an iteration02 run below {parent}, not a historical result.")
        manifest_path = self.path / "run_manifest.json"
        verification_path = self.path / "verification.json"
        if not manifest_path.is_file() or not verification_path.is_file():
            raise FileNotFoundError(
                f"A completed, verified run is required at {self.path}. "
                "Run the iteration02 producer and verifier before executing this notebook. "
                "No synthetic or historical fallback is permitted."
            )
        self.manifest = json.loads(manifest_path.read_text())
        self.verification = json.loads(verification_path.read_text())
        self.protocol_hash = str(self.manifest.get("protocol_hash", self.path.name))
        if not self.protocol_hash or "synthetic" in json.dumps(self.manifest.get("data_kind", "")).lower():
            raise ValueError("The notebook requires a named real-data protocol.")
        self.mode = str(self.manifest.get("mode", "full"))
        kind = str(self.manifest.get("data_kind", self.manifest.get("dataset", ""))).lower()
        if not any(token in kind for token in ("gavd", "real")):
            raise ValueError("run_manifest.json must explicitly identify real GAVD data.")
        complete = str(self.manifest.get("status", "")).lower() in {"complete", "completed", "verified"}
        if not complete:
            raise ValueError("The producer has not marked this run complete. Do not report partial results as final.")
        passed = self.verification.get("passed", self.verification.get("verified", False))
        if not passed and str(self.verification.get("status", "")).lower() not in {"passed", "verified"}:
            raise ValueError("The saved evidence verifier did not pass.")
        verified_identity = self.verification.get("protocol_hash")
        if verified_identity and str(verified_identity) != self.protocol_hash:
            raise ValueError("Verification belongs to a different protocol.")
        if self.mode not in {"full", "full_study"} and os.getenv("AAAI_ITERATION02_ALLOW_DEVELOPMENT") != "1":
            raise ValueError("For a development run, explicitly set AAAI_ITERATION02_ALLOW_DEVELOPMENT=1.")
        self.figure_dir = self.path / "figures/notebooks"
        self.figure_dir.mkdir(parents=True, exist_ok=True)
        self.loaded: dict[str, str] = {}
        self.assert_declared_hashes()

    @classmethod
    def from_env(cls) -> "EvidenceRun":
        value = os.getenv("AAAI_ITERATION02_RUN_DIR", "").strip()
        if not value:
            raise RuntimeError(
                "Set AAAI_ITERATION02_RUN_DIR to the exact verified run path. "
                "This notebook never chooses the newest directory automatically."
            )
        return cls(value)

    def assert_declared_hashes(self) -> None:
        """Check the producer's recorded immutable inputs, not generated figures."""
        entries = self.manifest.get("artifacts", {})
        hashes = self.manifest.get("artifact_hashes", self.manifest.get("artifact_sha256", {}))
        checks: dict[str, str] = {}
        if isinstance(entries, dict):
            for key, entry in entries.items():
                if isinstance(entry, dict) and entry.get("sha256"):
                    checks[str(entry.get("path", key))] = str(entry["sha256"])
        if isinstance(hashes, dict):
            checks.update({str(key): str(value) for key, value in hashes.items() if isinstance(value, str)})
        for relative, expected in checks.items():
            path = Path(relative)
            if not path.is_absolute():
                path = self.path / path
            if not path.is_file() or file_hash(path) != expected:
                raise ValueError(f"Saved artifact is missing or changed: {relative}")
        self.checked_hash_count = len(checks)

    def artifact_path(self, filename: str, *, required: bool = True) -> Path | None:
        entries = self.manifest.get("artifacts", {})
        entry = entries.get(filename, filename) if isinstance(entries, dict) else filename
        if isinstance(entry, dict):
            if entry.get("status") in {"unavailable", "not_run", "failed"}:
                if required:
                    raise FileNotFoundError(f"Required artifact {filename}: {entry.get('reason', entry['status'])}")
                return None
            entry = entry.get("path", filename)
        path = Path(str(entry))
        if not path.is_absolute():
            path = self.path / path
        # The run names its immutable cohort explicitly. This is a provenance
        # lookup, never a search for another run or a replacement dataset.
        if not path.is_file() and filename == "manifest.csv" and self.manifest.get("cohort_directory"):
            path = Path(self.manifest["cohort_directory"]) / "manifest.csv"
        if not path.is_file():
            if required:
                command = self.manifest.get("producer_command", "the iteration02 producer command")
                raise FileNotFoundError(f"Missing required real artifact: {path}. Producer: {command}")
            return None
        self.loaded[filename] = file_hash(path)
        return path

    def table(self, filename: str, *, required: bool = True, columns: Iterable[str] = ()) -> pd.DataFrame:
        path = self.artifact_path(filename, required=required)
        if path is None:
            return pd.DataFrame()
        frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"{filename} lacks required columns: {sorted(missing)}")
        return frame

    def json(self, filename: str, *, required: bool = True):
        path = self.artifact_path(filename, required=required)
        return None if path is None else json.loads(path.read_text())

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "protocol_hash": self.protocol_hash,
            "mode": self.mode,
            "status": self.manifest.get("status"),
            "run_directory": str(self.path),
            "producer_command": self.manifest.get("producer_command", "See run_manifest.json"),
            "artifact_hashes_checked": self.checked_hash_count,
            "data_kind": self.manifest.get("data_kind", self.manifest.get("dataset")),
        }])

    def record(self, notebook: str) -> Path:
        """Record read inputs without implying that the notebook fitted the model."""
        directory = self.path / "notebook_records"
        directory.mkdir(exist_ok=True)
        path = directory / f"{notebook}.json"
        path.write_text(json.dumps({
            "notebook": notebook, "protocol_hash": self.protocol_hash,
            "mode": self.mode, "data_kind": "real_gavd",
            "operation": "independent artifact review and recomputation",
            "inputs_read_sha256": self.loaded,
            "versions": package_versions().to_dict("records"),
        }, indent=2))
        return path


def show_status(run: EvidenceRun, filename: str):
    """Display a required status ledger, including unavailable comparisons."""
    from IPython.display import display, Markdown
    value = run.json(filename)
    if isinstance(value, list):
        display(pd.json_normalize(value))
    elif isinstance(value, dict):
        display(pd.json_normalize(value).T.rename(columns={0: "recorded value"}))
    else:
        display(Markdown(str(value)))
    return value


def optional_evidence(run: EvidenceRun, filename: str, description: str):
    """Make an unavailable analysis visible without inventing its output."""
    from IPython.display import display, Markdown
    path = run.artifact_path(filename, required=False)
    if path is None:
        display(Markdown(f"**Not recorded in this run: {description}.** No result is inferred. Consult the required comparison-status ledger before calling the study complete."))
        return None
    if path.suffix in {".csv", ".parquet"}:
        value = run.table(filename)
        display(value)
    else:
        value = run.json(filename)
        display(pd.json_normalize(value).T if isinstance(value, dict) else value)
    return value


def fold_json_records(run: EvidenceRun, relative: str, *, field: str | None = None):
    """Read named fold artifacts in place, retaining every producing fold."""
    records = []
    for fold in range(4):
        name = f"fold_{fold}/{relative}"
        value = run.json(name)
        value = value[field] if field else value
        records.append({"outer_fold": fold, "artifact": name, "record": value})
    return records


def fold_tables(run: EvidenceRun, relative: str) -> pd.DataFrame:
    return pd.concat([run.table(f"fold_{fold}/{relative}").assign(outer_fold=fold)
                      for fold in range(4)], ignore_index=True)


def cohort_table(run: EvidenceRun, filename: str) -> pd.DataFrame:
    path = Path(run.manifest["cohort_directory"]) / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    run.loaded[str(path)] = file_hash(path)
    return pd.read_csv(path)


def external_replication(run: EvidenceRun) -> dict:
    """Follow the core run's explicit, hashed independent-study linkage."""
    link = run.json("external_replication.json")
    path = Path(link["run_directory"]).resolve()
    if not path.is_relative_to(BASE / "artifacts/iteration02"):
        raise ValueError("The external linkage does not name an expected local evidence directory.")
    for name, expected in link["artifact_sha256"].items():
        actual = file_hash(path / name)
        if actual != expected:
            raise ValueError(f"The linked external artifact changed: {name}")
        run.loaded[str(path / name)] = actual
    completion = json.loads((path / "completion.json").read_text())
    if completion["status"] != "complete":
        raise ValueError("The linked external run is not complete.")
    if path.name != "external_" + completion["protocol_hash"]:
        raise ValueError("The external protocol identity does not match its directory.")
    return {name: json.loads((path / filename).read_text()) for name, filename in {
        "completion": "completion.json", "verification": "adversarial_verification.json",
        "metrics": "metrics.json", "paired": "paired_effects.json",
        "age_sex": "age_sex_sensitivity.json"}.items()}


def timing_video_example(run: EvidenceRun):
    """Preview one exact recorded parent, rather than substituting a nearby window."""
    paths = sorted((run.path / "timing").glob("fold_*/outer/target_rows/*.csv.gz"))
    if not paths:
        paths = sorted((run.path / "timing").glob("fold_*/test/target_rows/*.csv.gz"))
    if not paths:
        from IPython.display import display, Markdown
        display(Markdown("No outer-test target ledger is available for an exact timing-parent preview."))
        return None
    path = paths[0]
    run.loaded[str(path.relative_to(run.path))] = file_hash(path)
    target = pd.read_csv(path, nrows=1).iloc[0]
    parents = cohort_table(run, "timing_window_index.csv")
    key = "parent_index" if "parent_index" in parents else "window_index"
    matches = parents.loc[parents[key].eq(int(target.parent_index))]
    if len(matches) != 1 or str(matches.source_id.iloc[0]) != str(target.source_id):
        raise ValueError("The timing target does not identify one original parent.")
    preview = matches.iloc[0].copy()
    preview["start_seconds"] = preview.parent_start_seconds
    preview["end_seconds"] = preview.parent_end_seconds
    preview["window_id"] = preview.parent_window_id
    return show_video(run, preview)


def mask_bank_summary(run: EvidenceRun) -> pd.DataFrame:
    records = []
    for bank in ("U", "E_leg"):
        path = run.artifact_path(f"mask_bank_{bank}.npz")
        with np.load(path, allow_pickle=False) as values:
            hidden, targets = values["hidden"], values["targets"]
        assert hidden.shape == targets.shape == (16, 128)
        assert hidden.dtype == targets.dtype == bool
        assert np.all(hidden.sum(1) == 64)
        assert not (targets & ~hidden).any()
        expected_targets = 64 if bank == "U" else 8
        assert np.all(targets.sum(1) == expected_targets)
        records.append({"bank": bank, "draws": len(hidden), "hidden_per_draw": 64,
                        "requested_targets_per_draw": expected_targets,
                        "file_sha256": file_hash(path),
                        "token_hiding_frequency_min": float(hidden.mean(0).min()),
                        "token_hiding_frequency_max": float(hidden.mean(0).max())})
    return pd.DataFrame(records)


def measured_window(run: EvidenceRun, example: pd.Series) -> np.ndarray:
    metadata = run.table("window_metadata.csv")
    if "cohort_window_index" in example:
        matching = metadata.loc[metadata.window_index.eq(int(example.cohort_window_index))]
    else:
        matching = metadata.loc[metadata.sequence_id.eq(example.sequence_id)
                                & np.isclose(metadata.start_seconds, float(example.start_seconds))]
    if len(matching) != 1 or str(matching.source_id.iloc[0]) != str(example.source_id):
        raise ValueError("The example does not identify one exact original cohort window.")
    path = Path(run.manifest["cohort_directory"]) / "windows.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    run.loaded[str(path)] = file_hash(path)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if len(array) != len(metadata):
        raise ValueError("The cohort array and metadata have different lengths.")
    return np.asarray(array[int(matching.index[0])])


def show_diagram(name: str):
    from IPython.display import display, SVG, Markdown
    path = BASE / "images/iteration02" / f"{name}.svg"
    if not path.is_file():
        raise FileNotFoundError(path)
    display(SVG(filename=str(path)))
    display(Markdown("This diagram explains the design. Its labels do not establish an empirical result."))


def joint_table() -> pd.DataFrame:
    records = []
    for index, name in sorted(JOINT_NAMES.items()):
        feature = ("Primary lower-leg coordinate readout and context" if index >= 25
                   else "Shared body reference and own-leg context" if index in (23, 24)
                   else "Optional upper-body context; no primary readout")
        records.append({"BLAZEPOSE_33 index": index, "Keypoint name": name, "Features involved": feature})
    return pd.DataFrame(records)


def source_role_table(registry: pd.DataFrame) -> pd.DataFrame:
    columns = [name for name in registry if name.startswith("role_fold_")]
    if not columns:
        raise ValueError("Source registry has no explicit outer-fold roles.")
    result = registry.melt(id_vars=["source_id", "binary_label"], value_vars=columns,
                           var_name="outer_fold", value_name="role")
    result["outer_fold"] = result.outer_fold.str.rsplit("_", n=1).str[-1].astype(int)
    if result.duplicated(["source_id", "outer_fold"]).any():
        raise ValueError("A source has multiple roles in one fold.")
    if result.loc[result.role.isin(["train", "fit", "val", "reference"]), "binary_label"].ne(0).any():
        raise ValueError("An abnormal source entered normal-only fitting or calibration.")
    return result


def choose_window(run: EvidenceRun, label: str = "normal") -> pd.Series:
    windows = run.table("window_metadata.csv", columns=("source_id", "sequence_id", "start_seconds", "end_seconds"))
    if "label" not in windows:
        manifest = run.table("manifest.csv", columns=("sequence_id", "label"))
        windows = windows.merge(manifest[["sequence_id", "label"]], on="sequence_id", validate="many_to_one")
    registered = set(run.table("source_registry.csv", columns=("source_id",)).source_id)
    choices = windows.loc[windows.label.eq(label) & windows.source_id.isin(registered)].sort_values(["source_id", "sequence_id", "start_seconds"])
    if choices.empty:
        raise ValueError(f"The verified run has no eligible {label} video example.")
    return choices.iloc[0]


def show_video(run: EvidenceRun, row: pd.Series | dict | None = None, *, label: str = "normal", overlay: bool = True):
    """Display the exact stored interval and its extracted image-coordinate pose."""
    from IPython.display import display, Markdown
    from .media import show_window_context
    row = choose_window(run, label) if row is None else pd.Series(row)
    manifest = run.table("manifest.csv", columns=("source_id", "sequence_id"))
    show_window_context(row, manifest)
    if overlay:
        fig = pose_overlay(run, row)
        if fig is not None:
            display(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
    display(Markdown("The folder label is an annotation, not a diagnosis verified by this experiment."))
    return row


def pose_overlay(run: EvidenceRun, row: pd.Series | dict):
    import cv2
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown
    from .data import find_video, load_pose
    from .media import EDGES
    row = dict(row)
    pose_dir = Path(run.manifest.get("pose_dir", BASE / "artifacts/poses"))
    if not pose_dir.is_absolute():
        pose_dir = ROOT / pose_dir
    path = pose_dir / f"{row['sequence_id']}.npz"
    video = find_video(ROOT, str(row["source_id"]))
    if video is None or not path.is_file():
        display(Markdown("The matching local video or pose cache is unavailable. No alternative clip was substituted."))
        return None
    pose = load_pose(path)
    provenance = json.loads(str(pose["provenance"]))
    base = int(provenance["frame_number_base"])
    fps = float(pose["fps"])
    times = (pose["frame_numbers"] - base) / fps
    target_time = float(row["start_seconds"])
    idx = int(np.argmin(abs(times - target_time)))
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(pose["frame_numbers"][idx]) - base)
    ok, image = cap.read()
    cap.release()
    if not ok:
        display(Markdown("The requested source frame cannot be decoded. The missing overlay remains explicit."))
        return None
    xy = pose["raw_xy"][idx]
    valid = (pose["confidence"][idx] >= 0.5) & np.isfinite(xy).all(axis=1)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for left, right in EDGES:
        if valid[left] and valid[right]:
            ax.plot(xy[[left, right], 0], xy[[left, right], 1], color="#ffd166", linewidth=2)
    selected = [joint for joint in JOINT_NAMES if valid[joint]]
    ax.scatter(xy[selected, 0], xy[selected, 1], s=15, c="#5ce0d8", edgecolors="#172e47", linewidths=0.4)
    ax.set_title(f"Recorded pose at source time {times[idx]:.3f} s\n{row['source_id']} | {row['sequence_id']}")
    ax.axis("off")
    fig.tight_layout()
    return fig


def save_figure(run: EvidenceRun, fig, stem: str, *, caption: str, inputs: Iterable[str]):
    import matplotlib as mpl
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    paths = {}
    for suffix in ("svg", "pdf", "png"):
        path = run.figure_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=130)
        paths[suffix] = str(path)
    provenance = {
        "protocol_hash": run.protocol_hash, "data_kind": "measured_real_gavd",
        "caption": caption,
        "input_sha256": {name: file_hash(run.artifact_path(name)) for name in inputs},
        "outputs": paths,
    }
    (run.figure_dir / f"{stem}.json").write_text(json.dumps(provenance, indent=2))
    return paths


def plot_counts(run: EvidenceRun):
    import matplotlib.pyplot as plt
    registry = run.table("source_registry.csv", columns=("source_id", "binary_label"))
    windows = run.table("window_metadata.csv", columns=("source_id", "sequence_id", "label"))
    unique = windows.loc[windows.source_id.isin(registry.source_id), ["source_id", "label"]].drop_duplicates()
    counts = unique.groupby("label").source_id.nunique().sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, len(counts) * .34)))
    ax.barh(counts.index, counts.values, color=[COLORS[0] if name == "normal" else COLORS[1] for name in counts.index])
    ax.set_xlabel("Distinct source videos with this label")
    ax.set_title(f"Measured observation population: {registry.source_id.nunique()} registered sources")
    for i, value in enumerate(counts):
        ax.text(value + max(counts.max() * .01, .2), i, str(value), va="center", fontsize=10)
    ax.set_xlim(0, counts.max() * 1.16)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(run, fig, "12_source_population", caption="Measured eligible source counts. A multi-label source may appear in more than one row.", inputs=("source_registry.csv", "window_metadata.csv"))
    return fig


def within_fold_metrics(scores: pd.DataFrame, score: str, *, endpoint: str = "focal") -> tuple[pd.DataFrame, float]:
    """Recompute the prespecified fixed pair-count-weighted AUROC from source rows."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    if endpoint == "focal":
        membership = scores.focal_eligible.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
        if membership.isna().any():
            raise ValueError("focal_eligible contains an unknown or missing Boolean value.")
        table = scores.loc[membership].copy()
    elif endpoint == "broad":
        table = scores.copy()
    else:
        raise ValueError("Endpoint must be focal or broad.")
    if table.duplicated(["outer_fold", "source_id"]).any():
        raise ValueError("Select exactly one policy and one evaluation bank first.")
    if set(table.outer_fold.unique()) != {0, 1, 2, 3}:
        raise ValueError("A complete primary comparison requires every one of the four outer folds.")
    if "cluster_id" in table:
        # Independently reconstruct the declared mean-score reduction of confirmed duplicate footage.
        reduced = []
        for cluster, group in table.groupby("cluster_id", dropna=False, sort=True):
            if pd.isna(cluster):
                raise ValueError("Every source requires an explicit recording-cluster identity.")
            if group.outer_fold.nunique() != 1 or group.binary_label.nunique() != 1:
                raise ValueError("A known duplicate-recording group crosses folds or binary labels.")
            values = group[score].to_numpy(float)
            if not np.isfinite(values).all():
                raise ValueError("A recording-group score is not finite.")
            reduced.append({"source_id": str(cluster), "outer_fold": int(group.outer_fold.iloc[0]),
                            "binary_label": int(group.binary_label.iloc[0]), score: float(values.mean())})
        table = pd.DataFrame(reduced)
    records = []
    for fold, group in table.groupby("outer_fold", sort=True):
        y = group.binary_label.to_numpy(int)
        values = group[score].to_numpy(float)
        if len(np.unique(y)) != 2 or not np.isfinite(values).all():
            raise ValueError("Every scored fold requires both labels and finite scores.")
        negatives, positives = int((y == 0).sum()), int((y == 1).sum())
        records.append({"outer_fold": int(fold), "n_normal": negatives, "n_abnormal": positives,
                        "pair_count": negatives * positives, "prevalence": float(y.mean()),
                        "auroc": float(roc_auc_score(y, values)),
                        "average_precision": float(average_precision_score(y, values))})
    result = pd.DataFrame(records)
    value = float(np.average(result.auroc, weights=result.pair_count))
    return result, value


def plot_source_scores(run: EvidenceRun, *, policy: str = "U", bank: str = "U"):
    import matplotlib.pyplot as plt
    scores = run.table("source_scores.csv", columns=("policy", "bank", "binary_label", "M", "R", "S"))
    selected = scores.loc[scores.policy.eq(policy) & scores.bank.eq(bank)]
    if selected.empty:
        raise ValueError(f"No saved source scores for policy {policy}, bank {bank}.")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, column in zip(axes, ("M", "R", "S")):
        for label, color, name in ((0, COLORS[0], "normal"), (1, COLORS[1], "abnormal")):
            values = np.sort(selected.loc[selected.binary_label.eq(label), column].to_numpy(float))
            if values.size:
                ax.step(values, np.arange(1, len(values) + 1) / len(values), where="post", label=name, color=color)
        ax.set_title(column)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("Saved source score (symlog axis)")
        ax.set_ylabel("Fraction of sources at or below score")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle(f"Measured score distributions | policy {policy}, bank {bank}")
    fig.tight_layout()
    save_figure(run, fig, "15_score_distributions", caption="Descriptive empirical score distributions. Cross-fold pooled separation is not the primary AUROC estimator.", inputs=("source_scores.csv",))
    return fig


def plot_metric_table(run: EvidenceRun):
    import matplotlib.pyplot as plt
    table = run.table("metrics.csv", columns=("metric", "method", "endpoint", "value"))
    if "policy" in table:
        table = table.loc[table.policy.eq("U")]
    if "bank" in table:
        table = table.loc[table.bank.isin(["U", "E_U"])]
    auroc = table.loc[table.metric.isin(["auroc", "weighted_auroc", "pair_weighted_auroc"])]
    if auroc.empty:
        raise ValueError("metrics.csv contains no named primary AUROC rows.")
    endpoints = list(dict.fromkeys(auroc.endpoint))
    fig, axes = plt.subplots(1, len(endpoints), figsize=(6 * len(endpoints), max(4, auroc.method.nunique() * .3)), squeeze=False)
    for ax, endpoint in zip(axes.ravel(), endpoints):
        group = auroc.loc[auroc.endpoint.eq(endpoint)].sort_values("method")
        ax.scatter(group.value, np.arange(len(group)), c=COLORS[0], s=35)
        ax.set_yticks(np.arange(len(group)), group.method)
        ax.set_xlim(0, 1)
        ax.axvline(.5, linestyle=":", color="#778899")
        ax.set_xlabel("Recorded within-fold weighted AUROC")
        ax.set_title(str(endpoint))
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Measured source-level comparisons")
    fig.tight_layout()
    save_figure(run, fig, "16_source_metrics", caption="Recorded source-level AUROC estimates. Inspect the paired-difference interval table before claiming an improvement.", inputs=("metrics.csv",))
    return fig


def plot_paired_differences(run: EvidenceRun):
    import matplotlib.pyplot as plt
    intervals = run.json("paired_intervals.json")
    rows = []
    for item in intervals:
        if item.get("status") != "complete" or item.get("bank") != "U":
            continue
        if item.get("contrast") == "method" and item.get("policy") != "U":
            continue
        low, high = item["conditional_95_interval"]
        label = f"{item['first']} minus {item['second']}"
        if item.get("contrast") == "mask_policy":
            label = f"Mask {label}"
        rows.append({"endpoint": item["endpoint"], "label": label,
                     "estimate": item["observed"], "low": low, "high": high})
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("No complete fixed-U paired interval was saved.")
    endpoints = list(dict.fromkeys(table.endpoint))
    fig, axes = plt.subplots(1, len(endpoints), figsize=(12, 5), squeeze=False)
    for ax, endpoint in zip(axes.ravel(), endpoints):
        subset = table.loc[table.endpoint.eq(endpoint)].reset_index(drop=True)
        for i, row in subset.iterrows():
            ax.plot([row.low, row.high], [i, i], color=COLORS[0], linewidth=2)
            ax.scatter([row.estimate], [i], color=COLORS[0], s=25)
        ax.axvline(0, color="#778899", linestyle=":")
        ax.set_yticks(range(len(subset)), subset.label)
        ax.invert_yaxis()
        ax.set_title(str(endpoint))
        ax.set_xlabel("Paired difference in weighted AUROC")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Measured paired differences and conditional 95% intervals")
    fig.tight_layout()
    save_figure(run, fig, "16_paired_differences", caption="Paired source-cluster bootstrap intervals with fixed original fold weights. Intervals condition on the fitted models, selected split and exploratory protocol.", inputs=("paired_intervals.json",))
    return fig


def plot_condition_confusions(run: EvidenceRun):
    """Display actual held-out source counts, separate from anomaly metrics."""
    import matplotlib.pyplot as plt
    records = run.json("five_class_metrics.json")
    methods = ["random_forest", "learned_jepa_probe", "random_encoder_probe"]
    names = ["Random Forest", "Learned JEPA probe", "Random encoder probe"]
    labels = records[0]["labels"]
    assert all(item["labels"] == labels for item in records)
    matrices = []
    for method in methods:
        selected = [item for item in records if item["method"] == method]
        assert {item["outer_fold"] for item in selected} == {0, 1, 2, 3}
        matrix = np.sum([np.asarray(item["confusion_matrix"], int) for item in selected], axis=0)
        assert int(matrix.sum()) == sum(item["n_test_sources"] for item in selected)
        matrices.append(matrix)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    ticks = [name.replace(" ", "\n") for name in labels]
    maximum = max(matrix.max() for matrix in matrices)
    for ax, matrix, name in zip(axes, matrices, names):
        ax.imshow(matrix, vmin=0, vmax=maximum, cmap="Blues")
        for i, j in np.ndindex(matrix.shape):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if matrix[i, j] > maximum * .55 else "#172e47")
        ax.set_xticks(range(len(labels)), ticks, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)), ticks, fontsize=8)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("Dataset label")
        ax.set_title(name)
    fig.suptitle("Measured five-class held-out source counts | four folds combined")
    fig.tight_layout()
    save_figure(run, fig, "16_condition_confusions", caption="Summed held-out source confusion counts. These supervised probes use condition labels and are separate from normal-only anomaly detection.", inputs=("five_class_metrics.json",))
    return fig


def _array(value) -> np.ndarray:
    parsed = json.loads(value) if isinstance(value, str) else value
    array = np.asarray(parsed, dtype=float)
    if array.size != 8:
        raise ValueError("A four-frame 2D coordinate patch must contain eight values.")
    return array.reshape(4, 2)


def _column(table: pd.DataFrame, choices: tuple[str, ...]) -> str:
    for name in choices:
        if name in table:
            return name
    raise ValueError(f"Saved artifact needs one of these named columns: {choices}")


def prediction_example(run: EvidenceRun) -> tuple[pd.Series, dict[str, np.ndarray]]:
    table = run.table("target_predictions.csv", columns=("source_id", "sequence_id"))
    if "binary_label" in table:
        table = table.loc[table.binary_label.eq(0)]
    elif "label" in table:
        table = table.loc[table.label.eq("normal")]
    else:
        registry = run.table("source_registry.csv", columns=("source_id", "binary_label"))
        table = table.merge(registry[["source_id", "binary_label"]], on="source_id", validate="many_to_one")
        table = table.loc[table.binary_label.eq(0)]
    if "policy" in table:
        table = table.loc[table.policy.eq("U")]
    if "bank" in table:
        table = table.loc[table.bank.isin(["U", "E_U"])]
    if table.empty:
        raise ValueError("No deterministic real normal coordinate example was saved for the uniform reference.")
    keys = [name for name in ("outer_fold", "source_id", "sequence_id", "window_index", "joint", "patch_index", "mask_id") if name in table]
    row = table.sort_values(keys).iloc[0]
    fields = {
        "Observed target": ("Y", "target", "y", "target_json"),
        "Local baseline f": ("f", "f0", "pred_f", "prediction_f"),
        "Own-leg correction f+h": ("fh", "f_h", "pred_fh", "prediction_fh"),
        "Both-leg correction f+g": ("fg", "f_g", "pred_fg", "prediction_fg"),
    }
    expanded = {"Observed target": "Y", "Local baseline f": "base",
                "Own-leg correction f+h": "own", "Both-leg correction f+g": "both"}
    arrays = {}
    for name, candidates in fields.items():
        prefix = expanded[name]
        columns = [f"{prefix}_{index}" for index in range(8)]
        arrays[name] = (_array(row[columns].to_numpy(float)) if set(columns).issubset(table.columns)
                        else _array(row[_column(table, candidates)]))
    return row, arrays


def plot_prediction(run: EvidenceRun):
    import matplotlib.pyplot as plt
    row, arrays = prediction_example(run)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for axis, ax in enumerate(axes):
        for i, (name, values) in enumerate(arrays.items()):
            ax.plot(np.arange(4), values[:, axis], marker="o", markersize=4,
                    color="#172e47" if i == 0 else COLORS[i - 1],
                    linewidth=2 if i == 0 else 1.5, linestyle="-" if i == 0 else "--", label=name)
        ax.set_title(("Horizontal", "Vertical")[axis] + " coordinate")
        ax.set_xlabel("Frame within the same hidden patch")
        ax.set_ylabel("Declared body-normalized coordinate")
        ax.set_xticks(range(4))
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.04))
    title = f"Measured normal prediction | joint {row.get('joint', 'recorded in table')}"
    fig.suptitle(title)
    fig.tight_layout(rect=(0, .13, 1, .94))
    save_figure(run, fig, "13_normal_prediction", caption="One deterministic real normal coordinate target and saved fitted predictions. This selected illustration is not an accuracy estimate.", inputs=("target_predictions.csv",))
    return fig


def example_losses(run: EvidenceRun) -> pd.DataFrame:
    row, arrays = prediction_example(run)
    target = arrays["Observed target"]
    records = [{"prediction": name, "mean_squared_error": float(np.mean((values - target) ** 2))}
               for name, values in arrays.items() if name != "Observed target"]
    return pd.DataFrame(records)


def plot_adequacy(run: EvidenceRun):
    import matplotlib.pyplot as plt
    table = run.table("adequacy.csv")
    fold_col = _column(table, ("outer_fold", "fold"))
    policy_col = _column(table, ("policy", "training_policy"))
    own = _column(table, ("mse_fh", "fh_mse", "own_capacity_mse", "loss_fh", "own"))
    both = _column(table, ("mse_fg", "fg_mse", "both_leg_mse", "loss_fg", "both"))
    shown = table.copy()
    if "bank" in shown:
        shown = shown.loc[shown.bank.isin(["U", "E_U"])]
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (policy, group) in enumerate(shown.groupby(policy_col, sort=True)):
        group = group.sort_values(fold_col)
        ax.scatter(group[fold_col] + .08 * (i - 1), group[own] - group[both], label=policy, color=COLORS[i % len(COLORS)])
    ax.axhline(0, color="#778899", linestyle=":")
    ax.set_xlabel("Outer fold; normal development only")
    ax.set_xticks(range(4))
    ax.set_ylabel("Own-capacity MSE minus both-leg MSE")
    ax.set_title("Measured normal-development predictive benefit")
    ax.legend(title="Training mask", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(run, fig, "13_normal_adequacy", caption="Normal-development benefit on a common coordinate target. Positive values favor the both-leg correction; adequacy also requires the recorded gates.", inputs=("adequacy.csv",))
    return fig


def plot_timing(run: EvidenceRun):
    import matplotlib.pyplot as plt
    table = run.table("timing_pairs.csv")
    if table.empty:
        from IPython.display import display, Markdown
        display(Markdown("The timing artifact has no eligible paired observations. A null or positive timing effect cannot be claimed."))
        return None
    offset = _column(table, ("shift_frames", "offset_frames", "offset"))
    aligned = _column(table, ("aligned_loss", "loss_aligned", "mse_aligned", "loss_both_aligned"))
    shifted = _column(table, ("shifted_loss", "loss_shifted", "mse_shifted", "loss_both_edited"))
    shown = table.copy()
    if "variant" in shown:
        shown = shown.loc[shown.variant.eq("timing")]
    if "scope" in shown:
        shown = shown.loc[shown.scope.eq("outer_test_normal")]
    if "eligible" in shown:
        eligible = shown.eligible.astype(str).str.lower().isin(["true", "1"])
        shown = shown.loc[eligible]
    shown["effect"] = shown[shifted] - shown[aligned]
    # One source contributes one mean to this descriptive diagnostic, not many independent targets.
    keys = [name for name in ("outer_fold", "source_id", "policy", offset) if name in shown]
    source = shown.groupby(keys, as_index=False).effect.mean()
    if "policy" in source:
        source = source.loc[source.policy.eq("U")]
    if source.empty:
        from IPython.display import display, Markdown
        display(Markdown("There are no eligible U-policy timing pairs. No effect curve is drawn."))
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    offsets = sorted(source[offset].unique())
    for i, value in enumerate(offsets):
        values = np.sort(source.loc[source[offset].eq(value), "effect"].to_numpy(float))
        # Deterministic display spread carries no statistical meaning.
        spread = np.linspace(-.13, .13, len(values)) if len(values) > 1 else np.zeros(len(values))
        ax.scatter(i + spread, values, color=COLORS[i % len(COLORS)], s=20, alpha=.7)
        if len(values):
            ax.plot([i - .18, i + .18], [np.median(values)] * 2, color="#172e47", linewidth=2)
    ax.axhline(0, color="#778899", linestyle=":")
    ax.set_xticks(range(len(offsets)), [f"{int(value):+d} frames" for value in offsets])
    ax.set_ylabel("Frozen both-leg loss: shifted minus aligned")
    adequacy = run.table("timing/paired_support_adequacy.csv")
    failed = not adequacy.paired_support_adequacy_passed.all()
    ax.set_title("Measured timing challenge | " + ("sparse-input adequacy failed" if failed else "U policy"))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(run, fig, "14_paired_timing", caption="Source-level paired input sensitivity. Distal timing and geometry both change; support and drift checks determine interpretability.", inputs=("timing_pairs.csv",))
    return fig
