"""Read-only source-run diagnostics; no model fitting or outcome-based selection.

Trajectory panels use physical timestamps and image pixels. Missing observations
and invalid references remain gaps. Curves diagnose individual examples, not
population performance. Training losses are displayed separately by fit/phase.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import textwrap

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl


_CONTROLS = ("unchanged", "joint_offset", "joint_affine")
_NEURAL = ("initialized", "coordinate", "direct", "paired_jepa")
_LABELS = {"reference": "Reference (occluded proxies included)", "unchanged": "Unchanged input",
           "joint_offset": "Joint offset", "joint_affine": "Joint affine",
           "initialized": "Initialized encoder", "coordinate": "Coordinate",
           "direct": "Direct", "paired_jepa": "Paired JEPA"}
_COLORS = {"reference": "#111111", "unchanged": "#999999", "joint_offset": "#0072B2",
           "joint_affine": "#D55E00", "initialized": "#CC79A7", "coordinate": "#009E73",
           "direct": "#E69F00", "paired_jepa": "#0072B2"}


def select_windows(records, max_windows=4) -> list[dict]:
    """Select canonical-person/window pairs using metadata alone.

    If split metadata is supplied, only development records are eligible. Sorted
    people contribute one sorted window each before any contributes a second.
    Selection deliberately ignores array indices, errors and model names.
    The caller must save this selection before computing model diagnostics.
    """
    if isinstance(max_windows, bool) or not isinstance(max_windows, int) or max_windows < 0:
        raise ValueError("max_windows must be a nonnegative integer")
    records = list(records)
    has_split = any("split" in row for row in records)
    candidates = set()
    for row in records:
        if has_split and row.get("split") != "development":
            continue
        person, window = row.get("canonical_person_id"), row.get("window_id")
        if not isinstance(person, str) or not person.strip() or not isinstance(window, str) or not window.strip():
            raise ValueError("Trajectory selection requires canonical_person_id and window_id strings")
        candidates.add((person, window))
    people = sorted({person for person, _ in candidates})
    by_person = {person: sorted(window for owner, window in candidates if owner == person) for person in people}
    selected = []
    for rank in range(max((len(windows) for windows in by_person.values()), default=0)):
        for person in people:
            if rank < len(by_person[person]) and len(selected) < max_windows:
                selected.append(dict(canonical_person_id=person, window_id=by_person[person][rank]))
    return selected


def _slug(value):
    return (re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value)).strip(".-") or "unnamed")[:55]


def _stem(prefix, metadata):
    encoded = json.dumps(metadata, sort_keys=True, ensure_ascii=True).encode()
    return prefix + "-" + "-".join(_slug(v) for v in metadata.values()) + "-" + hashlib.sha256(encoded).hexdigest()[:10]


def _save_figure(figure, folder, stem):
    paths = []
    with mpl.rc_context({"svg.fonttype": "none", "font.family": "DejaVu Sans"}):
        for extension in ("svg", "png"):
            path = folder / f"{stem}.{extension}"
            figure.savefig(path, dpi=145, facecolor="white")
            paths.append(str(path))
    figure.clear()
    return paths


def _prediction_series(predictions, shape):
    """Accept exact method names or seed suffixes, but never select among seeds."""
    result = {}
    labels = {}
    for key, value in predictions.items():
        normalized = str(key).lower()
        for arm in _CONTROLS + _NEURAL:
            if normalized != arm and not re.match(re.escape(arm) + r"(?:[:/_.-]|\s)", normalized):
                continue
            if arm in result:
                raise ValueError(f"Multiple plot series for {arm}; caller must select one declared seed")
            array = np.asarray(value, dtype=float)
            if array.shape != shape:
                raise ValueError(f"Prediction {key} has shape {array.shape}, expected {shape}")
            result[arm], labels[arm] = array, str(key)
            break
    return result, labels


def plot_trajectories(output_dir, inputs, targets, records,
                      predictions: dict[str, np.ndarray], selection: list[dict]) -> list[str]:
    """Plot fixed windows with separate calibration/neural columns (4 x 2).

    Both ankle x/y trajectories are shown. Reference validity and observed-input
    masks are retained; model outputs are not hidden just because input is absent.
    One figure per window/extractor/render variant prevents nuisance averaging.
    """
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    xy, truth = np.asarray(inputs["xy"], float), np.asarray(targets["xy"], float)
    if xy.ndim != 4 or xy.shape[-2:] != (12, 2) or truth.shape != xy.shape:
        raise ValueError("Trajectory arrays must have matching [N,T,12,2] shape")
    n, t = xy.shape[:2]
    times = np.asarray(inputs["timestamps"], float)
    observed = np.asarray(inputs["observed"], bool)
    valid = np.asarray(targets["valid"], bool)
    visible = np.asarray(targets["visible"], bool)
    if (len(records) != n or times.shape != (n, t) or observed.shape != xy.shape[:-1]
            or valid.shape != xy.shape[:-1] or visible.shape != xy.shape[:-1]):
        raise ValueError("Trajectory metadata, times and masks must align with arrays")
    if not np.isfinite(times).all() or (np.diff(times, axis=1) <= 0).any():
        raise ValueError("Physical timestamps must be finite and strictly increasing")
    series, labels = _prediction_series(predictions, xy.shape)
    series["unchanged"] = np.where(observed[..., None], xy, np.nan)
    selected = [(row["canonical_person_id"], row["window_id"]) for row in selection]
    if len(selected) != len(set(selected)):
        raise ValueError("Trajectory selection contains duplicate windows")
    saved = []
    for person, window in selected:
        matching = [(index, row) for index, row in enumerate(records)
                    if (row.get("canonical_person_id"), row.get("window_id")) == (person, window)]
        if not matching:
            raise ValueError(f"Selected trajectory not present: {person} / {window}")
        strata = [(str(row.get("extractor", "unknown")), str(row.get("variant", "unknown"))) for _, row in matching]
        if len(strata) != len(set(strata)):
            raise ValueError("Repeated window/extractor/variant records: select one aligned dataset before plotting")
        for index, row in sorted(matching, key=lambda pair: (str(pair[1].get("extractor", "")), str(pair[1].get("variant", "")))):
            metadata = {"person": person, "window": window, "extractor": row.get("extractor", "unknown"),
                        "variant": row.get("variant", "unknown")}
            figure = Figure(figsize=(12.8, 11.5))
            FigureCanvasAgg(figure)
            axes = figure.subplots(4, 2, sharex=True, sharey="row", squeeze=False)
            trace_time = times[index] - times[index, 0]
            reference = np.where(valid[index, :, :, None], truth[index], np.nan)
            for column, arms in enumerate((_CONTROLS, _NEURAL)):
                shown = ["reference"] + [arm for arm in arms if arm in series]
                for axis_index, (joint, coordinate, title) in enumerate(((10, 0, "Left ankle x"), (10, 1, "Left ankle y"),
                                                                         (11, 0, "Right ankle x"), (11, 1, "Right ankle y"))):
                    axis = axes[axis_index, column]
                    for arm in shown:
                        values = reference[:, joint, coordinate] if arm == "reference" else series[arm][index, :, joint, coordinate]
                        values = np.where(np.isfinite(values), values, np.nan)
                        axis.plot(trace_time, values, label=_LABELS[arm], color=_COLORS[arm],
                                  linewidth=1.8 if arm == "reference" else 1.25,
                                  linestyle="--" if arm == "unchanged" else "-", alpha=.9)
                    axis.set_title(title, loc="left", fontsize=10, pad=5)
                    axis.grid(alpha=.18)
                    axis.tick_params(labelsize=8)
                    axis.spines[["top", "right"]].set_visible(False)
                    if column == 0:
                        axis.set_ylabel("Image pixels", fontsize=9)
                    if axis_index == 3:
                        axis.set_xlabel("Seconds from window start", fontsize=9)
                handles, legend_labels = axes[0, column].get_legend_handles_labels()
                figure.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.275 + .48 * column, .908),
                              frameon=False, ncol=2, fontsize=9, columnspacing=1.2, handlelength=2.5)
            heading = f"{person} | {window}\n{metadata['extractor']} | {metadata['variant']}"
            figure.suptitle(textwrap.fill(heading.split("\n")[0], width=105) + "\n" + heading.split("\n")[1],
                           fontsize=12, y=.991)
            figure.text(.275, .928, "Calibration controls", ha="center", fontsize=11, weight="bold")
            figure.text(.755, .928, "Learned restoration", ha="center", fontsize=11, weight="bold")
            keys = ", ".join(labels[arm] for arm in _NEURAL if arm in labels)
            bilateral = ((valid[index, :, 10:12] & visible[index, :, 10:12]).all(-1)
                         & np.isfinite(truth[index, :, 10:12]).all(axis=(1, 2)))
            caption = (f"Valid synthetic reference includes occluded proxies. Bilateral visible reference: {bilateral.sum()}/{t} frames. "
                       "Missing observations are gaps; no interpolation. Image y increases downward.")
            if keys:
                caption += "\nDisplayed model keys: " + keys
            figure.text(.065, .022, textwrap.fill(caption.split("\n")[0], width=148) +
                        ("\n" + textwrap.fill(caption.split("\n")[1], width=148) if "\n" in caption else ""), fontsize=8, va="bottom")
            figure.subplots_adjust(left=.07, right=.985, bottom=.095, top=.815, hspace=.32, wspace=.17)
            saved.extend(_save_figure(figure, folder, _stem("trajectory", metadata)))
    return saved


def _number(value):
    if isinstance(value, bool):
        return np.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return np.nan
    return number if np.isfinite(number) else np.nan


def _history_entries(report):
    history = report.get("history", [])
    if isinstance(history, list):
        return [dict(row) for row in history if isinstance(row, dict)]
    if isinstance(history, dict):
        for key in ("entries", "records"):
            if isinstance(history.get(key), list):
                return [dict(row) for row in history[key] if isinstance(row, dict)]
        return [dict(row, phase=row.get("phase", phase)) for phase, rows in history.items()
                if isinstance(rows, list) for row in rows if isinstance(row, dict)]
    return []


def _slope(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2 or np.ptp(x[ok]) == 0:
        return np.nan
    return float(np.polyfit(x[ok] - x[ok][0], y[ok], 1)[0])


def inspect_training(source_root, output_dir) -> pd.DataFrame:
    """Export fitting evidence and phase curves without judging convergence.

    One summary row per fit/phase. Elapsed time is fit-wide, repeated for context,
    and must not be summed over phases. Missing/nonfinite entries remain missing.
    The returned frame attrs contain ``figure_paths`` and ``history_path``.
    """
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    images = folder / "images"
    images.mkdir(exist_ok=True)
    summaries, histories, saved = [], [], []
    for path in sorted(Path(source_root).glob("fits/*/training.json")):
        report = json.loads(path.read_text())
        if not isinstance(report, dict):
            raise ValueError(f"Training report must be a JSON object: {path}")
        fit = path.parent.name
        identity = {"fit": fit, "arm": report.get("arm", fit), "seed": report.get("seed"),
                    "status": report.get("status", "unreported"), "termination": report.get("termination", "unreported")}
        entries = _history_entries(report)
        planned = {str(row["name"]): _number(row.get("updates")) for row in report.get("phases", [])
                   if isinstance(row, dict) and "name" in row}
        phases = list(planned) + sorted({str(row.get("phase", "unreported")) for row in entries} - set(planned))
        if not phases:
            phases = ["unreported"]
        figure = Figure(figsize=(min(18, max(6, 5.5 * len(phases))), 4.9))
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, len(phases), squeeze=False)[0]
        for phase, axis in zip(phases, axes):
            rows = [row for row in entries if str(row.get("phase", "unreported")) == phase]
            x = np.array([_number(row.get("phase_update", row.get("update", i + 1))) for i, row in enumerate(rows)])
            y = np.array([_number(row.get("loss")) for row in rows])
            order = np.argsort(x, kind="stable")
            x, y = x[order], y[order]
            ordered = [rows[i] for i in order]
            finite = np.isfinite(x) & np.isfinite(y)
            summary = dict(identity, phase=phase, planned_phase_updates=planned.get(phase, np.nan),
                           recorded_phase_updates=len(set(x[np.isfinite(x)])), history_entries=len(rows),
                           finite_loss_entries=int(finite.sum()), nonfinite_loss_entries=int((~np.isfinite(y)).sum()),
                           first_loss=float(y[finite][0]) if finite.any() else np.nan,
                           last_loss=float(y[finite][-1]) if finite.any() else np.nan,
                           loss_slope=_slope(x, y), tail_loss_slope=_slope(x[max(0, len(x) * 3 // 4):], y[max(0, len(y) * 3 // 4):]),
                           optimizer_updates=_number(report.get("optimizer_updates")), planned_updates=_number(report.get("planned_updates")),
                           elapsed_seconds=_number(report.get("elapsed_seconds")), objective=report.get("objective", "unreported"),
                           teacher_initialization_weight=_number(report.get("teacher_initialization_weight")),
                           teacher_mean_age_updates=_number(report.get("teacher_mean_age_updates")), source_path=str(path))
            for row, phase_update in zip(ordered, x):
                flat = dict(identity, phase=phase, phase_update=phase_update)
                for key, value in row.items():
                    if isinstance(value, dict):
                        flat.update({f"{key}_{field}": _number(number) for field, number in value.items()
                                     if not isinstance(number, (dict, list))})
                    elif key not in {"phase", "phase_update"}:
                        flat[key] = _number(value)
                histories.append(flat)
            for field in ("ema", "teacher_entropy", "teacher_initialization_weight", "teacher_mean_age_updates", "coordinate_dependence_rms"):
                numbers = [_number(row.get(field)) for row in ordered]
                numbers = [number for number in numbers if np.isfinite(number)]
                summary[f"phase_last_{field}"] = numbers[-1] if numbers else np.nan
            for group in ("online_features", "initialized_features", "teacher_features"):
                for field in ("count", "mean_std", "effective_rank"):
                    numbers = [_number(row[group].get(field)) for row in ordered if isinstance(row.get(group), dict)]
                    numbers = [number for number in numbers if np.isfinite(number)]
                    summary[f"last_{group}_{field}"] = numbers[-1] if numbers else np.nan
            summaries.append(summary)
            if finite.any():
                axis.plot(x, y, color="#0072B2", linewidth=1.35)
            else:
                axis.text(.5, .5, "No finite loss history", transform=axis.transAxes, ha="center", fontsize=11)
            axis.set(title=phase.replace("_", " "), xlabel="Phase optimizer update", ylabel="Recorded training loss")
            axis.grid(alpha=.2)
            axis.spines[["top", "right"]].set_visible(False)
            axis.tick_params(labelsize=8)
        figure.suptitle(textwrap.fill(f"{fit} | {identity['arm']} | seed {identity['seed']}", 100), fontsize=12, y=.97)
        figure.text(.055, .035, "Objectives differ across methods/phases; loss magnitudes are not comparable.\nTraining loss alone does not establish convergence, generalization or useful motion representations.", fontsize=9)
        figure.subplots_adjust(left=.09, right=.98, bottom=.22, top=.83, wspace=.38)
        saved.extend(_save_figure(figure, images, _stem("training", {"fit": fit})))
    summary_frame = pd.DataFrame(summaries)
    history_frame = pd.DataFrame(histories)
    if summary_frame.empty:
        summary_frame = pd.DataFrame(columns=["fit", "arm", "seed", "phase", "status", "history_entries"])
    if history_frame.empty:
        history_frame = pd.DataFrame(columns=["fit", "arm", "seed", "phase", "phase_update", "loss"])
    summary_frame.to_csv(folder / "training-summary.csv", index=False)
    history_path = folder / "training-history.csv"
    history_frame.to_csv(history_path, index=False)
    summary_frame.attrs.update(figure_paths=saved, history_path=str(history_path), summary_path=str(folder / "training-summary.csv"))
    return summary_frame
