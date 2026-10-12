"""Independently audit downloaded diagnostic tables; never mutate source files.

Run from the gavd6 checkout with an interpreter providing NumPy, pandas, and
Matplotlib. This reaggregates saved per-window scores, not raw pose predictions:
the original source bundle and neural prediction arrays were not downloaded.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs/synthetic-training-v2/source-smoke-01/diagnostics/20260919T083049222781Z-065a97"
STRATA = ["method", "split", "seed", "extractor", "evidence_status"]
UNIT = ["person_id", "motion_id", "window_id", "variant"]
METRICS = ["visible_nle", "lower_limb_nle", "missing_rate", "displacement_nle",
           "ankle_separation_mae", "amplitude_error", "amplitude_ratio", "event_timing_mae_s"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def complete_mean(series):
    return series.mean() if np.isfinite(series).all() else np.nan


def reaggregate(frame, metric):
    windows = frame.groupby(STRATA + UNIT[:-1])[metric].agg(complete_mean)
    motions = windows.groupby(level=STRATA + UNIT[:2]).agg(complete_mean)
    people = motions.groupby(level=STRATA + UNIT[:1]).agg(complete_mean)
    return people.groupby(level=STRATA).agg(complete_mean).rename("recomputed")


def audit_aggregates(frame, saved, condition=False):
    blocks = frame.groupby("variant") if condition else [(None, frame)]
    result = []
    for variant, block in blocks:
        for metric in METRICS:
            expected = saved.loc[saved.metric.eq(metric)]
            if condition:
                expected = expected.loc[expected.variant.eq(variant)]
            joined = reaggregate(block, metric).to_frame().join(
                expected.set_index(STRATA)[["value"]], how="outer", validate="one_to_one")
            assert len(joined) == len(expected)
            assert np.array_equal(joined.recomputed.isna(), joined.value.isna())
            np.testing.assert_allclose(joined.recomputed, joined.value, rtol=1e-12, atol=1e-14, equal_nan=True)
            joined["absolute_difference"] = abs(joined.recomputed - joined.value)
            joined["metric"] = metric
            joined["variant"] = variant if condition else "all"
            result.append(joined.reset_index())
    return pd.concat(result, ignore_index=True)


def main():
    # Keep file hashes as a reproducible local snapshot, not a claim of an
    # independent authenticated transfer manifest from HAIC.
    before = {str(p.relative_to(SOURCE)): sha(p) for p in SOURCE.rglob("*")
              if p.is_file() and "matplotlib-cache" not in p.parts}
    meta = json.loads((SOURCE / "analysis.json").read_text())
    status = json.loads((SOURCE / "status.json").read_text())
    assert status["status"] == "POSTRUN_CHECKS_COMPLETE" and status["checks"] == "all"
    code_checks = {name: sha(ROOT / name) == digest
                   for name, digest in meta["analysis_code_sha256"].items()}
    assert all(code_checks.values())
    f = pd.read_csv(SOURCE / "per-window.csv")
    summary = pd.read_csv(SOURCE / "per-person-balanced-summary.csv")
    conditions = pd.read_csv(SOURCE / "by-condition.csv")
    timing = pd.read_csv(SOURCE / "timing-per-window.csv")
    timing_summary = pd.read_csv(SOURCE / "timing-summary.csv")
    history = pd.read_csv(SOURCE / "training-history.csv")
    training = pd.read_csv(SOURCE / "training-summary.csv")
    assert training.status.eq("complete").all()
    models = json.loads((SOURCE / "calibration-models.json").read_text())
    assert not f.duplicated(STRATA + UNIT).any()
    assert f.groupby("method").size().eq(48).all() and len(f) == 336
    assert set(f.seed) == {17}
    assert f.missing_count.eq(0).all() and f.displacement_missing_count.eq(0).all()
    assert set(models["joint_offset"]["provenance"]["canonical_people"]).isdisjoint(f.person_id)
    for model in models.values():
        assert model["provenance"]["pooled_families"] == ["hrnet", "rtmpose"]
        assert not model["provenance"]["evaluation_labels_used"]
    for path in (SOURCE / "predictions").glob("*.npz"):
        with np.load(path, allow_pickle=False) as item:
            assert item["prediction"].shape == (48, 64, 12, 2)
            assert np.isfinite(item["prediction"]).all()

    audited = pd.concat([audit_aggregates(f, summary), audit_aggregates(f, conditions, True)])
    audited.to_csv(OUT / "aggregate-verification.csv", index=False)
    person_rows = []
    for metric in METRICS:
        windows = f.groupby(STRATA + UNIT[:-1])[metric].agg(complete_mean)
        motions = windows.groupby(level=STRATA + UNIT[:2]).agg(complete_mean)
        people = motions.groupby(level=STRATA + UNIT[:1]).agg(complete_mean)
        person_rows.append(people.rename("value").reset_index().assign(metric=metric))
    pd.concat(person_rows).to_csv(OUT / "person-balanced-metrics.csv", index=False)

    effects = []
    for metric in ("visible_nle", "displacement_nle"):
        wide = summary.loc[summary.metric.eq(metric)].pivot(index="extractor", columns="method", values="value")
        for extractor, row in wide.iterrows():
            for method in wide:
                effects.append(dict(metric=metric, extractor=extractor, method=method,
                    error=row[method], unchanged=row.unchanged,
                    reduction_vs_unchanged_percent=100 * (row.unchanged - row[method]) / row.unchanged,
                    reduction_vs_coordinate_percent=100 * (row.coordinate - row[method]) / row.coordinate))
    effects = pd.DataFrame(effects)
    effects.to_csv(OUT / "method-comparisons.csv", index=False)
    cw = conditions.loc[conditions.metric.eq("visible_nle")].pivot(
        index=["extractor", "variant"], columns="method", values="value")
    ce = cw.assign(best_reported_method=cw.idxmin(axis=1),
        paired_improvement_vs_coordinate_percent=100 * (cw.coordinate - cw.paired_jepa) / cw.coordinate)
    ce.to_csv(OUT / "condition-comparisons.csv")

    # References repeat across extractors; verify that duplication explicitly.
    oracle = timing.loc[timing.method.eq("reference_oracle")].copy()
    ref_columns = ["reference_visible_frames", "reference_valid_frames", "reference_finite_frames",
                   "reference_event_count", "reference_raw_event_count", "reference_amplitude_raw_px",
                   "reference_amplitude_normalized", "bbox_scale_cv", "reference_eligible"]
    for _, group in oracle.groupby(UNIT):
        assert len(group) == 3
        assert group[ref_columns].nunique(dropna=False).eq(1).all()
    ref = oracle.loc[oracle.extractor.eq("hrnet_w32")].copy()
    assert len(ref) == 16 and ref.reference_eligible.sum() == 4
    assert ref.reference_valid_frames.eq(64).all() and ref.reference_finite_frames.eq(64).all()
    assert ref.loc[~ref.reference_eligible, "reference_reasons"].str.contains("reference_visibility_gap").all()
    assert ref.loc[ref.variant.str.contains("obstruction"), "reference_visible_frames"].eq(0).all()
    ref[UNIT + ref_columns + ["reference_reasons"]].to_csv(OUT / "reference-coverage.csv", index=False)

    timing_check = []
    for (method, extractor, seed), group in timing.groupby(["method", "extractor", "seed"]):
        eligible = group.loc[group.reference_eligible]
        matched = int(eligible.matched_event_count.sum())
        reference = int(eligible.reference_event_count.sum())
        predicted = int(eligible.predicted_event_count.sum())
        assert eligible.prediction_finite_frames.eq(64).all()
        extra = int(eligible.extra_event_count.sum())
        missed = int(eligible.missed_event_count.sum())
        assert matched + missed == reference and matched + extra == predicted
        mae = (eligible.matched_timing_mae_s * eligible.matched_event_count).sum() / matched if matched else np.nan
        saved = timing_summary.loc[(timing_summary.method == method) &
            (timing_summary.extractor == extractor) & (timing_summary.seed == seed)].iloc[0]
        for name, value in {"reference_eligible": len(eligible), "matched_events": matched,
                            "reference_events": reference, "missed_events": missed,
                            "known_extra_events": extra}.items():
            assert saved[name] == value
        np.testing.assert_allclose(saved.conditional_timing_mae_s, mae, atol=1e-14, rtol=1e-12, equal_nan=True)
        timing_check.append(dict(method=method, extractor=extractor, seed=seed,
            eligible_records=len(eligible), records=len(group), matched=matched, reference=reference,
            predicted=predicted, extra=extra, missed=missed,
            precision=matched / predicted if predicted else np.nan,
            recall=matched / reference if reference else np.nan, conditional_timing_mae_s=mae))
    pd.DataFrame(timing_check).to_csv(OUT / "timing-verification.csv", index=False)
    amplitude = f.loc[f.amplitude_error.notna(), ["method", "extractor", *UNIT, "amplitude_error", "amplitude_ratio"]]
    amplitude.assign(scope="support_only_exploratory_not_primary_aggregate").to_csv(
        OUT / "supported-amplitudes.csv", index=False)
    assert amplitude.groupby(["method", "extractor"]).size().eq(4).all()

    trends = []
    for (arm, phase), group in history.groupby(["arm", "phase"]):
        group = group.sort_values("phase_update")
        s = training.loc[(training.arm == arm) & (training.phase == phase)].iloc[0]
        assert list(group.phase_update) == list(range(1, int(s.planned_phase_updates) + 1))
        assert np.isfinite(group.loss).all() and np.isfinite(group.gradient_norm).all()
        np.testing.assert_allclose([group.loss.iloc[0], group.loss.iloc[-1]], [s.first_loss, s.last_loss])
        losses = group.loss.to_numpy()
        trends.append(dict(arm=arm, phase=phase, updates=len(group), first_loss=losses[0], last_loss=losses[-1],
            previous25_mean=losses[-50:-25].mean(), last25_mean=losses[-25:].mean(),
            last25_reduction_percent=100 * (losses[-50:-25].mean() - losses[-25:].mean()) / losses[-50:-25].mean(),
            final_learning_rate=group.learning_rate.iloc[-1], max_preclip_gradient_norm=group.gradient_norm.max(),
            supported_examples_min=group.supported_examples.min(), targets_min=group.target_count.min()))
    assert len(history) == 3000 and history.groupby("fit").size().sum() == 3000
    pd.DataFrame(trends).to_csv(OUT / "training-verification.csv", index=False)
    pairs = history.loc[(history.arm == "paired_jepa") & (history.phase == "pretrain")]
    np.testing.assert_allclose(pairs.ema.prod(), pairs.teacher_initialization_weight.iloc[-1], atol=1e-14)

    joints = pd.read_csv(SOURCE / "per-joint-residuals.csv")
    joints.groupby(["method", "joint"])[["bias_x_px", "bias_y_px", "mae_x_px", "mae_y_px"]].mean().to_csv(
        OUT / "conditional-joint-residuals.csv")

    # A static scientific figure; no synthetic observations or uncertainty bars.
    methods = ["joint_offset", "joint_affine", "initialized", "coordinate", "direct", "paired_jepa"]
    labels = ["Joint offset", "Joint affine", "Initialized", "Coordinate", "Direct", "Paired JEPA"]
    extractors = ["hrnet_w32", "rtmpose_m", "vitpose_base"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    y = np.arange(len(methods))
    for ax, metric, title in zip(axes, ["visible_nle", "displacement_nle"],
                               ["Visible coordinate error", "0.20 s displacement error"]):
        for offset, extractor, color in zip([-.2, 0, .2], extractors, ["#0072B2", "#D55E00", "#009E73"]):
            a = effects.loc[(effects.metric == metric) & (effects.extractor == extractor)].set_index("method")
            ax.scatter(a.loc[methods, "reduction_vs_unchanged_percent"], y + offset, label=extractor, color=color, s=40)
        ax.axvline(0, color="black", lw=.8)
        ax.set(yticks=y, yticklabels=labels, title=title, xlabel="Error reduction versus unchanged (%)")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=.2)
    axes[1].legend(loc="lower right", fontsize=9)
    fig.suptitle("Synthetic development screen: two development people, seed 17", fontsize=13)
    fig.text(.5, .025, "Positive values indicate improvement. Descriptive comparisons; no uncertainty or confirmation claim.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .94))
    for suffix in ("png", "svg"):
        fig.savefig(OUT / f"coordinate-and-motion.{suffix}", dpi=160)
    plt.close(fig)

    result = dict(source=str(SOURCE.relative_to(ROOT)), source_status=status,
        analysis_script_sha256=sha(Path(__file__)),
        input_sha256=before, checked_diagnostic_code=code_checks,
        per_window_rows=len(f), methods=sorted(f.method.unique()),
        independent_development_people=int(f.person_id.nunique()),
        development_motions=int(f.motion_id.nunique()), unique_windows=int(f.window_id.nunique()),
        seed=17, checked_aggregate_cells=len(audited),
        aggregate_max_absolute_difference=float(audited.absolute_difference.max()),
        aggregate_unsupported_cells=int(audited.value.isna().sum()),
        checked_timing_summary_rows=len(timing_check), training_updates=len(history),
        training_phases=len(trends), best_affine_condition_cells=int(ce.best_reported_method.eq("joint_affine").sum()),
        reference_eligible_records_per_extractor=4, reference_records_per_extractor=16,
        raw_vs_normalized_peak_counts=ref.loc[ref.variant.eq("clean"),
            ["person_id", "window_id", "reference_raw_event_count", "reference_event_count", "bbox_scale_cv"]].to_dict("records"),
        limitations=["Saved per-window scores independently reaggregated; raw neural predictions/targets were not downloaded.",
            "HAIC source verification is retained remote evidence, not a fresh local replay of all source receipts.",
            "Calibration comparisons and support-only amplitude diagnostics are post hoc.",
            "Seven diagnostic methods have outcome tables; all eight learned arms have histories, but full 12-method source outcomes are absent.",
            "Two development people and one seed cannot establish a general JEPA effect."])
    (OUT / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    assert before == {str(p.relative_to(SOURCE)): sha(p) for p in SOURCE.rglob("*")
                      if p.is_file() and "matplotlib-cache" not in p.parts}
    print(json.dumps({k: v for k, v in result.items() if k not in
                      {"input_sha256", "source_status", "raw_vs_normalized_peak_counts", "limitations"}}, indent=2))


if __name__ == "__main__":
    main()
