"""Recompute paper evidence from retained OOF rows; no neural training.

From the experiment root:
    .venv/bin/python docs/wacv2007/evidence/results_audit.py

Writes only results-audit.json and results-audit.md beside this script. The small
RF refits verify the current baseline and the historical duplicated-column bug.
Bootstrap intervals are descriptive, conditional on fixed OOF predictions.
"""
from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import platform
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from sjepa.data import sliding_windows
from sjepa.splits import LABELS, digest, file_sha256, load_full_registry, partition_records

OUT = Path(__file__).parent
RUN_ROOT = ROOT / "artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5"
OLD = RUN_ROOT / "capstone-20260921T011305448893Z"
NEW = RUN_ROOT / "capstone-20261009T060955236580Z"
SYSTEMS = ["sjepa", "mean_pose", "rf", "visibility", "majority"]
BOOTSTRAP_SEED = 20260920  # Preserve existing audit's paired source draws.
BOOTSTRAP_REPEATS = 20000


def read(path):
    return json.loads(path.read_text())


def metrics(cm):
    """Independent confusion-matrix implementation, fixed three-class order."""
    cm = np.asarray(cm, dtype=float)
    diag = np.diagonal(cm, axis1=-2, axis2=-1)
    support = cm.sum(-1)
    predicted = cm.sum(-2)
    precision = np.divide(diag, predicted, out=np.zeros_like(diag), where=predicted != 0)
    recall = np.divide(diag, support, out=np.zeros_like(diag), where=support != 0)
    f1 = np.divide(2 * diag, support + predicted, out=np.zeros_like(diag), where=(support + predicted) != 0)
    return {"accuracy": diag.sum(-1) / cm.sum(axis=(-2, -1)),
            "macro_precision": precision.mean(-1), "macro_recall": recall.mean(-1),
            "macro_f1": f1.mean(-1), "precision": precision, "recall": recall,
            "f1": f1, "support": support}


def scalar_metrics(cm):
    values = metrics(cm)
    report = {k: float(values[k]) for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")}
    report["confusion"] = np.asarray(cm).tolist()
    report["labels"] = LABELS
    report["per_class"] = {label: {k: float(values[k][i]) for k in ("precision", "recall", "f1", "support")}
                           for i, label in enumerate(LABELS)}
    return report


def source_matrices(rows):
    sources = sorted({r["source_id"] for r in rows})
    matrices = np.zeros((len(sources), len(SYSTEMS), 3, 3))
    counts, labels = [], []
    for i, source in enumerate(sources):
        subset = [r for r in rows if r["source_id"] == source]
        assert len({r["true"] for r in subset}) == 1
        assert len({r["fold"] for r in subset}) == 1
        counts.append(len(subset)); labels.append(subset[0]["true"])
        for r in subset:
            for j, system in enumerate(SYSTEMS):
                matrices[i, j, LABELS.index(r["true"]), LABELS.index(r["pred_" + system])] += 1 / len(subset)
    return sources, np.array(labels), np.array(counts), matrices


def audit_predictions(rows, registry, saved):
    inventory = {r["clip"]: r for r in registry["inventory"]["cache"]}
    expected_fold = {c: f["fold"] for f in registry["folds"] for c in f["test_clips"]}
    assert Counter(r["clip"] for r in rows) == Counter({c: 1 for c in inventory})
    for row in rows:
        record = inventory[row["clip"]]
        assert (row["true"], row["source_id"], row["fold"]) == (record["label"], record["source_id"], expected_fold[row["clip"]])
        assert all(row["pred_" + s] in LABELS for s in SYSTEMS)
    _, _, counts, matrices = source_matrices(rows)
    result = {}
    for j, system in enumerate(SYSTEMS):
        source_cm = matrices[:, j].sum(0)
        clip_cm = (matrices[:, j] * counts[:, None, None]).sum(0)
        scores = {"source_weighted": scalar_metrics(source_cm), "clip_weighted": scalar_metrics(clip_cm)}
        for weight in scores:
            for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
                assert np.isclose(scores[weight][key], saved[system][weight][key], atol=1e-12)
            assert np.allclose(scores[weight]["confusion"], saved[system][weight]["confusion"], atol=1e-12)
        fold_scores = []
        for fold in range(5):
            _, _, _, mm = source_matrices([r for r in rows if r["fold"] == fold])
            fold_scores.append(scalar_metrics(mm[:, j].sum(0)))
        scores["fold_source_macro_f1"] = [r["macro_f1"] for r in fold_scores]
        scores["fold_macro_f1_mean"] = float(np.mean(scores["fold_source_macro_f1"]))
        scores["fold_macro_f1_population_std"] = float(np.std(scores["fold_source_macro_f1"], ddof=0))
        assert np.isclose(scores["fold_macro_f1_mean"], saved[system]["source_weighted_fold_mean_std"]["macro_f1"]["mean"])
        assert np.isclose(scores["fold_macro_f1_population_std"], saved[system]["source_weighted_fold_mean_std"]["macro_f1"]["std"])
        result[system] = scores
    return result


def bootstrap(rows, repeats=BOOTSTRAP_REPEATS, seed=BOOTSTRAP_SEED):
    sources, labels, _, matrices = source_matrices(rows)
    rng = np.random.default_rng(seed)
    draws = np.concatenate([rng.choice(np.flatnonzero(labels == label),
                            size=(repeats, int((labels == label).sum())), replace=True)
                            for label in LABELS], axis=1)
    multiplicities = np.stack([np.bincount(row, minlength=len(sources)) for row in draws])
    cm = np.einsum("bs,smij->bmij", multiplicities, matrices)
    scores = metrics(cm)["macro_f1"]
    point = metrics(matrices.sum(0))["macro_f1"]
    individual = {s: {"point": float(point[i]), "percentile_95": np.quantile(scores[:, i], [.025, .975]).tolist()}
                  for i, s in enumerate(SYSTEMS)}
    paired = {"sjepa_minus_" + s: {"point": float(point[0] - point[j]),
               "percentile_95": np.quantile(scores[:, 0] - scores[:, j], [.025, .975]).tolist()}
              for j, s in enumerate(SYSTEMS) if j > 0}
    return {"seed": seed, "repeats": repeats,
            "source_strata": dict(Counter(labels)), "individual": individual, "paired": paired,
            "method": "Paired class-stratified source bootstrap; each sampled occurrence carries total source weight one; percentile interval.",
            "qualification": "Retrospective descriptive sensitivity intervals conditional on fixed OOF predictions, models, split and class source counts. Assumes source clusters independent, although participant independence is unknown and training sets overlap. No retraining, model selection, split or seed uncertainty; not a confirmatory population confidence interval or an equivalence test."}


def original_draft_audit(old_rows, new_rows, old_metrics, new_metrics):
    """Forensic numerical reconstruction only; synthetic rows are never saved."""
    historical = bootstrap(old_rows, repeats=10000, seed=20261004)
    current = bootstrap(new_rows, repeats=10000, seed=20261004)
    # Search every possible one-clip prediction change for rounded table and
    # per-class agreement. This diagnoses the discrepancy, not a valid result.
    candidates = []
    target_signature = (.468, .406, .452, .189, .568, .479, .403, .396, .429, .611)
    for i, row in enumerate(old_rows):
        for label in LABELS:
            if label == row["pred_sjepa"]:
                continue
            synthetic = copy.deepcopy(old_rows)
            synthetic[i]["pred_sjepa"] = label
            _, _, count, matrices = source_matrices(synthetic)
            source_score = metrics(matrices.sum(0))
            clip_score = metrics((matrices * count[:, None, None, None]).sum(0))
            fold_scores = []
            for fold in range(5):
                _, _, _, fold_matrices = source_matrices([r for r in synthetic if r["fold"] == fold])
                fold_scores.append(float(metrics(fold_matrices.sum(0))["macro_f1"][0]))
            signature = tuple(round(float(v), 3) for v in [source_score["macro_f1"][0], clip_score["macro_f1"][0],
                np.mean(fold_scores), np.std(fold_scores), *source_score["precision"][0], *source_score["recall"][0]])
            if signature == target_signature:
                candidates.append({"clip": row["clip"], "source_id": row["source_id"], "fold": row["fold"],
                    "saved_prediction": row["pred_sjepa"], "counterfactual_prediction": label,
                    "source_weighted": scalar_metrics(matrices[:, 0].sum(0)),
                    "clip_weighted": scalar_metrics((matrices[:, 0] * count[:, None, None]).sum(0)),
                    "fold_macro_f1": fold_scores, "fold_mean": float(np.mean(fold_scores)), "fold_population_std": float(np.std(fold_scores)),
                    "bootstrap": bootstrap(synthetic, repeats=10000, seed=20261004)})
    assert len(candidates) == 1
    table_reported = {"sjepa": [.468, .406, .452, .189], "rf": [.411, .397, .395, .098],
                     "mean_pose": [.452, .410, .427, .132], "visibility": [.318, .318, .285, .104],
                     "majority": [.259, .202, .159, .022]}
    table = []
    for system in SYSTEMS:
        old, new = old_metrics[system], new_metrics[system]
        keys = ["pooled_source_macro_f1", "pooled_clip_macro_f1", "fold_mean", "fold_population_std"]
        old_values = [old["source_weighted"]["macro_f1"], old["clip_weighted"]["macro_f1"], old["fold_macro_f1_mean"], old["fold_macro_f1_population_std"]]
        new_values = [new["source_weighted"]["macro_f1"], new["clip_weighted"]["macro_f1"], new["fold_macro_f1_mean"], new["fold_macro_f1_population_std"]]
        for key, claimed, o, n in zip(keys, table_reported[system], old_values, new_values):
            table.append({"system": system, "quantity": key, "draft": claimed, "historical_saved": o, "latest_saved": n,
                          "draft_matches_historical_at_3dp": round(o, 3) == claimed,
                          "draft_matches_latest_at_3dp": round(n, 3) == claimed})
    correct_counts = {s: {"historical": sum(r["true"] == r["pred_" + s] for r in old_rows),
                           "latest": sum(r["true"] == r["pred_" + s] for r in new_rows)} for s in SYSTEMS}
    return {"draft_file": "docs/wacv2007/wacv2007-v1.pdf", "draft_sha256": file_sha256(ROOT / "docs/wacv2007/wacv2007-v1.pdf"),
            "reported_table_audit": table, "correct_clip_counts": correct_counts,
            "historical_saved_draft_seed_bootstrap": historical, "latest_saved_draft_seed_bootstrap": current,
            "single_clip_counterfactual_search": {"searched_changes": len(old_rows) * 2,
                "rounded_signature": list(target_signature), "matches": candidates,
                "qualification": "A unique single-label counterfactual reproduces the draft's complete rounded S-JEPA numerical signature and its RF/bootstrap values. This is forensic consistency evidence, not an observed experiment or proof of how the discrepancy arose. Both saved run files instead record ms for this clip; no artifact was edited and synthetic rows were never exported as data."},
            "root_causes": [
                "S-JEPA draft results are inconsistent with both retained OOF runs. A single unsupported ms-to-pd change on -oJM2wUUjws_P1_02 reconstructs all examined rounded S-JEPA results, including abstract, table, per-class values, fold spread and individual/RF bootstrap intervals.",
                "The identical RF and mean-pose paired intervals are erroneous. Even for the counterfactual rows explaining the draft numbers, mean-pose gives [-0.1478957, 0.1845686], whereas RF gives [-0.1209356, 0.2309749]. Equality is consistent with a copied or misassigned interval, but the editing history is unavailable.",
                "RF results in the draft describe the historical duplicated-ankle feature implementation. The corrected current implementation produces 12 different predictions and a lower score; all old and new RF predictions were independently reproduced by a one-column feature change.",
                "Dataset composition, mean-pose/visibility/majority rows, mean-pose superiority in four folds, and the distinction between pooled metrics and fold spread agree with retained artifacts."]}


def source_plurality_sensitivity(rows):
    """All possible tie decisions; no outcome-informed tie rule is selected."""
    sources = sorted({r["source_id"] for r in rows})
    output = {}
    for system in SYSTEMS:
        choices, true = [], []
        ties = []
        for source in sources:
            subset = [r for r in rows if r["source_id"] == source]
            count = Counter(r["pred_" + system] for r in subset)
            candidates = [i for i, label in enumerate(LABELS) if count[label] == max(count.values())]
            choices.append(candidates); true.append(LABELS.index(subset[0]["true"]))
            if len(candidates) > 1:
                ties.append(source)
        combinations = int(np.prod([len(c) for c in choices]))
        assert combinations < 1000000, "Tie enumeration would be too large"
        extrema = [float("inf"), -float("inf")]
        fixed = None
        for pred in product(*choices):
            cm = np.bincount(3 * np.array(true) + np.array(pred), minlength=9).reshape(3, 3)
            score = scalar_metrics(cm)
            if fixed is None:
                fixed = score
            extrema = [min(extrema[0], score["macro_f1"]), max(extrema[1], score["macro_f1"])]
        output[system] = {"tied_sources": ties, "n_tied_sources": len(ties), "tie_assignments": combinations,
                          "macro_f1_range_over_all_tie_assignments": extrema,
                          "fixed_tie_order_normal_ms_pd": fixed}
    return {"systems": output, "qualification": "Exploratory hard-label plurality aggregation. Saved OOF rows contain no probabilities, so calibrated or averaged probability aggregation cannot be recovered from those rows alone. The range is sensitivity to all possible tie decisions, not a confidence interval. Fixed label-order tie resolution is arbitrary and should not become a headline result."}


def compare_runs(old_rows, new_rows):
    old_map = {r["clip"]: r for r in old_rows}
    differences = {s: [r["clip"] for r in new_rows if r["pred_" + s] != old_map[r["clip"]]["pred_" + s]] for s in SYSTEMS}
    embeddings = []
    manifests = []
    for fold in range(5):
        for stage in ("ssl", "continued"):
            rel = Path(f"fold-{fold}/{stage}-embeddings.npz")
            with np.load(OLD / rel, allow_pickle=False) as a, np.load(NEW / rel, allow_pickle=False) as b:
                embeddings.append({"fold": fold, "stage": stage,
                    "arrays": {key: {"exactly_equal": bool(np.array_equal(a[key], b[key])),
                         "max_absolute_difference": float(np.abs(a[key] - b[key]).max())} for key in a.files},
                    "old_sha256": file_sha256(OLD / rel), "new_sha256": file_sha256(NEW / rel)})
    for run in (OLD, NEW):
        for path in sorted(run.glob("fold-*/*.json")):
            obj = read(path)
            assert obj["value_sha256"] == digest(obj["value"])
            checks = {name: file_sha256(path.parent / name) == expected for name, expected in obj["files"].items()}
            assert all(checks.values())
            manifests.append({"file": str(path.relative_to(ROOT)), "value_hash_valid": True, "artifact_hashes_valid": checks})
    return {"differing_predictions": {k: len(v) for k, v in differences.items()}, "differing_clips": differences,
            "embedding_comparison": embeddings, "verified_manifests": manifests,
            "interpretation": "Both executions report computed folds and have different checkpoint/embedding bytes and numerically different embedding arrays, with identical S-JEPA hard predictions. Their registry, data, seed and split are the same. This is repeat execution, not independent seed or cohort replication; direct cache reuse is not demonstrated."}


def verify_rf(records, registry, old_rows, new_rows):
    from sjepa.classical import build_feature_matrix, train_rf_and_predict
    from threadpoolctl import threadpool_limits
    import ambient
    X, _, _, names = build_feature_matrix(records)
    left, right = names.index("left_ankle_range"), names.index("right_ankle_range")
    assert not np.array_equal(X[:, left], X[:, right])
    legacy = X.copy(); legacy[:, left] = legacy[:, right]
    by_clip = {r.clip_name: i for i, r in enumerate(records)}
    expected = {"current": {r["clip"]: r["pred_rf"] for r in new_rows},
                "historical_duplicate": {r["clip"]: r["pred_rf"] for r in old_rows}}
    verified, varying = {}, {}
    with threadpool_limits(limits=1):
        for label, matrix in (("current", X), ("historical_duplicate", legacy)):
            verified[label], varying[label] = [], []
            for k in range(5):
                train, _, test = partition_records(records, registry, k)
                tr, te = [by_clip[r.clip_name] for r in train], [by_clip[r.clip_name] for r in test]
                pred = train_rf_and_predict(matrix[tr], [r.label for r in train], matrix[te], n_jobs=1)
                differences = [r.clip_name for r, p in zip(test, pred) if p != expected[label][r.clip_name]]
                assert not differences, (label, k, differences)
                verified[label].append({"fold": k, "all_predictions_match": True})
                varying[label].append(int((matrix[tr].std(0) > 0).sum()))
    feature_file = Path(ambient.__file__).parent / "classification/features.py"
    return {"n_columns": len(names), "varying_columns_by_fold": varying,
            "columns_varying_anywhere": [n for i, n in enumerate(names) if X[:, i].std() > 0],
            "current_ankle_columns_identical": False, "prediction_refits": verified,
            "feature_code_sha256": file_sha256(feature_file),
            "interpretation": "Current feature extraction reproduces all 88 latest RF predictions. Replacing only left_ankle_range with right_ankle_range reproduces all 88 historical RF predictions. Thus the historical versus current RF difference is reproducible from this one feature correction with the saved split and RF settings. This audit does not modify production features or train neural models."}


def main():
    records, registry = load_full_registry(ROOT)
    old_rows, new_rows = read(OLD / "oof.json"), read(NEW / "oof.json")
    old_result, new_result = read(OLD / "results.json"), read(NEW / "results.json")
    old_provenance, new_provenance = read(OLD / "provenance.json"), read(NEW / "provenance.json")
    assert old_provenance["inputs"] == new_provenance["inputs"]
    assert old_result["config"] == new_result["config"]
    counts = []
    for label in LABELS:
        subset = [r for r in records if r.label == label]
        counts.append({"label": label,
            "raw_clips": sum(r["label"] == label for r in registry["inventory"]["raw_clips"]),
            "usable_clips": len(subset), "source_recordings": len({r.source_id for r in subset}),
            "frames": sum(r.n_frames for r in subset), "minimum_frames": min(r.n_frames for r in subset),
            "maximum_frames": max(r.n_frames for r in subset), "padded_clips": sum(r.n_frames < 32 for r in subset),
            "windows": sum(len(sliding_windows(r.load_norm(), 32, 16)) for r in subset)})
    hashes = {str(p.relative_to(ROOT)): file_sha256(p) for run in (OLD, NEW)
              for p in (run / "oof.json", run / "results.json", run / "provenance.json")}
    hashes[str(Path(__file__).relative_to(ROOT))] = file_sha256(__file__)
    dates = {}
    for run in (OLD, NEW):
        instant = datetime.strptime(run.name.split("-", 1)[1], "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
        dates[run.name] = {"utc": instant.isoformat(), "America/Los_Angeles": instant.astimezone(ZoneInfo("America/Los_Angeles")).isoformat()}
    folds = [{"fold": f["fold"], **{part: {"clips": len(f[part + "_clips"]), "sources": len(f[part + "_sources"])} for part in ("train", "validation", "test")}} for f in registry["folds"]]
    report = {"audit_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "numpy": np.__version__, "primary_run": str(NEW.relative_to(ROOT)), "historical_run": str(OLD.relative_to(ROOT)),
              "hashes": hashes, "registry_sha256": registry["registry_sha256"], "dataset_sha256": registry["dataset_sha256"],
              "counts": counts, "folds": folds, "exclusions": registry["inventory"]["exclusions"],
              "date_interpretation": dates,
              "source_independence": "Source IDs come from filename prefixes. Registry asserts source separation, not participant separation. Raw video bytes, unique people and clinical diagnoses are not independently authenticated by this audit.",
              "primary_metrics": audit_predictions(new_rows, registry, new_result["metrics"]),
              "historical_metrics": audit_predictions(old_rows, registry, old_result["metrics"]),
              "bootstrap": bootstrap(new_rows), "source_plurality": source_plurality_sensitivity(new_rows),
              "run_comparison": compare_runs(old_rows, new_rows),
              "rf_verification": verify_rf(records, registry, old_rows, new_rows),
              "training_budget": {"base_updates": 800, "continuation_updates": 400, "seed": 42,
                  "selected_stage_by_fold": [s["selected"] for s in new_result["selections"]]},
              "metric_definition": "Source-weighted metrics pool clip confusion contributions with clip weight 1 / usable clips in its source. They retain clip decisions and are not one classification per source. Fold standard deviations use ddof=0 and are descriptive, not standard errors.",
              "historical_scope": "The g1/R1 cache has 47 clips and a different registry, training budget, and readout; its 0.4384 S-JEPA and 0.6666 RF scores cannot be merged with the 88-clip full-v1 evaluation."}
    report["notebook_primary_run"] = report["historical_run"]
    report["notebook_primary_metrics"] = report["historical_metrics"]
    report["corrected_verification_run"] = report["primary_run"]
    report["corrected_verification_metrics"] = report["primary_metrics"]
    report["source_hierarchy"] = ("The top-level notebook 06 prose reports the September 21 retained full-v1 run, including the duplicated-ankle RF; its seven code cells have no execution outputs. Use notebook_primary_metrics for the notebook-anchored manuscript and corrected_verification_metrics only as explicitly labeled reanalysis. The initial audit's primary_metrics key is retained for compatibility and refers to the corrected latest execution.")
    report["original_draft_audit"] = original_draft_audit(old_rows, new_rows, report["historical_metrics"], report["primary_metrics"])
    (OUT / "results-audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Reproducible results audit", "", f"Notebook-anchored execution: `{report['notebook_primary_run']}`.", "",
             f"Corrected verification execution: `{report['corrected_verification_run']}`.", "", report["source_hierarchy"], "",
             "All retained scalar metrics and confusion matrices were independently recomputed from OOF rows. Source/clip attribution and complete OOF coverage were checked against the frozen registry; all referenced checkpoint and embedding manifest hashes verified.", "",
             "## Dataset and estimand", "", "| Class | Raw clips | Usable clips | Sources | Frames | Windows | Padded clips |", "|---|---:|---:|---:|---:|---:|---:|"]
    for c in counts:
        lines.append(f"| {c['label']} | {c['raw_clips']} | {c['usable_clips']} | {c['source_recordings']} | {c['frames']} | {c['windows']} | {c['padded_clips']} |")
    lines += ["", report["source_independence"], "", report["metric_definition"], "", "## Verified notebook-anchored results", "",
              "| Model | Source-weighted macro-F1 | Clip-weighted macro-F1 | Source-weighted accuracy | Fold macro-F1 mean ± population SD |", "|---|---:|---:|---:|---:|"]
    for s in SYSTEMS:
        m = report["notebook_primary_metrics"][s]
        lines.append(f"| {s} | {m['source_weighted']['macro_f1']:.6f} | {m['clip_weighted']['macro_f1']:.6f} | {m['source_weighted']['accuracy']:.6f} | {m['fold_macro_f1_mean']:.6f} ± {m['fold_macro_f1_population_std']:.6f} |")
    paper_bootstrap = report["original_draft_audit"]["historical_saved_draft_seed_bootstrap"]
    lines += ["", "## Paired source-resampling sensitivity", "", paper_bootstrap["method"], "",
              "10,000 resamples; seed 20261004 (the draft's stated settings); class source counts fixed at 16 normal, 13 MS, 12 PD.", "",
              "| Difference in source-weighted macro-F1 | Observed | 95% percentile interval |", "|---|---:|---:|"]
    for s, b in paper_bootstrap["paired"].items():
        lines.append(f"| {s} | {b['point']:.6f} | [{b['percentile_95'][0]:.6f}, {b['percentile_95'][1]:.6f}] |")
    lines += ["", report["bootstrap"]["qualification"], "", "The near-zero S-JEPA versus mean-pose difference does not establish equivalence. These results do not establish that temporal order contributed to classification; a matched time-order ablation remains unrun.", "",
              "## Execution lineage and corrected RF", "", report["run_comparison"]["interpretation"], "",
              "All 88 S-JEPA predictions coincide between executions; 12 RF predictions change. Mean-pose, visibility and majority predictions also coincide.", "",
              report["rf_verification"]["interpretation"], "",
              "The corrected RF has source-weighted macro-F1 0.325184; the historical duplicated-column baseline has 0.411263. The latter must be identified as historical and should not be substituted for the current corrected result.", "",
              "The current extractor supplies 82 feature fields, of which 15 vary in each training fold. The remaining fields are constant for this extraction route. Neither the current nor historical RF should be described as a fully populated 82-feature gait model.", "",
              "## Corrected-execution source aggregation sensitivity", "", report["source_plurality"]["qualification"], "",
              "| Model | Tied sources | Macro-F1 with fixed normal/MS/PD tie priority | Range over every tie assignment |", "|---|---:|---:|---:|"]
    for s, m in report["source_plurality"]["systems"].items():
        lo, hi = m["macro_f1_range_over_all_tie_assignments"]
        lines.append(f"| {s} | {m['n_tied_sources']} | {m['fixed_tie_order_normal_ms_pd']['macro_f1']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines += ["", "## Dates and scope", "",
              "The latest timestamp, 2026-10-09 06:09:55 UTC, is 2026-10-08 23:09:55 in Los Angeles. The apparent next-day folder date is a timezone difference, not a future-dated experiment.", "",
              report["historical_scope"], "",
              "This audit reads existing data and model artifacts and refits only the small deterministic RF baselines. It does not establish participant-level generalization, clinically verified diagnoses, disease severity estimation, or external validity.", "",
              "Reproduction: `.venv/bin/python docs/wacv2007/evidence/results_audit.py`. Exact input hashes, per-class metrics, confusion matrices, fold scores, source-vote tie ranges and manifest checks are in `results-audit.json`.", ""]
    (OUT / "results-audit.md").write_text("\n".join(lines))
    audit = report["original_draft_audit"]
    forensic = audit["single_clip_counterfactual_search"]["matches"][0]
    forensic_bootstrap = forensic["bootstrap"]
    consistency = ["# Numerical consistency and bootstrap audit of the supplied draft", "",
        "The supplied PDF's main S-JEPA numbers are inconsistent with both retained full-v1 prediction files. A search over all 176 one-clip label changes finds one counterfactual that reproduces every examined rounded S-JEPA quantity: change the prediction for `-oJM2wUUjws_P1_02` from the saved `ms` to `pd`. This is a diagnostic reconstruction only. Both saved runs record `ms`; the manuscript must use the retained predictions, and no artifact was changed.", "",
        "The reconstruction also reproduces the draft's stated S-JEPA bootstrap interval and its RF-difference interval. The draft repeats the RF interval for mean pose, which fails independent calculation even for the reconstructed rows. The editing history is unavailable, so these checks locate the numerical discrepancy without claiming how it arose.", "",
        "## Complete scalar table audit", "",
        "| Model | Quantity | Draft | Notebook/Sep21 saved | Corrected/Oct09 saved | Matches notebook at 3 dp? |", "|---|---|---:|---:|---:|---|"]
    for entry in audit["reported_table_audit"]:
        consistency.append(f"| {entry['system']} | {entry['quantity']} | {entry['draft']:.3f} | {entry['historical_saved']:.6f} | {entry['latest_saved']:.6f} | {'yes' if entry['draft_matches_historical_at_3dp'] else 'NO'} |")
    consistency += ["", "## Per-class and fold checks", "",
        "For S-JEPA, the saved source-weighted recalls are 0.395833 (normal), 0.428994 (MS), and 0.569048 (PD), and precisions are 0.568189, 0.458888, and 0.385787. The PDF reports 0.611 PD recall, 0.479 MS precision and 0.403 PD precision; each is reproduced by the one-clip counterfactual. Normal and MS correct clip counts are 10/24 and 8/29; PD is 18/35, not 19/35. Consequently S-JEPA has 36 correct clips, while the historical RF and mean pose each have 37. The corrected RF has 30.", "",
        "The saved S-JEPA fold source-weighted macro-F1 values are 0.515263, 0.362559, 0.779598, 0.315654 and 0.217267. Their mean is 0.438068 and population SD is 0.195980. The counterfactual changes fold 3 to 0.386466, yielding the draft's 0.452230 mean and 0.189059 SD. The draft's range 0.217–0.780 remains correct. S-JEPA exceeds historical RF in two folds (0 and 2), not the draft's three; the counterfactual would add fold 3. It exceeds corrected RF in three folds and mean pose in only fold 2.", "",
        "All reported counts in dataset Table 1 agree with the registry and caches: 91 raw clips, 88 usable clips, 41 sources, 15 padded clips and 358 windows. The non-S-JEPA historical result rows agree to the shown precision. Stage selections remain ssl/ssl/ssl/continued/continued; original fold-0 and fold-4 validation examples agree with retained results. Those accurate quantities do not validate the inconsistent S-JEPA table entries.", "",
        "## Bootstrap verification with the draft's exact settings", "",
        "Every interval below uses 10,000 paired class-stratified source resamples with seed 20261004; repeated draws preserve weight one per occurrence. Label order is normal/MS/PD, sources are sorted, all methods share identical draws, and zero-denominator F1 is zero.", "",
        "| Prediction basis | Quantity | Point | 95% percentile interval |", "|---|---|---:|---:|"]
    for basis, b in [("Notebook/Sep21 saved", audit["historical_saved_draft_seed_bootstrap"]),
                     ("Corrected/Oct09 saved", audit["latest_saved_draft_seed_bootstrap"]),
                     ("Counterfactual explaining PDF — not data", forensic_bootstrap)]:
        for name, val in [("S-JEPA", b["individual"]["sjepa"]), ("S-JEPA minus RF", b["paired"]["sjepa_minus_rf"]), ("S-JEPA minus mean pose", b["paired"]["sjepa_minus_mean_pose"])]:
            consistency.append(f"| {basis} | {name} | {val['point']:.9f} | [{val['percentile_95'][0]:.9f}, {val['percentile_95'][1]:.9f}] |")
    consistency += ["", "The PDF's repeated [−0.121, 0.231] interval equals the counterfactual RF interval after rounding. The counterfactual mean-pose interval is [−0.148, 0.185], so the identical displayed intervals are unsupported. For the actual notebook predictions the correct differences are 0.045361 [−0.133447, 0.220114] versus historical RF, and 0.004731 [−0.160496, 0.173999] versus mean pose.", "",
        paper_bootstrap["qualification"], "", "## Root causes and disposition", ""]
    consistency.extend(f"{i}. {cause}" for i, cause in enumerate(audit["root_causes"], start=1))
    consistency += ["", "The abstract, Results, Discussion, Conclusion, main table, raw counts, per-class descriptions, fold comparisons and bootstrap statements must all be regenerated from a single declared saved run. The latest corrected RF belongs in a labeled reanalysis if the notebook-anchored study remains the main result. Notebook prose has historical claims about the RF defect; these should be described as historical context rather than as the current extractor's state.", "",
        "Reproduce with `.venv/bin/python docs/wacv2007/evidence/results_audit.py`; inspect `original_draft_audit` in `results-audit.json` for exact values and the exhaustive one-change search. The counterfactual never overwrites OOF files or enters the study's reported results.", ""]
    (OUT / "results-numerical-consistency.md").write_text("\n".join(consistency))
    print(json.dumps({"written": [str(OUT / "results-audit.json"), str(OUT / "results-audit.md")],
                      "bootstrap": report["bootstrap"]["paired"], "rf_verified": report["rf_verification"]["prediction_refits"]}, indent=2))


if __name__ == "__main__":
    main()
