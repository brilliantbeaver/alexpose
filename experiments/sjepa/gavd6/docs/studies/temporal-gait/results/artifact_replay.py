#!/usr/bin/env python3
"""Independently replay an explicitly supplied retained SYNTHETIC run.

Read-only: this program opens existing JSON/NPZ/checkpoint bytes and prints JSON
to stdout. It never imports the experiment package, fits a model, writes files,
accesses media, discovers private sources, or reports source/window identifiers.
Requires only the installed NumPy and Python standard library.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ATOL = RTOL = 1e-12
PAIRS = ((25, 26), (27, 28), (29, 30), (31, 32))


class Replay:
    def __init__(self, root):
        self.root = root.resolve(strict=True)
        self.files, self.counts = {}, defaultdict(int)
        self.discrepancy = defaultdict(float)
        self.rows, self.evaluations = {}, []

    def path(self, path):
        path = Path(path)
        path = (self.root / path if not path.is_absolute() else path).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError("An artifact reference escaped the explicitly supplied run root")
        return path

    def digest(self, path):
        path = self.path(path)
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
        value = h.hexdigest()
        self.files[str(path.relative_to(self.root))] = value
        return value

    def read(self, path):
        path = self.path(path)
        self.digest(path)
        return json.loads(path.read_text())

    def equal(self, first, saved, category):
        if first is None or saved is None:
            assert first is None and saved is None, (category, "unavailable score mismatch")
            return
        first, saved = np.asarray(first), np.asarray(saved)
        assert first.shape == saved.shape, (category, "shape mismatch")
        if first.dtype == bool or saved.dtype == bool:
            assert np.array_equal(first, saved), (category, "support mismatch")
            return
        assert np.array_equal(np.isnan(first), np.isnan(saved)), (category, "missingness mismatch")
        finite = np.isfinite(first) & np.isfinite(saved)
        difference = float(np.max(np.abs(first[finite] - saved[finite]))) if finite.any() else 0.
        self.discrepancy[category] = max(self.discrepancy[category], difference)
        assert np.allclose(first, saved, atol=ATOL, rtol=RTOL, equal_nan=True), (category, difference)

    @staticmethod
    def aggregate(errors, eligible, complete, records):
        """Independent equal-window-within-bout, equal-bout-within-video means."""
        hierarchy, groups = defaultdict(lambda: defaultdict(list)), {}
        for index, row in enumerate(records):
            hierarchy[row["video_id"]][row["sequence_id"]].append(index)
            groups[row["video_id"]] = row["group_id"]
        result = {}
        for horizon in range(errors.shape[1]):
            for video, bouts in sorted(hierarchy.items()):
                means, missing, supported, scored = [], 0, 0, 0
                for indices in bouts.values():
                    selected = [i for i in indices if eligible[i, horizon]]
                    supported += len(selected)
                    missing += sum(not complete[i, horizon] for i in selected)
                    values = [errors[i, horizon] for i in selected if np.isfinite(errors[i, horizon])]
                    scored += len(values)
                    if values:
                        means.append(float(np.mean(values)))
                result[(video, horizon)] = dict(
                    score=float(np.mean(means)) if means and not missing else None,
                    eligible_windows=supported, scored_windows=scored,
                    missing_predictions=missing, scored_bouts=len(means), group_id=groups[video])
        return result

    def check_scores(self, aggregate, saved, method, horizons, category):
        source_rows = [r for r in saved["sources"] if r["method"] == method]
        assert len(source_rows) == len(aggregate)
        rebuilt = []
        for row in source_rows:
            h = horizons.index(row["horizon_seconds"])
            values = aggregate[(row["video_id"], h)]
            for key, value in values.items():
                if key == "group_id":
                    assert value == row[key]
                else:
                    self.equal(value, row[key], category + "_source_" + key)
            rebuilt.append({**row, **values})
            self.counts[category + "_source_scores"] += 1
        summaries = [r for r in saved["summary"] if r["method"] == method]
        assert len(summaries) == len(horizons)
        for row in summaries:
            h = horizons.index(row["horizon_seconds"])
            items = [value for (_, hi), value in aggregate.items() if hi == h]
            scores = [value["score"] for value in items if value["score"] is not None]
            missing = sum(value["missing_predictions"] for value in items)
            self.equal(float(np.mean(scores)) if scores and not missing else None,
                       row["score"], category + "_summary")
            self.counts[category + "_summary_scores"] += 1
        return rebuilt

    def evaluation(self, path):
        result = self.read(path)
        meta = self.read(result["records_path"])
        saved = self.read(result["scores_path"])
        for name in ("predictions", "records", "scores", "fitted"):
            assert self.digest(result[name + "_path"]) == result[name + "_sha256"]
            self.counts["evaluation_artifact_hashes"] += 1
        fitted_path = self.path(result["fitted_path"])
        fitted = self.read(fitted_path)
        assert self.digest(fitted_path.parent / fitted["arrays"]) == fitted["arrays_sha256"]
        self.counts["fitted_array_hashes"] += 1
        if result["checkpoint_path"]:
            assert self.digest(result["checkpoint_path"]) == result["checkpoint_sha256"]
            self.counts["checkpoint_hashes"] += 1
        records, horizons = meta["records"], meta["horizons"]
        assert len({r["window_id"] for r in records}) == len(records)
        assert result["mode"] == "synthetic" and {r["mode"] for r in records} == {"synthetic"}
        assert {r["role"] for r in records} == {result["role"]}
        assert saved["clinical_endpoint"] is False and saved["manual_annotation_efficiency"] is False
        assert fitted["tree"]["train_role"] == "train"
        for field in ("groups", "videos", "windows"):
            key = {"groups": "group_id", "videos": "video_id", "windows": "window_id"}[field]
            assert not set(fitted["tree"]["train_" + field]) & {r[key] for r in records}
        with np.load(self.path(result["predictions_path"]), allow_pickle=False) as archive:
            y = archive["endpoint"].astype(float)
            valid, scale = archive["endpoint_valid"], archive["scale_valid"]
            assert np.isfinite(y[valid]).all()
            pairs, pair_count = np.zeros_like(valid), np.zeros(valid.shape[:2], int)
            for left, right in PAIRS:
                both = valid[:, :, left] & valid[:, :, right]
                pairs[:, :, left] = pairs[:, :, right] = both
                pair_count += both
            eligible = (pair_count >= 3) & scale[:, None]
            support = pairs & eligible[:, :, None]
            self.equal(support, archive["joint_support"], "primary_support")
            for method in meta["methods"]:
                pred = archive[method].astype(float)
                finite = np.isfinite(pred).all(-1)
                complete = (~support | finite).all(-1)
                delta = np.where(support[:, :, :, None] & finite[:, :, :, None], pred - y, 0)
                errors = np.linalg.norm(delta, axis=-1).sum(-1) / np.maximum(support.sum(-1), 1)
                errors[~eligible | ~complete] = np.nan
                aggregated = self.aggregate(errors, eligible, complete, records)
                self.rows[(result["role"], result["arm"], result["seed"], method)] = self.check_scores(
                    aggregated, saved, method, horizons, "primary")

                secondary_eligible = eligible & valid[:, :, 23] & valid[:, :, 24]
                pred_pelvis, true_pelvis = pred[:, :, [23, 24]].mean(-2), y[:, :, [23, 24]].mean(-2)
                hip_complete = np.isfinite(pred[:, :, [23, 24]]).all((-2, -1))
                root_error = np.linalg.norm(pred_pelvis - true_pelvis, axis=-1)
                root_error[~secondary_eligible | ~hip_complete] = np.nan
                root_support = np.zeros_like(valid)
                root_support[:, :, 23] = root_support[:, :, 24] = secondary_eligible
                limb_support = support & secondary_eligible[:, :, None]
                relative = (pred - pred_pelvis[:, :, None]) - (y - true_pelvis[:, :, None])
                relative = np.where(limb_support[:, :, :, None] & np.isfinite(relative), relative, 0)
                relative_error = np.linalg.norm(relative, axis=-1).sum(-1) / np.maximum(limb_support.sum(-1), 1)
                relative_complete = complete & hip_complete
                relative_error[~secondary_eligible | ~relative_complete] = np.nan
                for metric, e, c, joints in (
                    ("root_displacement_2d", root_error, hip_complete, root_support),
                    ("root_relative_lower_limb_2d", relative_error, relative_complete, limb_support)):
                    bucket = saved["secondary"][metric]
                    assert bucket["secondary_only"] and not bucket["used_for_selection_or_gate"]
                    for key, value in dict(errors=e, eligible=secondary_eligible,
                                           prediction_complete=c, joint_support=joints).items():
                        self.equal(value, archive[f"secondary__{metric}__{method}__{key}"],
                                   "secondary_window_" + key)
                        self.counts["secondary_window_arrays"] += 1
                    self.check_scores(self.aggregate(e, secondary_eligible, c, records),
                                      bucket, method, horizons, "secondary")
                self.counts["method_evaluations"] += 1
        self.counts["evaluations"] += 1
        self.evaluations.append(dict(path=str(self.path(path).relative_to(self.root)),
            sha256=self.digest(path), role=result["role"], arm=result["arm"], seed=result["seed"],
            windows=len(records), videos=len({r["video_id"] for r in records}),
            groups=len({r["group_id"] for r in records}), methods=len(meta["methods"])))

    def source_rows(self, role, arm, method, seeds):
        return [r for seed in seeds for r in self.rows[(role, arm, seed, method)]
                if r["horizon_seconds"] == .5 and r["score"] is not None]

    def decision(self, path):
        decision = self.read(path)
        assert decision["mode"] == "synthetic" and decision["status"] == "synthetic_only"
        assert decision["ready_for_expansion"] is False
        role, arm = decision["role"], decision["selected_arm"]
        method, baseline = decision["selected_method"], decision["baseline_method"]
        assert not method.startswith(("observed_", "initialized_"))
        seeds = decision["seeds"]
        initialized = "initialized_" + method.split("_", 1)[1]
        first = self.source_rows(role, arm, method, seeds)
        saved_pairs = self.read(decision["source_scores_path"])
        for name, m in (("candidate", method), ("baseline", baseline), ("initialized", initialized)):
            rows = self.source_rows(role, arm, m, seeds)
            stored = {(r["seed"], r["video_id"]): r for r in saved_pairs[name]}
            assert set(stored) == {(r["seed"], r["video_id"]) for r in rows}
            for row in rows:
                saved = stored[(row["seed"], row["video_id"])]
                assert row["group_id"] == saved["group_id"]
                self.equal(row["score"], saved["score"], "paired_source_score")
        output = []
        for name, stored in decision["comparisons"].items():
            if "bootstrap_differences" not in stored:
                continue
            reference_arm, reference_method = arm, baseline if name == "baseline" else initialized
            if name not in {"baseline", "initialized"}:
                reference_arm, reference_method = name, method
            reference = self.source_rows(role, reference_arm, reference_method, seeds)
            a = {(r["seed"], r["video_id"]): (r["group_id"], r["score"]) for r in first}
            b = {(r["seed"], r["video_id"]): (r["group_id"], r["score"]) for r in reference}
            assert set(a) == set(b)
            videos, groups = sorted({v for s, v in a}), sorted({g for g, value in a.values()})
            assert set(a) == {(s, v) for s in seeds for v in videos}
            assert all(a[key][0] == b[key][0] == a[(seeds[0], key[1])][0] for key in a)
            av = np.array([[a[(s, v)][1] for v in videos] for s in seeds])
            bv = np.array([[b[(s, v)][1] for v in videos] for s in seeds])
            video_groups = [a[(seeds[0], v)][0] for v in videos]
            rng = np.random.default_rng(stored["bootstrap_seed"])
            differences, gains = [], []
            for _ in range(stored["bootstrap_repetitions"]):
                selected = rng.choice(groups, size=len(groups), replace=True)
                # Independent expansion rather than the implementation's group
                # multiplicity weights; retain every video's paired seed errors.
                indices = [i for group in selected for i, g in enumerate(video_groups) if group == g]
                first_draw, reference_draw = av[:, indices].mean(), bv[:, indices].mean()
                differences.append(first_draw - reference_draw)
                gains.append((reference_draw - first_draw) / reference_draw)
            for key, computed in (
                ("bootstrap_differences", differences), ("bootstrap_relative_improvements", gains),
                ("first_mean", av.mean()), ("reference_mean", bv.mean()),
                ("difference", (av - bv).mean()),
                ("relative_improvement", (bv.mean() - av.mean()) / bv.mean())):
                self.equal(computed, stored[key], key)
            self.equal(np.quantile(differences, [.025, .975]),
                       [stored["lower_95"], stored["upper_95"]], "difference_CI")
            self.equal(np.quantile(gains, [.025, .975]),
                       [stored["relative_improvement_lower_95"], stored["relative_improvement_upper_95"]], "gain_CI")
            seed_effects = (av - bv).mean(1)
            self.equal(seed_effects, [stored["seed_differences"][str(s)] for s in seeds], "seed_effects")
            self.equal(seed_effects.std(ddof=1) if len(seeds) > 1 else 0., stored["seed_sd_difference"], "seed_sd")
            self.counts["paired_contrasts"] += 1
            self.counts["bootstrap_draw_pairs"] += len(differences)
            output.append(dict(path=str(self.path(path).relative_to(self.root)), role=role,
                selected_arm=arm, selected_method=method, contrast=name, reference=reference_method,
                seeds=seeds, videos=len(videos), groups=len(groups), draws=len(differences),
                first_mean=stored["first_mean"], reference_mean=stored["reference_mean"],
                difference=stored["difference"], difference_ci95=[stored["lower_95"], stored["upper_95"]],
                relative_improvement=stored["relative_improvement"],
                relative_improvement_ci95=[stored["relative_improvement_lower_95"], stored["relative_improvement_upper_95"]],
                status=stored["status"]))
        return output

    def run(self):
        cfg = self.read("config/resolved.json")
        assert cfg["mode"] == "synthetic", "This diagnostic is restricted to explicit synthetic evidence"
        verification = self.read("verification.json")
        identity = self.read("config/identity.json")
        identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                                  allow_nan=False).encode()).hexdigest()
        assert identity_hash == verification["frozen_identity"]
        development = self.read("evaluation/pilot/conditions.json")
        test = self.read("test/conditions.json")
        assert {(r["arm"], r["seed"]) for r in development} == {
            (arm, seed) for arm in cfg["arms"] for seed in cfg["pilot_seeds"]}
        self.evaluation("evaluation/E0/evaluation.json")
        for row in development + test:
            self.evaluation(row["artifact_path"])
        selection = self.read("decisions/pilot.json")
        comparisons = self.decision(selection["decision_path"]) + self.decision("test/summary/locked-test-summary.json")
        claim = self.read("reports/claim-audit.json")
        assert claim["mode"] == "synthetic" and claim["status"] == "software_verified_only"
        assert claim["confirmatory_claim"] is False and claim["real_runs_complete"] is False
        assert claim["development"]["scientific_status"] == "software_only"
        assert verification["core_status"] == "software_verified_only"
        lock = self.read("locks/analysis.json")
        for row in test:
            frozen = lock["selection"]["locked_predictors"][str(row["seed"])]
            assert row["fitted_reused"] is True
            for key in ("arm", "task_id", "fold", "checkpoint_sha256", "fitted_sha256"):
                assert row[key] == frozen[key], ("test lock", key)
            self.counts["locked_test_fits_verified"] += 1
        for path in sorted((self.root / "receipts").glob("*.json")):
            receipt = self.read(path)
            assert receipt["identity"] == verification["frozen_identity"]
            for relative, expected in receipt.get("outputs", {}).items():
                assert self.digest(relative) == expected, ("receipt hash", str(path.relative_to(self.root)), relative)
                self.counts["receipt_output_hashes"] += 1
        anchors = ("config/identity.json", "config/resolved.json", "verification.json",
                   "decisions/pilot.json", "locks/analysis.json", "reports/claim-audit.json",
                   "test/summary/locked-test-summary.json")
        anchor_hashes = {path: self.digest(path) for path in anchors}
        manifest = json.dumps(self.files, sort_keys=True, separators=(",", ":")).encode()
        return dict(schema_version="independent-artifact-replay-v1", status="passed",
            run_root="gavd6/outputs/temporal-gait/" + self.root.name,
            frozen_identity=verification["frozen_identity"],
            diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            environment=dict(python=sys.version.split()[0], numpy=np.__version__),
            tolerance=dict(absolute=ATOL, relative=RTOL, missingness="exact", boolean_support="exact"),
            counts=dict(self.counts), maximum_absolute_discrepancies=dict(self.discrepancy),
            evaluations=self.evaluations, paired_comparisons=comparisons,
            anchor_sha256=anchor_hashes, unique_verified_files=len(self.files),
            verified_file_manifest_sha256=hashlib.sha256(manifest).hexdigest(),
            manifest_digest_rule="SHA256(JSON mapping run-relative paths to SHA256, sorted keys, compact separators)",
            scientific_status=dict(mode="synthetic", status=claim["status"],
                development_scientific_status=claim["development"]["scientific_status"],
                confirmatory_claim=False, real_runs_complete=False, ready_for_expansion=False),
            limitations=["One seed, three videos/groups per held-out role, 50 bootstrap draws: software fixture only.",
                "Repeated methods and arms reuse observations; score-row counts are not independent sample sizes.",
                "Tiny periodic-fixture baseline error makes relative gains numerically large; not clinical performance.",
                "Replays retained predictions, arithmetic and provenance, not model retraining, real GAVD validity or GPU/Slurm execution."],
            mutation_policy="No files modified by diagnostic; stdout only; no experiment-package import or training")


def main():
    if not __debug__:
        raise RuntimeError("Do not run this assertion-based audit with -O or PYTHONOPTIMIZE")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path,
                        help="Explicit existing absolute synthetic run directory")
    args = parser.parse_args()
    if not args.run_root.is_absolute():
        parser.error("--run-root must be absolute; no implicit source discovery")
    print(json.dumps(Replay(args.run_root).run(), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
