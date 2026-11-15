"""Post-outcome paired completion summaries from frozen outer-test predictions.

No predictor fitting, target selection, or anomaly score construction occurs here.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
GAVD = BASE / "artifacts/iteration02/d66e1b126c9d9a9adf1b"
EXTERNAL = BASE / "artifacts/iteration02/external_fb9d0d23b56ce29fd846"
OUT = BASE / "docs/draft03/advisory/completion_audit_v1"
REPLICATES = 1000
SEED = 42


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(table, dataset, bank, unit, comparator, candidate):
    """Resample paired units within all four folds with fixed original weights."""
    if table.source_id.duplicated().any():
        raise ValueError("Each evaluation unit must have exactly one held-out row")
    grouped = [table.loc[table.outer_fold.eq(f)].sort_values("source_id") for f in range(4)]
    if any(group.empty for group in grouped):
        raise ValueError("All four original folds must be represented")
    counts = np.array([len(group) for group in grouped], dtype=int)
    weights = counts / counts.sum()
    rng = np.random.default_rng(SEED)
    boot_fold = []
    rows = []

    def record(a, b, boot, fold, n, improved):
        gain = a - b
        relative = gain / a
        absolute_ci = np.quantile(boot[:, 0] - boot[:, 1], [.025, .975])
        relative_ci = np.quantile((boot[:, 0] - boot[:, 1]) / boot[:, 0], [.025, .975])
        return dict(dataset=dataset, bank=bank, outer_fold=fold, evaluation_unit=unit,
                    count=int(n), comparator=comparator, candidate=candidate,
                    comparator_mean_mse=float(a), candidate_mean_mse=float(b),
                    absolute_mse_reduction=float(gain), relative_mse_reduction=float(relative),
                    absolute_ci_low=float(absolute_ci[0]), absolute_ci_high=float(absolute_ci[1]),
                    relative_ci_low=float(relative_ci[0]), relative_ci_high=float(relative_ci[1]),
                    units_with_lower_candidate_mse=int(improved))

    for fold, group in enumerate(grouped):
        values = group[[comparator, candidate]].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or (values[:, 0] <= 0).any():
            raise ValueError("Invalid raw mean squared error")
        indices = rng.integers(0, len(values), size=(REPLICATES, len(values)))
        boot = values[indices].mean(axis=1)
        boot_fold.append(boot)
        means = values.mean(axis=0)
        rows.append(record(*means, boot, fold, len(values), (values[:, 1] < values[:, 0]).sum()))
    combined_boot = np.einsum("f,fri->ri", weights, np.array(boot_fold))
    fold_means = np.array([[r["comparator_mean_mse"], r["candidate_mean_mse"]] for r in rows])
    combined_means = weights @ fold_means
    rows.append(record(*combined_means, combined_boot, "all", counts.sum(),
                       sum(r["units_with_lower_candidate_mse"] for r in rows)))
    draws = pd.DataFrame(dict(replicate=np.arange(REPLICATES),
                              comparator_mean_mse=combined_boot[:, 0],
                              candidate_mean_mse=combined_boot[:, 1],
                              absolute_mse_reduction=combined_boot[:, 0] - combined_boot[:, 1],
                              relative_mse_reduction=(combined_boot[:, 0] - combined_boot[:, 1]) / combined_boot[:, 0]))
    return rows, draws, weights.tolist()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "audit_manifest.json").exists():
        raise ValueError("Preserve the completed descriptive audit")
    inputs = {}
    verification = []

    def read(path, **kwargs):
        inputs[str(path.relative_to(BASE))] = sha(path)
        return pd.read_csv(path, **kwargs)

    # This derivative was itself independently checked against every frozen
    # target prediction and uses the original participant/angle aggregation.
    ext = read(EXTERNAL / "derivatives/interpolation_audit_v1/participant_comparison.csv")
    if len(ext) != 128 or ext.source_id.duplicated().any():
        raise ValueError("Unexpected external normal evaluation population")
    for fold in range(4):
        scores = read(EXTERNAL / f"fold_{fold}/scores.csv")
        normal_ids = set(scores.loc[scores.binary_label.eq(0), "source_id"])
        current = ext.loc[ext.outer_fold.eq(fold)].set_index("source_id").sort_index()
        if set(current.index) != normal_ids or len(current) != 32:
            raise ValueError("External derivative differs from frozen test normals")
        profiles = read(EXTERNAL / f"fold_{fold}/predictions/participant_angle_profiles.csv")
        test_profiles = profiles.loc[profiles.source_id.isin(normal_ids)]
        if test_profiles.groupby("source_id").angle.nunique().ne(6).any():
            raise ValueError("Incomplete external six-angle profile")
        original = test_profiles.groupby("source_id")[["loss_base", "loss_own", "loss_both"]].mean().sort_index()
        np.testing.assert_allclose(current[original.columns], original, rtol=1e-12, atol=1e-11)
    all_rows, draws, external_weights = summarize(ext, "external_angles", "U", "participant",
                                                "loss_own", "loss_both")
    draws.to_csv(OUT / "external_paired_bootstrap.csv", index=False)
    ext.to_csv(OUT / "external_normal_participant_losses.csv", index=False)
    print(pd.DataFrame(all_rows).to_string(index=False), flush=True)

    # Check both fixed evaluation banks. No bank is selected by its result.
    gavd_rows = []
    gavd_weights = {}
    target_columns = ["window_index", "joint", "patch_index", "mask_draws"]
    for bank in ["U", "E_leg"]:
        source_parts = []
        for fold in range(4):
            normal_profiles = {}
            support = {}
            targets = {}
            masks = {}
            normal_ids = None
            cluster_ids = None
            for policy in ["U", "A"]:
                folder = GAVD / f"fold_{fold}/{policy}_{bank}"
                scores = read(folder / "source_scores.csv")
                normal = scores.loc[scores.binary_label.eq(0)]
                if normal.outer_fold.ne(fold).any() or len(normal) != 7:
                    raise ValueError("Unexpected GAVD normal test ledger")
                ids = set(normal.source_id)
                if normal_ids is not None and ids != normal_ids:
                    raise ValueError("Training policies changed the evaluation sources")
                normal_ids = ids
                cluster_ids = normal.set_index("source_id").cluster_id
                if cluster_ids.duplicated().any():
                    raise ValueError("Source bootstrap would split duplicate recording clusters")
                profiles = read(folder / "source_joint_profiles.csv")
                profiles = profiles.loc[profiles.source_id.isin(normal_ids)].sort_values(["source_id", "joint"])
                if set(profiles.source_id) != normal_ids or profiles.groupby("source_id").joint.nunique().ne(8).any():
                    raise ValueError("Missing GAVD normal readout coverage")
                support[policy] = profiles[["source_id", "joint", "distinct_patches"]].reset_index(drop=True)
                normal_profiles[policy] = profiles.groupby("source_id").loss_both.mean().sort_index()
                metadata = read(folder / "window_metadata.csv")
                normal_windows = metadata.loc[metadata.source_id.isin(normal_ids), "window_index"]
                target_table = read(folder / "target_losses.csv.gz", usecols=target_columns)
                targets[policy] = target_table.loc[target_table.window_index.isin(normal_windows)].sort_values(target_columns).reset_index(drop=True)
                manifest_path = folder / "prediction_manifest.json"
                inputs[str(manifest_path.relative_to(BASE))] = sha(manifest_path)
                masks[policy] = json.loads(manifest_path.read_text())["bank"]["sha256"]
            pd.testing.assert_frame_equal(support["U"], support["A"])
            pd.testing.assert_frame_equal(targets["U"], targets["A"])
            if masks["U"] != masks["A"]:
                raise ValueError("Training policies used different evaluation masks")
            verification.append(dict(bank=bank, outer_fold=fold, normal_sources=len(normal_ids),
                                     unique_window_targets=len(targets["U"]),
                                     mask_target_exposures=int(targets["U"].mask_draws.sum()),
                                     target_keys_and_mask_draw_counts_identical=True,
                                     source_joint_distinct_patch_support_identical=True,
                                     evaluation_mask_sha256=masks["U"]))
            paired = pd.DataFrame({"loss_U_both": normal_profiles["U"], "loss_A_both": normal_profiles["A"]}).reset_index()
            paired["outer_fold"] = fold
            paired["bank"] = bank
            paired["cluster_id"] = paired.source_id.map(cluster_ids)
            source_parts.append(paired)
        table = pd.concat(source_parts, ignore_index=True)
        rows, draws, weights = summarize(table, "GAVD", bank, "recording_source",
                                         "loss_U_both", "loss_A_both")
        gavd_rows.extend(rows)
        gavd_weights[bank] = weights
        table.to_csv(OUT / f"gavd_{bank}_normal_source_losses.csv", index=False)
        draws.to_csv(OUT / f"gavd_{bank}_paired_bootstrap.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
    all_rows.extend(gavd_rows)
    pd.DataFrame(all_rows).to_csv(OUT / "completion_summary.csv", index=False)
    pd.DataFrame(verification).to_csv(OUT / "gavd_target_identity_audit.csv", index=False)
    manifest = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        status="complete_post_outcome_descriptive_only",
        producer_sha256=sha(__file__), inputs=inputs,
        original_runs={"GAVD": GAVD.name, "external": EXTERNAL.name},
        inference=dict(replicates=REPLICATES, seed=SEED, confidence_interval="percentile 2.5% and 97.5%",
                       resampling="paired evaluation units within each original outer fold; fixed original normal-count fold weights",
                       external_fold_weights=external_weights, gavd_fold_weights=gavd_weights,
                       relative_gain="difference of weighted mean losses divided by weighted mean comparator loss, not mean of individual ratios",
                       scope="Conditional on the fitted models, retained units, masks, folds and chosen descriptive comparisons. Not training uncertainty or a prospective primary test."),
        aggregation={"external": "mask draws within identical targets, targets within windows, windows within participant/angle, six angles equally within participant; native squared degrees",
                     "GAVD": "mask draws within identical targets, targets within windows, windows within sequences, sequences within source/joint, eight readout joints equally within source; normalized xy squared coordinates"},
        scientific_limits=["Matched residual input width does not imply equal effective capacity: external own/both design ranks are 44/62.",
                           "Completion improvements do not establish intrinsic information, neurological causation, disease diagnosis, or improved anomaly discrimination.",
                           "GAVD units are recording sources, not verified independent people; all normal sources were previously explored.",
                           "Both GAVD fixed banks are reported, with paired identical target identities and mask draw counts."],
        outputs={p.name: sha(p) for p in sorted(OUT.glob("*.csv"))})
    (OUT / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
