"""Refresh the current rows of result_history.csv from the artifacts on disk.

The ledger is the machine-readable record of what each result was before and after a change. Its current
rows used to be transcribed by hand, which is how they drifted out of step with the artifacts and started
failing make_figures.py's input contract. This script rewrites only the rows whose status is current,
reading them from the same CSVs the figures read, and leaves every superseded row untouched.

It also writes the companion file symmetry_verdicts.csv, which carries the three preregistered
reflection-symmetry verdicts. Those live in a separate file rather than in result_history.csv because
result_history.csv's three numeric columns are classifier accuracy, balanced accuracy, and macro-F1,
while a symmetry result is a preregistered verdict string over a ridge R-squared or a mirror residual.
Writing an R-squared of -0.602 into a column named accuracy would corrupt every consumer that reads
that column. Both files are regenerated from the same authoritative bundles.

Run:  python3 refresh_result_history.py            (rewrites both files in place)
      python3 refresh_result_history.py --check    (fails if a refresh is needed; for CI)

When the active checkpoint lineage changes, pass --supersede so the outgoing current rows are retained
under a version label naming their fingerprint instead of being overwritten.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from artifact_paths import load_contract, resolve_artifacts  # noqa: E402

ARTIFACTS = resolve_artifacts()
CONTRACT = load_contract(ARTIFACTS)
FINGERPRINT = CONTRACT["checkpoint_fingerprint"]
LEDGER_PATH = Path(__file__).with_name("result_history.csv")
SYMMETRY_PATH = Path(__file__).with_name("symmetry_verdicts.csv")

# comparison label -> (task, condition) in classifier_metrics.csv
CLASSIFIER_ROWS = {
    "exact_exp5": ("five_class_exp5_exact", "all"),
    "all_96": ("five_class_all_sequences", "all"),
    "one_vs_normal_parkinsons": ("one_vs_normal", "parkinsons"),
    "one_vs_normal_stroke": ("one_vs_normal", "stroke"),
    "one_vs_normal_myopathic": ("one_vs_normal", "myopathic"),
    "one_vs_normal_cerebralpalsy": ("one_vs_normal", "cerebralpalsy"),
}
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")


def current_rows() -> pd.DataFrame:
    metrics = pd.read_csv(ARTIFACTS / "classifier_metrics.csv")
    lane_c = pd.read_csv(ARTIFACTS / "lane_c_video_disjoint_metrics.csv")
    five_class = lane_c.loc[
        lane_c["task"].eq("five_class_classifier_video_disjoint_encoder_transductive")
    ].iloc[0]

    rows = []
    for comparison, (task, condition) in CLASSIFIER_ROWS.items():
        match = metrics.loc[metrics["task"].eq(task) & metrics["condition"].eq(condition)]
        if match.empty:
            raise RuntimeError(f"classifier_metrics.csv has no row for {task}/{condition}")
        row = match.iloc[0]
        rows.append({
            "comparison": comparison,
            "version": "current_five_stage",
            "status": "current",
            "model_changed": "yes",
            "evaluation_changed": "no",
            **{metric: float(row[metric]) for metric in METRICS},
            "checkpoint_fingerprint": FINGERPRINT,
            "scope_note": SCOPE_NOTES[comparison],
        })
    rows.append({
        "comparison": "lane_c_five_class",
        "version": "corrected_two_fold_mean",
        "status": "current",
        "model_changed": "no",
        "evaluation_changed": "yes",
        **{metric: float(five_class[f"{metric}_mean"]) for metric in METRICS},
        "checkpoint_fingerprint": FINGERPRINT,
        "scope_note": "Same checkpoint; every train and test fold contains all five labels",
    })
    rows.append({
        "comparison": "lane_c_five_class",
        "version": "corrected_pooled_oof",
        "status": "current_summary",
        "model_changed": "no",
        "evaluation_changed": "yes",
        **{metric: float(five_class[f"{metric}_pooled_oof"]) for metric in METRICS},
        "checkpoint_fingerprint": FINGERPRINT,
        "scope_note": "Pooled out-of-fold summary from the corrected two folds; encoder exposure remains "
                      "complete",
    })
    return pd.DataFrame(rows)


SCOPE_NOTES = {
    "exact_exp5": "Current five-stage model; exact 47/21 split remains video-confounded and "
                  "encoder-exposed",
    "all_96": "Current five-stage model; all 29 classifier-test rows trained the encoder",
    "one_vs_normal_parkinsons": "Current model; seven-row classifier test remains video-confounded and "
                                "encoder-exposed",
    "one_vs_normal_stroke": "Current model; seven-row classifier test remains video-confounded and "
                            "encoder-exposed",
    "one_vs_normal_myopathic": "Current model; eighteen-row classifier test remains video-confounded and "
                               "encoder-exposed",
    "one_vs_normal_cerebralpalsy": "Current model; nine-row classifier test remains video-confounded and "
                                   "encoder-exposed",
}

CURRENT_VERSIONS = {"current_five_stage", "corrected_two_fold_mean", "corrected_pooled_oof"}

R2_SCALE = "ridge R-squared; 1.0 recovers the target, 0.0 equals the fold mean"
RHO_SCALE = "mirror residual rho; 0 is mirror equivariant, 4 is mirror blind"
SYMMETRY_COLUMNS = [
    "experiment", "notebook", "status", "endpoint", "endpoint_scale",
    "treatment_lane", "treatment_value", "control_lane", "control_value", "control_role",
    "verdict", "verdict_basis", "checkpoint_fingerprint", "scope_note",
]
SYMMETRY_KEYS = ("experiment", "notebook")
SYMMETRY_VALUES = ("treatment_value", "control_value")


def _read_bundle(*parts: str) -> dict:
    path = ARTIFACTS.joinpath(*parts)
    if not path.is_file():
        raise FileNotFoundError(f"Run the symmetry notebooks before refreshing the ledger: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def symmetry_rows() -> pd.DataFrame:
    """Read the three preregistered symmetry verdicts out of their result bundles.

    Idea 5 and Idea 9 arm 1 read out of the frozen checkpoint, so they must carry the contract's own
    fingerprint. Arm 2 retrains one encoder per rung and per seed, so its bundle carries the shared
    fingerprint only as the baseline reference row; the scope note says so rather than implying that
    the rungs are that checkpoint.
    """
    idea5 = _read_bundle("idea5_signed_laterality_result.json")
    arm1 = _read_bundle("idea9_antisymmetric_readout_result.json")
    arm2 = _read_bundle("idea9_arm2", "idea9_arm2_evaluation_result.json")
    arm2_contract = _read_bundle("idea9_arm2", "idea9_arm2_contract.json")

    for name, fingerprint in (
        ("idea5_signed_laterality_result.json", idea5["fingerprint"]),
        ("idea9_antisymmetric_readout_result.json", arm1["fingerprint"]),
        ("idea9_arm2_contract.json", arm2_contract["contract"]["baseline_fingerprint"]),
    ):
        if fingerprint != FINGERPRINT:
            raise RuntimeError(f"{name} names fingerprint {fingerprint[:12]}, not {FINGERPRINT[:12]}")

    idea5_treatment = idea5["lanes"]["A_learned"]["r2"]
    idea5_control = idea5["lanes"]["C_floor"]["r2"]
    arm1_treatment = arm1["lanes"]["A_prime"]["r2"]
    arm1_control = arm1["lanes"]["E_pooled"]["r2"]
    y_fraction = arm1["y_variance"]["between_source_fraction"]
    y_threshold = arm1["y_variance"]["threshold"]
    primary = arm2["primary"]
    feature_std = next(g for g in arm2["guardrails"] if g["guardrail"] == "feature_std")
    seeds_run = len(arm2["seeds_observed"])
    seeds_registered = len(arm2["seeds_registered"])

    return pd.DataFrame([
        {
            "experiment": "idea5_signed_laterality",
            "notebook": idea5["notebook"],
            "status": "current",
            "endpoint": "ridge_r2_signed_laterality",
            "endpoint_scale": R2_SCALE,
            "treatment_lane": "A_learned",
            "treatment_value": idea5_treatment,
            "control_lane": "C_floor",
            "control_value": idea5_control,
            "control_role": "untrained-encoder floor",
            "verdict": idea5["verdict"]["PRIMARY_VERDICT"],
            "verdict_basis": (
                f"the treatment trails its own untrained floor by "
                f"{idea5_control - idea5_treatment:.3f}; all three preregistered gates failed; sign "
                f"consistency {idea5['verdict']['sign_consistency']:.3f} against a 0.75 gate; "
                f"anatomical mirror slope {idea5['mirror']['slope']:.3f} does not flip"
            ),
            "checkpoint_fingerprint": idea5["fingerprint"],
            "scope_note": (
                f"Transductive; {idea5['n_sequences']} canonical sequences from {idea5['n_sources']} "
                f"source videos over {idea5['n_splits']} source-disjoint folds; the encoder is frozen "
                f"and nothing is retrained; the measurement was valid and the answer was no"
            ),
        },
        {
            "experiment": "idea9_arm1_antisymmetric_readout",
            "notebook": arm1["notebook"],
            "status": "current",
            "endpoint": "ridge_r2_signed_laterality",
            "endpoint_scale": R2_SCALE,
            "treatment_lane": "A_prime",
            "treatment_value": arm1_treatment,
            "control_lane": "E_pooled",
            "control_role": "mirror-symmetrized, mathematically blind to left and right",
            "control_value": arm1_control,
            "verdict": arm1["verdict"]["PRIMARY_VERDICT"],
            "verdict_basis": (
                f"the side-blind control outscores the antisymmetric treatment by "
                f"{arm1_control - arm1_treatment:.3f}, so the lane is not admissible evidence about "
                f"sides; the wiring swap slope {arm1['wiring_identity']['swap_slope']:.3f} confirms the "
                f"head really was antisymmetric; permutation p "
                f"{arm1['permutation_null_A_prime']['p_value']:.3f}"
            ),
            "checkpoint_fingerprint": arm1["fingerprint"],
            "scope_note": (
                f"Transductive; the same frozen encoder and cohort as Idea 5, with only the readout "
                f"shape changed; the y-quality gate failed because {y_fraction:.3f} of the target's "
                f"variance lies between source videos against a {y_threshold:.2f} threshold, so a "
                f"held-out-source R-squared cannot support a positive laterality claim on this cohort"
            ),
        },
        {
            "experiment": "idea9_arm2_equivariant_retrain",
            "notebook": arm2["notebook"],
            "status": "current",
            "endpoint": primary["endpoint"],
            "endpoint_scale": RHO_SCALE,
            "treatment_lane": f"E1_equiv_weight_{arm2['equiv_weight']}",
            "treatment_value": primary["E1_mean"],
            "control_lane": f"D0_equiv_weight_{arm2_contract['pre_registered']['ladder']['equiv_weight']['D0']}",
            "control_value": primary["D0_mean"],
            "control_role": "the identical recipe with the equivariance term switched off",
            "verdict": arm2["PRIMARY_VERDICT"],
            "verdict_basis": (
                f"conditions 1 and 2 pass, improvement {primary['improvement']:.3f} against a control "
                f"seed spread of {primary['D0_seed_spread']:.3f} with "
                f"{primary['paired_bootstrap']['sources_improved']} of "
                f"{primary['paired_bootstrap']['sources']} source videos improving; condition 3 fails "
                f"because feature_std fell {feature_std['regression']:.4f} against a control seed "
                f"spread of {feature_std['D0_seed_spread']:.4f}; all three were required"
            ),
            "checkpoint_fingerprint": arm2_contract["contract"]["baseline_fingerprint"],
            "scope_note": (
                f"Transductive; the encoder is retrained per rung, {seeds_run} seeds run against "
                f"{seeds_registered} registered; the endpoint is label-free, so it sidesteps arm 1's "
                f"y-quality gate; the grouped five-class guardrail was not evaluable; the fingerprint "
                f"names the baseline reference row, not the rungs, which are freshly trained per seed"
            ),
        },
    ])[SYMMETRY_COLUMNS]


def supersede_label(fingerprints) -> str:
    """Name the outgoing lineage after the fingerprints it was measured on."""
    stale = {str(f)[:8] for f in fingerprints if str(f).strip() and str(f) != "nan"}
    return f"_{'_'.join(sorted(stale))}" if stale else "_previous"


def symmetry_differences(fresh: pd.DataFrame) -> list[str]:
    """Report how the on-disk symmetry file differs from the bundles, in the ledger's own idiom."""
    if not SYMMETRY_PATH.is_file():
        return [f"{SYMMETRY_PATH.name}: absent"]
    existing = pd.read_csv(SYMMETRY_PATH)
    missing = [column for column in SYMMETRY_COLUMNS if column not in existing.columns]
    if missing:
        return [f"{SYMMETRY_PATH.name}: missing columns {', '.join(missing)}"]

    differences = []
    for _, row in fresh.iterrows():
        match = existing
        for key in SYMMETRY_KEYS:
            match = match.loc[match[key].eq(row[key])]
        if match.empty:
            differences.append(f"{row['experiment']}: absent from {SYMMETRY_PATH.name}")
            continue
        before = match.iloc[0]
        for column in SYMMETRY_VALUES:
            if abs(float(before[column]) - float(row[column])) > 1e-9:
                differences.append(
                    f"{row['experiment']} {column}: {float(before[column]):.6f} -> {float(row[column]):.6f}")
        if str(before["verdict"]) != str(row["verdict"]):
            differences.append(
                f"{row['experiment']} verdict: {before['verdict']} -> {row['verdict']}")
    return differences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="report whether a refresh is needed without writing")
    parser.add_argument("--supersede", action="store_true",
                        help="retain the outgoing current rows under a fingerprint-named version label")
    args = parser.parse_args()

    ledger = pd.read_csv(LEDGER_PATH)
    if "checkpoint_fingerprint" not in ledger.columns:
        ledger["checkpoint_fingerprint"] = ""
    fresh = current_rows()
    fresh_symmetry = symmetry_rows()

    outgoing = ledger.loc[ledger["version"].isin(CURRENT_VERSIONS)].copy()
    kept = ledger.loc[~ledger["version"].isin(CURRENT_VERSIONS)].copy()

    differences = []
    for _, row in fresh.iterrows():
        match = outgoing.loc[outgoing["comparison"].eq(row["comparison"])
                             & outgoing["version"].eq(row["version"])]
        if match.empty:
            differences.append(f"{row['comparison']}/{row['version']}: absent from the ledger")
            continue
        for metric in METRICS:
            before, after = float(match.iloc[0][metric]), float(row[metric])
            if abs(before - after) > 1e-9:
                differences.append(
                    f"{row['comparison']}/{row['version']} {metric}: {before:.6f} -> {after:.6f}")

    differences.extend(symmetry_differences(fresh_symmetry))

    if args.check:
        if differences:
            print("the result ledger is STALE against the artifacts:")
            for difference in differences:
                print(f"  {difference}")
            return 1
        print(f"result_history.csv and {SYMMETRY_PATH.name} agree with the artifacts at {ARTIFACTS}")
        return 0

    if args.supersede and not outgoing.empty:
        label = supersede_label(outgoing["checkpoint_fingerprint"])
        outgoing["version"] = outgoing["version"] + label
        outgoing["status"] = "superseded"
        # A superseded row must not keep a note calling itself current, or the ledger contradicts itself.
        for prefix in ("Current five-stage model; ", "Current model; ", "Same checkpoint; "):
            outgoing["scope_note"] = outgoing["scope_note"].str.replace(prefix, "", regex=False)
        lineage = f"Superseded{label.replace('_', ' ')} lineage; "
        outgoing["scope_note"] = lineage + outgoing["scope_note"]
        kept = pd.concat([kept, outgoing], ignore_index=True)

    combined = pd.concat([kept, fresh], ignore_index=True)
    combined = combined.sort_values(["comparison", "status"], kind="stable")
    columns = ["comparison", "version", "status", "model_changed", "evaluation_changed",
               *METRICS, "checkpoint_fingerprint", "scope_note"]
    combined[columns].to_csv(LEDGER_PATH, index=False)

    symmetry = fresh_symmetry
    if args.supersede and SYMMETRY_PATH.is_file():
        previous = pd.read_csv(SYMMETRY_PATH)
        previous = previous.loc[previous["status"].eq("current")].copy()
        if not previous.empty:
            previous["experiment"] += supersede_label(previous["checkpoint_fingerprint"])
            previous["status"] = "superseded"
            symmetry = pd.concat([previous, fresh_symmetry], ignore_index=True)[SYMMETRY_COLUMNS]
    symmetry.to_csv(SYMMETRY_PATH, index=False)

    print(f"wrote {LEDGER_PATH} ({len(combined)} rows) for fingerprint {FINGERPRINT[:16]}")
    print(f"wrote {SYMMETRY_PATH} ({len(symmetry)} symmetry verdict rows)")
    print(f"contract checkpoint: {CONTRACT['encoder_checkpoint']}")
    for difference in differences:
        print(f"  refreshed {difference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
