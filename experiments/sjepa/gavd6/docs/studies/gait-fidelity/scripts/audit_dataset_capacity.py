"""Count metadata capacity without modifying cohorts or opening source motions.

Run from the repository root with .venv/bin/python. Counts deliberately omit
the remote reservation/review ledgers and are upper bounds, not authorization
to use protected people. This does not validate HAIC media availability.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from gavd6_sjepa.research_directions.gait_fidelity.cohort import plan_cohort


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs/studies/gait-fidelity/records/dataset-capacity-20260924.json"


def read(relative):
    return pd.read_csv(ROOT / relative, keep_default_na=False)


def population(table):
    return {
        "people": int(table.canonical_person_id.nunique()),
        "motions": int(table.relative_path.nunique()),
        "windows": len(table),
    }


def main():
    raw = read("manifests/amass/amass_raw_inventory.csv")
    eligible = read("manifests/amass/amass_raw_inventory_eligible.csv")
    registry = read("manifests/amass/amass_subject_registry.csv")
    splits = read("manifests/amass/amass_subject_splits.csv")
    converted = read("manifests/amass/amass_core11_conversion.csv")
    sequences = read("manifests/gavd/gavd_full_sequences.csv")
    videos = read("manifests/gavd/gavd_full_videos.csv")
    people = splits[["identity", "split"]].drop_duplicates()
    assert not people.identity.duplicated().any()
    assert set(sequences.video_id) == set(videos.video_id)
    assert sequences.groupby("video_id").dataset_annotation.nunique().max() == 1
    config = json.loads((ROOT / "outputs/gait-fidelity/config.json").read_text())
    config["preparation"]["manifest_dir"] = str(ROOT / "manifests/amass")
    config["preparation"]["amass_root"] = str(ROOT / "data/amass/extracted")
    for key in ("reservation_csv", "motion_review_csv"):
        config["cohort"].pop(key, None)
    plans, tables = {}, {}
    for preset in ("treadmill_walking", "named_walking", "all_eligible"):
        trial = deepcopy(config)
        trial["cohort"]["preset"] = preset
        result = plan_cohort(trial)
        table = tables[preset] = pd.DataFrame(result["records"])
        plans[preset] = {
            "roles": result["summary"]["roles"],
            "decisions": dict(Counter(r["decision"] for r in result["inventory"])),
            "by_source_and_role": [dict(source=source, role=role, **population(rows))
                                   for (source, role), rows in table.groupby(["source_dataset", "split"])],
        }
    additional = {}
    for role in ("train", "development", "confirmation"):
        named = tables["named_walking"].query("split == @role")
        all_rows = tables["all_eligible"].query("split == @role")
        fresh = all_rows.loc[~all_rows.canonical_person_id.isin(named.canonical_person_id)]
        additional[role] = [dict(person_id=person, **population(rows))
                            for person, rows in fresh.groupby("canonical_person_id")]
    scenarios = []
    samples, hz = int(config["data"]["samples"]), float(config["data"]["hz"])
    for fps in (25, 30, 60):
        minimum = math.ceil((samples - 1) / hz * fps) + 1
        subset = sequences.loc[sequences.n_annotated_frames >= minimum]
        scenarios.append(dict(assumed_native_fps=fps, minimum_native_frames=minimum,
                              sequences=len(subset), source_videos=int(subset.video_id.nunique()),
                              by_dataset_label=[dict(label=label, sequences=len(rows),
                                                     source_videos=int(rows.video_id.nunique()))
                                                for label, rows in subset.groupby("dataset_annotation")]))
    files = sorted((ROOT / "manifests/amass").glob("*.csv")) + sorted((ROOT / "manifests/gavd").glob("*"))
    files += [ROOT / "outputs/gait-fidelity/config.json",
              ROOT / "src/gavd6_sjepa/research_directions/gait_fidelity/cohort.py"]
    report = dict(
        schema="gait-fidelity-metadata-capacity-v1",
        limitations=["Metadata upper bounds before geometry QC, actual availability and exposure review.",
                     "Remote reservation and interval-review ledgers were intentionally omitted.",
                     "Window counts are nonoverlapping source intervals, not independent people.",
                     "All-action candidates are not automatically walking intervals.",
                     "GAVD native FPS and person crosswalk are absent from these manifests.",
                     "No source experiment was run or modified."],
        input_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},
        amass=dict(raw_motions=len(raw), eligible_motions=len(eligible),
                   registry_alias_rows=len(registry), split_alias_rows=len(splits),
                   canonical_people=len(people), original_splits=people.split.value_counts().to_dict(),
                   registry_status=registry.identity_audit_status.value_counts().to_dict(),
                   core11_conversion_rows=len(converted), core11_status=converted.status.value_counts().to_dict(),
                   core11_canonical_frames=int(converted.canonical_frames.sum()),
                   core11_fps=sorted(converted.canonical_fps.unique().tolist()),
                   raw_sources=[dict(source=source, motions=len(rows), candidate_aliases=int(rows.subject_id_candidate.nunique()))
                                for source, rows in raw.groupby("source_dataset")],
                   window_samples=samples, window_hz=hz, span_s=(samples - 1) / hz,
                   stride_s=samples / hz, plans=plans,
                   additional_people_all_eligible_vs_named=additional,
                   people_without_full_length_records=sorted(set(people.identity) - set(tables["all_eligible"].canonical_person_id))),
        gavd=dict(sequences=len(sequences), source_videos=len(videos),
                  annotation_rows=int(sequences.n_annotated_frames.sum()),
                  by_dataset_label=[dict(label=label, sequences=len(rows), source_videos=int(rows.video_id.nunique()),
                                         annotation_rows=int(rows.n_annotated_frames.sum()))
                                    for label, rows in sequences.groupby("dataset_annotation")],
                  duration_scenarios_not_measured_availability=scenarios),
    )
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(OUT.relative_to(ROOT))
    print(json.dumps({"amass_people": len(people), "plans": {k: v["roles"] for k, v in plans.items()},
                      "additional_people": {k: len(v) for k, v in additional.items()},
                      "gavd": report["gavd"]}, indent=2))


if __name__ == "__main__":
    main()
