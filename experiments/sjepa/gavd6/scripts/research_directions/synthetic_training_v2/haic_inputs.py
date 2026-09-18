"""Reuse the existing HAIC assets and check reviewed source inputs on the CPU."""
from __future__ import annotations

import csv
from dataclasses import MISSING, fields
import getpass
from pathlib import Path
from typing import Mapping


AUDIT_COLUMNS = (
    "relative_path", "start_s", "locomotion_status", "audit_reviewer",
    "audit_evidence", "audit_date", "exposure", "canonical_person_id",
)
RESERVATION_COLUMNS = (
    "person_id", "canonical_person_id", "original_split", "reserved", "exposure",
)
MANIFEST_FILES = (
    "amass_raw_inventory_eligible.csv", "amass_subject_registry.csv", "amass_subject_splits.csv",
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def preparation_config(repo: Path, work: Path, env: Mapping) -> dict:
    """Resolve one complete preparation file without changing existing ST_* settings."""
    repo, work = Path(repo).expanduser().resolve(), Path(work).expanduser().resolve()
    scratch = Path("/hai/scratch") / (env.get("USER") or getpass.getuser())

    def path(variable, default):
        return Path(env.get(variable) or default).expanduser().resolve()

    model = path("ST_MODEL_ROOT", scratch / "models")
    body = path("ST_BODY_MODEL_ROOT", scratch / "body_models")
    amass = path("ST_AMASS_ROOT", path("AMASS_ROOT", repo / "data/amass") / "extracted")
    config_root = model / "mmpose/configs/body_2d_keypoint"
    roster = (
        ("rtmpose_m", "rtmpose", "rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py", "rtmpose-m.pth"),
        ("hrnet_w32", "hrnet", "topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py", "hrnet-w32.pth"),
        ("vitpose_base", "vitpose", "topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py", "vitpose-base.pth"),
    )
    return dict(
        scope_config=str(work / "config/prepare-01.json"),
        manifest_dir=str(repo / "manifests/amass"), amass_root=str(amass),
        locomotion_audit=str(work / "inputs/locomotion-audit.csv"),
        reservation_csv=str(work / "inputs/person-reservations.csv"),
        body_model_root=str(body),
        dmpl_root=str(path("ST_DMPL_ROOT", body.parent / "dmpls" if body.name == "smplh" else body / "dmpls")),
        uv_path=str(path("ST_UV_PATH", model / "synthetic-rendering/smplitex/smpl_uv.obj")),
        texture_dir=str(path("ST_TEXTURE_DIR", model / "synthetic-rendering/smplitex/textures")),
        background_dir=str(path("ST_BACKGROUND_DIR", model / "synthetic-rendering/coco-backgrounds")),
        seed=17,
        estimators=[dict(student_id=student, family=family, config=str(config_root / name),
                         checkpoint=str(model / "pose" / checkpoint))
                    for student, family, name, checkpoint in roster],
    )


def create_drafts(config: dict, work: Path) -> Path:
    """Create worksheets once; never infer reviews, aliases or reservation decisions."""
    draft = Path(work).expanduser().resolve() / "inputs/review-drafts"
    if draft.is_dir():
        return draft
    if draft.exists():
        raise FileExistsError(f"Review draft path is not a directory: {draft}")
    from gavd6_sjepa.research_directions.motion_preservation.motion_data import load_amass_manifest

    table = load_amass_manifest(config["manifest_dir"], config["amass_root"])
    candidates = table.loc[
        table.original_split.isin(["train", "validation"])
        & table.available & table.duration_s.ge(63 / 25)
    ].copy()
    candidate_columns = ["person_id", "original_split", "relative_path", "raw_path", "duration_s"]
    candidates = candidates[candidate_columns]
    people = table[["person_id", "original_split"]].drop_duplicates().sort_values("person_id").copy()
    for field in ("canonical_person_id", "reserved", "exposure"):
        people[field] = ""
    try:
        draft.mkdir(parents=True)
    except FileExistsError:
        if draft.is_dir():
            return draft
        raise
    candidates.to_csv(draft / "motion-candidates.csv", index=False)
    people[list(RESERVATION_COLUMNS)].to_csv(draft / "person-reservations.draft.csv", index=False)
    with (draft / "locomotion-audit.draft.csv").open("x", newline="") as stream:
        csv.writer(stream).writerow(AUDIT_COLUMNS)
    return draft


def _asset_path(value, label):
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"Preparation configuration needs a nonempty {label} path")
    return Path(value).expanduser()


def _available_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _require_assets(files: list, directories: list):
    problems = [f"{label}: {path}" for label, path in files if not _available_file(path)]
    for label, path, image_directory in directories:
        try:
            available = path.is_dir()
            if available and image_directory:
                available = any(p.suffix.lower() in IMAGE_SUFFIXES and _available_file(p) for p in path.rglob("*"))
        except OSError:
            available = False
        if not available:
            suffix = " (needs at least one nonempty PNG/JPG/JPEG)" if image_directory else ""
            problems.append(f"{label}: {path}{suffix}")
    if problems:
        raise FileNotFoundError(
            "Source inputs are missing, empty or unreadable:\n  " + "\n  ".join(problems)
            + "\nComplete the reviewed CSVs using inputs/review-drafts, or correct their configured paths. "
            "Reuse the existing model and rendering assets; check the original synthetic-training guide for missing assets."
        )


def check_inputs(config: dict, held: str) -> dict:
    """Validate metadata and asset paths without CUDA, body models or raw motion arrays.

    Configuration inheritance files are small and hashed by the existing parser.
    Checkpoints and complete asset trees are not hashed or loaded by this check.
    The GPU preparation job remains responsible for actual model/render compatibility.
    """
    from gavd6_sjepa.research_directions.synthetic_training.estimators import StudentSpec
    from gavd6_sjepa.research_directions.synthetic_training_v2.preparation import (
        _config_files, audited_motion_windows, held_family,
    )

    specifications = config.get("estimators")
    if not isinstance(specifications, list) or not specifications:
        raise ValueError("Preparation requires a nonempty estimators list")
    allowed = {field.name for field in fields(StudentSpec)}
    required = {field.name for field in fields(StudentSpec)
                if field.default is MISSING and field.default_factory is MISSING}
    files = []
    for index, spec in enumerate(specifications):
        if not isinstance(spec, dict) or set(spec) - allowed or required - set(spec):
            raise ValueError(f"Estimator {index} must use StudentSpec fields {sorted(allowed)}; "
                             f"required fields are {sorted(required)}")
        StudentSpec(**spec)
        for key in ("config", "checkpoint", "head_checkpoint"):
            if key == "head_checkpoint" and not spec.get(key):
                continue
            label = f"estimator {spec['student_id']} {key}"
            files.append((label, _asset_path(spec[key], label)))

    for field in ("locomotion_audit", "reservation_csv", "uv_path"):
        files.append((field, _asset_path(config.get(field), field)))
    manifest = _asset_path(config.get("manifest_dir"), "manifest_dir")
    if (manifest / "amass").is_dir():
        manifest /= "amass"
    files.extend((name, manifest / name) for name in MANIFEST_FILES)
    body_root = _asset_path(config.get("body_model_root"), "body_model_root")
    body = body_root / "smplh" if (body_root / "smplh").is_dir() else body_root
    dmpl = _asset_path(config["dmpl_root"], "dmpl_root") if config.get("dmpl_root") else (
        body_root / "dmpls" if body != body_root else body_root.parent / "dmpls")
    directories = [("amass_root", _asset_path(config.get("amass_root"), "amass_root"), False),
                   ("body_model_root", body, False), ("dmpl_root", dmpl, False)]
    directories.extend((field, _asset_path(config.get(field), field), True)
                       for field in ("texture_dir", "background_dir"))
    _require_assets(files, directories)

    family = held_family(specifications, held)
    for spec in specifications:
        _config_files(spec["config"])
    table, excluded = audited_motion_windows(
        config["manifest_dir"], config["amass_root"], config["locomotion_audit"], config["reservation_csv"])
    if not {"train", "development"} <= set(table["split"]):
        raise ValueError("Both train and development splits need reviewed, admitted windows")
    by_person = table.groupby(["split", "canonical_person_id"]).size()
    training_counts = by_person.loc["train"]
    insufficient = training_counts[training_counts.lt(2)]
    if len(insufficient):
        raise ValueError("Each training person needs at least two distinct reviewed windows for shuffled_jepa; "
                         f"insufficient people: {insufficient.to_dict()}")
    if "gender" not in table:
        raise ValueError("AMASS inventory lacks gender metadata needed to check licensed body-model assets")
    genders = {str(value).strip().lower() for value in table["gender"]}
    if not genders <= {"male", "female"}:
        raise ValueError(f"Unsupported AMASS gender metadata: {sorted(genders)}; the source loader requires male or female")
    _require_assets([(f"{label} {gender}", root / gender / "model.npz")
                     for gender in sorted(genders) for label, root in (("SMPL-H", body), ("DMPL", dmpl))], [])

    return dict(
        status="pass", held_extractor_family=family, excluded_windows=len(excluded),
        counts=dict(admitted_windows=len(table), independent_people=int(table.canonical_person_id.nunique()),
                    train_windows=int(table.split.eq("train").sum()),
                    development_windows=int(table.split.eq("development").sum())),
        windows_by_person=[dict(split=split, canonical_person_id=person, windows=int(count))
                          for (split, person), count in by_person.items()],
        validation="CPU metadata and file availability; GPU model/render compatibility remains to be checked",
    )
