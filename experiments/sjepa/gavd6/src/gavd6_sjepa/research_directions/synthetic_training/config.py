"""Editable experiment settings shared by notebooks and HAIC jobs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path


@dataclass
class RunConfig:
    """Paths, finite training budgets, and source-only selection settings.

    A student has ``student_id``, ``family``, ``config``, ``checkpoint`` and
    ``role`` (train, validation, held). Validation students select the teacher's
    settings; held students never enter that process. All paths in JSON are
    resolved relative to GAVD6_ROOT by ``from_env``.
    """

    run_root: str = "outputs/synthetic-training/pilot-01"
    device: str = "cuda"
    seed: int = 17
    amass_manifest_dir: str = "manifests/amass"
    amass_root: str = "data/amass"
    body_model_root: str = ""
    dmpl_root: str | None = None
    render_texture_dir: str = ""
    render_background_dir: str = ""
    render_uv_path: str = ""
    render_width: int = 256
    render_height: int = 320
    clip_frames: int = 64
    clip_fps: float = 15.0
    clips_per_lesson: int = 4
    source_contexts_per_domain: int = 2
    coco_image_root: str = ""
    coco_annotations_json: str = ""
    coco_replay_images: int = 256
    gavd_manifest_dir: str = "manifests/gavd"
    gavd_video_root: str = "data/gavd_full/youtube/all"
    gavd_reservation_csv: str = ""
    gavd_view_csv: str = ""
    gavd_context_recordings: int = 3
    gavd_early_recordings: int = 4
    gavd_confirmation_recordings: int = 6
    gavd_annotation_frames: int = 4
    gavd_annotations_csv: str = ""
    students: list[dict] = field(default_factory=list)
    probe_steps: int = 10
    adaptation_steps: list[int] = field(default_factory=lambda: [25, 75])
    train_batch_size: int = 20
    predict_batch_size: int = 32
    synthetic_fraction: float = 0.1
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    context_kind: str = "vjepa"
    context_repo: str = ""
    context_checkpoint: str = ""
    context_builder: str = "vjepa2_1_vit_base_384"
    context_checkpoint_key: str = "ema_encoder"
    context_image_size: int = 384
    context_frames: int = 64
    image_checkpoint: str = ""
    selector_neighbors: list[int] = field(default_factory=lambda: [1, 3, 5])
    selector_ridge: list[float] = field(default_factory=lambda: [1.0, 10.0, 100.0])
    bootstrap_samples: int = 2000
    accurate_joint_threshold: float = 0.03
    missing_prediction_penalty: float = 1.0

    @property
    def root(self) -> Path:
        return Path(self.run_root).expanduser().resolve()

    @property
    def source_students(self) -> list[dict]:
        return [s for s in self.students if s["role"] in {"train", "validation"}]

    @property
    def deployment_students(self) -> list[dict]:
        return list(self.students)

    def student(self, student_id: str) -> dict:
        for item in self.students:
            if item["student_id"] == student_id:
                return item
        raise ValueError(f"Unknown student_id {student_id!r}; configure it in students.")

    def validate(self) -> None:
        ids = [s["student_id"] for s in self.students]
        if len(ids) != len(set(ids)):
            raise ValueError("Student identifiers must be unique.")
        for student in self.students:
            if student.get("role") not in {"train", "validation", "held"}:
                raise ValueError("Every student role must be train, validation, or held.")
            if not all(student.get(k) for k in ("student_id", "family", "config", "checkpoint")):
                raise ValueError("Each student needs student_id, family, config and checkpoint.")
            if not all(c.isalnum() or c in "_.-" for c in student["student_id"]):
                raise ValueError("Student IDs may contain letters, digits, underscores, dots and hyphens.")
        held = {s["family"] for s in self.students if s["role"] == "held"}
        if held & {s["family"] for s in self.source_students}:
            raise ValueError("Held architecture families must be absent from all source students.")
        if self.probe_steps <= 0 or not self.adaptation_steps or min(self.adaptation_steps) <= 0:
            raise ValueError("Probe and adaptation budgets must be positive.")
        if len(set(self.adaptation_steps)) != len(self.adaptation_steps):
            raise ValueError("Adaptation budgets must be distinct.")
        if not 0 < self.synthetic_fraction < 1:
            raise ValueError("synthetic_fraction must lie strictly between zero and one.")
        count = round(self.train_batch_size * self.synthetic_fraction)
        if not 0 < count < self.train_batch_size:
            raise ValueError("Batch size must allow both synthetic and replay examples.")
        if self.predict_batch_size <= 0 or self.clip_frames < 2:
            raise ValueError("Positive prediction batch size and at least two clip frames are required.")
        if self.context_kind not in {"vjepa", "image", "simple", "none"}:
            raise ValueError("context_kind must be vjepa, image, simple or none.")
        if self.missing_prediction_penalty <= 0 or self.bootstrap_samples < 1:
            raise ValueError("The missing-prediction penalty and bootstrap count must be positive.")

    def save(self) -> Path:
        self.validate()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "config.json"
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")
        return path

    @classmethod
    def from_env(cls) -> "RunConfig":
        """Explicit ST_* values override JSON; shared legacy paths fill gaps."""
        base = Path(os.environ.get("GAVD6_ROOT", Path.cwd())).expanduser().resolve()
        root = Path(os.environ.get("ST_RUN_ROOT", cls.run_root)).expanduser()
        if not root.is_absolute():
            root = base / root
        config = Path(os.environ.get("ST_CONFIG", str(root / "config.json"))).expanduser()
        if not config.is_absolute():
            config = base / config
        if "ST_CONFIG" in os.environ and not config.is_file():
            raise FileNotFoundError(f"ST_CONFIG does not exist: {config}")
        values = json.loads(config.read_text()) if config.is_file() else {}
        paths = {
            "ST_RUN_ROOT": "run_root", "ST_AMASS_ROOT": "amass_root",
            "ST_BODY_MODEL_ROOT": "body_model_root", "ST_DMPL_ROOT": "dmpl_root",
            "ST_TEXTURE_DIR": "render_texture_dir", "ST_BACKGROUND_DIR": "render_background_dir",
            "ST_UV_PATH": "render_uv_path", "ST_COCO_IMAGE_ROOT": "coco_image_root",
            "ST_COCO_ANNOTATIONS": "coco_annotations_json", "ST_GAVD_VIDEO_ROOT": "gavd_video_root",
            "ST_GAVD_RESERVATION": "gavd_reservation_csv", "ST_GAVD_VIEWS": "gavd_view_csv",
            "ST_GAVD_ANNOTATIONS": "gavd_annotations_csv", "ST_CONTEXT_REPO": "context_repo",
            "ST_CONTEXT_CHECKPOINT": "context_checkpoint", "ST_IMAGE_CHECKPOINT": "image_checkpoint",
        }
        for env, name in paths.items():
            if env in os.environ:
                values[name] = os.environ[env]
        fallbacks = {
            "amass_root": ("MP_AMASS_ROOT", "AMASS_EXTRACTED_ROOT"),
            "body_model_root": ("MP_BODY_MODEL_ROOT", "AMASS_BODY_MODEL_ROOT"),
            "dmpl_root": ("MP_DMPL_ROOT",),
            "gavd_video_root": ("MP_GAVD_VIDEO_ROOT", "GAVD_FULL_ROOT", "GAVD_VIDEO_ROOT"),
            "gavd_reservation_csv": ("MP_GAVD_RESERVATION",),
            "context_repo": ("VJEPA2_ROOT",),
            "context_checkpoint": ("FI_TEACHER_CHECKPOINT",),
        }
        for field_name, names in fallbacks.items():
            if field_name not in values:
                for name in names:
                    if os.environ.get(name):
                        values[field_name] = os.environ[name]
                        break
        for env, name in {"ST_DEVICE": "device", "ST_CONTEXT_KIND": "context_kind",
                          "ST_CONTEXT_BUILDER": "context_builder"}.items():
            if env in os.environ:
                values[name] = os.environ[env]
        for env, name in {"ST_PROBE_STEPS": "probe_steps", "ST_BATCH_SIZE": "train_batch_size",
                          "ST_SEED": "seed", "ST_CLIPS_PER_LESSON": "clips_per_lesson",
                          "ST_CLIP_FRAMES": "clip_frames"}.items():
            if env in os.environ:
                values[name] = int(os.environ[env])
        if "ST_ADAPTATION_STEPS" in os.environ:
            values["adaptation_steps"] = [int(x) for x in os.environ["ST_ADAPTATION_STEPS"].split(",")]
        cfg = cls(**values)
        for name in set(paths.values()) | {"amass_manifest_dir", "gavd_manifest_dir"}:
            value = getattr(cfg, name)
            if value:
                path = Path(value).expanduser()
                setattr(cfg, name, str(path if path.is_absolute() else base / path))
        for student in cfg.students:
            for name in ("config", "checkpoint", "head_checkpoint"):
                if student.get(name):
                    path = Path(student[name]).expanduser()
                    student[name] = str(path if path.is_absolute() else base / path)
        cfg.validate()
        return cfg
