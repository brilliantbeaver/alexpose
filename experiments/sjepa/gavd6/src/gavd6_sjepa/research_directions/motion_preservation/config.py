"""Small, editable experiment configuration shared by notebooks and Slurm."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path


@dataclass
class RunConfig:
    run_root: str = "outputs/motion-preservation/pilot"
    mode: str = "real"
    device: str = "cuda"
    amass_manifest_dir: str = "manifests/amass"
    amass_root: str = "data/amass"
    body_model_root: str = ""
    dmpl_root: str | None = None
    gavd_manifest_dir: str = "manifests/gavd"
    gavd_video_root: str = "data/gavd_full/youtube/all"
    gavd_pose_root: str = ""
    gavd_reservation_csv: str | None = None
    prior_backend: str = "momask"
    prior_id: str = "momask"
    momask_repo: str = ""
    momask_checkpoint: str = ""
    target_skeleton_path: str | None = None
    prior_predictions: str | None = None
    flow_backend: str = "sea_raft"
    flow_repo: str = ""
    flow_checkpoint: str = ""
    flow_config: str | None = None
    external_methods: dict[str, str] = field(default_factory=dict)
    fps: float = 20.0
    duration_s: float = 3.2
    image_size: int = 128
    max_motions_per_role: int = 32
    max_motions_per_person: int = 2
    max_gavd_sequences: int = 8
    seed: int = 17
    seeds: list[int] = field(default_factory=lambda: [17, 23, 42])
    epochs: int = 25
    batch_size: int = 16
    hidden_dim: int = 64
    learning_rate: float = 0.001
    noise_std_m: float = 0.025
    target_noise_removal: float = 0.25
    removal_tolerance: float = 0.10
    retention_gain: float = 0.15
    bootstrap_samples: int = 1000
    feature_modes: list[str] = field(default_factory=lambda: ["full", "coordinates", "shuffled_flow", "random_features"])
    strength_grid: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])

    @property
    def root(self) -> Path:
        return Path(self.run_root).expanduser().resolve()

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "config.json").write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def from_env(cls) -> "RunConfig":
        """Read MP_CONFIG or an existing run config, then explicit MP_* overrides.

        Demo mode uses only small simulated motions and classical estimators.
        It never falls back silently when research assets are missing.
        """
        selected_root = os.environ.get("MP_RUN_ROOT", cls.run_root)
        path = Path(os.environ.get("MP_CONFIG", str(Path(selected_root) / "config.json"))).expanduser()
        values = json.loads(path.read_text()) if path.is_file() else {}
        env = {
            "MP_RUN_ROOT": "run_root", "MP_MODE": "mode", "MP_DEVICE": "device",
            "MP_AMASS_ROOT": "amass_root", "MP_BODY_MODEL_ROOT": "body_model_root",
            "MP_DMPL_ROOT": "dmpl_root", "MP_GAVD_VIDEO_ROOT": "gavd_video_root",
            "MP_GAVD_POSE_ROOT": "gavd_pose_root", "MP_MOMASK_REPO": "momask_repo",
            "MP_GAVD_RESERVATION": "gavd_reservation_csv",
            "MP_MOMASK_CHECKPOINT": "momask_checkpoint", "MP_TARGET_SKELETON": "target_skeleton_path",
            "MP_PRIOR_BACKEND": "prior_backend", "MP_PRIOR_ID": "prior_id",
            "MP_PRIOR_PREDICTIONS": "prior_predictions", "MP_FLOW_BACKEND": "flow_backend",
            "MP_FLOW_REPO": "flow_repo", "MP_FLOW_CHECKPOINT": "flow_checkpoint",
            "MP_FLOW_CONFIG": "flow_config",
        }
        for source, target in env.items():
            if source in os.environ:
                values[target] = os.environ[source]
        for source, target in {"AMASS_EXTRACTED_ROOT": "amass_root", "AMASS_BODY_MODEL_ROOT": "body_model_root"}.items():
            if source in os.environ and target not in values:
                values[target] = os.environ[source]
        for key, target in {"MP_MAX_MOTIONS": "max_motions_per_role", "MP_EPOCHS": "epochs", "MP_IMAGE_SIZE": "image_size"}.items():
            if key in os.environ:
                values[target] = int(os.environ[key])
        if "MP_SEEDS" in os.environ:
            values["seeds"] = [int(x) for x in os.environ["MP_SEEDS"].split(",")]
        cfg = cls(**values)
        if cfg.mode not in {"real", "demo"}:
            raise ValueError("MP_MODE must be real or demo")
        if cfg.mode == "demo":
            cfg.device = os.environ.get("MP_DEVICE", "cpu")
            cfg.prior_backend = "demo_smoothing"
            cfg.prior_id = "demo_smoothing"
            cfg.flow_backend = "demo_farneback"
            if not path.is_file():
                cfg.max_motions_per_role = min(cfg.max_motions_per_role, 3)
                cfg.duration_s = 1.6
                cfg.image_size = 64
                cfg.epochs = int(os.environ.get("MP_EPOCHS", "2"))
                cfg.seeds = [cfg.seed]
                cfg.bootstrap_samples = 100
        return cfg
