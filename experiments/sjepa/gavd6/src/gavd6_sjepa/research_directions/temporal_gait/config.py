"""Strict, explicit configuration shared by CLI, notebooks and Slurm."""
from dataclasses import asdict, dataclass, field, fields
import json
import math
import os
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    schema_version: str = "temporal-gait-v1"
    mode: str = "real"
    run_root: str = ""
    video_manifest: str | None = None
    sequence_manifest: str | None = None
    pose_manifest: str | None = None
    identity_manifest: str | None = None
    exposure_manifest: str | None = None
    reservation_manifest: str | None = None
    split_manifest: str | None = None
    device: str = "cuda"
    seeds: list[int] = field(default_factory=lambda: [42, 43, 44, 45, 46])
    pilot_seeds: list[int] = field(default_factory=lambda: [42])
    development_seeds: list[int] = field(default_factory=lambda: [42, 43, 44])
    arms: list[str] = field(default_factory=lambda: ["masked_index", "masked", "future", "future_wrong_source", "future_wrong_time"])
    recipe: str = "historical_schedule_adaptation"
    cohort_scope: str = "full_allowed"
    updates: int = 1200
    checkpoint_updates: list[int] = field(default_factory=lambda: [0, 300, 1200])
    batch_size: int = 20
    hidden_dim: int = 96
    encoder_depth: int = 4
    predictor_depth: int = 2
    heads: int = 4
    patch_size: int = 4
    clock_channels: bool = False
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    ema_start: float = 0.999
    ema_end: float = 0.999
    lr_schedule: str = "constant"
    warmup_fraction: float = 0.05
    gradient_clip: float = 1.0
    objective: str = "centered_ce_v1"
    teacher_temperature: float = 0.06
    student_temperature: float = 0.1
    center_momentum: float = 0.9
    vicreg_weight: float = 0.05
    view_translation: float = 0.02
    mask_fraction: float = 0.5
    prefix_seconds: float = 2.56
    grid_hz: float = 25.0
    horizons: list[float] = field(default_factory=lambda: [0.25, 0.50, 1.0])
    target_interval_seconds: float = 0.08
    endpoint_tolerance_seconds: float = 0.020
    max_observation_age_seconds: float = 0.080
    window_stride_seconds: float = 0.50
    visibility_threshold: float = 0.45
    bootstrap_samples: int = 2000
    ridge_alphas: list[float] = field(default_factory=lambda: [0.1, 1., 10., 100., 1000., 10000.])
    direct_updates: int = 300
    precision: str = "float32"
    min_relative_improvement: float = 0.05
    video_time_check: str = "ffprobe_pts"
    ffprobe: str = "ffprobe"
    resume_from: str | None = None

    @property
    def root(self):
        if not self.run_root:
            raise ValueError("Explicit absolute run_root is required")
        return Path(self.run_root).resolve()

    def to_dict(self):
        return asdict(self)

    def scientific_dict(self):
        """Runtime checkpoint selection is logged but does not change the recipe."""
        values = self.to_dict()
        values["resume_from"] = None
        values["run_root"] = str(self.root)
        return values

    def validate(self, check_input_paths=False):
        if self.schema_version != "temporal-gait-v1" or self.mode not in {"real", "synthetic"}:
            raise ValueError("Unsupported schema/mode (no automatic data fallback)")
        if self.cohort_scope not in {"full_allowed", "historical_overlap"}:
            raise ValueError("Explicit full_allowed or historical_overlap cohort scope required")
        if not self.run_root or not Path(self.run_root).is_absolute():
            raise ValueError("run_root must be an explicit absolute path")
        if self.precision not in {"float32", "bfloat16"}:
            raise ValueError("Unsupported precision")
        if self.precision == "bfloat16" and not self.device.startswith("cuda"):
            raise ValueError("bfloat16 requires explicitly selected CUDA")
        for name in ("updates", "batch_size", "hidden_dim", "encoder_depth", "predictor_depth", "heads", "patch_size", "bootstrap_samples", "direct_updates"):
            val = getattr(self, name)
            if type(val) is not int or val < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.clock_channels) is not bool or self.batch_size < 2 or self.hidden_dim % self.heads:
            raise ValueError("Invalid model dimensions/batch/clock_channels")
        for name in ("learning_rate", "prefix_seconds", "grid_hz", "target_interval_seconds", "endpoint_tolerance_seconds", "max_observation_age_seconds", "window_stride_seconds", "teacher_temperature", "student_temperature", "gradient_clip"):
            val = getattr(self, name)
            if not isinstance(val, (float, int)) or not math.isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("mask_fraction", "ema_start", "ema_end", "center_momentum", "warmup_fraction", "visibility_threshold"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"Invalid {name}")
        for name in ("weight_decay", "vicreg_weight", "min_relative_improvement"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"Invalid {name}")
        if self.lr_schedule not in {"constant", "cosine"} or self.objective not in {"centered_ce_v1", "feature_regression_v1"}:
            raise ValueError("Unknown objective or schedule")
        if sorted(set(self.horizons)) != self.horizons or 0.5 not in self.horizons or min(self.horizons) <= self.target_interval_seconds:
            raise ValueError("Unique increasing horizons must include primary 0.50s")
        if any(not math.isfinite(h) for h in self.horizons):
            raise ValueError("Nonfinite horizon")
        if not math.isclose(self.target_interval_seconds * self.grid_hz, 2):
            raise ValueError("This recipe requires a separate two-sample future interval")
        count = self.prefix_seconds * self.grid_hz
        if not math.isclose(count, round(count)) or round(count) % self.patch_size:
            raise ValueError("Prefix bins must be integral and divisible by patch size")
        if self.endpoint_tolerance_seconds >= min(self.horizons) - self.target_interval_seconds:
            raise ValueError("Target tolerance could cross issue boundary")
        for name in ("seeds", "pilot_seeds", "development_seeds"):
            vals = getattr(self, name)
            if not vals or len(set(vals)) != len(vals) or any(type(s) is not int or s < 0 for s in vals):
                raise ValueError(f"Invalid {name}")
            if not set(vals) <= set(self.seeds):
                raise ValueError(f"{name} must be in frozen seeds")
        allowed = {"masked_index", "masked", "future", "future_wrong_source", "future_wrong_time"}
        if not self.arms or len(set(self.arms)) != len(self.arms) or not set(self.arms) <= allowed:
            raise ValueError("Unknown/duplicate arm; extensions require a separate protocol")
        if "masked_index" in self.arms and (self.patch_size != 4 or self.clock_channels):
            raise ValueError("First timing/support contrast holds patch4 and coordinate channels fixed")
        if sorted(set(self.checkpoint_updates)) != self.checkpoint_updates or min(self.checkpoint_updates) < 0 or max(self.checkpoint_updates) > self.updates or self.updates not in self.checkpoint_updates:
            raise ValueError("Checkpoints must be sorted and include frozen final update")
        if not self.ridge_alphas or any(not math.isfinite(a) or a <= 0 for a in self.ridge_alphas):
            raise ValueError("Invalid ridge penalties")
        if self.video_time_check != "ffprobe_pts":
            raise ValueError("Real video timing must be checked against original presentation timestamps")
        for name in MANIFEST_FIELDS:
            val = getattr(self, name)
            if self.mode == "real" and (not val or not Path(val).is_absolute()):
                raise ValueError(f"Real mode requires explicit absolute {name}")
            if val and check_input_paths and not Path(val).is_file():
                raise FileNotFoundError(f"Missing explicit {name}: {val}")
        if self.mode == "synthetic" and any(getattr(self, n) for n in MANIFEST_FIELDS):
            raise ValueError("Synthetic mode may not open private manifests")
        if self.resume_from and not Path(self.resume_from).is_absolute():
            raise ValueError("resume_from must be an explicitly selected absolute checkpoint path")
        return self

    @classmethod
    def from_env(cls, config_path=None):
        path = config_path or os.environ.get("TG_CONFIG")
        values = json.loads(Path(path).read_text()) if path else {}
        if not isinstance(values, dict):
            raise ValueError("Config must be a JSON object")
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        mapping = {"TG_RUN_ROOT": "run_root", "TG_MODE": "mode", "TG_DEVICE": "device", "TG_RESUME_FROM": "resume_from", **{"TG_" + n.upper(): n for n in MANIFEST_FIELDS}}
        for env, key in mapping.items():
            if env in os.environ:
                if key in values and values[key] not in (None, "") and values[key] != os.environ[env]:
                    path_key = key == "run_root" or key == "resume_from" or key in MANIFEST_FIELDS
                    if not path_key or Path(values[key]).resolve() != Path(os.environ[env]).resolve():
                        raise ValueError(f"Contradictory {env} and explicit config value")
                values[key] = os.environ[env]
        return cls(**values).validate()


MANIFEST_FIELDS = ("video_manifest", "sequence_manifest", "pose_manifest", "identity_manifest", "exposure_manifest", "reservation_manifest", "split_manifest")
