"""Build the three executable full-GAVD GaitParity feasibility notebooks."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def md(source: str) -> dict:
    source = source.strip() + "\n"
    return {"cell_type": "markdown", "id": hashlib.sha256(("md:" + source).encode()).hexdigest()[:12], "metadata": {}, "source": source}


def code(source: str) -> dict:
    source = source.strip() + "\n"
    return {"cell_type": "code", "id": hashlib.sha256(("code:" + source).encode()).hexdigest()[:12], "execution_count": None, "metadata": {}, "outputs": [], "source": source}


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_SETUP = r'''
from pathlib import Path
import json, math, os, sys

def find_notebook_root(start=None):
    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "gait_parity_jepa.py").exists() and (candidate / "nb_09a_equivariant_encoder_contract.ipynb").exists():
            return candidate
    raise FileNotFoundError("Run this notebook from experiments/sjepa/gavd6 or set the working directory there")

PROJECT_DIR = find_notebook_root()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from gait_parity_jepa import *

MODE = os.getenv("GAIT_PARITY_MODE", "smoke").strip().lower()
if MODE not in {"smoke", "real"}:
    raise ValueError("GAIT_PARITY_MODE must be smoke or real")
PROFILE_NAME = "smoke" if MODE == "smoke" else os.getenv("GAIT_PARITY_PROFILE", "cpu").strip().lower()
if PROFILE_NAME not in PROFILES or (MODE == "real" and PROFILE_NAME == "smoke"):
    raise ValueError("Real runs require GAIT_PARITY_PROFILE=cpu or gpu")
CONFIG = PROFILES[PROFILE_NAME]
MATCHING_REGIME = os.getenv("GAIT_PARITY_MATCHING", "exposure").strip().lower()
if MATCHING_REGIME not in {"exposure", "compute"}:
    raise ValueError("GAIT_PARITY_MATCHING must be exposure or compute")
RUN_ID = os.getenv("GAIT_PARITY_RUN_ID", "smoke")
if MODE == "real" and RUN_ID == "smoke":
    raise ValueError("Set a versioned GAIT_PARITY_RUN_ID for a real run")
SEEDS = [int(value) for value in os.getenv(
    "GAIT_PARITY_SEEDS", "7" if PROFILE_NAME in {"cpu", "smoke"} else "7,19,31"
).split(",")]
OUT_DIR = PROJECT_DIR / "work" / "artifacts" / "gait_parity" / MODE / RUN_ID / MATCHING_REGIME
OUT_DIR.mkdir(parents=True, exist_ok=True)

if MODE == "smoke":
    RECORDS = synthetic_records(frames=48)
    POSE_DIR = None
else:
    POSE_DIR = resolve_pose_dir(PROJECT_DIR)
    RECORDS = load_gavd_records(POSE_DIR)
WINDOWS, VALID_PATCH, WINDOW_TABLE = build_windows(RECORDS, CONFIG)
MANIFEST = cohort_manifest(RECORDS, WINDOW_TABLE, CONFIG, MODE)

print("scope            :", MANIFEST["scope"])
print("mode/profile     :", MODE, "/", PROFILE_NAME)
print("matching regime  :", MATCHING_REGIME)
print("run ID           :", RUN_ID)
print("device default   :", os.getenv("GAIT_PARITY_DEVICE", "cpu"))
print("records/windows  :", len(RECORDS), "/", len(WINDOWS))
print("source videos    :", MANIFEST["source_video_count"])
print("output           :", OUT_DIR)
'''


cells_09c = [
    md(r'''# 09c. Freeze the full-GAVD GaitParity training contract

This notebook turns the contracts in `nb_09a` and `nb_09b` into a concrete three-model
feasibility run over every available GAVD sequence:

1. a shared one-view **standard** encoder applied independently to both orbit members;
2. a **paired-unconstrained** encoder with branch-specific self- and cross-attention;
3. a **reflection-equivariant** encoder with shared self-attention and symmetric cross-attention.

This is not clinical evidence. GAVD lacks participant IDs, the encoder sees the local corpus, and all
health quantities come from the skeleton coordinates themselves. The allowed conclusion is only that
the full loop trains locally, remains non-collapsed, and satisfies its declared geometry.

### Safe startup

The notebook defaults to `GAIT_PARITY_MODE=smoke` and the CPU device. A full local run requires:

```bash
export GAIT_PARITY_MODE=real
export GAIT_PARITY_RUN_ID=gavd-full-v1
export GAIT_PARITY_PROFILE=cpu       # or gpu
export GAIT_PARITY_DEVICE=cpu        # CUDA is opt-in
export GAIT_PARITY_MATCHING=exposure # rerun as compute for the second fairness view
```
'''),
    code(COMMON_SETUP),
    md(r'''## 1. CPU and CUDA profiles

The CPU profile is intentionally modest but uses all 96 cached GAVD sequences. The GPU profile restores
the 96-wide, four-layer encoder scale, denser temporal windows, more epochs, more seeds, and mixed
precision. Changing hardware does not change the objective.
'''),
    code(r'''
profile_table = pd.DataFrame([asdict(PROFILES[name]) for name in ["cpu", "gpu", "smoke"]]).set_index("profile")
display(profile_table)
'''),
    md(r'''## 2. Freeze anatomical reflection and the full-cohort window manifest

Reflection negates the lateral coordinate and exchanges all 16 BlazePose left/right pairs. Windows are
formed only after short-gap interpolation, pelvis centering, and robust body scaling. Every eligible
sequence is retained. Conditions are recorded for provenance but never enter the objective.
'''),
    code(r'''
mirror_error = float((anatomical_mirror(anatomical_mirror(WINDOWS[:8])) - WINDOWS[:8]).abs().max())
assert mirror_error == 0.0
assert MANIFEST["record_count"] == (12 if MODE == "smoke" else 96)
display(WINDOW_TABLE.groupby("condition").agg(sequences=("sequence_id", "nunique"), windows=("window_id", "size")))
print("mirror involution max abs:", mirror_error)
print("cohort SHA256:", MANIFEST["cohort_sha256"])
print("window SHA256:", MANIFEST["window_sha256"])
'''),
    md(r'''## 3. Freeze the shared objective and both matching views

All variants optimize the same centered-and-sharpened masked-token JEPA loss. The EMA teacher is updated
with the same schedule. The same two orbit augmentations enter parity-resolved VICReg: invariance,
variance, and covariance are evaluated separately on the even and odd channels. This is deliberately
stronger than total-feature VICReg because `nb_09a` demonstrated that a perfectly equivariant odd
channel can collapse to zero.

`exposure` gives every variant the same optimizer updates, orbit windows, masks, and branch forwards.
`compute` holds a frozen analytic token-parameter budget approximately constant, so architectures with
different per-step costs receive different update counts. The proxy is not called FLOPs; measured wall
time, peak CUDA memory, parameter counts, and actual exposures remain separate manifest fields.
'''),
    code(r'''
models = {variant: build_model(CONFIG, variant, SEEDS[0]) for variant in VARIANTS}
updates = planned_updates(models, CONFIG, len(WINDOWS), MATCHING_REGIME)
architecture_rows = []
for variant, model in models.items():
    architecture_rows.append({
        "variant": variant,
        "trainable_parameters": parameter_count(model),
        "encoder_parameters": parameter_count(model.encoder),
        "compute_proxy_per_step": compute_proxy_per_step(model, CONFIG),
        "planned_updates": updates[variant],
        "planned_orbit_exposures": updates[variant] * CONFIG.batch_size,
    })
architecture_table = pd.DataFrame(architecture_rows).set_index("variant")
display(architecture_table)
'''),
    md(r'''## 4. Persist the immutable input to training

These gates are engineering feasibility thresholds, chosen without clinical outcomes. Passing them is
not evidence that a representation is useful for force prediction or unseen participants.
'''),
    code(r'''
HEALTH_GATES = {
    "minimum_feature_variance": 1e-5,
    "minimum_effective_rank": 1.05,
    "maximum_mean_pairwise_cosine": 0.995,
    "minimum_odd_to_even_energy_ratio": 1e-5,
    "maximum_odd_to_even_energy_ratio": 1e5,
    "equivariance_float32_atol": 5e-5,
    "unconstrained_control_minimum_residual": 1e-6,
}
contract = {
    "notebook": "nb_09c_gavd_matched_jepa_contract",
    "scope": MANIFEST["scope"],
    "run_id": RUN_ID,
    "matching_regime": MATCHING_REGIME,
    "seeds": SEEDS,
    "pose_dir": str(POSE_DIR) if POSE_DIR else None,
    "train_config": asdict(CONFIG),
    "cohort_manifest": MANIFEST,
    "architectures": architecture_table.reset_index().to_dict(orient="records"),
    "objective": {
        "jepa": "centered_sharpened_latent_cross_entropy",
        "anti_collapse": "even_and_odd_orbit_VICReg",
        "condition_labels_used": False,
    },
    "health_gates": HEALTH_GATES,
}
WINDOW_TABLE.to_csv(OUT_DIR / "window_manifest.csv", index=False)
write_json(OUT_DIR / "training_contract.json", contract)
print("Wrote", OUT_DIR / "training_contract.json")
'''),
]


cells_09d = [
    md(r'''# 09d. Train the three matched GAVD JEPAs

This notebook performs fresh, paired-seed pretraining for the standard, paired-unconstrained, and
reflection-equivariant variants. It consumes the frozen contract from `nb_09c`; it does not reuse the
historical one-view checkpoint audited by `nb_09a`.

The default CPU smoke run proves execution only. For a full run, execute `nb_09c` and this notebook with
the same `GAIT_PARITY_MODE`, `GAIT_PARITY_RUN_ID`, profile, seeds, and matching regime.
'''),
    code(COMMON_SETUP),
    md("## 1. Refuse cohort, configuration, or objective drift"),
    code(r'''
contract_path = OUT_DIR / "training_contract.json"
if not contract_path.exists():
    raise FileNotFoundError(f"Run nb_09c first: {contract_path}")
CONTRACT = json.loads(contract_path.read_text())
assert CONTRACT["cohort_manifest"]["cohort_sha256"] == MANIFEST["cohort_sha256"]
assert CONTRACT["cohort_manifest"]["window_sha256"] == MANIFEST["window_sha256"]
assert CONTRACT["train_config"] == asdict(CONFIG)
assert CONTRACT["matching_regime"] == MATCHING_REGIME
assert CONTRACT["seeds"] == SEEDS
print("Frozen contract accepted:", contract_path)
'''),
    md(r'''## 2. Build paired-seed models and freeze update allocations

In exposure mode, batch indices and mask RNG streams are identical across variants for each seed. In
compute mode, variants traverse the same deterministic stream for different prespecified numbers of
updates. Parameter counts are reported, not described as matched: exact weight tying necessarily changes
the number of independent parameters at fixed width.
'''),
    code(r'''
allocation_models = {variant: build_model(CONFIG, variant, SEEDS[0]) for variant in VARIANTS}
UPDATES = planned_updates(allocation_models, CONFIG, len(WINDOWS), MATCHING_REGIME)
display(pd.DataFrame({
    variant: {
        "parameters": parameter_count(model),
        "updates": UPDATES[variant],
        "orbit_exposures": UPDATES[variant] * CONFIG.batch_size,
        "compute_proxy_total": UPDATES[variant] * compute_proxy_per_step(model, CONFIG),
    } for variant, model in allocation_models.items()
}).T)
'''),
    md(r'''## 3. Train and checkpoint every variant

The EMA teacher receives no gradients. Every checkpoint carries the cohort hashes, complete objective,
hardware/profile information, exposure counts, compute proxy, and measured wall time. CUDA is used only
when `GAIT_PARITY_DEVICE=cuda`; the default remains CPU.
'''),
    code(r'''
DEVICE = device_from_environment()
print("training device:", DEVICE)
run_rows = []
for seed in SEEDS:
    for variant in VARIANTS:
        print(f"\n--- seed {seed} | {variant} | {UPDATES[variant]} updates ---")
        model = build_model(CONFIG, variant, seed)
        model, projector, history, wall_seconds = train_variant(
            model, WINDOWS, VALID_PATCH, CONFIG, DEVICE, UPDATES[variant], seed
        )
        stem = f"seed-{seed}_{variant}"
        history_path = OUT_DIR / f"{stem}_history.csv"
        checkpoint_path = OUT_DIR / f"{stem}.pt"
        history.to_csv(history_path, index=False)
        metadata = {
            "variant": variant,
            "seed": seed,
            "train_config": asdict(CONFIG),
            "matching_regime": MATCHING_REGIME,
            "cohort_sha256": MANIFEST["cohort_sha256"],
            "window_sha256": MANIFEST["window_sha256"],
            "optimizer_updates": UPDATES[variant],
            "orbit_exposures": UPDATES[variant] * CONFIG.batch_size,
            "branch_forward_exposures": UPDATES[variant] * CONFIG.batch_size * 8,
            "compute_proxy_per_step": compute_proxy_per_step(model, CONFIG),
            "trainable_parameters": parameter_count(model),
            "wall_clock_seconds": wall_seconds,
            "device": str(DEVICE),
            "amp_enabled": bool(CONFIG.amp and DEVICE.type == "cuda"),
            "scope": MANIFEST["scope"],
        }
        save_checkpoint(checkpoint_path, model, projector, metadata)
        row = {
            **metadata,
            "checkpoint": str(checkpoint_path),
            "history": str(history_path),
            "first_total_loss": float(history.total_loss.iloc[0]),
            "last_total_loss": float(history.total_loss.iloc[-1]),
            "first_jepa_loss": float(history.jepa_loss.iloc[0]),
            "last_jepa_loss": float(history.jepa_loss.iloc[-1]),
        }
        run_rows.append(row)
        print(f"loss {row['first_total_loss']:.4f} -> {row['last_total_loss']:.4f}; {wall_seconds:.1f}s")

training_manifest = {
    "notebook": "nb_09d_gavd_matched_jepa_training",
    "scope": MANIFEST["scope"],
    "contract": str(contract_path),
    "runs": run_rows,
}
write_json(OUT_DIR / "training_manifest.json", training_manifest)
display(pd.DataFrame(run_rows)[["seed", "variant", "optimizer_updates", "orbit_exposures", "trainable_parameters", "wall_clock_seconds", "first_total_loss", "last_total_loss"]])
print("Wrote", OUT_DIR / "training_manifest.json")
'''),
]


cells_09e = [
    md(r'''# 09e. Audit GAVD checkpoint health and reflection geometry

This notebook is intentionally separate from optimization. It reloads each saved checkpoint, measures
even/odd representation health, and evaluates the layerwise swap contract on both online and EMA
encoders. A passing local verdict means only that the implementation is ready for a non-clinical,
subject-disjoint pretraining study such as AMASS.
'''),
    code(COMMON_SETUP),
    md("## 1. Load the frozen contract and completed training manifest"),
    code(r'''
contract_path = OUT_DIR / "training_contract.json"
training_path = OUT_DIR / "training_manifest.json"
if not contract_path.exists() or not training_path.exists():
    raise FileNotFoundError("Run nb_09c and nb_09d with this exact run ID/regime first")
CONTRACT = json.loads(contract_path.read_text())
TRAINING = json.loads(training_path.read_text())
GATES = CONTRACT["health_gates"]
assert CONTRACT["cohort_manifest"]["cohort_sha256"] == MANIFEST["cohort_sha256"]
assert CONTRACT["cohort_manifest"]["window_sha256"] == MANIFEST["window_sha256"]
print("checkpoints:", len(TRAINING["runs"]))
'''),
    md(r'''## 2. Reload, audit health, and test commutation

The standard orbit shell commutes because the same ordinary encoder processes both members independently;
that fact is not called a paired equivariant architecture because it has no cross-branch interaction.
The paired-unconstrained control must produce a nonzero residual, demonstrating that the test can fail.
Only the reflection-equivariant encoder is required to combine cross-branch interaction with a passing
online and EMA layerwise contract.
'''),
    code(r'''
health_rows, geometry_rows = [], []
contract_size = min(8, len(WINDOWS))
contract_batch = WINDOWS[:contract_size]
contract_valid = VALID_PATCH[:contract_size]
contract_target_mask = sample_mask(
    contract_valid, CONFIG.mask_fraction, torch.Generator().manual_seed(909)
)
contract_keep_mask = ~contract_target_mask
AUDIT_DEVICE = device_from_environment()
precision_cases = [("float32", False)]
if AUDIT_DEVICE.type == "cuda" and CONFIG.amp:
    precision_cases.append(("float16_autocast", True))
for run in TRAINING["runs"]:
    model, projector, metadata = load_checkpoint(run["checkpoint"])
    even, odd = collect_parity_features(model.target_encoder, WINDOWS, batch_size=CONFIG.batch_size)
    even_metrics, odd_metrics = representation_metrics(even), representation_metrics(odd)
    ratio = odd_metrics["energy"] / max(even_metrics["energy"], 1e-12)
    health_pass = bool(
        even_metrics["feature_variance"] >= GATES["minimum_feature_variance"]
        and odd_metrics["feature_variance"] >= GATES["minimum_feature_variance"]
        and even_metrics["effective_rank"] >= GATES["minimum_effective_rank"]
        and odd_metrics["effective_rank"] >= GATES["minimum_effective_rank"]
        and even_metrics["mean_pairwise_cosine"] <= GATES["maximum_mean_pairwise_cosine"]
        and odd_metrics["mean_pairwise_cosine"] <= GATES["maximum_mean_pairwise_cosine"]
        and GATES["minimum_odd_to_even_energy_ratio"] <= ratio <= GATES["maximum_odd_to_even_energy_ratio"]
    )
    health_rows.append({
        "seed": metadata["seed"], "variant": metadata["variant"],
        **{f"even_{k}": v for k, v in even_metrics.items()},
        **{f"odd_{k}": v for k, v in odd_metrics.items()},
        "odd_to_even_energy_ratio": ratio, "health_pass": health_pass,
    })
    for encoder_name, encoder in [("online", model.encoder), ("ema_teacher", model.target_encoder)]:
        encoder = encoder.to(AUDIT_DEVICE)
        for mode_name, train_mode in [("eval", False), ("train", True)]:
            for mask_name, keep_mask in [("unmasked", None), ("masked", contract_keep_mask)]:
                for precision_name, use_amp in precision_cases:
                    report = commutation_report(
                        encoder, contract_batch, keep_mask=keep_mask, train_mode=train_mode,
                        device=AUDIT_DEVICE, mixed_precision=use_amp,
                    )
                    for row in report.to_dict(orient="records"):
                        geometry_rows.append({
                            "seed": metadata["seed"], "variant": metadata["variant"],
                            "encoder": encoder_name, "mode": mode_name, "mask": mask_name,
                            "precision": precision_name, **row,
                        })

health = pd.DataFrame(health_rows)
geometry = pd.DataFrame(geometry_rows)
display(health)
display(geometry.groupby(["seed", "variant", "encoder", "mode", "mask", "precision"]).max(numeric_only=True))
'''),
    md("## 3. Apply the predeclared local-feasibility gates"),
    code(r'''
maximums = geometry.groupby(["seed", "variant", "encoder", "mode", "mask", "precision"], as_index=False).max(numeric_only=True)
eq_geometry = maximums[maximums.variant == "reflection_equivariant"]
free_geometry = maximums[maximums.variant == "paired_unconstrained"]
geometry_pass = bool((eq_geometry.max_abs <= GATES["equivariance_float32_atol"]).all())
control_pass = bool((free_geometry.max_abs >= GATES["unconstrained_control_minimum_residual"]).all())
health_pass = bool(health.health_pass.all())
checkpoints_complete = len(health) == len(SEEDS) * len(VARIANTS)
verdict = {
    "checkpoints_complete": checkpoints_complete,
    "all_representation_health_gates_pass": health_pass,
    "equivariant_online_and_teacher_geometry_pass": geometry_pass,
    "unconstrained_control_demonstrates_test_power": control_pass,
    "local_feasibility_pass": bool(checkpoints_complete and health_pass and geometry_pass and control_pass),
    "claim_ceiling": MANIFEST["scope"],
}
print(json.dumps(verdict, indent=2))
'''),
    md(r'''## 4. Persist audit tables and diagnostic figure

No model is ranked by downstream accuracy here. GAVD can tell us whether the loop is technically viable;
it cannot decide whether encoder-wide parity beats output repair on held-out clinical force.
'''),
    code(r'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

health.to_csv(OUT_DIR / "checkpoint_health.csv", index=False)
geometry.to_csv(OUT_DIR / "checkpoint_geometry.csv", index=False)
figure, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for variant, frame in health.groupby("variant"):
    axes[0].scatter([variant] * len(frame), frame.odd_feature_variance, label=variant)
    axes[1].scatter([variant] * len(frame), frame.odd_effective_rank)
for (variant, encoder), frame in maximums.groupby(["variant", "encoder"]):
    axes[2].scatter([f"{variant}\n{encoder}"] * len(frame), frame.max_abs)
axes[0].axhline(GATES["minimum_feature_variance"], color="black", ls="--", lw=1)
axes[0].set(title="Odd-channel variance", ylabel="mean feature variance")
axes[1].axhline(GATES["minimum_effective_rank"], color="black", ls="--", lw=1)
axes[1].set(title="Odd-channel effective rank", ylabel="effective rank")
axes[2].set_yscale("symlog", linthresh=1e-10)
axes[2].axhline(GATES["equivariance_float32_atol"], color="black", ls="--", lw=1)
axes[2].set(title="Layerwise swap residual", ylabel="maximum absolute residual")
for axis in axes:
    axis.tick_params(axis="x", rotation=25)
    axis.grid(alpha=0.2)
figure.tight_layout()
figure_path = OUT_DIR / "checkpoint_audit.png"
figure.savefig(figure_path, dpi=160, bbox_inches="tight")
from IPython.display import Image, display
display(Image(filename=str(figure_path)))
plt.close(figure)

audit = {
    "notebook": "nb_09e_gavd_matched_jepa_audit",
    "verdict": verdict,
    "health": health.to_dict(orient="records"),
    "geometry": geometry.to_dict(orient="records"),
    "figure": str(figure_path),
}
write_json(OUT_DIR / "checkpoint_audit.json", audit)
print("Wrote", OUT_DIR / "checkpoint_audit.json")
'''),
]


outputs = {
    ROOT / "nb_09c_gavd_matched_jepa_contract.ipynb": cells_09c,
    ROOT / "nb_09d_gavd_matched_jepa_training.ipynb": cells_09d,
    ROOT / "nb_09e_gavd_matched_jepa_audit.ipynb": cells_09e,
}
for path, cells in outputs.items():
    path.write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
    print(f"Wrote {path} ({len(cells)} cells)")
