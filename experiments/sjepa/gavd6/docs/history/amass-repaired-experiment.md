# Repaired JEPA experiment status

> **Historical record.** This report predates the active
> [swap probe](../studies/latent-laterality/swap-probe.md).
> Its “no downstream probe” limitation describes the repository at the time of
> the repair experiment and is not a statement about the current workflow.

## Status

Architecture, metric, checkpoint-selection, evaluation, capacity, and smoke checks pass. The real AMASS seed-7 pilot has not run: this workstation has the 8,854-row manifest but no `core11/` tensor tree, and non-interactive access to `tedmui@haic.stanford.edu` is rejected with `Permission denied (keyboard-interactive)`.

No real validation or test result is reported below. The synthetic smoke is pipeline evidence only.

## Matched full-profile allocation

All arms use a 64-dimensional embedding and identical depth. Capacity is matched through feed-forward width so JEPA entropy and KL use the same output dimension.

| Variant | FF width | Trainable parameters |
|---|---:|---:|
| `paired_shared_no_cross` | 903 | 821,866 |
| `reflection_equivariant` | 773 | 821,860 |
| `paired_unconstrained` | 256 | 822,214 |

Maximum spread: 0.04%. The frozen EMA encoder is excluded; the online encoder, predictor, cross gates, and VICReg projector are included.

## Final synthetic smoke, seed 7

Three epochs, 16/8/8 synthetic train/validation/test windows. Best epoch is selected by eligible validation KL and reloaded before the one-time synthetic test evaluation.

| Variant | Params | Runtime (s) | Best epoch | Val CE | Val entropy | Val KL | Test KL | Val even/odd variance | Encoder / predictor / masked commutation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `paired_shared_no_cross` | 62,085 | 0.61 | 3 | 9.824 | 0.286 | 9.537 | 9.794 | 0.0122 / 0.0017 | 0 / 0 / 0 |
| `reflection_equivariant` | 62,082 | 0.84 | 2 | 13.532 | 0.233 | 13.299 | 12.606 | 0.0095 / 0.0009 | 0 / 0 / 0 |
| `paired_unconstrained` | 62,139 | 0.86 | 2 | 15.552 | 0.265 | 15.287 | 14.436 | 0.0085 / 0.0053 | 3.53 / 3.33 / 3.63 |

The original `outputs/repaired-smoke-seed7-final` artifact is not included in
this checkout. Its path is retained below only as the output location that the
historical smoke command would create.

## Commands

Synthetic smoke:

```bash
AMASS_RUN_TRAINING=1 AMASS_PROFILE=smoke AMASS_SYNTHETIC_SMOKE=1 \
AMASS_DEVICE=cpu AMASS_NUM_WORKERS=0 AMASS_SEEDS=7 \
AMASS_OUTPUT_DIR=outputs/repaired-smoke-seed7-final \
.venv/bin/python -m gavd6_sjepa.train_amass
```

Full seed-7 pilot on HAIC after interactive authentication:

```bash
export AMASS_RUN_ROOT=/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/data/amass/outputs
export AMASS_OUTPUT_DIR="$AMASS_RUN_ROOT/repaired-jepa-seed7"
AMASS_RUN_TRAINING=1 AMASS_PROFILE=full AMASS_DEVICE=cuda \
AMASS_SEEDS=7 AMASS_NUM_WORKERS=4 AMASS_EVALUATE_TEST=0 \
uv run --no-sync train-amass-core11
```

Only after inspecting the seed-7 validation curves, run the frozen three-seed protocol:

```bash
export AMASS_OUTPUT_DIR="$AMASS_RUN_ROOT/repaired-jepa-three-seed"
AMASS_RUN_TRAINING=1 AMASS_PROFILE=full AMASS_DEVICE=cuda \
AMASS_SEEDS=7,19,31 AMASS_NUM_WORKERS=4 AMASS_EVALUATE_TEST=1 \
uv run --no-sync train-amass-core11
```

Run these commands from a workspace containing the repaired source. The pilot never opens the real test split. The frozen three-seed run writes every seed and variant to `summary.csv` and evaluates each selected checkpoint on test once.

## Verification

- All 26 repository tests pass.
- CE = teacher entropy + KL is tested numerically; smoke-history residuals are at floating-point scale.
- Symmetric online encoder, EMA encoder, predictor, and masked predictions commute; the unconstrained arm does not.
- Validation evaluation is deterministic and does not mutate model, projector, target center, or EMA state.
- Full manifest dry run finds 79,535 train, 5,936 validation, and 8,220 test windows.

## Scientific limitations

- The synthetic smoke cannot rank architectures.
- Real seed 7 and seeds 19/31 remain blocked on data/HAIC access; the real test split has not been touched.
- Parameter and exposure matching do not guarantee matched FLOPs: tied modules reuse parameters across both branches, and the FF widths differ by design. Runtime is reported for this reason.
- No inexpensive, integrated downstream Core11 probe exists in this workflow. Conclusions must stay limited to masked prediction, geometry, and representation health until a common frozen-encoder probe is added.
- The default uses one deterministic evaluation draw per window; `AMASS_EVALUATION_DRAWS` can increase this before the real protocol is frozen.
