# Reproducing the latest downstream probes

The latest completed training checkpoint among `gavd4-vicreg`, `gavd5`, and
`gavd5-tm` is the augmented five-stage checkpoint under `gavd5/work`. Its file
SHA-256 is:

```text
6e67fc5c4a025375de24b46230ea0ec420a0ba360462d4b440c19e139577fbf0
```

Do not report a pre-existing classifier CSV without rerunning notebook 06.
The checkpoint, pose archives, notebook outputs, and CSVs have existed with
different timestamps in this workspace. A fresh run is the reproducible source
of truth.

## One-command probe rerun

From `experiments/sjepa/gavd5`:

```sh
GAVD_MODE=real \
GAVD_CACHE_DIR="$PWD/cache" \
GAVD_ARTIFACT_DIR="$PWD/work/artifacts" \
SJEPA_INCLUDE_AUGMENTED_NORMAL=1 \
MPLCONFIGDIR="$PWD/cache/matplotlib" \
.venv/bin/jupyter nbconvert \
  --execute \
  --to notebook \
  --output /tmp/gavd5-downstream-probe-reproduced.ipynb \
  --ExecutePreprocessor.timeout=1200 \
  06_capstone_health_condition_classifiers.ipynb
```

This freezes the saved EMA target encoder, rebuilds 384-dimensional pooled
embeddings, fits the seeded Random Forest readouts, and rewrites the probe CSVs
and contract under `work/artifacts/real`.

## Verified results

Two fresh processes produced the same canonical embedding-corpus SHA-256,
`0eff77b03ce19f38ec1209603b13301e09aecddc41f33b2c386b42d649c9a1e6`, and
the same metrics:

| Five-class probe | Accuracy | Macro-F1 |
|---|---:|---:|
| All 96 examples, random sequence split | 0.724 | 0.750 |
| Matched comparison, 47 training and 21 test examples | 0.762 | 0.765 |
| Videos kept separate, average across two splits | 0.564 | 0.495 |
| Videos kept separate, all held-out predictions combined | 0.566 | 0.496 |

The first two comparisons can share source videos between classifier training
and testing. The final two keep source videos separate during classification,
but the feature-learning stage had already used all 159 examples.
These are descriptive frozen-representation probes, not estimates of unseen
patient, unseen-video, or clinical performance.

## Rebuild the scorecard

```sh
MPLCONFIGDIR=cache/matplotlib \
.venv/bin/python docs/make_downstream_probe_figure.py
```

The generator uses the same DejaVu Sans typography, navy/blue/teal palette,
grid treatment, dimensions, and PDF/SVG/PNG export settings as
`docs/make_figures.py`. It rejects mixed-run artifacts, mismatched embedding
hashes, incomplete curricula, and non-real runs.
