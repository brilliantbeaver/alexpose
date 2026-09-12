# Implemented comparison, verification and the next scientific decision

The new source-held comparison ran to completion on the existing cached arrays. Its primary increment is **+0.001993744 R²**, with a 95% source-bootstrap interval **[−0.006672679, +0.012608431]** and **68.35%** positive draws. Under the prospectively frozen rule, this is `no_supported_temporal_lead`. It is a complete development measurement, not a positive student-distillation result. The original direct-v3 result remains a separate completed STOP.

## What changed and why

The laterality study tests access to a signed movement observable after skeleton-JEPA training. Future-innovation Experiment 0 tests what skeletons add after an RGB reference. A skeleton-only student cannot use the RGB reference. The new implementation therefore replaces RGB with two student-accessible references: observation support/timing, and support/timing plus the current normalized posture. It retains the existing teacher target and fitting machinery so that this change of question is explicit.

| Component | Implementation and evidence |
|---|---|
| New finite-family measurement | `src/gavd6_sjepa/research_directions/iclr_bridge/cached_panel.py`, version `student-accessibility-v1`; 331/397-dimensional references and unchanged 924-dimensional history summaries. |
| Safe fitting and controls | Reuses direct-v3 float64 joint ridge, input-support masks, separate target scaling, source partitions, raw controls, exact baseline option and paired score/bootstrap calculations. Historical FI code was not changed by this bridge. |
| Frozen design | [Prospective protocol](02_cached_panel_protocol.md), copied into [frozen-protocol.md](../../../outputs/iclr-bridge-cached-20260911/config/frozen-protocol.md) before fitting. SHA-256 `f91181e1b3a13a1d7f121052f9e650fa5a36bf67ecbed596e41f39780dbc70e3`. |
| Symmetry and order calibration | `symmetry_calibration.py`: complete channel reflection, physical-interval reversal, four parity classes, zero-feature control, paired pooled-teacher arithmetic and a source-held constructed continuation fixture. |
| Additional numerical verification | `verification_supplement.py` adds metadata, exact inventory, displayed candidate and outer training-diagnostic checks after adversarial review. It changes no fitting decision or sealed artifact. |
| Tutorial implementation | Canonical `scripts/research_directions/iclr_bridge/build_notebooks.py` generates notebooks 19–22 in `notebooks/iclr_bridge/`. All 23 code cells executed successfully. Notebook 21 defaults to CPU reconstruction refits without artifact writes; its explicit no-fit mode checks file integrity only. No notebook starts a new real comparison. |
| Vector graphics | `build_figures.py` generates five explanatory SVG/PDF pairs and a sixth results pair directly from the saved report. Trajectory illustrations are labeled synthetic. |

The primary fitted comparison is not an estimator of conditional mutual information. Real and no-skeleton models can choose different regularization, and their history blocks differ in coordinates and confidence. In particular, the reference includes **raw** confidence means, whereas real history also includes means conditioned on valid observations. These are not generally equal. Some confidence columns also duplicate reference columns: two identical routes penalized by λ_x and λ_s have effective penalty (1/λ_x + 1/λ_s)⁻¹ for their summed coefficient. Current reference inputs therefore do not exhaust observation-quality or regularization effects. A future study claiming a coordinate-motion effect should include confidence, validity and transition support once and identically across arms, vary only coordinate/displacement features, and control the reference route more tightly. The current evidence does not quantify the contribution of these routes to the observed estimate.

## Measured results

Every number below is linked to [panel-report.json](../../../outputs/iclr-bridge-cached-20260911/reports/panel-report.json). Models are deterministic; seed 0 is an identity, not one of three repeated optimization runs. All 256 target features survived the intersection of training-derived variance masks.

| Reference panel | Shared reference | Real history | No skeleton | Shuffle | Mismatch |
|---|---:|---:|---:|---:|---:|
| Posture + support, primary panel | 0.00589100 | 0.01326171 | 0.01126797 | 0.00769218 | 0.00589100 |
| Support only, secondary panel | 0.00822909 | 0.01386483 | 0.00729554 | 0.01243354 | 0.00822909 |

These are predictive R² values, using the outer-training target mean as the reference. They must not be ranked against the laterality study's differently defined R². The earlier RGB model's 0.33421517 also answers a different conditioning question; the lower student-accessible scores suggest limited usefulness under this finite feature family, not a proven absence of all learnable skeleton information.

| Paired contrast | Estimate | 95% source interval | Positive draws |
|---|---:|---|---:|
| **Posture: real − no skeleton, primary** | **+0.001993744** | **[−0.006672679, +0.012608431]** | **68.35%** |
| Posture: real − reference | +0.007370712 | [−0.002890440, +0.019967023] | 92.60% |
| Posture: real − shuffle | +0.005569524 | [−0.001754300, +0.014774816] | 92.50% |
| Support: real − no skeleton, secondary | +0.006569290 | [−0.000071482, +0.014709426] | 97.30% |
| Support: real − shuffle, secondary | +0.001431291 | [−0.003920839, +0.008541424] | 65.65% |

![Paired uncertainty in the new cached comparison.](figures/06_cached_panel_results.svg)

The positive fraction and two-sided interval answer different questions. The support-only result cannot replace the named posture primary after inspection. Its shuffle comparison is also uncertain. None of these intervals includes repeated fitting, hyperparameter selection or adaptation to the inspected sources.

Seventeen of 40 arm/fold selections chose exact baseline. All 1,480 pooled candidates were numerically valid. Forty selected arm models and ten shared reference models generated 102,400 long-form prediction rows; 16,000 saved bootstrap rows cover two panels, four arms and 2,000 draws. Mismatch selected baseline in every fold. In the posture panel, the real model's reference-block penalty differed from the shared baseline in two folds. Its real-minus-reference gain therefore includes a change in reference regularization and cannot be called a pure motion contribution.

## What was actually run

The real fitting/report stage took **13.17 seconds** on the local CPU. The machine used Python 3.12.10 on macOS arm64, NumPy 2.5.2 and pandas 3.0.5 as recorded in the run contract. No HAIC job, teacher encoding, raw-video loading or GPU allocation was performed for this comparison. The expensive features were reused from the parent cache through the existing verified lineage; the child did not regenerate teacher evidence.

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_iclr_bridge*.py' -v
.venv/bin/python scripts/research_directions/iclr_bridge/run_cached_panel.py freeze --source-run outputs/future-innovation-direct-v3-dev-20260911 --output-root outputs/iclr-bridge-cached-20260911 --protocol-document docs/studies/iclr/02_cached_panel_protocol.md
.venv/bin/python scripts/research_directions/iclr_bridge/run_cached_panel.py run --output-root outputs/iclr-bridge-cached-20260911
.venv/bin/python scripts/research_directions/iclr_bridge/run_cached_panel.py verify --output-root outputs/iclr-bridge-cached-20260911
```

`freeze` was run once, before revised real-data scores were inspected. `run` was exercised again after completion and verified reuse instead of fitting. The separate `verify` command also passed. Their measured durations were 4.38 and 4.61 seconds; all 129 child files retained their bytes and modification times. [The reuse record](../../../work/artifacts/iclr-bridge-2026-09-11/future/completed-stage-verification.json) preserves commands, return codes and times. Original fit-time code copies are retained in [fitting-code](../../../work/artifacts/iclr-bridge-2026-09-11/future/fitting-code).

Run the strengthened read-only check from Python:

```python
from gavd6_sjepa.research_directions.iclr_bridge.verification_supplement import verify_panel_supplement
result = verify_panel_supplement("outputs/iclr-bridge-cached-20260911")
assert result["status"] == "passed"
```

The [supplemental verification record](../../../work/artifacts/iclr-bridge-2026-09-11/future/verification-supplement.json) is outside the sealed run. It checks the exact 126-artifact sealed schema, all 1,480 displayed candidate records and all 40 outer training-diagnostic records, in addition to numerical reconstruction. The original verifier refits selected models and every saved inner/outer input/target preprocessor, reconstructs deterministic donor assignments, and recomputes predictions, metrics and bootstrap draws. It validates the finite search records and selection but does not independently refit every rejected candidate. Elapsed time can be checked for plausibility, not recovered arithmetically from a model.

## Calibration and regression evidence

| Check | Observed result |
|---|---|
| Reflection/reversal algebra, irregular time intervals and invalid-coordinate sentinels | Nine focused tests pass; algebraic/parity errors satisfy 10⁻¹² tolerance. |
| Zero representation | Exact transformation consistency with zero variance and zero predictive R² on the balanced nonzero fixture. |
| Constructed temporal ambiguity | 48 generated sources, 96 paired clips, 36 training/12 test sources. Current posture, support and order-even summaries each score 0; ordered signed velocity scores 0.999999999978224. This predicts the fixture's defined continuation, not real human motion. |
| Actual 32-frame representation and designated shuffle | Source-held planted-history test uses the complete 36-joint-candidate plus baseline search; real error is below half the reference and shuffle errors. Includes checkpoint reload and exact fallback. |
| Input support and source isolation | Stable 331/397 schemas, missing endpoint handling, unchanged fitted statistics under held-out changes, split and donor reconstruction. |
| Tampered numerical artifacts | Changed selected coefficients or grid identities rejected even after hashes are updated. |
| Tampered displayed evidence | Six supplemental tests demonstrate the original metadata omission and reject rehashed wrong counts/version, omitted seal entries, false candidate gains/selection labels and false training diagnostics. |
| Existing future-innovation suite | 125 tests run, 123 pass, two skip because the external V-JEPA checkout is not supplied (`FI_TEST_VJEPA_ROOT`). These cover training, scoring, causality, controls, resumption, notebooks, direct protocols and Slurm conventions. |
| New notebooks | Four notebooks, 23 code cells executed; zero errors. Source and executed code/Markdown match. |

The complete current bridge suite comprises 22 tests: 20 core/symmetry/supplemental tests plus two subsequently added no-fit inspection tests. All pass. Logs are [bridge-final-tests.log](../../../work/artifacts/iclr-bridge-2026-09-11/bridge-final-tests.log), [inspection-tests.log](../../../work/artifacts/iclr-bridge-2026-09-11/inspection-tests.log), and [future-regression-tests.log](../../../work/artifacts/iclr-bridge-2026-09-11/future-regression-tests.log). Synthetic sources and random draws add no independent real videos.

Historical notebook generation and execution, before the source files moved
into `notebooks/` on 12 September 2026:

```bash
.venv/bin/python scripts/research_directions/iclr_bridge/build_notebooks.py
.venv/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 --output-dir work/artifacts/iclr-bridge-2026-09-11/executed_notebooks 19_evidence_and_observability.ipynb 20_symmetry_and_temporal_information.ipynb 21_student_accessible_future_features.ipynb 22_selective_future_distillation.ipynb
.venv/bin/python scripts/research_directions/iclr_bridge/build_figures.py
.venv/bin/python scripts/research_directions/iclr_bridge/check_deliverables.py --output work/artifacts/iclr-bridge-2026-09-11/deliverable-audit.json
```

Notebook 21 was re-executed after integrating the supplemental verifier. Figure PDFs and manuscript pages were rendered and visually checked. The [delivery audit](../../../work/artifacts/iclr-bridge-2026-09-11/deliverable-audit.json) compares source/executed cells, validates notebook schemas and local links, and verifies the external source snapshot.

The current [notebook guide](../../../notebooks/iclr_bridge/README.md)
gives commands using the new source paths and a fresh output directory. During
the relocation check, all four notebooks were executed again from
`notebooks/iclr_bridge/`, including notebook 21's default numerical
reconstruction. The [new executed copies](../../../work/artifacts/notebook-organization-20260912/)
are separate from the historical batch. The audit now requires exactly one of
each notebook 19–22 and accepts `--executed-dir` to check the matching batch;
an empty directory cannot silently pass. This organization changes no experiment
definition or measured result.

## Building the documents

The final paper contains eight main-text pages, one reference page and one evidence/ethics page. The extended abstract is one page. Their canonical text is Markdown; `build_manuscripts.py` generates the corresponding TeX and compiles with the unmodified official ICLR 2027 style. Only the status label is changed from “under review” to “research draft — not submitted.” No margins or font sizes were reduced to force the page count.

```bash
MPLCONFIGDIR=/tmp/iclr-mpl .venv/bin/python scripts/research_directions/iclr_bridge/build_figures.py
.venv/bin/python scripts/research_directions/iclr_bridge/build_manuscripts.py --compile
.venv/bin/python scripts/research_directions/iclr_bridge/build_evidence.py
```

These commands were exercised. Rendering uses ReportLab 5.0.1, PyMuPDF 1.28.2, Pandoc 3.10.2 and Tectonic 0.17.0. NumPy/pandas and the project environment versions remain recorded in the frozen experiment contract. The first typesetting attempts failed on missing `eso-pic` and font resources; genuine CTAN packages and a workspace-local Tectonic cache resolved them without modifying the conference style. The final compile is offline (`--only-cached`). [Dependency provenance](../../../work/artifacts/iclr-bridge-2026-09-11/tex-deps/README.md) and [the build manifest](../../../work/artifacts/iclr-bridge-2026-09-11/manuscripts/build-manifest.json) retain archive hashes, actual compiler commands, page counts and rendered-page inventories. The final logs contain no undefined citations, missing glyphs, font-load errors or overfull boxes. Underfull layout notices were visually checked.

The [document evidence folder](evidence/README.md) contains byte-identical JSON copies of the measured report, supplemental verification, retained laterality arithmetic, synthetic calibration and post-hoc confidence diagnostic, together with source hashes. It is a portable reading aid, not a replacement for the full saved model and prediction artifacts.

## Preservation, recovery and limits of portability

The original laterality PDF, matching Markdown/TeX and 19 notebooks retain their SHA-256 digests, sizes and modification times: 22 files checked against the pre-work snapshot. The 57-source laterality evidence manifest is retained separately. All 45 relevant future/S-JEPA notebooks were inventoried; this includes historical executed copies and source tutorials, not 45 independent training runs. Laterality raw prediction/checkpoint archives remain unavailable locally, so this work recomputed retained aggregate arithmetic only. It did not reconstruct historical laterality inference or bootstrap draws.

Both existing future-innovation source trees are compared with frozen digest/size/mtime snapshots whenever the child loads. Interrupted folds have atomic completion receipts and can be fitted again through `run`; completed-stage reuse was exercised. The new wrapper rejects overlapping parent/child roots and reuses stage ownership locks. The broader direct-v3 suite covers interrupted writes and duplicate writers. A process killed during this new real pilot was not separately simulated; that should not be inferred from successful completed-stage reuse.

The frozen child records absolute local parent locations and binds to the fitting implementation. Copying this child to another machine does not automatically relocate its contract. On HAIC, first prepare and verify the corresponding parent/cache lineage using the existing future-innovation guide, then freeze a **new** bridge run from that machine's source root. Use the same CLI `run` command in a CPU Slurm job, without GPU requests, from the repository environment. This path is documented for reproduction, not claimed as exercised here. Changed targets, physical-time windows, teacher layers/projection or reflected inputs require new cache/audit evidence.

## The next experiment that could justify the method claim

1. **Strengthen the measurement boundary.** Specify a confidence/validity-only history control, a tightly matched current-state reference and a direction-sensitive endpoint. Freeze how reference coefficients are controlled. Use the existing negative results to set scope, not to select the best-looking secondary analysis.
2. **Measure a future state with an interpretable clock.** Define a physical-time horizon and valid adjacent-frame motion target. First establish that history helps direct kinematics beyond current posture. If it does not, investigate the horizon, measurement and pose quality before adding teacher supervision.
3. **Generate the teacher evidence required by that question.** Encode a bounded future video block and actual mirrored-video pairs with matching model/projection and temporal audits. The present contextual cache cannot supply these targets by arithmetic alone. Controlled rendered motion can vary view and appearance independently before spending on a large real study.
4. **Implement one training-only target selector.** Compare a finite list of ranks including zero, use source-held accessibility beyond current state/support, and never rank directions on outer scores. Cross-fitted residuals must share raw units before combining. Distillation loss on a smaller target is not a common utility metric.
5. **Train matched students and test a shared endpoint.** Compare ordinary S-JEPA, full-target transfer, selected-target transfer, rank-matched random/variance targets and no transfer, with matched initialization and direct kinematics. Keep compute and source exposure comparable. This is the missing empirical method experiment.
6. **Confirm on independently selected sources or participants.** Freeze one successful development method and endpoint, plan the number of independent units for that endpoint's meaningful effect, and evaluate once. Additional excerpts or seed labels do not increase participant diversity.

The most plausible contribution is a tested rule for transferring extra student-accessible temporal information while permitting no transfer. Its distinction from existing future privileged supervision and spectral distillation must be demonstrated experimentally. Without actions or interventions, the resulting model remains an observational motion predictor; an agentic planning or causal world-model claim would require a separate study.
