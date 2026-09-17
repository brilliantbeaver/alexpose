# Temporal gait: starting evidence ledger

Reviewed: 2026-09-15. This is the implementation session's read-only historical audit plus a new ledger; it reports no new model training, data extraction, HAIC access, or forecast evaluation. Paths in code spans are relative to the `sjepa` repository root. The approved [plan](../../../../../gavd5-drift/notes/47-improvement-plan.md), [implementation audit](../../../../../gavd5-drift/notes/47-improvement-audit.md), [literature review](../../../../../gavd5-drift/notes/47-improvement-literature.md), and [execution specification](../../../../../gavd5-drift/notes/47-improvement-experiments.md) were read before implementation decisions.

The latest manuscript currently present is `gavd5-drift/neurips-laterality/docs/fmts_revisions/paper_v9.md`, **Evaluating Predictive Gait Representations Beyond Latent-Feature Matching**. Versions 1–9 are present; no later numbered Markdown manuscript was found in that specific manuscript directory. The working tree already contains extensive unrelated edits and untracked artifacts, including the v9 manuscript. This ledger does not modify or take ownership of those files.

## Evidence levels

| Status | Meaning |
| --- | --- |
| `retained_real_aggregate` | A saved real-data table or historical report is available; inference has not been replayed in this session. |
| `aggregate_arithmetic_verified` | Arithmetic was independently recalculated from the retained per-seed arrays in this session. This is not a checkpoint or bootstrap replay. |
| `code_observed` | The stated behavior was traced in the inspected implementation; empirical magnitude is not inferred from code alone. |
| `synthetic_only` | The evidence is a generated software example and cannot establish GAVD efficacy. |
| `historical_scientific_stop` | A recorded scientific advancement rule failed; later confirmation remains stopped under that study's protocol. |
| `proposed` | A new requirement or hypothesis; no empirical result exists yet. |

The exact local `gavd5-drift/neurips-laterality/artifacts/` directory is absent. This is an observation about the inspected checkout, not a claim that the user has no artifacts elsewhere. No alternative artifact directory or HAIC source location was searched. The numerical evidence's own `evidence_status` states that raw predictions and checkpoints are not included and that intervals cannot be regenerated from seed aggregates.

## Claim-to-evidence ledger

| Claim / finding | Source and location | Status | Independent unit / population | Model or pipeline version | Reproducibility here | Limitation / implementation consequence |
| --- | --- | --- | --- | --- | --- | --- |
| Latest motion cohort: 625 accepted clips, 93 source videos, five folds and seeds 42–46 | `gavd5-drift/neurips-laterality/docs/fmts_revisions/review/numerical_evidence.json`, `cohort`; Notebook 18 cell index 4 | `retained_real_aggregate` | Source videos, not verified people | Motion/regions 15–18; CUDA BF16 training, FP32 evaluation | Stored values inspected; raw cohort/inference not reloaded | Full-video availability cannot be inferred from these counts. |
| Initialization expanded readout R² = 0.22254357355538518; mean-only R² = 0.070827155920603 | Same JSON, `readout_rows` | `aggregate_arithmetic_verified` | Source-balanced held-out predictions pooled across folds within each seed, then seed mean | Initial online encoder, width 96; 2,890/960 feature readouts | Both means independently match the five stored seed values | Initialization includes architecture, supplied joint identities, preprocessing, pretrained pose detector and supervised ridge fitting. |
| Final teacher expanded readouts reach R² 0.10077719741431432–0.11420943905203321 and lose to their matched initial encoder in every seed | Same JSON, five `pretrained_teacher__mean_motion` rows | `aggregate_arithmetic_verified` | Same 93 source videos, five repeated training initializations | Five motion/region mask conditions; 1,200 updates/encoder | All 25 paired seed differences recomputed; each is negative | This establishes the specified readout deficit, not general information loss or collapse. |
| Five teacher-minus-initial intervals are below zero; mask-versus-uniform intervals cross zero | Same JSON, `trained_minus_initial` and `mask_intervals`; paper v9 Table 1 | `retained_real_aggregate` | Joint source resampling across paired methods/seeds | 2,000 bootstrap resamples conditional on fitted models | Endpoints inspected, not regenerated | Excludes retraining, split changes and development-selection uncertainty. Do not convert null mask evidence into equivalence. |
| Same-clip diagnostic: 375/375 trained aggregate rows favorable; 33/75 unique initial rows favorable | Same JSON, `correspondence`; `laterality_extensions/comparative_evaluation.py:520–754` | `retained_real_aggregate` + `code_observed` | Reused fold × seed × arm × evaluation-mask summaries | Each model is compared within its own teacher space | Retained count and source-selection code inspected | Not 100% clip retrieval accuracy, 375 independent trials, or real forecasting. |
| Direct-pose summary baseline R² = 0.03454120636613722 in latest grid | Numerical evidence, `direct_pose` row; `laterality_extensions/masked_learning.py:421–440` | `aggregate_arithmetic_verified` + `code_observed` | Same source-balanced cohort | 264 mean/SD/displacement/validity features | Five duplicated seed entries are identical | Seed-independent control duplication is not replication; this is not an information ceiling. |
| Input-preparation diagnostic: R² ≈0.218, correlation ≈0.652, MAE ≈0.041, sign agreement ≈70.4%; finite overlap 623 clips/92 sources | `RESEARCH_DIRECTIONS.md`, “A new, inexpensive diagnostic”; paper v9 secondary checks | `retained_real_aggregate` | Finite common overlap, fewer sources than full cohort | Same speed formula applied to prepared 64-position inputs | Historical aggregate inspected only | Several preparation factors changed; it cannot identify causation or an information-theoretic bound. |
| Target uses original positive time differences and bilateral observed support | `laterality/geometry.py:217–269`; `laterality/data.py:475–483` | `code_observed` | Five bilateral pairs, at least eight common transitions each | Historical signed normalized median-speed contrast | Formula and validity path inspected | Pose-derived and reversal invariant; not an independent clinical label. |
| Uniform time scaling approximately cancels in this normalized target | `laterality/geometry.py:255–264` | `code_observed` + algebraic implication | Same trajectory and support | `(median(vL)-median(vR))/(median(vL)+median(vR)+epsilon)` | Algebra follows directly; epsilon prevents exact invariance | Missing absolute duration alone cannot explain the historical deficit. Nonuniform timing/support changes remain candidates. |
| Encoder interpolation/resize uses sample indices and differs from target support/normalization | `laterality/geometry.py:83–107,153–214` | `code_observed` | Whole selected clip →64 positions | Historical input preparation | Direct implementation inspected | Do not reuse this whole-clip path inside a past-only forecasting claim. |
| Historical SSL cohort excludes noncomputable laterality targets | `laterality/data.py:527–538` | `code_observed` | Locked 642-archive inventory before QC | Protocol v2.1 | Exact exclusion inspected | New SSL eligibility and endpoint support must be distinct. |
| Historical augmentation rotates x–z and then translates x/y | `laterality/model.py:274–288` | `code_observed` | Geometric views of pseudo-3D detector coordinates | Historical local S-JEPA | Exact operation inspected | It is not a 2D image-plane rotation; preserve only in the exact historical arm. |
| Actual motion-mask audit hides about 17% of all valid tokens; regions about 9.9% | `tutorials/motion_results_20260908.py`, analysis for notebooks 15–16; `laterality_extensions/motion_structured_masks.py:194–205` | `retained_real_aggregate` + `code_observed` | Repeated source-balanced mask draws | Count from batch-minimum 12-gait-joint support, then all-33-joint candidates | Formula and retained audit inspected | Configured fraction 0.5 is not a 50% whole-skeleton mask. |
| Current loss is centered/sharpened feature CE plus pooled VICReg; EMA teacher receives no gradients | `laterality_extensions/motion_runtime.py:41–71`; `motion_structured_training.py:343–381` | `code_observed` | Mean targets/clip then clips; full-prefix pooled regularizer | Temperatures 0.06/0.10; multiplier 0.05; EMA 0.999 | Code path inspected | Token/within-source temporal usefulness is not forced by pooled noncollapse alone. |
| 125 encoders ×1,200 updates =150,000 arm-updates; batch 20 gives 3,000,000 total presentations | Notebook 18 cell index 6; `docs/physworld_evidence_recomputed.json` | `retained_real_aggregate` + arithmetic | Repeated presentations of 625 clips from 93 sources | Latest five-condition grid | Workload arithmetic verified | Presentations are not independent data and “300 epochs” means source-pass sampling, not full passes over every clip. |
| At fixed EMA 0.999, teacher initial coefficient after 1,200 updates ≈0.301; half-life ≈693 updates | EMA recurrence in `motion_runtime.py:74–80` | `code_observed` + algebraic implication | Optimizer steps | Fixed-EMA latest grid | Recurrence inspected | Teacher lag is a hypothesis; online encoders also underperform. |
| Largest ridge alpha 10,000 selected in 49/125 teacher and 96/125 online expanded-readout fits | Paper v9 Appendix A; `motion_results_20260908.py` readout analysis | `retained_real_aggregate` | Repeated fold/seed fitted readouts | Expanded grid | Retained counts inspected only | Wider alpha need not improve outer-test scores; report train/development curves. |
| Ridge preprocessing is fitted on source-separated inner-training partitions; SSL can have seen inner-validation examples unlabeled | `comparative_evaluation.py:89–162` | `code_observed` | Sources | Three source-separated ridge folds | Exact fitting path inspected | This selects a readout conditional on a fixed encoder, not a fully nested SSL recipe. |
| Source-video training/test overlap is explicitly checked | `laterality/splitting.py:48–143`; `motion_structured_training.py:86–152` | `code_observed` | Video IDs | Historical outer-fold training | Code assertions inspected | Videos can share unrecorded people or reuploads; new known connected groups must inherit one role. |
| Full constant-feature collapse is not supported by retained diagnostics | `motion_results_20260908.py:492–500` | `retained_real_aggregate` | Standardized summary features and raw predictor diagnostics | Latest grid | Retained values inspected | Effective rank after scaling can hide weak raw variation or lost useful directions. |
| Earlier Notebook 12 teacher results are about −0.023/−0.010, initialized +0.048, direct-pose +0.130 | `docs/TUTORIAL.md`, “Movement prediction in the completed Notebook 12 comparison”; `docs/figures/tutorial_comparative_masking_summary.json` | `retained_real_aggregate` | Same historical source cohort; a different comparison | Two-arm, 50-encoder grid with different readout configuration | Separate aggregate artifacts inspected | Never pool with Notebook 18 or claim the numerical shift isolates masking. |
| Earlier original native readout R²≈0.060; reflection predictive effect≈+0.004, interval crosses zero | `RESEARCH_DIRECTIONS.md`, “What the completed experiment supports” | `retained_real_aggregate` | Historical source-video folds | Original 00–06 symmetry protocol | Retained text inspected | This is not the later expanded-readout experiment. |
| Exact oddness can be constructed independently of learning | `laterality/evaluation.py:334–446`; `laterality/symmetry.py`; `symmetry_learning.py:111–146` | `code_observed` | Two-pass paired features | Constructed odd/zero-intercept lanes | Algebra/code inspected | Native behavior, enforced behavior and useful content need separate controls. |
| Real forecasting remains untested in the inspected suite | Notebook 14 cell index 23; `RESEARCH_DIRECTIONS.md`; paper v9 Discussion | `synthetic_only` for executed forecast demonstrations | Toy examples; separate historical feasibility census | Tutorials 10/14 | Explicit output labels inspected | A real feasibility count is not forecasting accuracy; observed-future decoding is privileged. |
| New primary metric uses 2D future detector coordinates, prefix projected chain length, 0.50 s horizon, common valid bilateral support | Approved plan §§5.2/8 and execution contract §§4/10 | `proposed` | Window→bout→video point estimate; connected-source bootstrap groups | New temporal-gait protocol | No new empirical score yet | Keep historical target/normalization unchanged; this is not motion-capture or a clinical threshold. |
| SG-JEPA's informative probability mechanism failed its seed-7 advancement rule | `gavd6/slurm/latent-laterality/README.md`; `gavd6/docs/studies/latent-laterality/results/validation.md` | `historical_scientific_stop` | 15 AMASS validation people; test sealed | AMASS paired v2, seed 7 | Retained report only; report notes missing linked artifacts | Do not submit its jobs 15/16 or use its result as positive mechanism evidence. |

## Exact aggregate verification performed in this implementation session

The saved evidence JSON has eight readout rows, each with five seed values. Recomputing each arithmetic mean produced a **maximum absolute discrepancy of 0** under the local JSON arithmetic check. The 25 teacher-versus-matched-initial seed comparisons were all negative. The directly recomputed mean paired differences are:

| Experiment / condition | Teacher mean R² | Mean paired teacher-minus-initial R² | Negative paired seeds |
| --- | ---: | ---: | ---: |
| Motion / uniform | 0.11372460890473453 | −0.10881896465065063 | 5/5 |
| Motion / MAMP | 0.11289028578638174 | −0.10965328776900343 | 5/5 |
| Motion / robust mixture | 0.11420943905203321 | −0.10833413450335197 | 5/5 |
| Regions / uniform | 0.10944588689877721 | −0.11309768665660795 | 5/5 |
| Regions / connected region | 0.10077719741431432 | −0.12176637614107085 | 5/5 |

Differences at approximately 1e-16 from a previously printed difference-of-means result reflect floating-point operation ordering, not a new scientific estimate. MAE means and confidence interval endpoints were inspected as retained fields; this audit did not reconstruct MAE from missing per-example predictions or regenerate intervals.

`docs/physworld_evidence_recomputed.json` records a **prior** audit of 125,000 prediction rows, 200 pooled rows, 125 training histories and ten hashed grid files, with historical largest R² discrepancy about 9.19e-17. This historical record is not relabeled as verification performed in the current session.

Content identities inspected now:

| File | SHA-256 |
| --- | --- |
| `gavd5-drift/neurips-laterality/docs/fmts_revisions/paper_v9.md` | `b5681f0398812b7d6bcbf1a33076a91a36274fa109444922748817b080dcd8a3` |
| `gavd5-drift/neurips-laterality/docs/fmts_revisions/review/numerical_evidence.json` | `528f0ce3f7be59975e0453ce777368efdcd8d433860fb3da0ad117011e3c9347` |
| `gavd5-drift/neurips-laterality/config/protocol.json` | `7356cb2e9167e13a85a5190a7a4f3aca8962d7e7c48449feca8f902038efd234` |

These hashes identify inspected local text, not the absent video corpus or original model checkpoint. Future implementation changes and scientific artifacts need their own identities.

Handoff recheck, **2026-09-15 16:17 UTC**: v9 is still the highest numbered Markdown manuscript, but its shared-workspace bytes changed during this task; its current SHA-256 is `adffb481435944bfac7af2a5ad3b86c0d3cad4bce37e0cb780d86dde34cc0d5a`. The initial hash above remains the audit's original observation, not a claim of immutable manuscript contents. This temporal-gait task did not write or revert the manuscript. The current prose still reports the625/93 cohort and trained-readout deficit. The numerical-evidence JSON and historical protocol hashes remain unchanged. No authorship or cause of the concurrent manuscript change is inferred.

## Notebook provenance to preserve

Notebook cell indices are zero-based. The current output inventory was read directly from notebook JSON.

| Notebooks | Code cells / code cells with output | Evidence interpretation |
| --- | --- | --- |
| 00–06 | Each 3/0 | Canonical historical sources are output-free; no fresh historical rerun is implied. |
| 07 | 6/0 | Diagnostic authoring source; retained historical conclusions live in supporting docs. |
| 08, 09 | Each 8/0 | Source notebooks have no inline run evidence. |
| 10 | 7/0 | Past-only helper/tutorial source, not a real forecast result. |
| 11 | 9/9 | Generated mask-construction examples. |
| 12 | 6/6 | Mixed evidence: early synthetic exercises; final cell index 13 retains completion of 25 real jobs/50 encoders. |
| 13 | 11/11 | Synthetic teaching run; historical real claims in prose refer to the separate Notebook 12 grid. |
| 14 | 12/12 | Synthetic forecast demonstration; cell index 23 explicitly leaves real loading/training disabled. |
| 15 | 6/0 | Current source lacks retained execution; dated interpretation references an archived mask audit. |
| 16 | 6/5 | Real cohort and structure-mask audit; no training efficacy from an audit alone. |
| 17 | 9/0 | Declares training; completion evidence is retained in 18. |
| 18 | 10/9 | Real grid outputs in cells 4–13; cell 16 generated amplitude control; cell 18 optional reanalysis not configured. |

`scripts/build_notebooks.py` owns 00–06. `tutorials/research_*.py` plus `scripts/build_research_notebooks.py` own continuation authoring; `tutorials/motion_results_20260908.py` deliberately stores dated historical prose. Do not edit old JSON outputs independently or insert new synthetic values into historical result cells. Current output-bearing files are not automatically fresh-kernel executions of their present source text.

## Preserved scientific stops

The latent-laterality AMASS v2 validation reported correction-first side-sensitive/insensitive errors **0.9300/0.7472**, SG-JEPA **0.8671/0.6696**, and uniform-uncertainty control **0.8668/0.6690**. The uniform control reproduced the apparent SG-JEPA gain. Its preset rule stopped additional seeds 19/31 and the sealed-test jobs 15/16. This temporal-gait implementation has no authority to reopen that stopped study or recycle its reserved sources silently. New source-reservation and exposure manifests must be explicit; unknown exposure is not fresh-test certification.

## Engineering constraints carried into the new implementation

1. Consume explicitly supplied full-video/bout/provenance/reservation/exposure paths only. No remote/local source search, cache fallback or inferred HAIC root; preserve protected media until the locked test stage.
2. Distinguish exact historical whole-clip replay from a new same-window preprocessing mechanism experiment. Keep four-sample patches, coordinate channels, augmentations, source draws and optimizer recipe fixed in the first mechanism contrast. Clock channels, two-sample patches, 2D coordinate conversion and revised optimization are separate named changes.
3. Represent forecast issue time `b`, half-open prefix `[b−2.56,b)`, 64 bin-left queries at 25 Hz and the 40 ms final-query offset explicitly. Query future endpoints at `b+h`; nearest observed scoring is within 20 ms with earlier tie breaking. Actual target time/deviation/validity must not enter the predictor.
4. New primary endpoint uses joints `[25,26,27,28,29,30,31,32]`, at least three common bilateral pairs, aspect-correct original pixels and a prefix chain-length scale. Require eight valid chain samples at distinct original times per side and scale at least `max(20 px,0.02×frame diagonal)`. Keep raw versus held/interpolated support distinct.
5. Prefix normalization, crop selection, pose extraction/smoothing, masks, cached features and predictor queries must remain unchanged under mutations of future observations. Teacher/loss/scoring access to future targets is separate from context access.
6. Compute each window's mean error over common valid joints, then mean windows within bout, bouts within video, and videos equally. Bootstrap complete connected-source groups with all their videos; do not silently replace the equal-video point estimate with equal-group or window weighting.
7. Fit JEPA decoders and direct forecasting controls on the same training-source/endpoints. Select on development, not calibration or test. Pose-derived readout-fit budgets are not manually annotated label efficiency. Report context-feature decoding and predicted-future-feature decoding separately.
8. Include initialized, persistence, robust velocity, prefix periodic, direct ridge, direct learned forecasting and nuisance/mismatch controls. Lower latent loss, greater effective rank, or imposed oddness is not downstream success.
9. Save model/readout states, initial/online/teacher weights, input/split/code/config identities, per-window target/support/prediction rows, all declared failed/missing members and deterministic analysis inputs. Successful execution and scientific advancement are different statuses.

No new efficacy claim is authorized by this ledger. Its purpose is to preserve what is known while making new implementation and evaluation choices traceable.
