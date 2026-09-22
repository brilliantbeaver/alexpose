# Implemented study and executed evidence

The new suite is implemented and fixture-validated. It includes paired body-12 data contracts and source preparation, score-preserving pose extraction, eight learned restoration/control arms, prediction-based grouped evaluation, calibrated gate machinery, blank annotation tools, immutable receipts/resumes, notebooks and bounded HAIC commands.

The [notebook guide](../../../../notebooks/synthetic_training_v2/README.md) explains each stage. The [full fixture report](../../../../outputs/synthetic-training-v2/fixture-notebooks-20260918-final/report.md) links saved predictions, grouped summaries, uncertainty and the accuracy–preservation plot. [Review dispositions](../review-dispositions.md) record independent findings and fixes.

Completed verification: **54 new contract tests, 37 relevant historical regression tests, and all nine notebooks executed from scratch in fresh kernels** under isolated local Torch **2.6.0 CPU**. The original project environment was not replaced. HAIC still requires **2.6.0+cu124**, Torchvision **0.21.0+cu124** and compatible MMCV **2.1.0**; no remote interpreter, allocated GPU operator or source runtime was verified.

All 72 inventoried pre-existing/unrelated files retain their SHA256 values. Canonical notebooks are output-free and deterministic; executed copies use a unique run identity. Batch shell syntax and submission dry runs pass. See [machine-readable verification](software-verification.json).

The analytic fixture contains 60 input rows across 14 windows and 7 analytic person IDs: 24 training rows and 36 development rows. It has one training seed, two simulated extractor IDs and no real people or images. Twelve methods/controls produce 432 window-score rows and 24 primary method/extractor summaries. These counts are software coverage, not empirical sample sizes.

The historical CSV audit independently reconstructed 2,304 source outcomes, 6,912 selector decisions and 144 configurations. Frozen 75-update errors are shown below; they are visible 2D errors normalized by the reference-box diagonal, not clinical gait scores.

| Historical policy | Recomputed error |
|---|---:|
| replay | 0.02732054 |
| full_replay | 0.02731485 |
| pooled | 0.02703672 |
| front | 0.02689916 |
| before | 0.02702576 |
| after | 0.02702576 |
| response | 0.02702220 |
| source_progress | 0.02702576 |
| full | 0.02702648 |
| magnitude | 0.02702576 |
| weakness | 0.02693726 |
| domain | 0.02678074 |
| no_context | 0.02694202 |
| simple_context | 0.02684218 |
| image_context | 0.02684708 |
| shuffled_context | 0.02706667 |
| source_progress_matched | 0.02702576 |
| shared_scene_oracle | 0.02667588 |
| student_specific_oracle | 0.02666506 |

The shared-scene oracle is 0.02667588; the student-specific oracle is 0.02666506. Extra retrospective relative reduction is **0.0405718%** on this inspected development panel. It is neither an achieved method nor a population bound.

The saved pilot used 10 common-probe optimizer updates with 20 synthetic examples, followed by 75 branch updates; full replay therefore used 85 total updates. The new report preserves this correction without altering historical files. Missing historical feature/checkpoint/per-frame caches prevent diagnoses of weak gradients, scaling or collapse. Prior future-feature and accessibility STOPs remain unchanged.

| Scientific gate | Result |
|---|---|
| Image-estimator adaptation | `insufficient_evidence` |
| JEPA advantage and preservation | `insufficient_evidence` |
| Real transfer | `insufficient_evidence` |
| Personalization on historical panel | `fail` |
| Optional video increment | `insufficient_evidence` |

Fixture learning is not evidence of source improvement. Gate A is a separately planned augmented-COCO/adaptation comparison. Source Gate B requires real paired inputs, independent calibration and sufficient groups/seeds. Independent real temporal annotations and genuinely held recordings are required for real preservation. Personalization can reopen only after a new development oracle shows useful headroom. The video branch is skipped. A failed JEPA branch must not suppress a qualifying direct denoiser.

The literature review adds MotionBERT, SynSP and PS-Mamba to the closest restoration precedents. Generic synthetic denoising, estimator-independent correction and masked MoCap pretraining are established; the controlled empirical advantage remains a hypothesis. No novelty, clinical usefulness or scientific success is claimed.

Next actions, in order:

1. Complete the locomotion audit and authoritative reservation/exposure CSVs; verify licensed body/render assets, explicit source extractor checkpoints and the excluded family. The preparation template currently contains required placeholders, not invented assets.
2. Supply an explicitly authorized measured GPU scope and prior cost ledger. Run the allocated HAIC preflight, record its cost, then render the paired smoke panel and inspect its overlays/proxy landmark convention.
3. Run the one-seed source screen, then expand only after measured throughput/learning curves and the relevant gate justify it. Calibrate preservation margins before decisive three-seed adjudication.
4. Time independent annotation on four development clips and double annotation on at least 20% before real temporal evaluation. Confirmation scoring remains disabled pending independent references, exposure review, frozen margins and scorer review.

The exact safe next command previews the prepared jobs without submitting:

```bash
PYTHONPATH=src /private/tmp/gavd6-stv2-cpu/bin/python scripts/research_directions/synthetic_training_v2/submit.py \
  --config slurm/synthetic-training-v2/source.example.json --dry-run
```

The [HAIC runbook](../../../../slurm/synthetic-training-v2/README.md) contains the explicit preflight/preparation/submission commands, resources, dependencies and resume behavior. Default GPU authorization is zero. The suggested 48-hour ceiling has not been treated as spending permission. CPU fixture throughput is not an H100 runtime estimate.
