# Literature update — what transfers to a 2D, 35-source MS/PD/normal gait study

_Compiled 2026-08-03. Decision-critical sources only (per scope). Machine-readable records:
[`../artifacts/research/evidence_ledger.yaml`](../artifacts/research/evidence_ledger.yaml)._

Every source below was verified from a primary document. Where a publisher page was uncrawlable
(IEEE Xplore, Elsevier are JavaScript-only; CVF returned HTTP 403), we fell back to the arXiv
full text, the NeurIPS proceedings PDF, or the PubMed/Semantic Scholar record, and labeled the
fallback. The S-JEPA **project page is unreliable** for mechanics and is not cited for method.

## The one-paragraph summary

The core S-JEPA learning mechanics are well specified and worth reproducing faithfully: mask the
target encoder's **output** (not its input), use factorized spatial+temporal position embeddings,
condition the predictor on **each target slot's position** (the I-JEPA pattern), and use a
centered, sharpened latent cross-entropy loss with an EMA teacher. What does **not** transfer is
the *scale*: the paper's 90% mask, 1,200 epochs, 0.9999 EMA, and batch 256 were tuned on ~57k 3D
action samples, not 35 gait sources. Two more cautions matter for us specifically: motion-aware
(high-motion) masking is contraindicated because reduced motion is the clinical signal in MS/PD,
and every result must be reported at a participant/group-disjoint level because random splits
inflate clinical-gait scores by about 0.13 F1.

## What transfers (mandatory correctness, dataset-independent)

| Mechanism | Source + locator | Why it is mandatory |
|---|---|---|
| Mask the target encoder **output** | S-JEPA, Table 6: input-mask 78.1 vs output-mask 85.3 (**+7.2**) | Output masking gives the predictor richer global targets. The repo already does this; keep it. |
| Predictor conditions on target-slot position | I-JEPA, Method "Prediction": "mask tokens ... a shared learnable vector with an added positional embedding" | This is the authoritative spec for our **D1** fix. Without it, all hidden predictions are identical (we measured std = 0). |
| Factorized spatial + temporal position embeddings | S-JEPA, Sec 4.2 | Present in the encoder; must also reach the predictor's hidden slots. |
| Centered (target-only, β=0.9) + sharpened latent CE, τ_p=0.1 > τ_t=0.06 | S-JEPA, Eq. 2–3, Table 5 (beats MSE) | The anti-collapse objective. The repo matches this; keep it. |
| EMA target + stop-gradient | S-JEPA, Table 5 (removing it collapses) | The load-bearing anti-collapse mechanism. |

## What does NOT transfer (retune, do not copy)

| Paper setting | Value | Why not here |
|---|---|---|
| Mask ratio | 0.90 | Tuned on 57k NTU samples; overfits 35 gait groups. Search 0.40/0.60/0.75 on inner folds. |
| Pretrain length | 1,200 epochs | We express budget in optimizer **updates** (300/1k/3k) and stop when memorization rises. |
| EMA schedule | 0.9999 → 1.0 | Too slow for a short run; the teacher would barely move. Use ~0.990 → 0.9995 by step half-life. |
| Effective batch | 256 | Reach ~128 by accumulation only if it helps; do not distort source balance. |
| 3D SO(3) rotation views | — | Invalid for 2D + visibility. Use 2D-safe augments. Our pipeline is a **2D variant** of a 3D method. |
| Motion-aware (high-motion) masking | MAMP Sec 3.4 | **Clinically wrong** here: reduced motion (hypokinesia, short steps, reduced arm swing) is the MS/PD signal. Use uniform / low-motion-inclusive masks. |

## Exploratory (later phases, not R1)

- **Motion (velocity) targets.** MAMP Table 7: predicting frame-to-frame joint differences beats
  coordinate reconstruction by +10.1/+6.1. This is 2D-safe (velocity in x,y) and is an R2 idea.
- **External clinical-motion pretraining.** GaitForeMer: NTU pretraining lifted few-shot PD
  severity macro-F1 from 0.60 to 0.76. CARE-PD (362 subjects, 8,477 walks, SMPL) is the largest
  public substrate. Both are **severity**, not diagnosis; CARE-PD ships no S-JEPA checkpoint and is
  non-commercial with a different skeleton topology, so any transfer is real work and unproven.

## What the clinical evidence tells us about evaluation and claims

- **Participant/group-disjoint is mandatory.** An FG-2024 clinical-gait benchmark reports F1 0.62
  under leave-one-subject-out versus 0.75 under random splitting — a ~0.13 inflation on a task very
  like ours. CARE-PD's leave-one-dataset-out (~0.50 macro-F1) and Kaur et al.'s new-subject
  protocol point the same way.
- **Acquisition and body shape are known shortcuts.** "The Paradox of Motion" shows static pose and
  height alone drive strong gait-recognition accuracy. In our data this is acute: **all MS clips
  are 60fps and square**, so "60fps ⇒ MS" scores 100% with zero pose information. Every headline
  score is reported next to these shortcut controls.
- **Do not overclaim.** GaitForeMer and CARE-PD estimate *severity*, not diagnosis; the MS
  markerless studies are small associational cohorts (n = 20–25). Distal shank/foot elevation-angle
  range, tandem heel-to-toe distance, and reduced stride length/time are the clinically meaningful
  markers, and they make a good feature-interpretation story — but nothing here establishes
  diagnostic validity from in-the-wild video. Our work is a methods/feasibility demonstration.

## How this reshaped the plan (contradictions with the prior synthesis)

`02-0802-NEXT_STEPS.md` was accurate on the mechanics. Two refinements from this pass:

1. The prior text left "predictor target positions" as a repository-specific inference. The I-JEPA
   text makes it an **explicit, ablated, dataset-independent spec** — so we implement it as core
   correctness, not a tuning choice.
2. The data audit found the identity problem is **worse** than "source_id ≠ participant_id": a
   single source can contain up to five different labeled patients, and the MS test-retest videos
   suggest one person appears across several sources that land in different folds. We therefore
   label every result **provisional source-grouped**, and a proper participant registry is flagged
   as required future work rather than something this session resolves.
