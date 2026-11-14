# Verification checklist for the standalone Section 4.1 correction

This check reread the original PDF's Section 4.1, notebook 02's relevant code and saved outputs, the current masking implementation, the masking/augmentation call sequence, and the retained full primary S-JEPA paper. Notebook cell numbers are zero-based and include Markdown cells. No notebook, model code, or saved experimental result was changed. The top-level notebooks remain the source of truth; the separate 5,000-mask audit is labeled additional verification.

## Facts the replacement section should retain

| Item | Verified statement | Exact evidence |
|---|---|---|
| Input and tokens | All 33 landmarks enter the model. Four consecutive frames for one landmark form a token; 32 frames therefore yield eight time blocks and 264 tokens. | Notebook 02 cells 7 and 9; `sjepa/tokenizer.py:39–50` and `sjepa/config.py:48–53`. |
| Twelve-landmark set | The selected indices are 11, 12, 23–32: bilateral shoulders, hips, knees, ankles, heels and foot indices. The set guides sampling weights; it does not restrict the input or define a permanent mask. | Notebook 02 cells 10–11; `sjepa/config.py:30`; `sjepa/masking_v2.py:55`. |
| Region groups | Seven groups cover all 33 landmarks: head 0–8, mouth 9–10, left/right arms, trunk, and left/right legs. Shoulders occur in arm and trunk; hips occur in leg and trunk. | `sjepa/masking_v2.py:43–50`; notebook 02 cells 12–13. |
| Sampling rule | Independently for each training example, draw a region and a time interval, union their tokens with the targets, and repeat until 158 targets are reached or 36 region draws have occurred. Interval length is uniform over 1–6 blocks; start is uniform among starts that keep that interval inside the eight-block window. | `sjepa/masking_v2.py:128–147`; `sample_mask_batch` at lines 168–190. |
| Budget | 158 is `round(0.6×264)`, a stopping target rather than an exact final count. A final whole region can overshoot; overlaps and the finite draw limit also mean this is not a fixed 60/40 partition. | `sjepa/masking_v2.py:135–147`; notebook 02 cell 13 reports 0.63 mean target fraction. |
| Context guarantee | A visible token from the twelve-landmark set remains somewhere in each window, and target/context sets are nonempty. The visible token can be a shoulder. No hip or leg is guaranteed visible anywhere in an individual window; an entire time block may lack context. | `sjepa/masking_v2.py:149–164`; notebook 02 cells 12, 16–17. |
| Per-draw versus full-window span | Six blocks is the maximum for one region draw. Several overlapping draws can hide the same joint in all eight blocks. | Notebook 02 cells 12 and 17; union operation at `sjepa/masking_v2.py:145–147`. |
| Uneven time coverage | Uniform start/length selection does not make token coverage uniform; middle blocks belong to more admissible intervals. | `sjepa/masking_v2.py:124–126`; notebook 02 cell 13's exact-time hip table. |

## The original account omitted the membership correction

For group G, the unnormalized weight is the clinical factor divided by the average number of groups containing its joints:

`w(G) = b(G) / mean[m(j) for j in G]`,

where `b(G)=1.5` when G contains any of the twelve selected landmarks, otherwise 1, and `m(j)` is joint j's group-membership count. Normalize these weights across the seven groups to obtain each region draw's probabilities. This is exactly `_region_weights`, `sjepa/masking_v2.py:84–105`.

| Region | Unnormalized weight | Probability per region draw |
|---|---:|---:|
| Head | 1 | 12.7854% |
| Mouth | 1 | 12.7854% |
| Left arm | 1.285714 | 16.4384% |
| Right arm | 1.285714 | 16.4384% |
| Trunk | 0.75 | 9.5890% |
| Left leg | 1.25 | 15.9817% |
| Right leg | 1.25 | 15.9817% |

These probabilities were recomputed from the existing function. They describe which region is selected on a draw, **not the fraction of that region's tokens ultimately hidden**. Arms receive the clinical factor because they contain a shoulder, so their elbows, wrists and hand landmarks inherit the boost. The factor 1.5 is neither a final probability nor a guarantee that each selected joint is hidden 1.5 times as often as another joint. The original section and figure caption need this distinction; describing “1.5× sampling weights” alone is incomplete.

## Keep the two diagnostic banks separate

### Notebook-authoritative bank: 512 masks, seed zero

Notebook 02 cell 13 calls `mask_bank_stats(..., n_masks=512, seed=0)` and has saved executed output. It reports:

- Mean target fraction **0.63**, rounded to two decimals. Recalculation of this same defined diagnostic yields **0.6327607126**, or **63.3%** to one decimal percent.
- Every joint is visible in at least one time block in at least **74.0%** of windows and targeted in at least one block in at least **80.9%**. “Visible somewhere” is not the same statistic as “fraction of visible tokens.”
- Visible-token percentages are 22.3%/20.2% for left/right shoulder and 22.8%/23.0% for left/right hip. Thus these joints are mostly hidden at the token level even though they are visible somewhere in most windows.
- Left-hip context at the first block is **51.6%**, versus **7.0%** at block five; corresponding right-hip values are **45.5%** and **7.6%**. These printed examples demonstrate time bias directly.

Cell 13 also prints six distinct masks from six samples, with nonempty context and targets. Cell 17's separate eight-mask illustration prints no visible hip blocks for either hip in samples 1 and 6; all joints nevertheless occur in both roles across its eight samples. These checks support varied masks across draws, not a claim that every joint appears as context in every example.

### Additional verification: 5,000 masks, seed zero

The original PDF cites a 5,000-mask bank but notebook 02's saved diagnostic uses 512. During the audit, a new deterministic 5,000-mask bank was generated from the current function, without training a model. It reproduces the original section's approximate marginal percentages. This supports those numbers as a **separately reproduced sampler diagnostic**, not proof of the historical bank's exact provenance or observed capstone-training exposure. Its overall target fraction is **63.3161%**.

| Landmark category in the separate bank | Target-token frequency |
|---|---:|
| Left/right shoulders | 77.8425% / 78.3100% |
| Left/right hips | 77.2625% / 76.8700% |
| Remaining leg landmarks | Approximately 62.06–62.54% |
| Left/right elbows, wrists and hand landmarks | 63.6050% / 64.3000% |
| Each of the nine head landmarks, 0–8 | 59.6500% |
| Each of the two mouth landmarks, 9–10 | 52.1800% |
| Equal-landmark average over all eleven facial landmarks | **58.2918%** |

The original “58% for facial landmarks” is numerically supported **only as a pooled/equal-landmark average across indices 0–10**: `(9×59.65 + 2×52.18)/11 = 58.2918`. It is misleading if read as the frequency of every facial landmark, and it is not the mean of the two region percentages with equal region weights. Prefer: “Targeting was 59.7% for the nine head landmarks and 52.2% for the two mouth landmarks; their pooled mean was 58.3%.” If space is limited, give the two region values and omit the pooled mean. Never mix this 5,000-mask table with the notebook's 512-mask table without naming both sample sizes.

## Attribution to the original S-JEPA method

The full primary paper is retained at `evidence/sjepa-eccv2024-primary.pdf`: Abdelfattah and Alahi, *S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition*, ECCV 2024. Its Section 3, “Targets” paragraph on printed page 5, describes motion-aware sampling inherited from MAMP, giving joints with substantial motion higher selection probability. The official [ECCV paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) supports **probabilistic preference**, not deterministic selection of the highest-motion joints. Section 4's implementation paragraph on printed page 8 uses mask ratio 0.9 and segment length four in that original study; those are not this repository's 0.6 region-mask setting.

The original PDF's Section 4.1 wording “preferentially masks joints with greater motion” is consistent with that source. Stronger wording elsewhere in v1 that the method simply takes the highest-motion joints should not be carried into the standalone correction. Say that this implementation replaces motion-related sampling with a region-based rule. Do not imply that the comparison experimentally establishes which rule is better.

## Clinical claim boundary

Clinical gait literature can motivate attention to shoulders and lower limbs and explain why these regions are plausible choices. The notebook evidence does not validate this exact twelve-landmark set, the 1.5 factor, the grouping, or their effect on classification. The code comment calling high-motion masking “contraindicated” is a proposed rationale, not a finding from these experiments. Reduced motion being clinically relevant does not establish that motion-aware masks are unsuitable. Keep the explicit statement that there was no mask ablation, and avoid “clinically validated mask,” “optimal clinical joints,” or a claim that this sampler preserves disease-specific information.

## Reflection caveat and the meaning of hidden

The mask marks joint/time **slots** to exclude as student context. `train_v2.py:205–216` samples that mask, takes its complement as context, then augments the input. `augment.py:80–87` can reflect the input and swap left/right landmark slots, including visibility, but the sampled mask is not permuted. The teacher stays in the original joint indexing. A physical joint intended as a teacher target can therefore appear as transformed input in the student's opposite-side context.

For a standalone correction, either include one sentence identifying this correspondence limitation or explicitly point to the separate training/limitations discussion. It is important when explaining what the mask actually withholds. It is **not** a test-source leak, condition-label leak, or measured explanation for classification performance. The deterministic region diagnostics above do not measure this defect's downstream effect.

## Editorial acceptance checklist for the replacement

- Remove the original editing marker “Rewrite:” and explain mask, target and context before detailed sampling rules.
- State all 33 input landmarks, eight blocks, 264 tokens and the twelve-landmark weighting role without conflating them.
- Include the average-membership correction, whole-region overshoot, and limited context guarantee.
- Report the notebook's 512-mask evidence as primary; label the separate 5,000-mask verification if retained.
- Split head and mouth frequencies, or explicitly identify the 58.3% pooled facial average.
- Describe original S-JEPA motion-aware selection as probabilistic and distinguish this adaptation.
- Keep the clinical motivation as a hypothesis and state that no masking-performance comparison was run.
- Bound “hidden” by the known reflection correspondence issue without treating it as held-out leakage.

The verified facts support a more complete and readable explanation of the evaluated sampler. They do not turn that sampler into a demonstrated clinical or methodological improvement.

## Review of the standalone replacement

The standalone `wacv2007-v1-section-4.1-refined.md` was subsequently checked against this evidence. Its regional probabilities, 512-mask figures, separate 5,000-mask percentages, head/mouth distinction and 58.29% pooled facial average are correct. Its account of limited context guarantees, all-33 input, 161-position output readout, clinical motivation, original S-JEPA motion-aware attribution and reflection limitation is supported. No material numerical or algorithmic correction is required.

Two small precision changes were recommended to the author of the standalone document:

1. In the replacement paragraph, distinguish **availability as context under the mask** from MediaPipe's separate visibility input channel. “Visibility was uneven across time” can become “Availability as context varied across time.” The following hip example already measures mask context correctly.
2. In the explanatory evidence note, state that notebook 02's saved output prints the 512-mask mean as `0.63`; deterministic recalculation of that same specified bank yields `0.6327607126`, reported as 63.3%. The separately generated 5,000-mask diagnostic remains distinct. This clarifies the precision source without changing the supported rounded result.

The standalone document and manuscript were not edited during this independent review.
