# Inference review of the current manuscript

Reviewed 16 October 2026. Target: `docs/wacv2027-v1/wacv-v1-revision.tex`, SHA-256 `67a5ed3f42f3a01b40f458e8ee0e71d00f0f3516655390604cf4a796b86385b3`. Line numbers refer to that version. This is a fresh check of inferential claims against the top-level notebooks' code and retained outputs, with the imported implementation inspected where the notebooks delegate work. It does not rely on the earlier editorial approval. No notebook cells were executed, no model or classifier was fitted, and the manuscript was not edited.

## Material errors

**No material inference error found in the current wording.** The central claim is an observed result for the trained encoder and its supervised readout: its features support prediction of collection labels on excluded source recordings. The manuscript does not claim that the experiment isolates a causal benefit of pretraining, proves a temporal mechanism, validates a clinical measurement, or establishes superiority in a new population. Those distinctions are explicit, and the mean-pose comparison is prominent.

“Useful trained features” is defensible here when read in its stated operational sense: features used by this fitted classifier for held-out collection-label prediction. It is not evidence that pretraining improved prediction relative to random features, or that the features represent neurological impairment. The current abstract, results and discussion preserve that boundary. An untrained-encoder experiment would answer the stronger causal question; its absence does not invalidate the narrower descriptive result already reported.

## Primary evidence inspected

Notebook cell numbers below are one-based and count both code and Markdown cells. Code and saved numerical outputs take precedence over surrounding notebook narrative.

| Primary record | Direct observation and inferential relevance |
|---|---|
| `03_sjepa_model_and_pretrain_normal.ipynb`, cells 8, 11 and 15 | Fold 0 has 51 training clips from 24 sources. The executed call passes `train_recs` to `train_checkpoint` for 800 updates. The saved final effective rank is 12.134888648986816. The filename's “normal” wording does not describe a Normal-only training population: the split output includes all three labels. |
| Notebook 03, cell 19 | The retained 800-update history reports mean losses 3.858368 and 1.022220 for the first and last 50 updates; corresponding effective ranks are 8.132109 and 10.939291, and mean feature standard deviations are 0.153868 and 0.525584. These are training-batch diagnostics, not test scores. |
| `04_progressive_finetune_ms_pd_vicreg.ipynb`, cells 9, 11 and 14 | The second stage loads the initial checkpoint, calls the same training function for 400 updates, fits a separate probe for each stage using training features, and chooses by validation F1. Fold-0 validation scores are 0.293979 for the initial stage and 0.276105 for continuation. The executed path does not use the VICReg objective suggested by the filename. |
| `05_representation_visualization.ipynb`, cells 10, 12, 15 and 18 | Visualization and silhouette use 51 **training** clips. Silhouettes are −0.004 for the initial stage, −0.012 for continuation, and −0.004 for visibility. These do not establish disease-separated clusters. They also do not contradict the held-out classifier result: global class clustering in feature space and prediction by a fitted classifier are different questions. The manuscript correctly does not use these plots as clinical validation. |
| `06_capstone_rf_vs_sjepa.ipynb`, cell 11 | The executed code calls `run_cross_validation` with 800 initial and 400 continuation updates, not smoke settings. The saved output identifies `capstone-20261009T060955236580Z`. All five folds execute both stages; validation retains the initial stage in folds 0–2 and continuation in folds 3–4. |
| Notebook 06, cell 14 | The current output reports source-weighted F1 of 0.456624 for S-JEPA, 0.325184 for RF, 0.451893 for mean pose, 0.317565 for visibility, and 0.259398 for the majority rule. Equal-clip scores are 0.397034, 0.320940, 0.409840, 0.318057 and 0.202125, respectively. These support the stated observed ranking, not a general ordering of methods. |

The October run's `results.json`, at the output location printed by notebook 06, agrees with those scores and stage selections. Its final losses decrease from initial to continuation in all five folds, while validation F1 improves in two. Reading its `oof.json` gives 88 rows, 88 distinct clips and 41 sources, with each source appearing in one test fold. Direct counts give 36 correct S-JEPA clips, 30 RF clips and 37 mean-pose clips.

The code path establishes the fitting boundaries used to interpret these observations:

- `sjepa/full_experiment.py:91–105` accepts only the registry's training clips and sources for encoder training. `sjepa/train_v2.py:178–220` draws pose windows and computes a feature-prediction loss; collection labels are not passed to that loss.
- `sjepa/full_experiment.py:56–64` fits the scaler and logistic-regression classifier on the supplied training features and labels. Lines 163–183 select between stages using validation predictions before extracting test features for the selected stage.
- `sjepa/splits.py:146–169` checks disjoint source and clip partitions and one test assignment per usable clip. `sjepa/full_experiment.py:208–230` checks complete, correctly attributed out-of-fold predictions before pooling their scores.
- `sjepa/full_experiment.py:194–200` evaluates mean pose, visibility and the majority rule. There is no random-encoder, shuffled-order or alternative-mask comparator in this executed comparison.

## Assessment of the manuscript's inference boundaries

| Manuscript location | Assessment |
|---|---|
| Lines 19, 30–32, 182, 224 and 239: useful representations and held-out prediction | Supported as a description of the trained feature-and-classifier pipeline on this collection. The classifier is supervised, and its inputs come from an encoder trained without condition labels. The wording does not imply that the whole prediction procedure is label-free. |
| Lines 19, 215–226 and 236: contribution of pretraining | Adequately bounded. The abstract calls for controls to isolate pretraining, the training-results paragraph explicitly says it does not measure improvement over an untrained encoder, and the discussion proposes that comparison. Do not reinterpret the RF difference as the benefit caused by pretraining; the manuscript does not do so. |
| Lines 184, 210 and 226: temporal information and mean pose | Adequately bounded. Mean pose's nearly equal pooled source score, better equal-clip score and four fold wins are disclosed. The paper acknowledges differences in visibility, feature dimension and averaging samples. It does not claim that average pose is an otherwise matched temporal ablation or that temporal learning has been proved useless. |
| Lines 182, 207, 210 and 231: superiority | Supported as the observed comparison with the specified RF pipeline. The manuscript reports fold variation, distinguishes the historical RF result, and limits generalization beyond the implemented pipelines. The supplement, lines 126–128, explicitly limits the fixed-prediction resampling intervals and does not claim a reliable advantage in new populations. Statistical significance is not required to report the observed score difference; it would be needed for a stronger reliability claim that the text does not make. |
| Lines 124, 143–148, 224 and 229–231: excluded sources and generalization | Supported for the held-out source IDs within this internal cross-validation. This is not verified participant independence or an untouched external collection. The manuscript explicitly acknowledges possible participant overlap, recording cues, limited extension to new settings, 8–9 additional validation sources, and the single training seed. |
| Lines 215–226: training behavior | The loss and rank values support a description of optimization and retained feature variation on sampled training batches. They do not establish semantic content, held-out feature-prediction performance or a classification gain from learning. The manuscript already distinguishes the diagnostics from an untrained-encoder comparison. Two optional wording refinements below would make the measured quantities even clearer. |
| Lines 41, 58, 212 and 229–236: clinical interpretation | Adequately bounded. Labels are described as title-derived collection categories without independent clinical verification. Clinical diagnoses, biomarkers, screening performance and calibrated gait measurements are not asserted. |

## Optional precision improvements

These would tighten wording; they are not grounds for rejecting the current descriptive result and do not require new experiments.

1. **Identify validation explicitly in the continuation conclusion (line 217).** The last sentence says further optimization does not consistently improve “label prediction.” The preceding sentences make validation the intended meaning, but the code evaluates test features only for the selected stage (`sjepa/full_experiment.py:179–183`). There is no paired initial-versus-continued test comparison. A more exact replacement is:

   > The restarted continuation improves source-weighted validation macro-F1 in only two of five folds. Lower final training-batch loss therefore does not consistently coincide with better validation classification; the validation step determines which stage is used for each test fold.

2. **Keep training-loss interpretation tied to evolving targets (lines 215 and 226).** The target encoder changes during training (`sjepa/train_v2.py:227–237`), and the target distribution also depends on its running center (`sjepa/losses.py:82–92`). Consequently, falling cross-entropy is not performance on a fixed prediction test. The current “fits the hidden-feature prediction task” language is normal shorthand and is bounded by the surrounding text, but an especially precise opening for line 215 would be:

   > Training-batch feature-prediction loss decreases while the sampled target-encoder features retain variation across inputs.

   The held-out classifier results, rather than the loss decrease alone, remain the evidence for label-prediction utility.

3. **Describe the proposed untrained comparison as estimating predictive benefit (line 226).** “Would establish how much of that information is gained” sounds more definitive than the outcome of one additional comparison and does not name the observable quantity. A more precise replacement is:

   > A matched untrained-encoder comparison would estimate the effect of pretraining on held-out label prediction under the same evaluation procedure.

## Notebook narrative that should not govern the manuscript

The raw notebooks contain older commentary alongside newer executed outputs. Notebook 06 Markdown cells 15, 19 and 20 still describe RF as 0.411 or its correction as outstanding, while code cell 14 reports the corrected 0.325184 result and cell 11 identifies the October execution. Notebook 03 Markdown cell 18 says only one rank value was printed and quotes an earlier endpoint, while executed cell 19 now contains exact summaries. These are documentation inconsistencies in the notebooks, not reasons to revert the manuscript to the older result or to withhold its bounded positive conclusion.

The manuscript's strongest defensible claim remains: **the trained S-JEPA features support held-out collection-label prediction in this source-grouped experiment, exceed the specific corrected RF under both weighting schemes, and perform close to mean pose. The design does not isolate why those features are predictive.**
