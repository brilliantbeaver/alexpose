# Advanced notebook evidence audit

Audit date: 2026-09-05. Scope: complete saved source, markdown, and text outputs of notebooks `08`, `09`, and `nb_05a`–`nb_05d`; their current and historical numerical artifacts; the shared split/QC and scientific-visualization helpers; and relevant replacement laterality geometry/evaluation code. No training, source-notebook edits, or upstream artifact writes were performed.

The strongest usable observation in this subset is a **descriptive, source-disjoint normal-reference representation audit**. In the one completed fold and seed, final validation cosine is 0.7011 when uploads receive equal weight, compared with 0.8891 when clips receive equal weight. One upload contributes 60 of the 64 validation clips. The same unchanged embeddings produce both numbers. This illustrates why the unit of analysis matters. It does not establish clinical deterioration, catastrophic forgetting, or a successful consolidation intervention.

There is **no current forecasting, external reflection, clinical laterality, or consolidation-repair result** in these notebooks. Notebook 09 is blocked, 05a/05c/05d are archived transductive analyses, and 05b contains simulations only.

## 1. Evidence status and source hierarchy

| Notebook | Actual current execution | Numerical evidence appropriate for the new paper |
|---|---|---|
| `08_normal_anchor_drift_and_consolidation.ipynb` | Completed fold 0, seed 42 under protocol v2 | Development normal-reference cosine curve and one final normal test estimate; descriptive pilot only |
| `09_predictive_surprise_world_model.ipynb` | Protocol gate executes; future-trained checkpoint absent; evaluation skips | Missing-evidence status, not forecasting performance |
| `nb_05a_signed_laterality_probe.ipynb` | E1 gate blocked; script missing; historical markdown retained | No current empirical result |
| `nb_05b_reflection_reach_and_futures.ipynb` | Four simulated decision patterns and synthetic multiview fixture execute | Workflow illustration only; no empirical checkpoint or external-cohort result |
| `nb_05c_reflection_equivariant_readout.ipynb` | E2 gate blocked; script missing | Algebraic construction may be explained with proper assumptions; historical performance excluded |
| `nb_05d_reflection_equivariant_encoder.ipynb` | E3 gates blocked; scripts missing | Algebraic construction may be explained; historical augmentation/performance excluded |

The current authoritative numerical report is `work/artifacts/real/fold_evaluation/outer_fold_0/normal_anchor_drift_seed_42.json`. Historical files named `anchor_guard_results.json`, `predictive_surprise_results.json`, `idea5_signed_laterality_result_hardened.json`, `idea9_equivariant_readout_result.json`, and `idea9_equivariant_encoder_result.json` remain on disk. Their presence must not be interpreted as current execution. The last three declare `transductive: true`; their notebook gates do not rerun them.

Saved figure-only cells have execution count 1 after the main cells ran. These cells render reports through `scientific_visuals.py`; a newly generated figure does not imply a newly executed experiment. In particular, the laterality renderer intentionally reuses archived JSON and labels it archived.

## 2. Notebook 08: verified current result

### Model and computation actually used

The consumer instantiates `_FoldSJEPA` in cell 11, not the older, unused `SJEPAGait` classes retained in early cells. The current checkpoint config is 64 resampled frames, four-frame segments, 64-dimensional embeddings, two encoder layers, and four attention heads. Each joint's four-frame coordinates are averaged before a `Linear(3,64)` projection. Sixteen segments and 33 joints yield 528 positioned tokens. The predictor pools visible context and applies a position-conditioned MLP. Describing flattened 12-coordinate patches and a transformer predictor would describe the unused historical implementation.

For each sequence, `prepare_sequence` applies visibility threshold 0.45, interpolates only short internal gaps up to four frames, preserves the original validity mask, centers at the pelvis, scales by a sequence-wide shoulder/hip width statistic, and resamples to 64 frames. A patch is counted valid only if every frame in the patch is valid. `target_embeddings` averages the EMA target encoder tokens over the 12 selected shoulder/lower-body joints and valid patches. It returns one 64-vector per sequence. Missing tokens can still affect contextualized tokens through the encoder; validity-weighted output pooling does not establish immunity to missingness.

Let \(z_i^{(0)}\) be a sequence's pooled Stage-0 target embedding and \(z_i^{(s)}\) its later-stage embedding. The notebook computes

\[
c_i^{(s)}=\frac{\langle z_i^{(0)},z_i^{(s)}\rangle}{\|z_i^{(0)}\|\,\|z_i^{(s)}\|},\qquad
C_{\mathrm{source}}^{(s)}=\frac1V\sum_{v=1}^{V}\frac1{n_v}\sum_{i\in v}c_i^{(s)}.
\]

This is the average of per-sequence cosines, first within upload and then across uploads. It is **not** cosine between two cohort centroids, a cosine between source-mean embeddings, or an average over individual joints. There is no fitted feature standardizer, alignment, or whitening in this computation.

### Numerical results

| Stage | Cumulative annotations present in training | Train source-equal cosine | Validation source-equal cosine |
|---:|---|---:|---:|
| 0 | normal | 1.000000 | 1.000000 |
| 1 | + Parkinson's | 0.995692 | 0.992833 |
| 2 | + stroke | 0.984225 | 0.969039 |
| 3 | + myopathic | 0.892483 | 0.737187 |
| 4 | + cerebral palsy | 0.867689 | 0.701058 |

The normal training reference includes 156 sequences from 18 uploads. The normal validation reference includes 64 sequences from five uploads. These counts remain fixed across the five stage comparisons. The final test result is 0.8496323824 on 56 normal sequences from seven uploads. The test source means range from 0.7165420651 to 0.9405758977. Only fold 0 and seed 42 are represented; no across-fold or across-seed confidence claim follows.

The largest stage-to-stage validation decrease is 0.2318518162 between stages 2 and 3. Stage 3 simultaneously changes the data mix and continues optimization; this is not an identified causal effect of myopathy, a disease-severity hierarchy, or evidence that a specific annotation damages the encoder.

The current available checkpoint directory contains one final objective, `jepa_vicreg`, and its five stage checkpoints. The notebook's default candidate list contains only that objective, and the saved report supplies no alternative candidate score. Calling this a successful consolidation comparison would therefore be inaccurate. “Final-model normal-reference audit” is the appropriate description. No current AnchorGuard intervention is evaluated.

### Direct weighting sensitivity using identical cached embeddings

The audit script recomputed both aggregations from the same Stage-0 and final cached 64-vectors. It did not refit, realign, standardize, or change the reference. The alternative clip-weighted value is

\[
C_{\mathrm{clip}}=\frac1N\sum_{i=1}^{N}c_i=\sum_v\frac{n_v}{N}\bar c_v.
\]

| Normal validation upload | Clips \(n_v\) | Mean cosine \(\bar c_v\) | Clip weight | Source weight |
|---|---:|---:|---:|---:|
| `WpARylM4UYU` | 1 | 0.5433649421 | 1.5625% | 20% |
| `_-Ubl8iD2B0` | 1 | 0.7553919554 | 1.5625% | 20% |
| `hSIYGZhRGd4` | 60 | 0.9049939513 | 93.75% | 20% |
| `sk0EU7MNt78` | 1 | 0.6033803225 | 1.5625% | 20% |
| `yFBy0X0D-w8` | 1 | 0.6981571913 | 1.5625% | 20% |

Final validation source-equal cosine is **0.7010577321**; clip-equal cosine is **0.8890614510**; the absolute difference is **0.1880037189 cosine units**. Source averaging changes the estimand: it estimates a mean across observed uploads, whereas clip averaging gives the clip-rich upload nearly all the weight. Neither should be described as an accuracy percentage, a clinical retention rate, or an estimated bias relative to an unknown population truth. The comparison is a post hoc descriptive reaggregation of existing development embeddings.

| Role and stage | Source-equal mean | Clip-equal mean |
|---|---:|---:|
| Validation, Stage 1 | 0.992833 | 0.996290 |
| Validation, Stage 2 | 0.969039 | 0.987301 |
| Validation, Stage 3 | 0.737187 | 0.906119 |
| Validation, Stage 4/final | 0.701058 | 0.889061 |
| Test, final | 0.849632 | 0.853572 |

The test difference is much smaller, 0.0039392114. This prevents a universal claim that clip weighting always strongly inflates this metric. The defensible general lesson is that upload concentration and weighting can materially change the reported quantity, demonstrated concretely on this validation subset.

### Verification and provenance

`review/audit_advanced_artifacts.py` reads local checkpoints and cached embeddings, checks hashes and role subsets, recomputes the published source means, and independently computes clip means and source counts. It ran successfully; all six checkpoint hash/role checks and all cached-cosine comparisons passed. Its derived output is `review/advanced_artifact_checks.json`.

| Item | SHA-256 |
|---|---|
| Input manifest | `7fd559e5105b11011a3e5c194b7ccc29729c56491c424745834df39884123b5a` |
| Source split | `ff3518b87b1d1fa7d95efb1aea1711773137a21699967cb8015edb8d845ccbe1` |
| Stage-0 checkpoint | `03c4414cd99009f198bcf421e30c11301df14f8b976fabef994e588f7020c4c2` |
| Final checkpoint | `f510be2a0453dda0d6698780fcd998835db213e39d94e345b227a5ed8ec648ac` |

Before attrition, the frozen fold has 60/20/20 train/validation/test uploads and 392/134/131 sequence rows. The checkpoint's post-QC roles contain 59/18/20 uploads. Stage fit-source counts are 18, 25, 36, 53, and 59; corresponding validation-selection counts are 5, 7, 11, 16, and 18. The recorded fitting IDs are subsets of training roles and never intersect test IDs. A source-disjoint split does not imply person-disjoint data: identities across uploads are not resolved.

The two exact normal-validation cache paths are:

- `work/artifacts/real/fold_evaluation/outer_fold_0/embedding_cache/normal_validation_stage_00_03c4414cd990.npz`
- `work/artifacts/real/fold_evaluation/outer_fold_0/embedding_cache/normal_validation_stage_99_f510be2a0453.npz`

Sequence IDs in those cache metadata are identical and map to the frozen manifest. The script also verifies each metadata checkpoint hash against the actual checkpoint file. The final and Stage-4 validation vectors produce the same numbers despite distinct serialized checkpoint hashes.

### Inferences that must be withheld

Cosine in a fixed coordinate basis measures representation change. A change of latent basis can lower cosine without losing task information. Conversely, unchanged or collapsed representations can have high cosine. To claim forgetting, one needs an appropriate retained-task evaluation, and ideally basis-insensitive geometry checks and a plasticity comparison. To claim a successful repair, one needs a separate intervention, matched training exposure, a comparator, noncollapse checks, and retained downstream performance.

The loss is self-supervised, but choosing a normal-only initial stage and a condition-ordered curriculum uses dataset annotations. “Label-free objective with annotation-informed curriculum” is precise. “No labels used in training” is not.

The normal-annotated reference is neither a patient-specific baseline nor a clinically validated normative population. Its drift does not indicate disease progression, safety, or an agent's awareness of deterioration. The seven test uploads also do not support a normative clinical threshold.

## 3. Notebook 09: blocked and not yet causally valid

The saved gate explicitly reports the absence of `sjepa_outer_fold_0_seed_42_jepa_vicreg_future_mask.pt` and its JSON sidecar. Horizon scoring prints “SKIPPED,” the test cell prints “outer test was not opened,” and the report cell prints “no predictive-surprise report was written.” The current fold-evaluation directory has no predictive-surprise report. A world-model performance or AUROC table must not be populated from historical files.

The notebook makes a sound conceptual distinction between spatial infilling and future prediction. However, merely supplying a future-named checkpoint would not make the present evaluation a causal forecast. Code review found the following issues before that experiment should be run:

1. **Future information remains visible.** `future_mask` hides only the 12 selected joints in a suffix; all 21 other joints remain visible at future times. The context encoder attends over those future joints. This is future-position conditional imputation, unless the intended information set explicitly permits those measurements.
2. **The copy-last baseline is contaminated by the complete clip.** `copy_last_cosine` calls `model.target_encoder(batch)` with the entire unmasked sequence. The target encoder has noncausal attention. Therefore the copied “last observed” latent can depend on the future; it is not a prefix-only persistence baseline. The previous token's validity is not explicitly required either.
3. **Preprocessing uses future observations.** Short-gap interpolation, whole-sequence scale/fallback statistics, and whole-clip temporal resizing can depend on values or endpoints beyond a causal cutoff. Online forecasting requires a stated observation boundary and prefix-only preparation, or an explicit retrospective conditional-imputation interpretation.
4. **Latent similarity alone does not demonstrate dynamics.** Target tokens encode spatial, temporal-position, and contextual information. A position-only, mean-pose, random-feature, or static-context baseline may perform well without modeling dynamics. The predictor's pooled-context MLP does not by itself establish a transition model, action conditioning, intervention response, or a meaningful multi-step rollout.
5. **The positive gate is too weak.** Code sets `positive_claim_gate = mean_advantage > 0` in a single fold. The markdown correctly asks for evidence across predeclared folds/seeds. A positive point estimate without uncertainty, robustness, or a valid baseline is insufficient even after causal inputs are repaired.
6. **There is no calibration result.** `surprise = 1 - cosine` is a discrepancy, not a predictive probability, likelihood, calibrated uncertainty, clinical risk, or abstention policy. AUROC, if eventually computed, would test ranking against observational condition labels. It would not establish calibration, diagnosis, or safe autonomous action.
7. **Missing-target behavior needs specification.** All-invalid selected patches can yield `nanmean` outputs; source averaging may skip NaNs, changing the scored cohort. Non-finite scores and retained counts should be explicit. The spatial-control checkpoint also needs config/cohort compatibility checks with the future checkpoint.
8. **Horizon units need care.** Candidate horizons are 2, 4, and 8 patches. For the present 64-frame/16-patch model these are fractions of a temporally resized clip, not fixed seconds into a natural-time future.

For a later study, a defensible first forecasting experiment would freeze a prefix-only preprocessing contract; hide all unavailable joints at future times; ensure the predictor has no future inputs; construct persistence from the same prefix information; compare against simple dynamical and static controls; predeclare horizons in physical time where possible; and evaluate source-aggregated errors with uncertainty. These are proposed experiments, not completed work.

## 4. Laterality notebooks: what the archived numbers mean

### 05a, signed-axis readout

The historical cohort is 626 encoder-trained sequences from 93 uploads; a 642-sequence/94-upload robustness cohort adds 16 rows. Both differ from the current 639-sequence/97-upload QC cohort. The historical representation is transductive even though the readout groups by upload. Historical embedding checkpoint fingerprint `7d13841a…` must not be confused with the current fold-local checkpoint.

| Historical E1 lane | One-partition \(R^2\) | Repeated-partition mean | Reported 95% partition-stability interval |
|---|---:|---:|---|
| Learned | 0.268394 | 0.198190 | [0.175460, 0.220920] |
| Untrained | 0.228682 | 0.244785 | [0.214297, 0.275274] |
| Missingness only | 0.162295 | 0.202191 | [0.172928, 0.231455] |
| Global pooled | 0.106662 | 0.101190 | [0.088565, 0.113815] |
| Raw target components | approximately 1 | not repeated | direct target-recovery oracle |

The historical learned mirror slope is −0.703492; one-partition sign consistency is 0.548387; repeated sign consistency is 0.576344 [0.548212, 0.604476]. All three registered historical gates fail. The learned-minus-untrained ordering reverses between the one-partition and repeated-partition means. The alpha sweep ranges from −2.035863 at 0.001 to 0.268395 at 1,000. The 642-row robustness view gives learned 0.240901, untrained 0.189972, missingness 0.046562, pooled 0.102620, mirror slope −0.627364, and sign consistency 0.553191.

These are useful historical warnings about unstable probes and potential nuisance signal; they are not current estimates. Repeated repartitioning of the same cohort produces dependent estimates. The reported intervals assess partition stability, not population sampling uncertainty, and interval overlap or nonoverlap is not a valid test of a population effect. Failure of one readout and selected gates does not prove that the entire encoder contains no laterality information.

The target is a signed difference in observed coordinate-motion statistics. It is mechanically computed from the same poses being encoded. Near-perfect regression from its own components is a self-consistency oracle, not a competitive clinical baseline. A missingness-only mean near the learned mean suggests an alternative explanation worth testing; it does not prove the learned model uses that shortcut or identify its cause.

### 05c, constrained readout

For any deterministic feature map \(A\) and involution \(M^2=I\), \(\Phi(x)=A(x)-A(Mx)\) is odd and \(\Psi(x)=A(x)+A(Mx)\) is even. A homogeneous linear map \(w^\top\Phi\) is exactly odd. A shared nonlinear map difference \(m(A(x))-m(A(Mx))\) is also exactly odd.

| Historical E2 lane | One-partition \(R^2\) | Repeated mean and partition interval | Mirror slope |
|---|---:|---|---:|
| Free learned ridge | 0.268394 | 0.198190 [0.175460, 0.220920] | −0.703492 |
| Odd learned projection | 0.314076 | 0.273368 [0.252685, 0.294051] | approximately −1 |
| Odd untrained projection | 0.120798 | 0.219322 [0.175036, 0.263609] | approximately −1 |
| Even learned projection | 0.014857 | not repeated | +1 |
| Shared-map odd MLP | 0.243311 | not repeated | approximately −1 |
| Free two-pass MLP | 0.047275 | not repeated | −0.430357 |

Odd learned sign consistency is 0.567742 [0.553330, 0.582154]. Its difference from the random odd mean is 0.054045, but the stability intervals overlap. The raw component oracle remains approximately 1. The current experiment is blocked.

Exact oddness requires attention to the complete pipeline: a fitted intercept or centering that is not handled symmetrically can break it. A regression slope of −1 alone does not prove \(f(Mx)=-f(x)\), because an intercept can remain; directly check residuals and nonzero output variance. A zero predictor satisfies oddness without usefulness. A symmetric feature cannot consistently represent a nonzero odd target on both members of every mirror pair, but it may correlate with that target on an asymmetric original-only sample. Thus the historical statement “symmetric features cannot predict the target” needs this qualification.

Global pooling of a nonequivariant, joint-position-aware encoder is not automatically mirror-invariant. Removing explicit left/right columns does not remove all laterality information from already contextualized features. A successful pooled control warrants investigation, not an impossibility proof or a causal attribution to acquisition artifacts.

### 05d, constrained encoder and augmentation

The construction \(E'(x)=\frac12[E(x)+\sigma E(Mx)]\), where \(\sigma^2=I\), satisfies \(E'(Mx)=\sigma E'(x)\) by direct algebra. This known symmetrization does not require learning and is not evidence that JEPA learned the symmetry. A free readout from all channels need not be odd.

Historical E3a records zero token-equivalence/difference-block/sum-block residuals. The odd learned block has repeated \(R^2=0.253349\) [0.226408, 0.280289], and the odd random block has 0.249558 [0.200850, 0.298267]. One-partition values are 0.288084 and 0.186879, respectively. The full free readout gives 0.276094 and mirror slope −0.767332. The near-identical learned and random repeated means strongly limit a learning-benefit interpretation even within that historical cohort.

Historical E3b retrains one normal-only cohort, 270 sequences/29 uploads, for 300 epochs with a single seed. Reflection probability 0 versus 0.5 gives final JEPA values 0.720310 versus 0.806860. The final objective value is not a performance score across independently calibrated targets.

| Historical E3b arm | Free-readout repeated \(R^2\) | Partition interval | Free mirror slope |
|---|---:|---|---:|
| Original Stage-0 | −0.022007 | [−0.074934, 0.030920] | −0.553414 |
| Reflection off | −0.048665 | [−0.112947, 0.015617] | −0.753366 |
| Reflection on | 0.062343 | [0.011182, 0.113503] | −0.509636 |
| Untrained | 0.078260 | [0.012714, 0.143806] | −0.818023 |

The corresponding odd-projection repeated means are 0.162989, 0.072585, 0.138468, and 0.200319. They are all exact-odd by construction. The observation that a random model has a slope closer to −1 is a warning that slope alone is not a learning metric. These single-seed transductive values cannot support the general statement that reflection augmentation fails to induce symmetry across architectures, objectives, or data.

Relevant replacement code exists under `neurips-laterality/laterality/`, but that does not repair the notebook gates automatically. The current gates look for absent `work/experiments/e1_laterality_hardened.py`, `e2_equivariant_readout.py`, `e3a_frame_averaging_wrapper.py`, and `e3b_probe_and_merge.py`. They also glob all final and stage checkpoints and demand exactly one match; the saved result reports six matches. Even supplying scripts and enabling the flag would require fixing this selection to identify an exact final objective and validating its contract. The retained hard-coded numerical assertions expect historical results and should not govern a new scientific rerun.

## 5. Notebook 05b: executed simulation, not empirical evidence

Its four hard-coded scenarios are:

| Scenario | Learned \(R^2\) | Raw | Random | Pooled | Sign fraction | Mirror slope |
|---|---:|---:|---:|---:|---:|---:|
| F1 clean example | 0.44 | 0.50 | 0.06 | −0.02 | 0.83 | −1.02 |
| F2 nonflipping example | 0.41 | 0.50 | 0.05 | 0.01 | 0.80 | −0.25 |
| F3 null example | 0.12 | 0.48 | 0.08 | 0.00 | 0.55 | −0.60 |
| F4 nuisance example | 0.47 | 0.50 | 0.06 | 0.46 | 0.82 | −0.90 |

All are invented numbers. “Above raw” is an incorrect label for passing the rule \(R_A^2\ge0.8R_B^2\): the illustrated learned scores are below raw scores. Thresholds of 0.05 improvement, 80% of the component oracle, 75% sign correctness, and slope within [−1.25, −0.8] are declared decision conventions, not validated clinical cutoffs. Describing them as preregistered requires dated independent evidence; prose saying “preregistered” is insufficient on its own.

The plotting generator also does not produce the labelled prediction \(R^2\). It sets \(\hat y=y+\epsilon\) with variance \((1-r)/r\), a relationship motivated by squared correlation. For unit-variance truth, prediction \(R^2\) tends toward \(2-1/r\), not \(r\). Reproducing the exact 40-point seeded code gives actual prediction \(R^2\) values 0.006228, −0.333470, −12.371052, and −0.286188 for labels 0.44, 0.41, 0.12, and 0.47. This is a simulation-label issue; none of these figures belong in an empirical results section.

The multiview fixture contains six artificial subjects, three views, and 18 artificial clips. Its reported correlation between 54° and 126° is +1.000. Both real external loaders return no data by default and raise `NotImplementedError` when given a path. No CASIA-B or OU-MVLP-Pose run occurred.

A camera rotation or view swap is not equivalent to anatomical coordinate reflection plus left/right landmark relabeling. The historical target sums coordinatewise temporal standard deviations, which is generally not invariant to camera rotations. The synthetic fixture and the statement that no suitable clinical public cohort exists do not establish either physical equivalence or a complete literature survey. Omit these assertions from the GenAI4Health draft.

## 6. Historical artifacts most likely to be accidentally promoted

`anchor_guard_results.json` stores a retired cosine curve near 0.700151, 0.502113, 0.396213, and 0.296638, and an AnchorGuard final value 0.478629. Its feature-standard-deviation gate fails (final 0.251647 versus threshold 0.35), and retention fails its 0.85 gate. It also contains attractive but transductive downstream macro-F1 values, including five-class 0.817162 and binary 0.970793. These are neither comparable to protocol-v2 estimates nor evidence of a current successful repair. Its gate fields and downstream numbers are internally awkward enough that it should remain excluded rather than rationalized.

`predictive_surprise_results.json` stores a retired 626-sequence/93-upload spatial-predictor probe. Normal-vs-condition AUROCs are 0.501916 (Parkinson's), 0.408046 (stroke), 0.460591 (myopathic), and 0.547893 (cerebral palsy), with broad intervals crossing 0.5. It also stores a small surprise/missingness correlation 0.080205 and old rollout demonstration cosines. These are transductive, use a predictor not established as future-trained, and are not outputs of the current blocked notebook. Do not cite them as current forecasting, calibrated surprise, or disease-discrimination results.

The old and new studies change cohort, checkpoint, preprocessing/evaluation contract, aggregation, and sometimes model architecture. Reporting their differences as a measured impact of removing leakage would be a confounded comparison. A proper leakage ablation requires matched data, model, training budget, and evaluation units.

## 7. Concrete contribution and presentation recommendations

Use the verified normal-reference audit as a **case study in evidence requirements for health-facing predictive representations**. A concise figure can show the same five validation source means with their clip counts, and two final aggregate values. A second panel can show the development source-equal trajectory, clearly marking annotation-driven curriculum stages. The caption should name fold 0/seed 42, five validation uploads, the absence of uncertainty across runs, and the fact that aggregation alone changes the first comparison.

Recommended wording:

> On the five normal-annotated validation uploads, the mean cosine between Stage-0 and final sequence embeddings was 0.701 when uploads received equal weight and 0.889 when clips received equal weight. One upload supplied 60 of the 64 clips. This descriptive reaggregation changes only the weights on the same per-sequence similarities. It illustrates how clip concentration can obscure heterogeneity across sources; it does not measure clinical decline or establish catastrophic forgetting.

For the test statement:

> The selected final checkpoint had a source-averaged normal-reference cosine of 0.850 on seven held-out uploads. This single-fold representation audit does not demonstrate normative clinical validity or a successful consolidation intervention.

For future prediction:

> Spatial latent infilling supplies the current representation-learning objective. Causal future prediction remains untested: the required future-trained checkpoint is absent, and the candidate protocol also needs stricter controls on future information before it could support a forecasting claim.

The clinical-trust contribution can be a carefully argued distinction among reliable measurement, discriminative association, predictive calibration, and action authorization. The completed experiments bear on the first two at most; they implement no clinician-facing agent, monitoring intervention, validated uncertainty threshold, or improved patient outcome. A missingness baseline is evidence that the available acquisition information has predictive value, not proof that the learned model uses a shortcut. State these boundaries in the abstract as well as the discussion.

## 8. Adversarial checks and remaining work

| Tempting claim | Adversarial objection | Required edit or evidence |
|---|---|---|
| “Representation retention is 89%.” | Cosine is not a percentage; clip concentration determines the summary. | Report 0.889 clip-equal and 0.701 source-equal as distinct estimands. |
| “The encoder catastrophically forgets normal gait.” | Coordinate drift need not imply information loss. | Say representation drift; add retained-task performance before using forgetting. |
| “Consolidation repairs drift.” | Only one current final objective has a score. | Remove repair claim; require a matched intervention comparison. |
| “This is a health world model.” | Current objective is spatial infilling; forecast gate is blocked and future inputs leak in the proposed code. | Specify latent representation learner; describe forecasting as future work. |
| “Surprise measures confidence.” | Cosine discrepancy is not calibrated predictive uncertainty. | Remove confidence/clinical-risk wording; require explicit calibration and decision evaluation. |
| “Laterality is learned by JEPA.” | Current probes are blocked; exact oddness can be imposed on random features. | Exclude archived metrics; separate construction from learned utility. |
| “Missingness proves a shortcut.” | Predictive nuisance features do not identify the learned pathway. | Use possible nuisance signal; require intervention or incremental-control analysis. |
| “Source-disjoint means unseen patients.” | Reuploads or multiple uploads of one person are unresolved. | Say unseen uploads, not unseen patients. |
| “Five-fold results support generalization.” | Five folds are planned; only one has current model evidence. | Explicitly state one completed fold/seed. |

Before a future empirical extension, strengthen notebook 08's report to include the selected checkpoint and Stage-0 hashes, model seed, all candidate scores, and per-source retention rows. The current report omits some of these fields despite the surrounding provenance contract. Cached-embedding metadata binds checkpoint/manifest/split and sequence order but not the preprocessing implementation or pose-file content hash; a changed preprocessing function could reuse stale embeddings. The read-only checks in this audit establish consistency with the stored caches and checkpoints, not an independent re-extraction and forward-pass reproduction from raw video.

The stage gate checks fewer fields than its explanatory prose suggests, although the audit independently verified the actual current stage role contracts. The candidate-stage-0 path is loaded without explicitly checking its recorded sidecar SHA at that point; this matters if multiple future objectives are introduced. The supervised-objective helper checks `group_loss`, whereas current producer metadata uses `group_loss_enabled`; `label_free` currently disambiguates the single objective, but prospective ablation labeling should use the producer schema directly. The notebook's “test once” sequencing describes the code path; immutable experiment logs would be needed to demonstrate a single use across arbitrary reruns.

These are implementation-hardening recommendations for later experiments. They do not require withholding the verified descriptive reaggregation, provided the paper states its source, scope, and limitations as above.
