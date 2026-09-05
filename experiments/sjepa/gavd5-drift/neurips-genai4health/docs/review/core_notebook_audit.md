# Core notebook and artifact audit for GenAI4Health

Audit date: September 5, 2026. Scope: all source cells and saved numerical outputs in numbered notebooks 00–07; `evaluation_protocol.py`, `pose_geometry.py`, `pose_cache.py`; current protocol-v2 manifests, checkpoint sidecar and checkpoint tensors, stage histories, readout predictions/metrics, and temporal-probe artifacts. No training or source-notebook modification was performed. Cell numbers below are zero-based notebook cell indices. Shared setup/model cells were checked against their counterparts; scientific conclusions are tied to the producing notebook and saved artifact, not an older tutorial description.

## Executive assessment

The defensible submission is a small, exploratory case study about evidence requirements for learned movement representations. The strongest completed results are the dated data funnel and the contemporaneous comparison of three feature lanes on the same 20 held-out source videos. Raw kinematics obtained macro-F1 0.440513, compared with 0.292424 for the learned latent and 0.251111 for missingness alone. The saved per-source predictions reproduce these metrics exactly. This supports the statement that the learned representation did not outperform this direct baseline in the observed split. It does not support a general ranking of methods, a statistically established difference, diagnostic validity, or an operational health agent.

One protocol-v2 outer fold and one model seed exist. The five-fold registry is complete, but the experiment grid is not. Source IDs, checkpoint and stage bytes, and relevant downstream artifacts agree on the current lineage. Several manuscript descriptions nevertheless disagree with the executed code. The objective, tokenization, predictor, preprocessing, validation weighting, readout refitting, and claims about opening test data must be corrected. These are substantive scientific details, not copy-editing issues.

The latest BrainBodyFM draft is already much more restrained than earlier papers, but its equations and some pipeline statements still describe an earlier implementation. The GenAI4Health manuscript should report the audited current implementation and its limits explicitly. A paper about trust loses credibility if its own provenance claims overstate the guarantees that are implemented.

## 1. Verified current evidence and lineage

### Data population

| Gate | Sequences | Source videos | Annotated frame rows |
|---|---:|---:|---:|
| Raw annotation inventory | 666 | 103 | 140,641 |
| Dated metadata-public inventory | 657 | 100 | 137,690 |
| Decoded-span eligible | 655 | 98 | 135,804 |
| Pose-QC eligible | 639 | 97 | 134,259 |

The first three rows were recomputed directly from protocol manifests. The last row follows the QC ledger joined to the manifest. Frame totals are sums over annotation sequences; they are not demonstrated counts of globally unique source frames if sequences overlap. Sources are uploads, not verified independent people.

| Folder annotation | Raw sequences / sources | Metadata-public | Decode eligible | Pose-QC eligible |
|---|---:|---:|---:|---:|
| Normal | 291 / 32 | 291 / 32 | 290 / 31 | 276 / 30 |
| Parkinson's | 47 / 11 | 47 / 11 | 47 / 11 | 46 / 11 |
| Stroke | 76 / 19 | 75 / 18 | 75 / 18 | 74 / 18 |
| Myopathic | 188 / 30 | 184 / 29 | 183 / 28 | 183 / 28 |
| Cerebral palsy | 64 / 11 | 60 / 10 | 60 / 10 | 60 / 10 |

The additional decoded exclusions are one terminal-short myopathic source (`n93bgWhLZk4`, 228 decoded frames versus annotation through 458) and one retryable normal-source acquisition failure (`hGNKzkCF4J8`). The latter must not be described as permanent source disappearance. Sixteen decoded sequences fail the declared 0.50 neurologic-joint coverage criterion; fourteen are normal, one Parkinson's, one stroke. The current pose inventory has 655 structurally ready locked entries plus one excluded legacy extra. All 655 locked caches have `resolution_safe_geometry=False`; the newly implemented resolution-explicit schema was not used to recompute these scientific arrays.

The metadata snapshot is named September 4 in local time, while its entries use UTC timestamps on September 5. State the timezone when a precise audit date matters.

### Fold 0 after QC

| Role | Sequences | Sources | Normal | Parkinson's | Stroke | Myopathic | Cerebral palsy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 377 | 59 | 156 / 18 | 28 / 7 | 44 / 11 | 111 / 17 | 38 / 6 |
| Validation | 131 | 18 | 64 / 5 | 9 / 2 | 15 / 4 | 35 / 5 | 8 / 2 |
| Test | 131 | 20 | 56 / 7 | 9 / 2 | 15 / 3 | 37 / 6 | 14 / 2 |

Condition cells give sequences / sources. The source registry originally assigns 60 / 20 / 20 sources in every fold. Later acquisition/QC attrition does not redraw those assignments. All 20 fold-0 test sources survive the gates; this unusually complete test retention should not be assumed for other folds.

Verified lineage:

- Manifest digest: `7fd559e5105b11011a3e5c194b7ccc29729c56491c424745834df39884123b5a`.
- Split digest: `ff3518b87b1d1fa7d95efb1aea1711773137a21699967cb8015edb8d845ccbe1`.
- Final checkpoint SHA-256: `f510be2a0453dda0d6698780fcd998835db213e39d94e345b227a5ed8ec648ac`.
- Split version: `brainbody-source-constrained-v2`; split seed 20260904.
- Checkpoint contract: `sjepa_fold_contract_v2`; outer fold 0; model seed 42; objective `jepa_vicreg`.

The final checkpoint and all five stage checkpoint byte hashes match their sidecar entries. Its serialized access log contains 377 train and 131 validation sequence IDs, with zero test IDs. This corroborates the notebook-04 training access path. It does not prove test outcomes were never inspected in earlier projects or after later notebook reruns.

Authoritative artifacts are under [work/artifacts/real](../../../work/artifacts/real/): `evaluation_protocol/{raw_sequence_manifest,metadata_public_sequence_manifest,eligible_sequence_manifest,source_split_registry,pose_qc_eligibility_outer_fold_0}.csv`, `checkpoints/sjepa_outer_fold_0_seed_42_jepa_vicreg.{pt,json}`, `sjepa_outer_fold_0_seed_42_jepa_vicreg_history.csv`, `readout_outer_fold_0_seed_42_jepa_vicreg_{metrics,source_predictions}.csv`, and the temporal files in `fold_evaluation/outer_fold_0/`.

## 2. What each notebook actually does

### Notebook 00: first-principles tutorial, not the current trained architecture

The tutorial explains skeleton token masking, the EMA teacher, a centered latent cross-entropy, geometric augmentation, and the distinction between same-clip infilling and future prediction. Its smoke trace uses two synthetic 32-frame sequences, a 32-dimensional encoder, and 57 masked tokens per sample. The trace reports prediction/target shapes `(2,57,32)`, finite loss, and zero target-encoder gradients. Realized global masking is 0.215909; the eligible-joint ratio is 0.59375. These are software checks, not health results.

Its implementation flattens four frames of coordinates to a 12-value patch, removes hidden tokens from the online encoder, and uses a Transformer predictor. It defines teacher centering, soft-target cross-entropy, geometric transforms, and a cosine EMA schedule. Its prose also describes two-view VICReg and a label-aware centroid loss. Those choices differ materially from the current notebook-04 training run. The tutorial and its architecture SVG cannot be treated as a canonical method specification for protocol v2.

The useful paper contribution here is a clear claim boundary: random target tokens are from the observed clip, so latent infilling alone is not evidence of forecasting, environment simulation, planning, or agentic behavior.

### Notebook 01: data gates and fixed source roles

Scans all five annotation directories, validates sequence and source identities, joins a dated metadata snapshot, constructs a deterministic source allocation, and attaches download/decode status afterward. The split algorithm works on source-level groups with exact per-condition source quotas differing by at most one; constrained optimization minimizes sequence-count imbalance. Five outer folds are created. Each outer development set receives a four-way inner partition, but only one inner validation partition (`outer_fold % 4`) is selected. Thus this is a five-fold outer design with one grouped validation split per fold, not a full four-fold inner hyperparameter cross-validation.

The download logic distinguishes terminal source attrition from bounded retryable acquisition failure. It checks a video opens, has positive FPS, has enough frame count, and can decode first and last required frames. It does not by itself decode every annotated frame; notebook 02's extraction/validation is a separate gate. The source census, split table, and acquisition status are appropriate main-text evidence.

The manifest fingerprint includes condition, source ID, sequence ID, first/last frame, and annotated-frame count. It does not hash complete annotation CSV bytes, bounding boxes, decoded-video bytes, availability status fields, or pose tensors. A matching digest is a precise agreement about selected fields, not a complete content-integrity guarantee.

### Notebook 02: pose extraction and cache audit

Defines a fixed MediaPipe Pose Landmarker Lite extraction in VIDEO mode, one pose, confidence thresholds 0.45, CPU delegate, and 15% padded annotation crops. It preserves the original 1-based annotated frame IDs and converts to zero-based decoder seeking. Landmarks are stored as normalized image x/y, relative z scaled by crop width, and visibility. These are monocular detector coordinates, not measured metric 3D joint locations.

`pose_geometry.py` normalizes boxes in the annotation's stated image dimensions before projection to the currently decoded rendition. It validates finite, positive dimensions and bounds. The new cache path stores frame sizes and normalized crop bounds. The preview fixes a real coordinate-space mismatch: legacy crop pixels from a 640×360 rendition were overlaid unchanged on a 1280×720 rendition. The example is a preview geometry correction, not evidence that all old trajectories were recomputed or retrospectively crop-validated.

`pose_cache.py` validates identity, exact annotation frame arrays, shape, positive FPS, expected pose-model hash, visibility threshold, and split provenance before migrating metadata. It preserves the scientific arrays. Its current schema allows legacy caches without resolution fields. It does not compare a complete source-video hash, exact crop coordinates against annotations, or all landmark finite/range conditions. Avoid claiming that structural readiness establishes extraction accuracy.

The recorded execution reused 655 cached files, with no extraction needed and no retries. Earlier migration described in markdown updated 641 legacy metadata records; 14 already satisfied the new provenance schema. QC uses the fraction of the 12 allowed joints whose visibility is at least 0.45, retaining a sequence when that fraction is at least 0.50. This is a technical detector-coverage rule, not expert assessment of gait quality. QC visibility does not itself require finite xyz; training validity subsequently adds that requirement.

Fixed extraction may legitimately cache all roles because no GAVD model is fitted in this stage. The relevant claim is no test-dependent fitted preprocessing or model selection, not that no test image was ever decoded. One preview and structural checks are also not a complete visual verification of all sources.

### Notebook 03: target-mask invariants

The allowed indices are `[11,12,23,24,25,26,27,28,29,30,31,32]`: bilateral shoulders, hips, knees, ankles, heels, foot tips. An internal MS/PD feature mapping motivates the choice; it does not clinically validate the whitelist for the five GAVD folder categories. A four-sample synthetic validity test gives a 0.216 global target fraction and approximately 0.594–0.600 eligible-target fractions, with no forbidden targets and reproducible seed behavior.

The mask samples uniformly from valid allowed tokens, without motion magnitudes. It uses a common target count per batch based on the smallest valid-eligible count. A nominal mask fraction 0.60 is therefore not 60% of all 528 tokens and is not exactly 60% of each sample's eligible tokens. The theoretical allowed-joint ceiling over all tokens is 12/33. The current training implementation reproduces this mechanism, though it is a separately copied function rather than literally importing notebook 03's function.

The notebook says the mapping hash should be frozen before folds, but its code checks text content and indices without writing that mapping-file hash into the current checkpoint contract. Describe a fixed engineering prior, not externally registered clinical specification.

### Notebook 04: actual protocol-v2 training

It reads the fixed metadata registry and authoritative QC ledger, validates role identity, loads train and validation tensors only, and trains a cumulative normal-first curriculum. All previously introduced conditions remain eligible for replay. This is cumulative replay under a fixed condition order, not a no-replay continual-learning setting or longitudinal progression within patients.

Current architecture: 64 resampled frames; four-frame segments; 16×33=528 tokens. Each four-frame joint patch is averaged to three coordinates before a linear projection to width 64. Learned temporal and joint position embeddings are added. The encoder has two pre-norm Transformer blocks with four heads, feed-forward width 128, GELU, and zero dropout. Hidden token contents become a learned mask vector but remain in the encoder sequence. The predictor pools visible valid contextual tokens to one vector, adds a learned per-position vector, and applies an MLP 64→128→64. A separate projector MLP 64→64→64 supplies variance/covariance regularization. There is no current coordinate decoder, Transformer predictor, geometric-view augmentation, teacher center, or sharpened probability matching.

Verified parameter counts from saved tensor states: online encoder 70,528; EMA target encoder 70,528; predictor 50,368; projector 8,320. The primary optimizer updates 129,216 parameters. An unused five-class linear head with 325 parameters is also serialized, but is not in the primary optimizer. The representation is a compact model; calling it foundation scale is unsupported.

For masked valid tokens, the exact primary objective is

\[
L=L_{\mathrm{SmoothL1}}(\widehat Z_M,\operatorname{sg}(Z_M)) + 0.10\,v(g(h)) +0.01\,c(g(h)).
\]

Here `h` is visible-valid online token pooling; `v` is mean `relu(1-sqrt(var+1e-4))` over projected dimensions, with biased batch variance; `c` is the sum of squared off-diagonal sample covariance divided by projection width. Covariance uses denominator B−1. No two-view VICReg invariance term is present. Say “VICReg-inspired variance/covariance regularization.” The optional supervised ablation adds `0.10 × cross_entropy(group_head(h), condition)`. The current default has zero group-loss weight. Existing draft equations using 0.05 VICReg and 0.25 group loss are incorrect for these artifacts.

The objective does not use condition labels, but the schedule does. The correct phrase is “label-free objective with an annotation-informed cumulative curriculum.” Source stratification and supervised readout fitting also use labels. Uniform-source then uniform-sequence sampling avoids upload-length weighting during gradient updates; it does not equalize condition counts because condition source counts differ.

Current source defaults are AdamW learning rate 0.001, weight decay 1e-4, gradient clipping 1.0, EMA decay 0.996, batch size 32, 100 optimizer steps per epoch, 20 epochs per stage, and nominal mask fraction 0.60. The 100-row history confirms 20 epochs in each of five stages. The checkpoint does not persist batch size, steps per epoch, learning rate, EMA, mask fraction, optimizer state, or all environment overrides. These defaults should not be represented as independently verified runtime values until the run environment is recovered.

| Stage | Fitting sources | Selection sources | Selected epoch (zero-based) | Saved best validation objective |
|---|---:|---:|---:|---:|
| Normal only | 18 | 5 | 7 | 0.151007153094 |
| Add Parkinson's | 25 | 7 | 0 | 0.129813320935 |
| Add stroke | 36 | 11 | 0 | 0.122365179161 |
| Add myopathic | 53 | 16 | 4 | 0.114262960851 |
| Add cerebral palsy | 59 | 18 | 0 | 0.125512136519 |

Selection uses the arithmetic mean of per-batch validation losses, not an equal-source mean. Sequence IDs determine batching; the final smaller batch receives equal batch weight. Mask RNG changes with epoch, so validation objective variation includes new mask draws. Comparing loss across stages also changes the population and the target encoder; decreasing loss is not itself preserved clinical or temporal function.

After each stage the best model weights are restored, but AdamW moments are not rewound to that best epoch. Later stages therefore start with best-epoch parameters and last-epoch optimizer moments. This is an implementation detail to document and improve before a new frozen experiment, not evidence that saved predictions are numerically wrong.

### Notebook 05: descriptive latent geometry

Loads and verifies the fold checkpoint, freezes the EMA encoder, and constructs 256 features: global token mean/std followed by neurologic token mean/std. It creates the normal reference from equally weighted training-source embeddings and derives scale from training sources. Validation and test are reported separately. Same-source nearest neighbors are explicitly excluded.

| Role | Sources | Mean standardized distance to train-normal reference | Condition silhouette |
|---|---:|---:|---:|
| Train | 59 | 15.383401 | −0.108600 |
| Validation | 18 | 19.494282 | −0.264261 |
| Test | 20 | 16.719585 | −0.193533 |

The shown validation query is cerebral palsy but retrieves a myopathic training source at cosine 0.988058; the shown normal test query retrieves a normal source at 0.982134. These single examples do not measure retrieval accuracy. Negative silhouettes do not establish meaningful class separation. High cosines alone may reflect anisotropy, common offsets, or limited diversity and cannot validate clinical similarity. This is secondary diagnostic evidence, not an abstract headline.

### Notebook 06: same-cohort baseline comparison

Uses three lanes: 256-dimensional frozen latent; 97-dimensional missingness vector (33 joint means plus 64 time-frame means); and 144-dimensional raw features (12 joints × three coordinates × four position/velocity moments). Raw “velocity” is a difference on the 64-frame resampled index, not calibrated speed in meters/second. All lanes share normalized pose inputs and the same source roles.

The readout is `StandardScaler` plus class-balanced logistic regression, `max_iter=2000`, seed 42. Each training and validation source is represented by its mean feature vector. `C∈{0.1,1,10}` is selected on validation source macro-F1; ties favor smaller C. Final scaler and classifier are refit using train plus validation source means (77 sources), not train alone. This is a legitimate conventional final refit after selection, but must be disclosed accurately.

There is an aggregation mismatch: validation scoring applies the classifier once to each source's mean feature; test scoring applies it to each sequence and averages the resulting probabilities within source. Logistic regression plus softmax is nonlinear, so these estimators are not interchangeable. No test-dependent hyperparameter tuning is shown, but a future run should use one aggregation rule throughout and evaluate after freezing that rule. Do not silently recompute a replacement score and portray it as the original prespecified experiment.

| Lane | Selected C | Validation source macro-F1 | Test accuracy | Test balanced accuracy | Test macro-F1 |
|---|---:|---:|---:|---:|---:|
| Learned latent | 1 | 0.132143 | 0.300000 | 0.257143 | 0.292424 |
| Missingness only | 1 | 0.318974 | 0.300000 | 0.247619 | 0.251111 |
| Raw kinematics | 10 | 0.339394 | 0.500000 | 0.442857 | 0.440513 |

All test values use 20 sources. Latent versus raw macro-F1 differs by 0.148088578089 (14.81 percentage points); this is an observed difference, not a significance claim. Raw source accuracy is 10/20 and latent/missingness accuracy 6/20. All three lanes miss every stroke source (0/3 recall). The latent lane also misses both cerebral-palsy sources (0/2 recall). Small class counts make individual errors influential. The saved source-prediction CSV omits probabilities, precluding probability calibration and some uncertainty analyses from that CSV alone.

Sequence-level values are secondary dependent-observation diagnostics: latent macro-F1 0.327030, missingness 0.208486, raw 0.409580 over 131 sequences. Do not substitute them for source-level scores.

The archived accuracies/F1s in cell 3 are explicitly retired. Historical sequence splits had 16 or nine overlapping training/test source IDs and encoder exposure to eventual evaluation rows. The historical cohorts, architecture, and objectives also differ. Thus the change from an old approximately 0.75 F1 to current 0.292 is not an identified effect of removing leakage. Only the current three-lane comparison is controlled for cohort and split.

### Notebook 07: temporal proxy recoverability

Strictly loads the current encoder state using a compatible wrapper, avoiding its older unused architecture definitions. Compares three deterministic 256-dimensional pooling lanes: A global/neuro mean and std; B replacing the final neuro std block with a signed temporal moment; C four ordered temporal-bin means. Ridge probes use 25 log-spaced alpha values from 1e-4 to 1e4. Each alpha is fitted on sequence-level train rows, selected by source-equal validation MAE, and refitted on train+validation sequences. Training itself is sequence weighted, even though scoring is source equal.

Targets are pose-derived proxies from the same clips: normalized time of maximum ankle separation, log second-half/first-half lower-limb motion ratio, and bounded circular cross-correlation lag between ankle y coordinates. They are not adjudicated clinical gait-event timings. Phase lag uses `np.roll`, so wraparound can affect the result. Targets use original detector coordinates and NaN summaries; they do not uniformly apply the visibility gate used in feature preprocessing. The code ignores absolute `frame_numbers` when calculating targets, assuming sequence rows constitute the relevant temporal grid.

Unlike notebooks 04–06, preprocessing here interpolates internal low-visibility gaps of up to four original frames before centering/scaling and resampling. Thus it is an additional preprocessing variant, not a pure pooling-only comparison to notebook 06. Within notebook 07 all three lanes do use identical prepared tensors.

| Proxy | Lane A R² / MAE | Lane B R² / MAE | Lane C R² / MAE |
|---|---:|---:|---:|
| Peak phase | 0.173011 / 0.092086 | 0.052344 / 0.091527 | 0.317745 / 0.075263 |
| Energy ratio | 0.105126 / 0.141453 | 0.053575 / 0.148818 | 0.176175 / 0.137220 |
| Phase lag | −0.070535 / 0.109191 | −0.053989 / 0.107476 | −0.052171 / 0.107240 |

All use 20 test sources. Every phase-lag probe selects alpha=10,000 at the search-grid ceiling. This suggests very strong shrinkage and does not establish useful phase-lag recovery. Negative R² means squared error exceeded the oracle constant equal to the mean observed test target; it is not directly an evaluated train-mean baseline.

The permutation check gives mean/std error 4.77e-7 and signed-moment change 0.162. It permutes already contextualized token tensors together with their masks. This proves the symmetry of the pooling operator. It does not prove that the entire encoder representation is invariant to permuting input frames: positional encoding and contextual attention already contain temporal information.

The code materializes all roles, including test, in cell 15 and encodes all role tensors in cell 17, before alpha tuning in cell 19. No test values enter the fitting or alpha-selection expressions, but the literal claim “test tensors remain unopened until tuning completes” is false here. The report's one-pass language is a within-function convention, not an access-control mechanism.

## 3. Required manuscript corrections

| Existing or tempting statement | Audited replacement |
|---|---|
| `L_JEPA + 0.05 L_VICReg`; supervised `+0.25 L_group` | Smooth-L1 masked latent loss +0.10 variance +0.01 covariance; optional +0.10 condition cross-entropy, not executed in the primary run |
| Current pipeline is fully label-free | Prediction/regularization objective excludes labels; fixed curriculum and split stratification use folder annotations |
| Four-frame flattened patches and Transformer predictor | Four-frame coordinate means; pooled-context MLP predictor in the current checkpoint |
| Two geometrically augmented views and teacher centering | Absent from current training; do not borrow old tutorial mechanics |
| All notebooks use short-gap interpolation | Current training/latent/readout do not; temporal/retention consumers must be checked separately |
| Readout fitted only on outer-training sources | Tuned using train/validation and finally refit on both roles before test |
| Source-equal validation throughout | True for classifier/probe metrics, false for training checkpoint selection and temporal probe fitting |
| Test data are globally sealed and opened only once | Notebook 04 has no test tensor access; downstream code has no global one-use ledger and NB07 materializes test before tuning |
| Cryptographically complete dataset provenance | Digests bind selected manifest fields, source-role assignments, and checkpoint bytes; not all media/annotation/pose bytes |
| All pose caches have corrected resolution metadata | All 655 current locked caches are legacy geometry; current preview reprojects correctly; newly extracted future caches include extra fields |
| The protocol reduced F1 from historical 0.75 to 0.29 | Historical/current runs change many factors; report current controlled baseline comparison only |
| Mean/std pooling erases all time information | It is permutation invariant over already encoded tokens; the encoder can still encode time |
| Future world-model or clinical trust validated | Neither future prediction, agent performance, clinical trust, nor clinical validity is demonstrated by notebooks 00–07 |

The phrase “fully traced execution” should be narrowed to the available artifacts: architecture/objective/IDs and hashes are traceable; the complete runtime training configuration and optimizer trajectory are not serialized. The same distinction applies to “reproducible”: numerical score reconstruction from saved predictions is verified; full training reproduction has not been performed here.

## 4. Remaining methodological threats and proportionate improvements

1. **Source grouping is necessary but incomplete.** Same person across uploads remains possible. Use “source-held-out” and report source class counts; never substitute subject/patient independence. A source can also contain several people.
2. **Choose scientific scope before adding experiments.** A six-page trust/evaluation case study can honestly report the existing negative result and limitations. A method-performance paper requires all folds, multiple seeds, uncertainty, and missing baselines. Do not claim those experiments are done.
3. **A new frozen run should use consistent preparation and aggregation.** Centralize producer/consumer preprocessing, preserve a meaningful validity treatment, align source aggregation across validation and test, and persist the exact configuration. Any resulting retraining belongs to a new experiment version.
4. **Uncertainty should be paired and source based.** With only 20 fixed test sources, confidence intervals from resampling can describe this split's finite-sample variability but cannot account for training-seed or split variability. Paired intervals for lane differences are preferable to unpaired bars. Preserve all methods and negative findings.
5. **Needed controls include an untrained encoder and matched training schedules.** Raw kinematics and missingness are valuable; without an untrained-encoder probe, any latent signal is not uniquely attributable to self-supervised learning. Continued-normal and joint-training controls are needed for curriculum-specific claims. No health-condition progression follows from the chosen folder order.
6. **Missingness remains present inside the learned system.** Invalid tokens are not attention-masked in the encoder, and finite low-visibility coordinates can still affect centering, scale, and contextual features. Similar latent/missingness accuracy suggests caution, but does not prove the latent uses the same shortcut. Viewpoint/crop/visibility stratification and controlled missingness interventions would test that explanation.
7. **Runtime records need strengthening.** Save LR, batch/steps, mask/EMA settings, seed, deterministic flags, preprocessing version, pose and annotation content hashes, code commit, environment, optimizer states, and evaluation-selection record. Current provenance is useful but partial.
8. **Do not infer pre-registration retrospectively.** Comments call gates and lanes predeclared; the local artifacts do not independently establish when those choices were frozen relative to historical data inspection. Call the design fixed for this execution unless an earlier registration exists.

## 5. Recommended GenAI4Health evidence selection

Use the data funnel and the three-lane source-level comparison as the main quantitative story. The practical contribution is an explicit boundary between model fitting, source-held-out evaluation, data acquisition/QC, and clinical interpretation. Present that boundary as a prerequisite for future generative or agentic health systems, not as proof that such a system has been built.

Use one architecture/protocol figure reflecting notebook 04's actual model, one compact data-funnel figure or table, and one source-level comparison table/figure with all three lanes. If space permits, add the phase-lag negative result or the pooling invariance demonstration as a bounded representation diagnostic. Avoid using the more favorable historical accuracy or a large panel of exploratory geometric plots. Every included result should expose denominator, split, seed, and evidence status.

The extended abstract should focus on the completed main result, not a catalogue of every notebook. One direct sentence about the limits is stronger than a list of aspirational future capabilities: the evaluation concerns 20 held-out uploads with weak folder annotations, and no clinical or agentic outcomes were tested.

## Verification record

Read-only checks performed during this audit:

- Recomputed manifest and registry fingerprints and ran registry validation.
- Recomputed raw, metadata-public, and decoded cohort counts from saved manifests.
- Confirmed QC role counts and all 655 legacy geometry flags.
- Hashed final and five stage checkpoint files and compared with saved metadata.
- Loaded local checkpoint tensor states to count model parameters and inspect training-access IDs.
- Recomputed accuracy, balanced accuracy, macro-F1, and confusion matrices from each saved source-prediction lane; all match the metrics CSV to floating-point precision.
- Compared saved temporal metrics with notebook outputs and inspected alpha selection, preprocessing, weighting, and target definitions.

The diagnostic scripts are [audit_core_verify.py](../../../tmp/audit_core_verify.py) and [audit_core_inspect.py](../../../tmp/audit_core_inspect.py). They read existing artifacts only. They are audit aids, not replacements for the project's training/evaluation entry points.
