# Should we build out the external evaluation?

Assessment prepared September 7, 2026. This is a research plan, not an external result. No dataset was downloaded, no external model was run, and no permission record was created or changed.

## Recommendation

An external evaluation is worth pursuing when it tests whether the proposed learning method improves a meaningful movement outcome on genuinely held-out participants. That could strengthen this project substantially. Expanding the existing readiness display would add little scientific value, because Notebook 06 already explains and checks the administrative and manifest prerequisites clearly.

Keep [Notebook 06](../06_external_subject_gate.ipynb) as the entry check. Build a separate evaluation notebook once a dataset, representation, and primary outcome have been selected. A sensible next deliverable is a small, verified external-data adapter and baseline study; a large external training grid should follow only when those checks succeed.

The main opportunity is to connect the methods developed in Notebooks 07–10 to independent evidence. For a health-oriented contribution, that could be a clinician-rated gait score. For a stronger world-model contribution, it could be prediction of subsequent movement or an independently measured physical quantity, such as ground-reaction force. Repeating the coordinate-derived laterality formula on another dataset would improve the scope of the geometric test, but would leave its clinical meaning unresolved.

## What Notebook 06 already does

The current notebook discovers the optional configuration, checks dataset-scoped review records, and validates a subject-indexed manifest. Its [validator](../laterality/external.py) requires unique sequence identifiers and pose paths, existing files inside the permitted directory, a supported joint layout, and no subject appearing in more than one partition. Both training and test partitions must contain a subject; validation is optional. The [preparation guide](EXTERNAL_EVALUATION_GATE.md) already documents these requirements.

The local configuration check performed for this assessment found neither required setting, so the current state is **not configured / not run**. This is the expected optional state and does not invalidate the completed GAVD experiment.

Several boundaries are important when deciding what to implement:

- The validator checks file existence and manifest fields, but does not establish that a pose file contains the expected numerical arrays, valid timestamps, correct coordinate conventions, or usable observations.
- Distinct paths can contain duplicate data. A future adapter needs a content-duplication check as well as path checks, without attempting to identify people from their appearance or gait.
- A `resolved` record with a reference and date records a supplied determination; software cannot authenticate the underlying institutional decision. A release review can conclude that artifacts must remain private, and does not have to authorize public redistribution.
- The minimum of one training subject and one test subject is a structural requirement, not a scientifically adequate sample-size criterion.
- The gate currently accepts only `BlazePose33`. Writing that name on another joint layout would conceal a representation mismatch.
- A completely test-only external dataset does not satisfy the current train/test contract. A genuine zero-shot study would need a separately specified contract, rather than invented training records.

The current geometric thresholds also do not define external clinical success. The token-error margin of 0.10 is an operational tolerance under one feature transformation rule; the eight-common-token minimum is a support check; and the constructed-output tolerance of 10⁻⁶ tests numerical implementation. None is a validated threshold for gait impairment. A clinical score or physical prediction task needs its own outcome, baselines, and prespecified interpretation.

## What an external study would add

| Study design | Main additional evidence | What would remain unsupported |
|---|---|---|
| Repeat the current geometric test on another pose collection | Whether the measured reflection behavior persists under a new acquisition setting | Clinical usefulness, future prediction, and general world-model competence |
| Predict an independently rated gait score with subject-disjoint testing | Whether the representation supports the specified rating beyond simple baselines | Diagnosis, treatment benefit, or readiness for clinical deployment |
| Predict subsequent movement or independently measured forces from past pose | Whether learned features support a defined dynamics task beyond simple continuation | Action-conditioned control, intervention effects, or a general physical simulator |

The second and third designs offer the most useful next evidence. They also let us test the scientific ingredient that is supposed to help—such as gait-informed masking—rather than adding another dataset without a clear comparison.

Subject-disjoint evaluation and dataset transfer should be described separately. An encoder frozen after GAVD pretraining and tested through a newly fitted external read-out is a transfer experiment. An encoder trained only on the external training participants is a new within-benchmark learning experiment. Both can be informative, but they answer different questions.

External subject identifiers establish non-overlap within the supplied external partitions. They do not by themselves prove that none of those people appeared in GAVD, which lacks persistent participant identifiers. A strict claim of participants unseen across every stage of pretraining would require appropriate provenance or custodian confirmation. We should not infer identities to fill that gap.

## Two candidates with primary-source support

### CARE-PD: strongest first candidate for a clinical representation study

CARE-PD provides a clinically relevant alternative to the current coordinate-derived target. Its official project describes nine cohorts from eight clinical centers, with clinical gait-score and motion-representation benchmarks. This supports a concrete question: does a prespecified JEPA training change improve the usefulness of learned motion features for an independently supplied gait rating? [CARE-PD project](https://neurips2025.care-pd.ca/).

The dataset card describes custodian-provided subject identifiers, walking-trial identifiers, frame rates, and gait scores where available. It provides SMPL and other motion formats, including H36M, with subject-based split options. These are useful ingredients for evaluation, but are not the BlazePose33 input expected by the current suite. [Official dataset card](https://huggingface.co/datasets/vida-adl/CARE-PD/blob/main/README.md).

The benchmark paper reports 362 participants overall, but only four component datasets carry the gait-score endpoint. That total therefore cannot be used as the labeled sample size. Labels may describe a walking trial or a participant's medication session, depending on the cohort; the adapter must preserve that distinction. The published work already evaluates frozen motion encoders and engineered gait features, so using CARE-PD alone would not be the novelty. Our contribution would need to come from a controlled learning comparison or a useful explanation of its outcome. [CARE-PD benchmark paper](https://proceedings.neurips.cc/paper_files/paper/2025/file/bedc73979a95be7727af0c9a99c675ce-Paper-Datasets_and_Benchmarks_Track.pdf).

Cross-cohort identity bookkeeping needs attention even inside this collection. The paper describes T-SDU-PD as a subset of T-SDU. Different cohort names therefore cannot guarantee different participants, and an external test participant must not reappear in an unlabeled pretraining pool under another cohort label. Use the supplied provenance and custodian guidance to resolve this without attempting identification. [CARE-PD cohort descriptions](https://proceedings.neurips.cc/paper_files/paper/2025/file/bedc73979a95be7727af0c9a99c675ce-Paper-Datasets_and_Benchmarks_Track.pdf).

There is an access detail to resolve before choosing this route. The official pages use inconsistent license wording: the dataset-card metadata identifies CC BY-NC-ND 4.0, while some accompanying prose abbreviates the license as CC BY-NC 4.0. The dedicated terms also spell out a NoDerivatives license while using the shorter abbreviation. Record the applicable dataset-specific terms and seek clarification about the proposed conversion and artifact handling; do not assume the least restrictive interpretation. This assessment does not make a legal determination. [Official terms](https://neurips2025.care-pd.ca/terms-of-use.html), [dataset license metadata](https://huggingface.co/datasets/vida-adl/CARE-PD/blob/main/README.md).

For an initial experiment, prefer the benchmark's supported representation and an explicitly adapted, matched set of models. If the joint layout changes, describe this as evaluating the method on a new representation. It is not direct transfer of the existing 33-joint checkpoint unless a defensible adapter has been established. Do not invent absent foot landmarks or label estimated replacements as observed points.

Clinical gait severity also differs mathematically from signed laterality: exchanging anatomical sides should not reverse a severity score. A useful representation may need both side-sensitive and side-insensitive information, so an odd-only read-out is not an appropriate universal constraint for this endpoint.

### BMCLab walking data: a smaller route toward measured body dynamics

The original BMCLab release contains full-body motion-capture and force-platform data from 26 people with Parkinson's disease, measured in ON- and OFF-medication sessions. The dataset record provides raw and processed data with participant metadata and lists CC BY 4.0. Its 44-marker layout also differs from BlazePose33. These properties make it a candidate for testing whether pose history can predict a physical measurement obtained from another sensor. [Original dataset record](https://figshare.com/articles/dataset/A_dataset_of_overground_walking_full-body_kinematics_and_kinetics_in_individuals_with_Parkinson_s_disease/14896881).

The accompanying paper describes participant identifiers, clinical assessments, timed kinematics and forces, and the acquisition procedure. It reports no freezing-of-gait episodes in the recorded trials, despite including participants with a history of freezing. A freezing-event detector would therefore be a poor proposed endpoint for this release. The sample is also small enough that many trials cannot substitute for independent participants. [Original data paper](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2023.992585/full).

One potentially informative experiment would use past marker trajectories to predict a prespecified future force quantity, after verifying synchronization and sufficient valid force-platform observations. This would require a new training and measurement pipeline, not a direct application of the current contrast read-out. Its appeal is that the target could come from an independent sensor. Predicting a formula calculated from the same input coordinates would provide a weaker test of physical learning.

BMCLab contributes to CARE-PD, so these should not be presented as two independent replications without checking and excluding overlapping participants and walks. The two releases offer different representations and measurements of related underlying data. [CARE-PD cohort inventory](https://huggingface.co/datasets/vida-adl/CARE-PD/blob/main/README.md).

## Minimum useful build

Keep the implementation small until the dataset and primary question are fixed. The useful next notebook would consume an approved contract from 06, validate the numerical data, and evaluate simple baselines before launching any JEPA comparison. Its output should be a subject-level outcome table and an explicit explanation of which generalization setting was tested.

### 1. Specify one outcome and its independent unit

For a clinical study, choose a documented gait-rating item and preserve whether it was assigned per walk or per session. Select the primary metric before test access; an ordinal error in rating points and a class-balanced secondary measure may be informative, but class support must be checked first. Do not silently replace the gait item with a total motor score.

For a dynamics study, choose a fixed horizon in seconds and a defined physical or pose-derived quantity. State its units and ensure any normalization uses training information or the observed input alone. Different force directions and anatomical sides have different reflection behavior; that transformation should follow the quantity's definition.

Use the participant as the uncertainty unit when repeated trials or sessions belong to the same person. Keep all that person's sessions, medication conditions, and derived windows in one outer partition. For a held-out-center claim, exclude that center from representation training and selection as well as from read-out fitting, then report center-specific results rather than treating every frame as independent evidence.

### 2. Build an honest adapter

The adapter should check coordinate semantics, joint identities, sampling times, finite values, visibility or validity conventions, label provenance, and repeated observations. It should describe which landmarks are measured, inferred, unavailable, or differently defined. Conversion quality needs its own checks before model performance is interpreted.

A future-prediction task must split raw observations into past and future before normalization or gap filling. Changing future coordinates or future validity must not change the predictor input or its output. The existing full-clip 64-frame arrays are unsuitable for this causal guarantee because their preparation uses the complete clip.

### 3. Establish baselines before comparing representations

For clinical ratings, include a training-only reference prediction, direct movement features, the paired untrained architecture, and the ordinary pretrained encoder. Then compare the proposed training change with identical splits, read-out selection, sampling, and compute. If a matched-budget mask is the proposed contribution, keep the number of hidden valid tokens and the rest of the recipe comparable. The internal [matched-budget parameter reference](MATCHED_BUDGET_MASKING.md) explains which Notebook 08 settings match the completed training and which remain exploratory choices.

For future movement, include persistence and a simple velocity-based continuation where those baselines fit the endpoint. A force-prediction baseline must receive the same available input as the learned model; do not provide measured past force to a baseline while describing the task as pose-only. A useful temporal control can test whether genuine history improves prediction over reduced or disrupted history.

A frozen encoder followed by a clinical-score read-out fitted on external training subjects is a reasonable first study. Fine-tuning the encoder or fitting a cross-schema adapter creates additional learning stages that must stay within those same training partitions and be reported separately.

### 4. Freeze the comparison, run it, and report every planned outcome

Select the pipeline from training and validation data, record it, and retain the external test data for final assessment. Use paired comparisons between methods on the same participants. Report the number of participants with usable outcomes, exclusion reasons, baseline scores, confidence intervals, and optimization-seed variation where relevant.

An external null result remains informative if the measurement and comparison are credible. Do not repeatedly adjust the method against the external test set until the transfer result becomes favorable. If repeated exploration has already used that dataset, describe it as development evidence and seek a separate confirmation set.

## Phases and stopping decisions

| Phase | Required outcome before proceeding | Reason to pause |
|---|---|---|
| Dataset and endpoint selection | A specific meaningful outcome, documented subject IDs, compatible or defensibly adaptable measurements, and scoped permission records | Unclear terms, unavailable labels, unidentified overlap, or a conversion that invents essential inputs |
| Training-data pilot | Numerical adapter checks pass; outcome support is described; simple baselines and aggregation work | Label semantics are ambiguous, usable participants are too few for the intended inference, or acquisition artifacts dominate the task |
| Matched development comparison | The proposed method and controls run under the same selected protocol | A comparison changes several ingredients at once, future information leaks, or the primary metric is ill-defined |
| External confirmation | A fixed pipeline produces results on untouched participant or center partitions | Test data were already used to choose the method; a new confirmation plan is needed |

No universal subject-count threshold is justified from the current information. Precision depends on outcome variation, participant clustering, label balance, and the effect worth detecting. Estimate the required precision using appropriate training or pilot information, and be explicit if the available dataset can support only an exploratory result.

## Priority relative to the new notebooks

The immediate highest-return step remains Notebook 07's measurement and read-out diagnosis. Its new exploratory calculation shows a sizable difference between the target computed from observed poses and the same formula applied to prepared inputs. Resolving that difference reduces the risk of carrying a measurement problem into another dataset.

Notebook 08's masking comparison can then test a concrete learning change. Notebook 09's representation-symmetry work can clarify or alter how geometric consistency is represented, provided usefulness is evaluated alongside consistency. These are plausible methodological contributions that an external study could subsequently test. Notebook 10's past-only prediction provides the stronger route toward a world-model claim, but requires more substantial timing and leakage controls.

External dataset selection and permission clarification can begin in parallel with these internal experiments. Once a meaningful endpoint and compatible data are available, the external evaluation should take priority over adding many more internal variants. Under present conditions, it would be premature to commit to a broad external training pipeline merely to make 06 report a completed study.

## What we should build next

Preserve 06 and add a later tutorial notebook for a selected external experiment, tentatively `11_external_subject_evaluation.ipynb`. Its first real deliverable should be an approved, numerically checked dataset adapter and a subject-disjoint baseline result. For a near-term health-learning paper, CARE-PD is the stronger first candidate to investigate. For a longer-term body-dynamics contribution, the independently measured kinetics in the original BMCLab release offer an interesting, smaller study.

The decision to proceed should be driven by whether the external experiment can test a useful scientific claim with interpretable measurements. More readiness checks would improve administration; a well-designed external comparison could change what we know about the learned representation.
