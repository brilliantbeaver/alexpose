# Requirements for the S-JEPA tutorial research team

The user confirmed [the populated ASTAD tutorial brief](../../../notes/aaai-astad/02-create-initial-tutorials.md) as the source for this team and requested that the definitions be saved separately from existing experiments. This document maps that brief to the ten roles in [team.json](team.json). It describes future work and the evidence required to accept it. Creating these definitions does not mean that the tutorials, training runs or scientific evaluations have been carried out.

The observed brief has SHA256 `9d2c611072addaa6b830412a09d6f92d7d3f6de8a080663ad381ea828931dfb4`. The coordinator must compare that identity with the file at the start of a future run and reconcile any changes. The similarly named `notes/aaai-worldmodels/02-create-initial-tutorials.md` was empty during this review and is not the confirmed brief.

## Scope and output location

The brief asks for a progressive course that learns from skeleton motion, starts with normal gait, considers adaptation to four other gait labels, and compares JEPA with a conventional Random Forest. It also asks for illustrated notebooks, a README, local video context, slides and independent review. The course should support a carefully qualified anomaly-detection study rather than promise a successful paper or a clinically validated classifier.

The tutorial output location remains unresolved. The brief names `experiments/aaai`, while the current repository contains an extensive historical study under `experiments/aaai-astad` and existing deleted tracked paths under `experiments/aaai`. Those facts do not authorize restoring the old path, replacing the historical study or relabeling its outputs. Before future tutorial writes, `sjepa_coordinator` must establish the output directory and study identity from the user's current instructions, record the decision, and inspect any instructions that apply there. The definitions themselves live in this folder and in `.codex/agents/`.

## Coverage and acceptance evidence

The role named first in each row is responsible for the handoff. Other named roles contribute or review it. The coordinator assigns exact file ownership through the task process; this table does not give several agents concurrent permission to edit the same file. Task identifiers refer to the dependency plan in `team.json`.

| Requirement | Responsible roles | Expected output | Evidence needed before acceptance | Tasks |
| --- | --- | --- | --- | --- |
| `requirements_binding`: Use the confirmed brief and track changes to its requirements. | `sjepa_coordinator`, `sjepa_reviewer` | A run record identifying the brief and this coverage table. | The recorded brief path and fingerprint match the intended input, and changed requirements are reconciled explicitly. | T00, T15 |
| `output_identity`: Choose the tutorial output directory without replacing existing research. | `sjepa_coordinator`, `sjepa_reproducibility` | A declared output directory, study identity and preservation decision. | Git status and directory instructions were inspected; no old result is presented under a new method identity. | T00 |
| `literature`: Read the requested background, papers, project documentation and relevant implementations. | `sjepa_literature`, `sjepa_reviewer` | A source-linked evidence table and method review. | Every material claim identifies its supporting passage or implementation; unavailable references remain marked as unread. | T01 |
| `ambient_reuse`: Understand and reuse the shared AlexPose pipeline. | `sjepa_pose`, `sjepa_data`, `sjepa_reproducibility` | An API map and small adapters where needed. | Signatures and actual callers were inspected; adapters use shared functionality and introduce no shared-code dependency on an experiment. | T03, T06 |
| `fresh_gavd`: Use the current selected category CSVs and their provenance. | `sjepa_data`, `sjepa_reviewer` | A fresh annotation inventory and reconciliation report. | Current files were parsed, `source/` was excluded from transformed counts, and labels, sequence IDs, hashes and frame ranges were checked. | T02 |
| `media`: Reuse verified full videos, retrieve missing sources when authorized, and preserve acquisition failures. | `sjepa_data`, `sjepa_reproducibility` | A source and clip manifest with available, missing and excluded records. | Source IDs, file fingerprints, decoded frame coverage and exclusion reasons are explicit; no unavailable frames are fabricated. | T02 |
| `generalization`: Assess what the available recordings can establish. | `sjepa_evaluation`, `sjepa_data`, `sjepa_literature`, `sjepa_reviewer` | A cohort discussion and limitations in the README and results. | Counts distinguish recordings, sequences, windows and people; uncertainty, confounding, repeated subjects and limits on clinical claims are addressed. | T02, T10, T14 |
| `pose`: Make pose quality and temporal alignment explicit. | `sjepa_pose`, `sjepa_data`, `sjepa_reproducibility` | Pose and window records, coordinate definitions and a quality report. | Pose format, actual image sizes, frame origins, decoded times, confidence, missingness and preprocessing are recorded; exclusions are counted by class and source. | T03, T06 |
| `joints`: Produce a sorted, deduplicated feature-to-joint table and use only its joints. | `sjepa_pose`, `sjepa_evaluation`, `sjepa_model`, `sjepa_reviewer` | The required BlazePose table and a machine-readable list of allowed joints. | Each joint is justified by an accepted feature definition; duplicate indexes are removed; a test rejects masks or model inputs outside the accepted list. | T03, T04, T06 |
| `splits`: Create training, validation and test sets without source leakage. | `sjepa_evaluation`, `sjepa_data`, `sjepa_reviewer` | A fixed source assignment and fitting rules. | Related clips, duplicate footage and overlapping windows remain together; preprocessing and selection use only their permitted partitions; small-class feasibility is reported. | T04, T08 |
| `sjepa`: Explain and implement the context encoder, target encoder and predictor. | `sjepa_model`, `sjepa_literature`, `sjepa_reviewer` | Model code, a tensor contract and a diagram. | Shapes, visible inputs, target inputs, target gradients, target updates, losses and missing-token handling are explicit and checked. | T07 |
| `masking`: Sample allowed joints across frames uniformly without motion guidance. | `sjepa_model`, `sjepa_pose`, `sjepa_reviewer` | A mask sampler, examples and focused checks. | Selection does not depend on motion intensity, velocity, labels or attention; allowed positions and the sampling distribution are tested; the difference from the cited methods is disclosed. | T01, T07 |
| `normal_only`: Begin encoder training with normal training sources. | `sjepa_model`, `sjepa_evaluation`, `sjepa_reviewer` | A distinct initial-training run and checkpoint. | Recorded source membership contains only fitting normals; validation and test sources do not enter encoder training, including unlabeled training. | T04, T09 |
| `adaptation`: Introduce additional condition data in explicit later stages. | `sjepa_model`, `sjepa_evaluation`, `sjepa_tutorials` | Separate adaptation runs and a comparison with the initial model. | Each stage identifies its classes, source membership, label use and checkpoint ancestry; an adapted model is not described as normal-only. | T09 |
| `vicreg`: Use an anti-collapse mechanism with observable checks. | `sjepa_model`, `sjepa_literature`, `sjepa_reviewer` | An accepted loss definition and training diagnostics. | Variance, redundancy, finite gradients and collapse behavior are examined; VICReg is not described as a guarantee of disease-class separation. | T07, T09 |
| `stochasticity`: Decide when positional randomness and latent variables are useful. | `sjepa_literature`, `sjepa_model`, `sjepa_evaluation` | A reasoned design decision and separate controlled comparisons where justified. | Mask randomness, positional embeddings and latent variables are distinguished; comparisons hold data and evaluation inputs fixed; uncertainty claims require appropriate evidence. | T01, T07, T10 |
| `rf_comparison`: Build an exp5-style Random Forest comparison on the same data. | `sjepa_evaluation`, `sjepa_pose`, `sjepa_reviewer` | A current feature baseline and capstone results. | Feature names and units are declared; historical exp5 scores are not treated as matched results; both methods use the same eligible sources, partitions and evaluation units. | T04, T08, T10 |
| `metrics`: Evaluate anomaly detection and five-class classification as distinct tasks. | `sjepa_evaluation`, `sjepa_model`, `sjepa_reviewer` | Fixed scoring, aggregation and uncertainty rules, followed by result tables. | Binary detection and five-class metrics have clear targets; thresholds and tuning avoid test data; resampling respects sources; negative and inconclusive results remain visible. | T08, T10 |
| `uv`: Provide one coherent dependency declaration for local use. | `sjepa_reproducibility`, `sjepa_tutorials` | A project `pyproject.toml` and `uv` setup commands. | Dependencies and Python compatibility are checked; there is no competing requirements file; applicable existing lockfiles are preserved. | T05 |
| `env`: Resolve environment loading without exposing secrets. | `sjepa_reproducibility`, `sjepa_coordinator`, `sjepa_reviewer` | An explicit `.env` location and notebook loading example. | The real python-dotenv functions or valid aliases are used; root discovery is deterministic; credentials are neither printed nor copied into tracked examples. | T00, T05 |
| `laptop`: Keep execution practical on the user's laptop. | `sjepa_reproducibility`, `sjepa_model`, `sjepa_evaluation` | A bounded smoke mode and measured runtime/resource record. | A small run establishes feasibility before longer authorized runs; unavailable hardware and unexecuted paths are reported. | T05, T09, T13 |
| `notebooks`: Create a progressive, detailed tutorial course. | `sjepa_tutorials`, `sjepa_reproducibility` | Notebook generators and generated notebooks. | The course covers inventory, video and pose, joint tokens, masks, JEPA, normal-only learning, adaptation, anomaly scoring, RF comparison, ablations and limitations with explained examples. | T11 |
| `video_context`: Let readers view the walking sequence behind each example. | `sjepa_tutorials`, `sjepa_data`, `sjepa_pose` | Inline or clearly linked local video examples. | The displayed clip has a known source and frame range; unavailable or excluded media are identified; demonstrations do not silently trigger bulk downloads. | T06, T11 |
| `language`: Explain the work in connected, accessible prose. | `sjepa_tutorials`, `sjepa_visuals`, `sjepa_reviewer` | Complete sentences in notebooks, README and slides. | Explanations build across paragraphs; technical terms are explained; em dashes, slogans and staccato phrasing are avoided. | T11, T12, T14 |
| `graphics`: Illustrate the concepts with clear vector diagrams and flowcharts. | `sjepa_visuals`, `sjepa_tutorials`, `sjepa_reviewer` | Editable vector sources, generators and exports under `images/`. | Diagrams agree with the implemented behavior; a reviewer inspects rendered images for text overlap, clipping, ambiguous arrows and legibility. | T12, T13 |
| `slides`: Explain the setup and experiments step by step. | `sjepa_visuals`, `sjepa_tutorials`, `sjepa_reviewer` | Slide sources and a documented local export under `slides/`. | The slides include clear diagrams, tables and flowcharts; measured charts point to accepted result identities; rendered pages are inspected. | T12 |
| `independent_execution`: Make every notebook executable from a fresh kernel. | `sjepa_reproducibility`, `sjepa_tutorials`, `sjepa_reviewer` | Per-notebook execution reports. | Each notebook declares imports, setup and artifact prerequisites; missing inputs identify producer commands; no preceding notebook's hidden state is required. | T13 |
| `adversarial_review`: Challenge the scientific and software work independently. | `sjepa_reviewer`, `sjepa_coordinator` | Findings, repairs and recheck records. | Review uses exact artifact identities and reproducible evidence; authors do not provide the only approval of their own work; changed artifacts are checked again. | T14 |
| `documentation`: Explain setup, organization, results and limitations. | `sjepa_coordinator`, `sjepa_tutorials`, `sjepa_reproducibility` | The final README and notebook index. | Counts match the current manifests; commands match actual paths; executed results, missing inputs and proposals are clearly distinguished. | T11, T15 |
| `acceptance`: Finish only when the required evidence exists. | `sjepa_coordinator`, `sjepa_reviewer` | A completion record covering every applicable requirement. | The record lists accepted artifacts, executed checks, failed or unverified items and unresolved limitations without claiming false completeness. | T15 |

## References assigned for scientific review

The following URLs are copied from the confirmed brief. Their inclusion here records the reading assignment. These scientific references were **not independently opened and reverified while writing this requirements document**. Existing repository discussion is useful context, but it does not replace reading the cited source. `sjepa_literature` must record access, identity, version, claims and relevant code before using a reference as support for a new method.

| Reference from the brief | Required investigation | Status of this assignment |
| --- | --- | --- |
| [ECCV 2024 paper 04755](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) | Verify the paper identity, skeleton representation, encoder and predictor design, targets, losses and masking policy. | Assigned to `sjepa_literature`; not reverified here. |
| [S-JEPA project](https://sjepa.github.io/) | Read the project documentation and follow its implementation links; record the code revision used for comparisons. | Assigned to `sjepa_literature`; not reverified here. |
| [MAMP, arXiv 2308.07092](https://arxiv.org/pdf/2308.07092) | Establish the motion-prediction objective and mask policy, and explain which parts can support the requested uniform-mask adaptation. | Assigned to `sjepa_literature`; not reverified here. |
| [OpenReview paper jLnygpRFYm](https://openreview.net/pdf?id=jLnygpRFYm) | Verify how stochastic positional embeddings work and what evidence supports transferring the idea to skeleton sequences. | Assigned to `sjepa_literature`; not reverified here. |
| [AlphaXiv paper 2611.09302](https://www.alphaxiv.org/pdf/2611.09302) | Confirm identity and version, then examine the evidence for latent stochasticity, varied completions, predictive uncertainty and representation quality. Do not adopt the brief's benefit claim without support. | Assigned to `sjepa_literature`; not reverified here. |
| [Scientific Reports article s41598-026-39330-9](https://www.nature.com/articles/s41598-026-39330-9) | Verify the article's task, population, model, evaluation and relevance to the proposed course. | Assigned to `sjepa_literature`; not reverified here. |
| [JEPA explanatory article](https://www.thecprogramminglanguage.com/jepa.html) | Read it as explanation and trace technical claims to primary research or code where possible. | Assigned to `sjepa_literature`; not reverified here. |

The requested local background `/Users/pmui/vaults/worldmodels/gait/skeleton-jepa/background` was unavailable in the current workspace. The literature researcher may locate an authorized project copy or record the missing input. No agent should claim to have read its contents. The brief's `/Users/pmui/dev/alexpose` prefix must likewise be resolved against the actual repository instead of being copied into runnable commands.

The [ASTAD4 workshop website](https://ismart.ece.mcgill.ca/astad4/) is a separate source for current submission rules and dates. The phrase “in a few days” is not a durable deadline. A future submission plan requires a fresh check of the official page, while this task creates definitions only.

## Decisions that must remain explicit

### Labels and scientific claims

The unique requested labels are `normal`, `parkinsons`, `stroke`, `cerebral palsy` and `myopathic`. Repeated occurrences of `stroke` do not create another class. Use the CSV's `gait_pat` field, since the broader `dataset` field can differ from the intended class label. The team should explain why this focal population is useful and what it excludes. The labels do not establish independently verified diagnoses, participant identities or a shared neurological cause.

Initial encoder training must use fitting normals only. Subsequent training with other conditions receives its own run identity and clear account of whether labels affect the encoder, classifier or both. A classifier trained on all five labels is a supervised recognition experiment even if its encoder began with self-supervised training. Its abnormal score is not evidence for a purely normal-only detector. Stochastic masks that allow observations on both sides of a hidden target implement completion; a forecasting claim additionally requires causal inputs and preprocessing that cannot inspect the future.

Existing ASTAD documentation identifies the original S-JEPA and MAMP masking as motion-aware. The brief instead requires uniform, motion-independent masks. The literature review must verify this distinction and describe the new implementation as an adaptation where appropriate. VICReg addresses collapse and redundant representations, but it does not by itself guarantee separation between disease labels. Positional randomness, latent variables and random mask selection need separate explanations and controls.

### Feature definitions and the allowed joints

The brief requests a joint table but does not provide a complete list of features or joints from which to derive it. The pose and evaluation roles must choose and document the actual feature set before the model role treats a joint list as final. The table must use this exact header:

| BLAZEPOSE_33 index | Keypoint name | Features involved |
| --- | --- | --- |

Rows must be unique and sorted numerically. The accepted table becomes the only allowed mask list. Historical code uses sixteen BlazePose indexes, `11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32`, but that historical choice is evidence to review rather than an automatic requirement for every new study. Any conversion from COCO joint indexing must be explicit.

### Environment loading and notebook independence

The brief first places `.env` in the output folder and later asks for the repository-root `.env`. The reproducibility role should record the repository-root convention, following that later instruction, unless the user's current directions specify otherwise. Root discovery must be deterministic and must not silently load an unrelated ancestor's environment file.

The actual python-dotenv function names are `load_dotenv` and `find_dotenv`. To retain the requested spelling, notebooks can use valid aliases:

```python
from dotenv import load_dotenv as load_env, find_dotenv as find_env

# Resolve and verify the intended repository root before this call.
load_env(find_env())
```

The implementation must verify the file that will be loaded or pass the explicitly resolved path. It must never print credential values or copy them into notebooks or tracked configuration. A notebook can depend on declared data artifacts while still being independently executable. Independence means a fresh kernel with explicit prerequisites, not a requirement to repeat every expensive download and training run whenever a lesson opens.

### Shared interfaces and coordinate conventions

Start with `ambient.utils.youtube_cache.download_youtube_urls`, `ambient.pose.factory.PoseEstimatorFactory`, `ambient.pose.keypoint_extractor.SequenceKeypointExtractor`, `ambient.analysis.feature_extractor.FeatureExtractor`, `ambient.pose.joint_angles.get_joint_angles` and `ambient.classification.features.GaitFeatureVector`. Inspect each actual interface and its callers before writing adapters. Examples in `experiments/multiple-sclerosis` and `experiments/exp5` explain usage; their historical results and configuration choices are not new-run evidence or universal defaults.

Original GAVD frame numbers and the shared sequence extractor's arguments are one-based. Decoded OpenCV indexes are zero-based. A clip extracted from source frames `s` through `e` has clip frame 1 corresponding to source frame `s`; source frame `f` therefore becomes clip frame `f - s + 1`, or zero-based clip index `f - s`. Never send original source frame numbers unchanged to an extractor reading that trimmed clip. Record the source origin and test a known inclusive range.

The downloaded image size can differ from CSV `vid_info`. Frame trimming does not resolve an incorrect bounding-box reference, changed framing or a source-content mismatch. Confidence thresholds, missingness, normalization and resampling belong to the accepted study protocol. Image coordinates and projected angles must not be described as physical gait speed or measured three-dimensional anatomy.

## Current data baseline to recount before execution

The following figures describe the local preparation reviewed during team definition on November 16, 2026 UTC. They are starting evidence, not constants to embed in the future training code. The data role must reread the selected CSVs and reconcile the current [media manifest](../../../data/videos-ms/gavd-sequences/manifest.json), [media report](../../../data/videos-ms/gavd-sequences/README.md) and [annotation inventory](../../../data/gavd/README.md).

| Quantity | Observed value | Interpretation |
| --- | ---: | --- |
| Requested sequence CSVs | 666 | These span the five unique selected gait labels. |
| Referenced YouTube IDs | 103 | These identify recordings, not verified independent people. |
| Retained full source videos | 100 | One retained source is too short for its requested sequence. |
| Completed sequence videos | 656 | Their 137,232 output frames passed exact source-frame comparison. |
| Missing sequence videos | 10 | Nine depend on three inaccessible sources, and one requests frames absent from its retained source. |
| Completed clips with an explicit content exclusion | 4 | These contain 144 frames and carry `exclude_from_ml: true`. |
| Completed clips without that explicit exclusion | 652 | This is an upper bound before pose quality and study eligibility checks, not a validated training cohort. |

The inaccessible IDs are `sf5X4YYkWUA`, `yULxvDc9e8c` and `YjRoLtP1di0`. The retained source `n93bgWhLZk4` has 228 decoded frames while its CSV requests frames 1 through 458. The four flagged normal clips belong to `2LbERGI34lA`, whose requested ranges show a map introduction or street views that conflict with the annotated close-up walking subject. The current manifest is the authority for the exact excluded sequence identities and evidence.

The normal source `EHx08gSNfxQ` was recovered during the recent media task. Older experiment reports that call it unavailable, or that report an older usable pose cohort, are historical snapshots. Neither a successful source-frame comparison nor an older pose cache is sufficient to override the current content exclusions or establish current pose eligibility.

## What counts as completion

The team definitions are complete when the ten roles, workflow, requirement mapping and review process are persisted and checked. A future tutorial project has a different completion condition: accepted notebooks and teaching materials must exist, their independent execution must be documented, scientific outputs must point to the inputs and run that produced them, and every missing or inconclusive part must remain visible. A passing software fixture demonstrates implementation behavior; it does not demonstrate classifier performance, patient generalization or clinical value.
