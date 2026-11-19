# Primary-source check for GAVD scale and sequence-led reporting

This check supports emphasizing GAVD's clinical annotation and sequence coverage. The defensible size claim is specific to publicly accessible abnormal-gait videos. It does not extend to every clinical gait modality or establish independence between walking passes.

## What the GAVD sources establish

Ranjan and colleagues describe GAVD as the largest online-accessible abnormal-gait video collection annotated by expert clinicians. Their release contains 1,874 normal and abnormal sequences and 458,116 annotated frames. The collection protocol uses expert visual screening, manually marked boundaries, at least two gait cycles, and one walking direction. These are properties of the original annotation protocol; the local audit must check actual retained clips. The paper appeared as an arXiv preprint in 2024 and an IEEE Access article in 2025. [Primary paper](https://arxiv.org/html/2407.04190v1)

The current author repository gives a single-direction meaning to its sequence identifier and retains the linked clinical-video size claim. It distributes annotations and video addresses rather than the original videos. Clinical expert annotation is a substantive strength. It is different from having patient records or independently reassessing each diagnosis in the present study. [Author repository and data dictionary](https://github.com/Rahmyyy/GAVD)

Recommended manuscript wording is: “GAVD was introduced as the largest publicly accessible collection of abnormal-gait videos annotated by expert clinicians. It contains 1,874 normal and abnormal walking sequences. Our study evaluates 456 unique annotated walking sequences, using 2,815 forecast windows.” The full resource and the evaluated subset should remain distinguishable. This formulation attributes the superlative and makes the study scale explicit.

## Why the scope of “largest” matters

A blanket cross-modality statement would be misleading. The public GaitRec resource contains 75,732 bilateral force-plate walking trials from 2,084 patients and 211 healthy controls. It is a different measurement modality, but it demonstrates why a clinical gait size claim needs an explicit scope. [Horsak et al., 2020](https://www.nature.com/articles/s41597-020-0481-z)

CARE-PD makes another clearly bounded size claim for public three-dimensional mesh data in Parkinson's disease, with nine cohorts from eight clinical centers. Its modality and evaluation protocols differ from this video-pose study. Neither resource supplies a matched performance comparison here. [Adeli et al., 2025](https://arxiv.org/abs/2510.04312), [dataset owner](https://neurips2025.care-pd.ca/)

No exhaustive current census was performed. The historically attributed GAVD wording is therefore more defensible than an unqualified present-tense world record.

## Sequences are meaningful observations, not automatic independent replicates

The original sequence definition supports leading the cohort description with unique walking sequences. Separate passes can cover different posture, timing, or viewing conditions, so they add useful movement observations. However, boundaries and one-way walking do not remove dependence from a shared person, camera, environment, recording episode, or pose estimator. This last statement is a methodological application of the following primary sources, rather than a measured estimate of this cohort's correlation.

| Primary source | What it supports for this manuscript |
| --- | --- |
| [Lazic, 2010, BMC Neuroscience](https://link.springer.com/article/10.1186/1471-2202-11-5) | The number of repeated observations can exceed the independent sample size. Ignoring nesting can give false precision. |
| [Saeb et al., 2017, GigaScience](https://pubmed.ncbi.nlm.nih.gov/28327985/) | The empirical evaluation split must match the deployment use case. Records from the same person do not establish generalization to new people. |
| [Saravanan, Berman, and Sober, 2020](https://nbdt.scholasticahq.com/article/13927-application-of-the-hierarchical-bootstrap-to-multi-level-data-in-neuroscience) | Resampling can respect the hierarchy of nested observations. It cannot create independent participants or guarantee validity without appropriate assumptions. |

The clearest reporting distinction is that sequences describe movement coverage, original recordings define the available dependence groups, and verified participants define a stronger clinical generalization unit. Reporting 456 sequences prominently is compatible with retaining 91 original recording groups, including 29 normal groups, in the methods and uncertainty descriptions. A video identifier remains a grouping proxy, not proof of a unique patient.

Changing the primary score from a recording average to a sequence-level ranking changes the endpoint and its weighting. It requires a separately identified analysis that retains the recording/participant clusters. Existing recording-level AUROCs and intervals cannot simply be relabeled as sequence-level results.

## Workshop fit

The official Med-WM call explicitly includes datasets, benchmarks, uncertainty, and clinical validation. This supports a movement-observation benchmark even though short video forecasts do not establish treatment-response simulation. Full papers use 6–8 body pages; short, poster, and position papers use 2–4, excluding references and appendices. The required format is AAAI two-column, with single-blind review. Scope fit and page count do not predict acceptance. [Official call](https://med-wm.github.io/aaai27-mwm/)

## Access and evidence boundaries

The versioned GAVD full text and current owner repository were accessible. The direct IEEE page returned a minimal shell, so final publication metadata were checked against the author citation. The Saeb publisher page failed, but its indexed primary PubMed abstract and figure descriptions were available. The Lazic publisher full text and Saravanan publisher abstract were accessible. No model was fitted, no video was acquired, and no frozen result was changed. The accompanying JSON records titles, authors, versions, identifiers, URLs, and the pinned local annotation revision.
