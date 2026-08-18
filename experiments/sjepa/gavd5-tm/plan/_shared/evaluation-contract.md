# Shared evaluation contract

Every proposal in this `plan/` folder must follow this contract instead of inventing its own evaluation rules. The point is simple: this project's own history shows that when each study designs its own split and its own baseline, small measurement mistakes get mistaken for real findings. Following one shared contract means a reader can compare across the seven proposals and trust that a claimed effect is not just an artifact of a looser evaluation.

## Why this contract exists (the evidence behind it)

GAVD5's canonical labeled cohort is 96 sequences, but those 96 sequences come from only 18 source videos: 12 normal sequences from 1 video, 9 Parkinson's from 2 videos, 12 stroke from 3 videos, 47 myopathic from 10 videos, and 16 cerebral palsy from 2 videos. The full curriculum (159 sequences, 35 videos) adds more normal sequences but does not fix this: normal gait still traces back to very few underlying people and cameras.

Because of this, every accuracy or AUC number produced so far in this project and its predecessor lineage (a related, separate codebase called `skeleton-jepa/gavd` through `gavd4`) has had some form of leakage: either the same video appears on both sides of a train/test split, or the self-supervised encoder was trained on the very sequences later used to "test" a downstream probe. A registered, multi-seed ablation study in that predecessor lineage found that once evaluation was tightened, a promising-looking architecture change shrank to a statistically indistinguishable null result (+0.0083 accuracy, 95% confidence interval crossing zero). The lesson: an evaluation must be strict enough that a fake finding cannot survive it, or the proposal is not worth running.

## The rule: split by source video, first

A "source video" is the original YouTube video a sequence was cut from. Before any preprocessing, any representation training, any checkpoint selection, or any probe fitting happens, decide which source videos go into an outer-training group and which go into an outer-held-out group. Nothing downstream is allowed to touch the held-out group's data until the very last evaluation step.

This is necessary but not sufficient. Two classes (Parkinson's and cerebral palsy) have only 2 source videos each in the canonical cohort, and normal gait traces to very few source videos overall. A single video-disjoint split for these classes is unstable: swapping which one video is "held out" can flip the result. So in addition to one primary split, every proposal must report:

- the exact list of which source videos were assigned to training versus held-out, so the split is auditable;
- a leave-one-source-out sensitivity check for any class with fewer than 4 source videos, showing the result under every possible choice of held-out video, not just one lucky (or unlucky) choice;
- whether each held-out test row's source video, and its underlying data-collection pathway (canonical GAVD extraction versus the separately-collected "added normal" pathway), was ever seen during self-supervised pretraining. If the encoder was pretrained on the same videos later used for testing, that must be stated plainly next to the reported number, not hidden in a footnote.

## Baselines every proposal should compare against, where relevant

- A missingness-only control: a model trained only on the fraction of frames/joints that MediaPipe detected, with zero gait geometry. If a proposed method does not clearly beat this, it may be picking up which joints the pose detector tends to lose, not gait itself.
- A provenance-only control where relevant: since canonical and "added normal" sequences use different extraction pipelines, check whether a result changes when provenance is controlled for.
- The existing frozen S-JEPA checkpoint's current numbers (documented in `notes/02_paper_draft.md` and `docs/staged_sjepa_gait.md`), clearly labeled as coming from an evaluation with encoder-side exposure to the test data, not as a fair external baseline.

## Statistics

- Use at least 3 seeds to screen an idea, and at least 5 seeds before treating any result as a finding worth writing up.
- Treat checkpoints or evaluation folds coming from the same training run as correlated, not independent.
- Report an effect size and a confidence interval, not just a point estimate and not just a p-value.
- State the primary metric in advance, before looking at results, and do not silently substitute a friendlier metric after the fact.

## Claim hygiene (say this, do not say that)

- A prediction-error spike is evidence of surprise to the model, not evidence of danger, pathology, or physical impossibility.
- An embedding being compact does not make it private by default.
- A folder condition label (e.g. "stroke") is a dataset annotation carried over from GAVD, not a clinical diagnosis produced by this project.
- A result computed on a video-disjoint probe split is still not a fair test of generalization if the self-supervised encoder itself was trained on the held-out videos.
- Unusual-but-real gait (for example, genuine cerebral palsy or myopathic gait) is not the same thing as an invalid or impossible motion. A method that cannot tell these apart is not measuring what it claims to measure.

## What each proposal should cite from this contract

Each `plan/<NN>-<slug>/README.md` should reference this file by name in its evaluation section (for example: "This study follows the shared evaluation contract in `plan/_shared/evaluation-contract.md`: source-video-disjoint splits, leave-one-source-out sensitivity for small classes, and the missingness-only baseline.") rather than re-explaining the leakage problem from scratch. Idea 1 (`01-honest-video-disjoint-anomaly-screening`) is the proposal that actually builds the corrected split as reusable code; ideas 2 through 7 should specify that they reuse that split rather than each building their own.
