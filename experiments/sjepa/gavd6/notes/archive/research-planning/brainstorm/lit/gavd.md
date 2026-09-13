# GAVD: A Gait Abnormality Video Dataset

## Problem

**Paper fact.** Clinical gait research lacks accessible, expert-annotated monocular video spanning natural settings and diverse abnormal presentations. GAVD targets this gap and supports coarse abnormality recognition, while the paper argues that future systems must localize when and where motion deviates.

## Method in exactly five sentences

1. Clinical experts manually screen publicly available online videos and label gait as visually normal, abnormal, or pathological.
2. They retain sequences with at least two gait cycles, minimal obstruction, and one walking direction, then annotate start and end frames, per-frame person boxes, camera view, and gait class.
3. The release stores ten fields: `seq`, `frame_num`, `cam_view`, `gait_event`, `dataset`, `gait_pat`, `bbox`, `vid_info`, `id`, and `url`.
4. For the binary baseline, the authors sample 291 normal and 291 abnormal sequences and mask pixels outside each annotated person region.
5. Kinetics-400-pretrained SlowFast and TSN models are fine-tuned for up to 200 epochs with batch size 16 and stratified ten-fold validation, then tested on held-out GAVD, GPJATK, and CASD sequences.

## Headline numbers and benchmark results

**Paper facts.** Table 3 reports 452 URL links, 1,874 sequences, and 458,116 annotated frames, split into 291 normal and 1,583 abnormal sequences; the abstract claims more than 400 subjects. Table 4 defines one normal label plus 11 abnormal or pathological subclasses. Table 7 reports SlowFast/TSN accuracy of 0.94/0.92 on GAVD, 0.76/0.59 on GPJATK, and 0.87/0.73 on CASD. Camera dependence is severe: SlowFast is 0.56 to 0.66 on side views versus 0.91 to 0.97 on front/back views, while TSN is 0.25 to 0.30 versus 0.86 to 0.95 (Table 8, Section 5).

**Benchmark inference from Tables 6 and 7.** GAVD testing contains only 1,132 abnormal examples and GPJATK only 618 normal examples, so constant-label classifiers score 1.00 on each. CASD has 94 normal and 644 abnormal sequences, making its 87.3 percent abnormal majority baseline effectively equal to SlowFast's 0.87. These accuracies therefore do not establish balanced binary discrimination.

## Reconciliation with the local audit

**Local evidence, not a paper citation.** The current five CSVs reproduce 458,116 rows and 1,874 sequences but only 348 unique YouTube source IDs, so the paper's 452-link and over-400-subject claims are not recoverable from the release. There is no subject, person, or split field and no official split; source-ID grouping is only a leakage-control surrogate. The observed `gait_pat` taxonomy has 12 labels, but the separate binary field is saturated at 1,620 abnormal versus 254 normal sequences and conflicts with `gait_pat` for 37 sequences. Paper Table 5 also totals 1,877 view assignments, not 1,874.

## What it enables here

**Repo inference.** GAVD provides source manifests, temporal boxes, views, and coarse labels for full-video retrieval, pose extraction, in-the-wild gait JEPA pretraining, view-robustness tests, and descriptive 12-label probes. Claims must remain source-grouped and non-diagnostic because subject-held-out evaluation is impossible.

## Limitations and public access

Labels are expert visual observations, not measured kinematics, kinetics, force, or independently verified diagnoses; `gait_event` is almost empty in the local audit. The MIT-licensed CSV/PKL annotations are public, but raw videos, CASD, baseline code, official splits, and checkpoints are not. Researchers must retrieve YouTube videos independently under applicable terms and ethics, and the README explicitly warns of link removal. It also advertises severity and gait notes, although the ten-column schema exposes neither as a named field.

## Sources

- Full paper: https://arxiv.org/pdf/2407.04190
- Official repository and README: https://github.com/Rahmyyy/GAVD
- Official annotation files: https://github.com/Rahmyyy/GAVD/tree/main/data
- Fetch date: 2026-09-03
