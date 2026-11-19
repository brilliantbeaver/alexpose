<!-- iteration02-execution-status -->
> This is the preserved design or feasibility record. Consult the [implemented protocol](protocol.md), [verified results](results.md), and [external execution report](external_replication_results.md) for current execution status.

# A bounded external replication using laboratory gait data

**Recommendation:** reserve a two-hour feasibility check, followed by at most one working day if it passes. This is a promising independent participant study of the proposed measurement, not evidence that GAVD-trained weights transfer. No scientific data files were downloaded or models run for this review. Public metadata and three small documentation files were inspected on November 15, 2026.

## What is available

Van Criekinge and colleagues released 138 able-bodied participants and 50 people after stroke. They walked barefoot at their preferred speed in the same laboratory. Mean ages differ, 51 versus 64 years. Stroke participants were selected during early recovery, so this is not a representative sample of all stroke survivors. The acquisition provides 100 Hz motion capture, clinical information, and synchronized EMG where available. C3D exports already include Plug-In Gait processing; they are not untouched sensor recordings. [Primary dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC10692332/).

| Public files | Exact combined size | Best use |
|---|---:|---|
| `138_HealthyPiG.zip` and `50_StrokePiG.zip` | 779,608,713 bytes | Preferred path to continuous, common-clock trajectories and joint angles. |
| Two full MAT structures | 9,150,652,601 bytes | Rich stride-level data, events, demographics, and matching limb signals. |
| Two summary XLSX files | 52,478,900 bytes | Participant-average waveform checks, not continuous gait prediction. |
| Schema XLSX, stride-count XLSX, example TXT | 41,443 bytes | Documentation inspected in this review. |

All 13 Figshare article records report CC0. The example script separately contains an MIT license. Preserve attribution and inspect file-specific notices when downloading. Full URLs, sizes, supplied MD5 values, and licenses are in [metadata](external_dataset_metadata.json). The downloaded documentation has SHA-256 hashes and extracted text in [documentation evidence](external_dataset_documentation.json). These records come from [collection version 1](https://doi.org/10.6084/m9.figshare.c.6503791.v1).

## A crucial distinction about timing

The authors' schema explicitly provides both limbs under one stride anchor, such as left-segmented left-limb data and left-segmented right-limb data. It also describes original sample rates, trial indexes, absolute initial-contact boundaries, and contralateral contact indexes. Thus MAT data may preserve bilateral phase **within the same anchored stride**. This is materially better than assuming every normalized export destroys phase. [Author schema](https://doi.org/10.6084/m9.figshare.24192504.v1).

However, pairing a left-anchored left waveform with a separately right-anchored right waveform aligns away their relative timing. Participant-average XLSX curves also cannot recover individual continuous trials or stride-to-stride variation. Before using MAT, verify the actual field names, paired columns, trial IDs, and event indexes. The schema and example use slightly different field spellings. Reconstruct seconds only from verified original boundaries and sample rates. Never concatenate unrelated strides and call the result continuous walking.

## The smallest defensible experiment

1. Freeze the GAVD hypothesis, target, score definition, and comparator before external outcomes are inspected. Test whether relative bilateral timing adds useful information beyond movement amplitude and a simple own-joint predictor. A favorable outcome is optional; this is a falsifiable replication.
2. If this branch proceeds, download only the two C3D archives into a separate versioned external-data directory. Verify their supplied checksums. In the first two hours, inspect one able-bodied participant and one stroke participant. Establish participant IDs, walking versus static trials, timestamps, angle units, missingness, and simultaneous left/right signals. Check clinical metadata joins separately.
3. Prefer six concurrent sagittal hip, knee, and ankle angle signals for this bounded method test. Do not rename these as BlazePose joints. Confirm the angle convention from actual file labels and documentation. Fit any normal reference, scaling, and model using external training participants only. Lock evaluation participants before tuning. Keep all trials and both limbs of each person together.
4. Compare the proposed timing-aware score with the frozen simple baselines on exactly the same eligible participants. Report paired participant-level uncertainty, the number excluded and why, and age/sex sensitivity analyses. Treat self-selected speed as a potential mediator as well as a confound; retain the unadjusted result and label adjusted analyses explicitly.
5. Keep external results separate from GAVD. Call this replication of the measurement or algorithm on a different representation. It does not establish transfer of the learned representation, video-pose robustness, or performance on Parkinson's disease, cerebral palsy, or myopathy.

## Mapping and interpretation limits

| GAVD input | Relevant marker names in the author schema | Why direct substitution is unsafe |
|---|---|---|
| Shoulders, elbows, wrists: 11–16 | SHO, ELB, WRA, WRB | Surface markers and wrist pairs are not identical to estimated joint centers. |
| Hips: 23–24 | ASI, SACR | Pelvic landmarks are not hip joint centers. |
| Knees and ankles: 25–28 | KNE, ANK | Lateral markers differ from anatomical joint centers. |
| Heels and foot indexes: 29–32 | HEE, TOE | Heel and toe definitions require explicit anatomical checks. |

Never zero-fill absent BlazePose semantics to force a 16-joint comparison. A future explicit marker-to-joint adapter needs validation. Synthetic camera projections can test geometric sensitivity, but cannot validate MediaPipe failures, clothing, or real video viewpoints. Existing smoothing also needs an audit before claiming strictly prospective forecasting.

EMG and clinical associations are useful later, after participant matching and missingness are verified. They are not necessary for this deadline branch. A recent [study using these spreadsheet exports](https://www.frontiersin.org/journals/medical-technology/articles/10.3389/fmedt.2026.1863908/full) already examines normative waveform deviations and asymmetry. Dataset reuse alone is therefore not novelty.

**Stop rule:** abandon the branch for this submission if the two-hour pilot cannot establish identity, common time, and signal semantics, or if the complete replication exceeds one working day. Document feasibility instead of substituting independently aligned average waveforms. Additional laboratory participants strengthen an independent test; they do not increase the number of GAVD participants or repair GAVD's source confounding.
