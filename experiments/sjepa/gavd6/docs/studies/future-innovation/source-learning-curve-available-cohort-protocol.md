# Source learning curve with available manifest recordings

This prospective cohort amendment uses the `source-learning-curve-v1` numerical
protocol with **`available-development-v1`** as an explicit cohort policy. The
user authorized this amendment after discovery found 12 missing development
recordings. It changes which recordings can enter the study, not the predictor,
features, targets, controls, selection, source weights or decision thresholds.

1. Read and retain the full sequence and recording manifests and known exposure
   inventories. Assign confirmation roles, participant-connected source groups
   and outer folds using the existing metadata-only rules **before filtering**.
2. Resolve manifest recording IDs against declared local storage. Prefer an
   existing explicit full-recording path; otherwise require an exact filename
   stem and supported video extension. Duplicate links to the same file count
   once. Missing, empty, unreadable or ambiguous matches are unavailable. Extra
   storage files that are not in the manifests do not enter the study.
3. Freeze a source availability table with roles, inclusion flags and exclusion
   reasons. The processing set is exactly the available development recordings.
   Available confirmation files remain outside processing. Keep all manifest
   sequences belonging to an included recording, subject to the original later
   alignment/pose eligibility checks.
4. Write separate processing manifests containing only those recordings and
   sequences. Their video paths are explicit, absolute paths to selected files.
   Record file sizes and modification times at discovery; decoded frames and
   teacher arrays retain the existing checksums. Metadata checks do not prove
   that every frame decodes or that a file is the authentic original recording.
5. Candidate construction receives only the processing manifests. Verify inherited
   window alignment only for parent candidates in this included set. Reuse parent
   teacher arrays only for included windows, preserving original cache bindings.
6. A retry uses the frozen set and paths. New files do not add sources. A selected
   file that disappears or changes before preparation completes requires restoring
   that input or starting another run; it cannot silently alter a frozen cohort.
   Completed decoded/cache stages retain their existing verified reuse behavior.
7. Final pose eligibility still determines the usable cohort. Keep source/fold
   isolation, minimum group requirements and the 40/80/160/all learning-curve
   plan. Unattainable nominal sizes remain explicitly unavailable. Insufficient
   valid data or failed teacher checks cannot produce a complete scientific result.

The original manifests, missing IDs and confirmation assignments remain audit
evidence. They are not executable processing inputs. Missing media is an explicit
cohort exclusion under this amendment rather than a reason to abort a new run.
The exclusion list must remain visible in notebook 23 and the frozen cohort audit.

Public `launch/submit.sh all` applies this policy automatically to a new study,
using the existing environment variables. A preflight failure before any freeze
can be retried in the same intended directory. Already frozen studies retain
their recorded policy and code identity; use a fresh root for this amendment.
Manual freeze supports `--video-root` for the amended policy. Omitting it preserves
the historical metadata-only freeze path and its original strict preparation.

The scientific conclusion concerns **eligible available development recordings**.
It does not establish representativeness of the full GAVD corpus. Unavailable
recordings may differ systematically from available ones. The inspected sources
remain development data; any promising skeleton increment still requires an
independent-source confirmation study before student training.

See the [base numerical protocol](source-learning-curve-protocol.md) and
[execution guide](../../../slurm/future-innovation/SOURCE_LEARNING_CURVE.md).
