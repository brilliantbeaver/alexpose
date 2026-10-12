# HAIC workflow simplification and independent review

Review date: 18 September 2026. Scope: make the source development study easier to run without changing its empirical claims or inventing reviewed inputs. No HAIC job was submitted during this revision.

The unattended workflow is reviewed in **Unattended development revision** below, followed by the latest **Git retrieval revision**. Earlier sections retain the history of the human-reviewed workflow and its checks. The [single operating guide](../../../../slurm/synthetic-training-v2/README.md) is authoritative for the current command sequence.

## Workflow decisions

At this implementation review, the guide introduced four steps: initialize from existing `ST_*` exports, complete/check reviewed inputs, prepare/inspect paired tracks, and train/read results. The current [HAIC guide](../../../../slurm/synthetic-training-v2/README.md) leads with resuming the existing run and places new-run initialization separately. Its input instructions distinguish actual worksheet filenames, reviewed contents and configured paths. A single optional transfer helper previews and copies the required code and exact preserved evidence, backs up replacements, and keeps existing study profiles.

The managed `submit.sh` interface generates valid configurations, the RTMPose-m/HRNet-W32/ViTPose-Base roster, a saved session, reviewed-input worksheets, immutable per-attempt scopes, allocation ledgers and job history. CUDA checks run inside preparation. It does not generate reviewer decisions or infer that an identity is available for use from its filename.

The budget remains explicit. Preparation and each of the two training GPU stages reserve at most one H100-hour, with a 54-minute Python guard inside the allocation. Costs use top-level Slurm allocation records, including failures and a conservative one-second rounding allowance; overlapping Python timers are not added again. The workflow cannot infer independent jobs or another run's costs, so those require an explicit prior ledger.

## Independent adversarial review

Three initial reviewers separately examined usability, runtime/scheduling, and scientific correctness. Implementation responsibilities were split. Follow-up review examined other contributors' changes; reviewers also used mocked scheduler failures and small CPU fits to test their hypotheses. This was not a review performed only by the author of each change.

| Finding | Resolution and evidence |
| --- | --- |
| Hardcoded user paths and manual estimator JSON caused setup errors. | Generated paths derive from the existing exports. Shell and configuration tests cover custom roots, preserved old settings, quoting, and the complete official estimator roster. |
| The selected interpreter could be stale despite a plausible environment name. | Initialization runs the established package/import checker before creating the run. Failure names the explicit `STV2_PYTHON` correction. |
| Importing scientific packages before environment diagnostics could hide the useful setup error. | The management entry point defers those imports. Its help and environment-check path remain usable with site packages disabled. |
| Required audit CSVs were referenced but not created or explained clearly enough. | Initialization generates clearly labeled drafts, with real manifest paths and blank review decisions. CPU checks report missing prerequisites before GPU submission. |
| Repeated low-level submission could launch duplicate chains and overwrite the first job IDs. | Submission is locked; accepted job IDs are saved atomically. Repeated submission is rejected. Managed retries preserve history and rebuild only unfinished stages. |
| A failed scheduler response could be mistaken for a rejected job. | The managed launcher persists an uncertainty token before submission. Recovery verifies an accepted job's name and owner, or records explicit reconciliation that no job was accepted. |
| Slurm requeue could hide earlier allocation costs under the same job ID. | Launchers and batch defaults request `--no-requeue`; managed accounting requests duplicate records and refuses ambiguous repeated allocations. |
| Inherited `SBATCH_*` settings could change resource requests or create arrays. | Submission environments remove those overrides while preserving study settings. Accounting retains any unexpected GPU allocation. |
| Total and typed GPU counts could conflict; a subsecond allocated failure could be charged as zero. | Accounting parses and checks the counts and distinguishes an actual allocation from a canceled pending job. |
| Strict `[0,1]` confidence validation was incompatible with native MMPose scores. | Extraction preserves native scores, including values above one. Nonpositive or nonfinite scores are unobserved; raw values remain in artifacts. Tests cover extraction, normalization, model inputs, and all eight arms in a small CPU fit/predict probe. |
| A misspelled held extractor silently produced no held-family panel. | Roster validation rejects missing, ambiguous, duplicate, incomplete and all-held rosters. Source submission also requires the excluded family in development data. |
| Negative or malformed seeds could fail only after paid preparation. | Shared run configuration validates both seeds and the seed list before initialization writes files. |
| A raw or model input could claim `observed=True` with a nonpositive score. | Both contracts now enforce the same positive native-score observation rule as extraction while retaining unobserved native values. |
| Mutable Slurm logs could invalidate scientific stage receipts. | Operational top-level logs/submission records are excluded; scientific artifacts, configurations and cost records remain checked. Existing regressions cover both cases. |

The native-score correction follows the official MMPose v1.3.2 [maximum decoders](https://github.com/open-mmlab/mmpose/blob/v1.3.2/mmpose/codecs/utils/post_processing.py), [HeatmapHead](https://github.com/open-mmlab/mmpose/blob/v1.3.2/mmpose/models/heads/heatmap_heads/heatmap_head.py), and [SimCC decoder](https://github.com/open-mmlab/mmpose/blob/v1.3.2/mmpose/codecs/simcc_label.py). These outputs are native maxima, not calibrated probabilities. The change does not clip scores, infer confidence from true errors, or change the training losses.

Requeue handling follows Slurm's [sbatch options](https://slurm.schedmd.com/sbatch.html) and [sacct duplicate-record behavior](https://slurm.schedmd.com/sacct.html). The tests simulate scheduling and do not establish access to the real cluster.

## Scientific limits retained

- A 200-update, single-seed run is an engineering feasibility check. It does not establish an ICLR result or a successful scientific gate.
- The study restores 2D synthetic joint-center proxies from a whole observed window. Its supplied boxes and fixed front-camera renders do not establish autonomous tracking, real gait measurement, 3D reconstruction, forecasting, or world-model capability.
- Paired JEPA versus coordinate reconstruction compares the declared objective package, including the JEPA recipe's regularization. A narrower claim about latent prediction alone needs its own matched ablation.
- The low-level `equal_total_compute` flag remains bounded by update limits. A faster baseline can stop at that cap before spending the time allowance. Actual timing and termination records must support any matched-compute claim; the simplified default uses matched data and steps.
- Obstruction can remove the complete bilateral reference support required for amplitude/timing metrics. A perfect prediction can therefore still leave aggregate preservation evidence insufficient. Do not relax support requirements after observing favorable results.
- Previously inspected ViTPose assets remain exposed development evidence, even when excluded from training. Human reservation/exposure records, independent real references, calibrated thresholds and repeated seeds remain prerequisites for stronger claims.

## Verification

Targeted regressions reproduce the confirmed failures and verify the fixes. The managed workflow tests cover initialization without submission, read-only previews, allocation accounting, explicit recovery, original-configuration preservation, successful sequential submission, and rejection of duplicate attempts. The shell transfer test verifies the actual selected dependency set and historical files without network access.

Final verification: **113 v2 tests passed**. A fresh CPU fixture completed all ten workflow stages and all retained stage receipts validated. Its scientific gates correctly remained insufficient or closed. All 25 shell examples across the four HAIC pages passed syntax checks, embedded Python/JSON and local links were checked, and every v2 shell/profile/batch script passed `bash -n`. The historical audit still verified all 72 preserved files and reconstructed the retained pilot. The final independent follow-up also reran the 13 managed-workflow and eight submission-safety tests after the last accounting and bootstrap fixes.

Actual HAIC CUDA/EGL rendering, licensed asset compatibility, released checkpoint loading, Slurm policy and throughput remain to be verified by the first allocated preparation job. CPU fixtures and scheduler mocks are not substitutes for that evidence.

The subsequent documentation revision starts from the reported passing environment/history and unresolved CSV inputs. It separates resuming `source-smoke-01` from creating a new run, recognizes the user's actual reservation filename, validates CSVs before binding their paths, and distinguishes failed input checks from failed Slurm allocations. Local checks passed for 23 shell examples, two embedded Python blocks and 14 relative links/anchors. The path-binding example was exercised with the reported and fresh-initializer filenames, including backup preservation and repeat execution, and rejected header-only audits, blank reservation decisions, ambiguous filenames and existing submission history without changing the configuration. Historical preservation verification also passed. These were documentation checks with temporary software fixtures, not a new independent adversarial review or HAIC execution; the user's actual CSV contents remain uninspected.

## Unattended development revision

The user subsequently required one instruction document, minimal commands, and no manual CSV editing or review. The implementation now provides three primary blocks in the Slurm README: transfer, launch and status. Former input, cost and historical instruction pages redirect there. The managed source graph, eight learned comparisons, four baselines and metrics remain unchanged. An explicit pre-fit protocol amendment distinguishes automated development from the human-reviewed study.

The controller submits a bounded CPU motion screen, validates its generated inputs, submits GPU preparation, waits for actual accounting completion, submits all ten source stages, and verifies the saved results. No new GPU allowance is granted. The human-reviewed path remains available in the underlying launcher but is not falsely recorded by automation.

Three agents worked independently on motion screening/playback, scheduling/execution, and adversarial policy/result verification. They reviewed one another's boundaries and the root author's controller, scientific-status changes and consolidated guide. The final adversarial disposition was **approved**, with no remaining blocking discrepancy identified in the reviewed workflow.

| Finding | Resolution |
| --- | --- |
| Human review was mandatory but no source playback command existed. | Added actual SMPL-H joint playback and a separately labeled, bounded kinematic screen for unattended development. No filename is treated as a completed review. |
| Automatically filling unknown reservations as false would fabricate clearance. | Preserve literal `unknown`, import existing decisions/aliases, exclude known reserved and original test identities, and restrict the resulting evidence to automated development. |
| Automated preparation could be misrepresented as a human overlay review. | Added distinct `--automated-screen` acceptance; incompatible mode/record/evidence combinations and human-review flags are rejected. |
| Favorable machine-screened metrics could be confused with a scientific pass. | Evidence is `automated-source-screen`; Gate B cannot advance even if a decision specification is supplied. Reports describe the missing independent review and confirmation limits. |
| A permissive reservation truthiness check accepted empty or malformed values. | Automatic bundle records now require exactly boolean false or literal `unknown`; malformed values are rejected. |
| A broad retry handler treated accounting corruption as a retryable job failure. | Only terminal nonzero job outcomes raise the dedicated retryable phase error. Accounting inconsistencies stop without submission. |
| Unknown scheduler states could wait indefinitely. | Recognized active states are explicit; unfamiliar states produce a diagnostic. Transient accounting-command failures receive bounded read retries. |
| A saved PID could be reused or refer to a different login host. | The shared automation lock determines controller activity. Stale PIDs cannot block resume. |
| Failed dependencies could leave the source chain pending forever. | Cancel only recorded source jobs still shown as pending in the live queue; preserve all job/cost history. Paid retries remain explicit. |
| A report filename alone was an inadequate completion test. | Verify latest successful allocations, current run identity, all receipts, complete fits, expected prediction files and reconstructed per-window metrics. |
| A copied `srun` GRES reset could be inappropriate in a batch header. | The CPU batch requests CPUs/memory and omits GPU GRES, consistent with the existing CPU stages and documented [sbatch resource syntax](https://slurm.schedmd.com/sbatch.html#OPT_gres). |

Final local verification: **166 v2 tests passed** under the existing local Torch 2.6.0 CPU interpreter. Full ten-stage CPU fixtures completed eight learned fits, twelve prediction methods and a report; a second fixture exercised the recommended eight-window/four-person panel with batch size 64. Actual generated machine-screen inputs passed the production input validator. Both source-status variants of the result checker passed software simulations, rejected failed/missing scheduler records and corrupted artifacts, and made no file changes. These simulations used software-generated trajectories and scheduler mocks; they are not empirical AMASS or HAIC results.

All six shell blocks in the consolidated README and all v2 shell/batch/profile files passed syntax checks. Five command entry points parsed for Python 3.11 and exposed `--help` without site packages. All 72 preserved historical files matched. The browser playback was exercised and visually inspected without JavaScript errors, clipped controls or overlapping labels. `git diff --check` passed.

No remote jobs were submitted. Actual HAIC scheduling, licensed body-model execution, CUDA/EGL rendering, released checkpoint compatibility and throughput remain unverified. The first real run checks these dependencies and stops with an explicit error when they fail; no local review can guarantee cluster availability or a completion time.

## Git retrieval revision

At the user's request, Step 1 now runs `git pull --ff-only origin main` on HAIC. The guide explicitly requires the complete revision to be committed and published first; it was still uncommitted/untracked locally during this review. No commit, push, remote update or job submission was performed as part of the documentation revision.

An independent adversarial reviewer required checks for a started run, an active controller, wrong branch, tracked edits, local-ahead/diverged history and missing published entry points. The literal README command holds the run's existing nonblocking management locks across the pull and historical verification. The guide also states that these locks do not protect other experiments using the same checkout. It makes no claim to identify a particular unpublished release or guarantee runtime compatibility merely from file presence.

Review found that the former historical recovery command could copy unpublished Mac code over the freshly pulled HAIC revision. The new `--history-only` transfer mode selects exactly the verified preservation-manifest paths, retains backup behavior, and does not traverse source trees. The actual 72-file manifest contains no `src/`, `scripts/` or `slurm/` entries. Full transfer behavior remains available separately in the helper.

Verification executed the literal Step 1 Python block against temporary local Git repositories for **11 scenarios**: published fast-forward success; unpublished code with untracked lookalikes; tracked edits; wrong branch; local-ahead history; divergent history; recorded jobs; uncertain submission; existing source configuration; active controller lock; and historical-check failure. Each produced its expected success or stop, preserved run state, and retained local work. These probes used a historical-check stub; separately, the real checker verified all 72 retained files and reconstructed the original pilot successfully. All seven README shell blocks parsed.

The shell regression suite passed **8 tests**, including exact history-only file selection in preview/apply modes, exclusion of unpublished scientific code, backup retention and rejection of corrupted evidence before any remote call. The independent reviewer reran that suite and approved the revised workflow with no remaining blocking finding in scope. Actual HAIC/GitHub access, scheduling and CUDA execution were not exercised. Publication remains a prerequisite, not an accomplished step.
