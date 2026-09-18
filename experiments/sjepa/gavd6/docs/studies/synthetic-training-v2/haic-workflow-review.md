# HAIC workflow simplification and independent review

Review date: 18 September 2026. Scope: make the source development study easier to run without changing its empirical claims or inventing reviewed inputs. No HAIC job was submitted during this revision.

## Workflow decisions

The main guide now has four steps: initialize from existing `ST_*` exports, complete/check reviewed inputs, prepare/inspect paired tracks, and train/read results. A single optional transfer helper previews and copies the required code and exact preserved evidence, backs up replacements, and keeps existing study profiles.

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
