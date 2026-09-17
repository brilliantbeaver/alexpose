# Independent local review: data and workflow boundaries

Date: **2026-09-15 UTC**. Reviewer: the model/training implementation agent, reviewing **coordinator-owned data, configuration, manifests and workflow code**, not its own model/training implementation. This is a bounded cross-owner review, not the separate external Codex gate and not validation on GAVD/HAIC.

Reviewed `config.py`, `contracts.py`, `video.py`, `preprocessing.py`, `windows.py`, `manifests.py`, `information_audit.py`, and `workflow.py`. Findings were sent to the coordinator before the fixes. The coordinator made the implementation changes; this reviewer added independent [regression tests](../../../../tests/temporal_gait/test_review_regressions.py).

## Findings and dispositions

| ID | Severity | Failure found | Disposition and evidence |
| --- | --- | --- | --- |
| R1 | P0 | Successful training returned `status="trained"`; the workflow copied it into a receipt, while receipt verification accepted only `complete`. Every successful training task would fail downstream verification. | Fixed: workflow translates trained→complete and interrupted→incomplete. `test_training_status_translates_to_receipt_completion` tests both paths. |
| R2 | P1 | Latent target sampling and the primary endpoint reused the same ±20 ms nearest-observation query. The final teacher observation could occur after the declared `(b+h−.08,b+h]` interval. | Fixed: teacher sampling has explicit interval bounds; the primary endpoint remains independently tolerance-matched. `future_times` is retained. The regression constructs a 3.50 s query whose primary observation is 3.503 s but teacher observation is 3.493 s, and checks every teacher timestamp lies within the interval. |
| R3 | P1 | Direct cached-test loading checked the analysis lock but not a test-open marker, unlike test preparation; merely testing marker existence also allowed a stale marker. | Fixed: shared `verify_test_open` checks analysis hash, frozen run identity, mode and status. Missing-marker and independently corrupted-marker tests verify both content readers remain uncalled. |
| R4 | P2 | Task locks included execution phase, allowing the same task in pilot/develop/confirm to write the same output directory concurrently. | Fixed: the lock key uses the task identity independent of phase. Regression checks identical keys across all three phases. This does not certify remote-filesystem lock behavior. |
| R5 | P2 / scope | The original information audit checked group counts and endpoint/scale support, not full raw→prepared→encoder mechanism preservation. | Coordinator added raw/time-grid/index-grid signed-speed diagnostics. A delegated descriptive summary now compares common finite support using windows→bouts→equal-video means, without changing a training/selection gate. It explicitly identifies the proxy as observation-weighted, not duration-weighted on VFR. This is still not complete encoder/readout mechanism localization or a clinical endpoint. |

Additional checks pass: verified person links within one partition are allowed while another partition remains disjoint; metadata-only inventory does not call video probing, pose loading, NPZ loading or a decoder; real mode with missing explicit manifests fails before manifest reading; unsupported cohort scope fails rather than substituting a convenient cohort.

## Follow-up and independence boundary

The stale-marker follow-up is closed by the coordinator's shared verifier and the independent regression for all four bound fields. No unresolved P0/P1 defect remains among this review's findings; this is not a whole-repository approval.

After the review, the coordinator delegated the small `information_audit.measurement_differences` helper to this agent. That helper and its arithmetic/no-future-read test are implementation work, **not independently reviewed by this same agent**. The external review should assess them. The summary adds no inference-time covariates, threshold or model-selection rule.

## Executed verification

```bash
cd gavd6
PYTHONPATH=src .venv/bin/python -m unittest discover \
  -s tests/temporal_gait -p test_review_regressions.py -v
```

Result after follow-up: **9 tests passed**. The tests use synthetic in-memory poses, temporary local manifests and access-denying spies. No private media was located or opened. No Slurm job, CUDA/bfloat16 run, remote lock-contention test, real-patient evaluation or pretrained-video experiment was performed by this review.

Upstream pose/tracker causality remains an audited provenance requirement: mutating future rows of a completed cache verifies downstream preparation, not whether the upstream estimator had already smoothed future evidence into its past rows. Source-held-out partitions are not participant-held-out unless verified identity links establish that stronger claim.
