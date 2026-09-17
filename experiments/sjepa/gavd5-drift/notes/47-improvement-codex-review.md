# Adversarial review: temporal gait JEPA

You are an independent reviewer of a proposed and/or implemented temporal gait JEPA study. Review critically and read-only. Your objective is to find evidence-backed defects that would invalidate a claim, comparison, implementation, or execution plan. Do not modify files, locate HAIC assets, submit jobs, download data, or train models.

The repository root contains `gavd5-drift/` and `gavd6/`. Review these explicitly scoped files and any directly supporting code:

- `gavd5-drift/notes/47-improvement-plan.md`
- `gavd5-drift/notes/47-improvement-audit.md`
- `gavd5-drift/notes/47-improvement-literature.md`
- `gavd5-drift/notes/47-improvement-experiments.md`
- `gavd5-drift/notes/47-improvement-llm-instructions.md`
- `gavd5-drift/neurips-laterality/laterality/` and relevant `laterality_extensions/`, tutorials, tests, retained outputs, and latest FMTS manuscript.
- If implemented, `gavd6/{src/gavd6_sjepa/research_directions,scripts/research_directions,notebooks,tests}/temporal_gait/`, `gavd6/slurm/temporal-gait/`, and `gavd6/docs/studies/temporal-gait/`.

Inspect the working-tree status but exclude unrelated user changes. Establish whether the reviewed item is a plan, implemented software, or empirical result; absent future code is not a bug in a clearly labeled planning deliverable. Conversely, fail any claim that future example commands have already been implemented or verified.

Prioritize these attacks:

1. Does any context input depend on future observations through preprocessing, crop tracking, normalization, target validity, hidden-value interpolation, a full-sequence encoder, or cached features?
2. Do windows, duplicates, known subjects, previous development sources, or prior reservations cross a claimed evaluation boundary? Does all-GAVD pretraining quietly become transductive evaluation?
3. Are “full videos/sequences” actually a capped selection or first crop? Are all exclusions, seconds, bouts, and independent source counts visible?
4. Is the laterality target a deterministic function of available poses? Does a proposed explanation incorrectly attribute it to duration despite uniform-speed cancellation? Are cancellation across pairs and reversal invariance acknowledged?
5. Do differences in support, model capacity, valid-target exposure, teacher scale, readout dimension, schedule, or compute confound the treatment?
6. Could a support-only, static, random, direct-forecast, periodic, or auxiliary-only baseline reproduce the claimed improvement? Are comparisons paired and nuisance controls meaningful?
7. Does the actual latent predictor support observable forecasts, or are only context/observed-future features evaluated? Are future target tolerance and horizon definitions consistent across documents and code?
8. Is checkpoint/readout/threshold selection test-contaminated? Are clustered intervals, seed variation, overlapping CV training sets, multiple comparisons, and post-development hypotheses handled accurately?
9. Are original S-JEPA and video JEPA claims faithful to primary sources? Does dense loss get advertised as universally beneficial? Are identity/clinical gait tasks or pretrained contamination conflated?
10. Do artifacts permit rerunning inference and intervals? Are missing seeds, stale caches, code/config mismatches, or notebook synthetic outputs able to produce a false complete/real result?
11. Do the Slurm stages obey their documented interfaces, afterok dependencies, final-data gates, resource declarations, and resume contracts? Do any scripts search for unspecified data paths?
12. Can the experimental plan distinguish success, failure, and uncertainty without choosing the outcome after the test is opened?

Return findings in descending severity. For each include an exact path/line or named protocol section, evidence, why it matters, the smallest correction, and a check that could falsify your concern. Distinguish confirmed defects from plausible risks requiring evidence. Do not demand new unrelated research or arbitrary architecture changes.

Conclude with blocking issues, nonblocking limitations, checks actually performed, and checks impossible with available artifacts. No approval score. If no blocking defect is found, state the remaining evidence limits; do not assert that unrun experiments succeed.
