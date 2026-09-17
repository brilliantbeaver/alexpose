# Independent implementation-fix and software-evidence review

Read gavd5-drift/notes/47-improvement-codex-review.md completely. Read the47 plan, audit, literature and experiments, current implemented protocol and governing decisions. Apply the relevant adversarial checklist to CURRENT CODE and actual retained artifacts, not only documentation. This is a follow-up to the protocol and implementation reviews, NOT an empirical gate3 approval. No real GAVD, HAIC, CUDA training or foundation-model transfer has occurred. Historical manuscript and outputs are unchanged.

Read-only authority: do not edit, discover HAIC/private assets, connect remotely, submit jobs, download data/checkpoints or run expensive training. You may inspect the explicitly named local synthetic artifact root and repository software. Short existing unittest checks are permitted if scratch is allowed; sandbox failure is an evidence limit, not a test failure or pass. Do not invoke broad dirty-diff review (unrelated user changes exist). No arbitrary score or demand to agree.

Review priorities:
1. Recheck all four supported findings in gavd5-drift/notes/47-reviews/codex-implementation-review.md against CURRENT code and regression tests: prepared-cache receipt access, partial-summary retry preserving locks/fits, Slurm explicit single-task resume, and governing-document identity binding.
2. Inspect latest cross-check fix: audit-baselines.sbatch now requests GPU because E0 direct MLP uses cfg.device=cuda. The real pilot resource dry-run regression must verify emitted launchers without real submissions.
3. Inspect information boundaries, target intervals, source/readout/seed aggregation and final test lock for any remaining material defects reachable in real mode. Distinguish proven upstream causality from manifest attestations, and no-cap indexing from unmeasured RAM scalability.
4. Assess software-only claims against gavd6/docs/studies/temporal-gait/results/README.md and software-verification.json, gavd6/outputs/temporal-gait/software-20260915-02/verification.json, reports/claim-audit.json, config/identity.json, receipts, predictions and selection artifacts. The coordinator observed 77 study tests pass, 6 layout tests pass, 15 fresh kernels pass and all13 receipts verify. Raw unit stdout was not separately retained; do not imply you reran it unless you do.
5. Earlier software-20260915-01 was successful but intentionally superseded by the launcher/protocol-decision fix. Its artifacts are preserved, not compatible resumes. E3, RGB, raw-pose extraction, calibrated uncertainty and exact historical replay are explicitly not implemented; evaluate the E0-E2 milestone honestly rather than demanding its missing empirical results or judging a negative score as failure.

For each remaining finding require severity, exact path/line or protocol section, evidence, consequence, minimal fix and falsification test. For fixed earlier findings state what actually verifies the fix and its limits. Separate code inspection from executed checks. Conclude with remaining blockers for software handoff versus real HAIC submission versus empirical paper claims. Do not revise implementation.

Exact implementation-created/modified file list at dispatch:
- gavd6/docs/studies/temporal-gait/README.md
- gavd6/docs/studies/temporal-gait/development/decisions.md
- gavd6/docs/studies/temporal-gait/development/evidence-ledger.md
- gavd6/docs/studies/temporal-gait/development/interfaces.md
- gavd6/docs/studies/temporal-gait/development/literature-refresh.md
- gavd6/docs/studies/temporal-gait/development/local-review-model-agent.md
- gavd6/docs/studies/temporal-gait/development/review-dispositions.md
- gavd6/docs/studies/temporal-gait/development/worktree-state.json
- gavd6/docs/studies/temporal-gait/execution/haic.md
- gavd6/docs/studies/temporal-gait/protocol/protocol.md
- gavd6/docs/studies/temporal-gait/results/README.md
- gavd6/docs/studies/temporal-gait/results/software-verification.json
- gavd6/notebooks/temporal_gait/00_inventory_and_split.ipynb
- gavd6/notebooks/temporal_gait/01_full_bout_timing_and_windows.ipynb
- gavd6/notebooks/temporal_gait/02_information_and_baseline_audit.ipynb
- gavd6/notebooks/temporal_gait/03_time_faithful_masked_jepa.ipynb
- gavd6/notebooks/temporal_gait/04_causal_future_jepa.ipynb
- gavd6/notebooks/temporal_gait/05_optional_dense_and_video_transfer.ipynb
- gavd6/notebooks/temporal_gait/06_development_comparison.ipynb
- gavd6/notebooks/temporal_gait/07_locked_calibration_and_test.ipynb
- gavd6/notebooks/temporal_gait/08_aggregate_and_claim_audit.ipynb
- gavd6/notebooks/temporal_gait/README.md
- gavd6/scripts/research_directions/temporal_gait/build_notebooks.py
- gavd6/scripts/research_directions/temporal_gait/execute_notebook.py
- gavd6/scripts/research_directions/temporal_gait/run_stage.py
- gavd6/scripts/research_directions/temporal_gait/submit.py
- gavd6/scripts/research_directions/temporal_gait/validate_run.py
- gavd6/scripts/research_directions/temporal_gait/verify_software.py
- gavd6/slurm/temporal-gait/README.md
- gavd6/slurm/temporal-gait/aggregate.sbatch
- gavd6/slurm/temporal-gait/audit-baselines.sbatch
- gavd6/slurm/temporal-gait/cache-video-teacher.sbatch
- gavd6/slurm/temporal-gait/calibrate.sbatch
- gavd6/slurm/temporal-gait/common.sh
- gavd6/slurm/temporal-gait/confirmatory.example.json
- gavd6/slurm/temporal-gait/evaluate-development.sbatch
- gavd6/slurm/temporal-gait/evaluate-test.sbatch
- gavd6/slurm/temporal-gait/full-cohort.example.json
- gavd6/slurm/temporal-gait/inventory.sbatch
- gavd6/slurm/temporal-gait/pilot.example.json
- gavd6/slurm/temporal-gait/prepare-bouts.sbatch
- gavd6/slurm/temporal-gait/submit.sh
- gavd6/slurm/temporal-gait/synthetic.example.json
- gavd6/slurm/temporal-gait/train-extensions.sbatch
- gavd6/slurm/temporal-gait/train-future.sbatch
- gavd6/slurm/temporal-gait/train-masked.sbatch
- gavd6/src/gavd6_sjepa/research_directions/temporal_gait/__init__.py
- gavd6/src/gavd6_sjepa/research_directions/temporal_gait/config.py
- gavd6/src/gavd6_sjepa/research_directions/temporal_gait/contracts.py
- gavd6/src/gavd6_sjepa/research_directions/temporal_gait/evaluation.py
- gavd6/src/gavd6_sjepa/research_directions/temporal_gait/fixtures.py
- gavd6/src/gavd6_sjepa/research_directions/temporal_gait/information_audit.py
- gavd6/src/gavd6_sjepa/research_directions/temporal_gait/manifests.py
- gavd6/src/gavd6_sjepa/research_directions/temporal_gait/masking.py
- gavd6/src/gavd6_sjepa/research_directions/temporal_gait/models.py
- gavd6/src/gavd6_sjepa/research_directions/temporal_gait/objectives.py
- gavd6/src/gavd6_sjepa/research_directions/temporal_gait/plots.py
- gavd6/src/gavd6_sjepa/research_directions/temporal_gait/preprocessing.py
- gavd6/src/gavd6_sjepa/research_directions/temporal_gait/statistics.py
- gavd6/src/gavd6_sjepa/research_directions/temporal_gait/training.py
- gavd6/src/gavd6_sjepa/research_directions/temporal_gait/video.py
- gavd6/src/gavd6_sjepa/research_directions/temporal_gait/windows.py
- gavd6/src/gavd6_sjepa/research_directions/temporal_gait/workflow.py
- gavd6/tests/temporal_gait/__init__.py
- gavd6/tests/temporal_gait/test_data_contracts.py
- gavd6/tests/temporal_gait/test_evaluation.py
- gavd6/tests/temporal_gait/test_hpc_notebooks.py
- gavd6/tests/temporal_gait/test_review_regressions.py
- gavd6/tests/temporal_gait/test_training.py
- gavd6/tests/temporal_gait/test_video_contracts.py
- gavd6/tests/temporal_gait/test_workflow_integrity.py
- gavd6/docs/repository/studies.json (only the additive temporal-gait entry is ours; other changes predate this task)
- gavd5-drift/notes/47-reviews/codex-protocol-review.md (prior independent output)
- gavd5-drift/notes/47-reviews/implementation-prompt.md
- gavd5-drift/notes/47-reviews/codex-implementation-review.md (prior independent output)
- gavd5-drift/notes/47-reviews/software-evidence-prompt.md (this prompt)

The six47-improvement planning documents were already present and remain unmodified. This review's output is a new codex-software-evidence-review.md. No manuscript revision is part of this gate.

