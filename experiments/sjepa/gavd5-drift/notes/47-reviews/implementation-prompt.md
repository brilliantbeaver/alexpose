# Independent Codex implementation gate

Read gavd5-drift/notes/47-improvement-codex-review.md completely and apply its adversarial checklist. Read the47 plan and its audit/literature/experiment contracts plus current gavd6/docs/studies/temporal-gait/protocol/protocol.md and development/decisions.md. This is implemented E0-E2, not a completed real experiment. E3/RGB/rawpose extraction are expressly deferred. Historical code/outputs remain preserved; plan commonwindow adapter is not exacthistorical replay.

You are read-only. Do not edit, locateHAIC assets, submit jobs, download, or run expensive training. Small existing unittest diagnostics may be attempted read-only with PYTHONPATH=gavd6/src and gavd6/.venv/bin/python if sandbox allows scratch; inability to execute is an evidence limit. No arbitrary approvalscore. Examine data information boundaries, loss semantics, source access, paired evaluation/readout fit, fullbout uncapped coverage, exactresume, final locks, artifact recovery and CLI/Slurm. Prioritize confirmed defects with path/line/evidence/consequence/minimalfix/falsification test. Independently inspect currentcode, not only docs/agentclaims. Keep output concise but sufficient. Scientific empirical gate3 remains pending realartifacts. You may find emerging code changes during review; anchor findings to exact functions/lines and describe any ambiguity.

Exact study files created/modified for this implementation at review dispatch (all are study-scoped; preserve all unrelated dirty worktree changes):

- gavd6/docs/studies/temporal-gait/README.md
- gavd6/docs/studies/temporal-gait/development/decisions.md
- gavd6/docs/studies/temporal-gait/development/evidence-ledger.md
- gavd6/docs/studies/temporal-gait/development/interfaces.md
- gavd6/docs/studies/temporal-gait/development/literature-refresh.md
- gavd6/docs/studies/temporal-gait/development/local-review-model-agent.md
- gavd6/docs/studies/temporal-gait/development/review-dispositions.md
- gavd6/docs/studies/temporal-gait/execution/haic.md
- gavd6/docs/studies/temporal-gait/protocol/protocol.md
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
- gavd6/slurm/temporal-gait/README.md
- gavd6/slurm/temporal-gait/aggregate.sbatch
- gavd6/slurm/temporal-gait/audit-baselines.sbatch
- gavd6/slurm/temporal-gait/cache-video-teacher.sbatch
- gavd6/slurm/temporal-gait/calibrate.sbatch
- gavd6/slurm/temporal-gait/common.sh
- gavd6/slurm/temporal-gait/confirmatory.example.json
- gavd6/slurm/temporal-gait/evaluate-development.sbatch
- gavd6/slurm/temporal-gait/evaluate-test.sbatch
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
- gavd6/tests/temporal_gait/test_data_contracts.py
- gavd6/tests/temporal_gait/test_evaluation.py
- gavd6/tests/temporal_gait/test_hpc_notebooks.py
- gavd6/tests/temporal_gait/test_review_regressions.py
- gavd6/tests/temporal_gait/test_training.py
- gavd6/docs/repository/studies.json (only new temporal-gait entry; other dirty changes preexisting)

The six gavd5-drift/notes/47-improvement*.md planning documents predate this implementation turn. Only review outputs under47-reviews are new there. No historical manuscript/notebook was revised. Use a new review response; do not demand a positive model result.

