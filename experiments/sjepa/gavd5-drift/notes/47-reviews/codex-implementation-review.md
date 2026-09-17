**Implementation gate remains blocked by artifact-integrity and recovery defects.** This is implemented E0–E2 software, not a completed experiment. E3/RGB/extraction deferrals and the common-window adapter’s nonhistorical semantics are not defects.

Findings below are confirmed from code; scratch-dependent reproductions were sandbox-blocked.

1. **P1 — Downstream stages can consume modified caches without verifying their preparation receipt.**  
   [workflow.py:144](/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6/src/gavd6_sjepa/research_directions/temporal_gait/workflow.py:144) checks receipt `02`, then loads training windows; development evaluation does likewise. But `02` does not bind the prepared archives, and [contracts.py:141](/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6/src/gavd6_sjepa/research_directions/temporal_gait/contracts.py:141) verifies only the requested receipt’s outputs, without predecessor verification. `load_role` directly opens the cache.
   
   **Consequence:** changing a valid NPZ after the information audit can silently change training or development data under the same run identity. A newly computed training dataset fingerprint records the changed data; it does not reject the change against preparation.  
   **Minimal fix:** verify receipt `01` before consuming its caches, or bind and recursively verify predecessor receipts.  
   **Falsification test:** complete preparation/audit, alter one cached coordinate, and require training/development evaluation to fail before model execution.

2. **P2 — A failure during summary publication makes the stage unrecoverable through its normal retry path.**  
   [_finish_comparison, evaluation.py:720](/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6/src/gavd6_sjepa/research_directions/temporal_gait/evaluation.py:720) writes immutable `paired-source-scores.json`, then the decision JSON. The stage receipt comes later. [workflow.py:198](/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6/src/gavd6_sjepa/research_directions/temporal_gait/workflow.py:198) creates fresh retry directories for individual evaluations but reuses the same comparison directory.
   
   **Consequence:** interruption after the first summary file—or after the decision but before the receipt—causes every retry to raise `FileExistsError`. The locked-test summary has the same failure window.  
   **Minimal fix:** publish each complete comparison transactionally, or validate and recover identical existing summary artifacts.  
   **Falsification test:** inject failures between summary writes and before receipt creation; retry successfully while preserving the original test lock and choices.

3. **P2 — Explicit checkpoint resume conflicts with the immutable Slurm task grid.**  
   [run_stage.py:34](/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6/scripts/research_directions/temporal_gait/run_stage.py:34) freezes the entire `cfg.to_dict()`, including `resume_from`. [submit.py:90](/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6/scripts/research_directions/temporal_gait/submit.py:90) checks that grid before every actual submission. Setting `resume_from` therefore conflicts with the original grid containing `None`, although run identity and training signatures deliberately exclude this runtime field.
   
   **Consequence:** a compatible resume can work through direct execution yet be rejected by the submission wrapper. The documented notebook retry alone also does not select a checkpoint.  
   **Minimal fix:** canonicalize runtime-only resume fields consistently across grid and run identities; document explicit checkpoint selection.  
   **Falsification test:** freeze a grid, change only `resume_from`, and mock-submit the compatible task successfully; changed scientific settings must still fail.

4. **P2 — The analysis identity does not bind the written protocol.**  
   [contracts.py:69](/Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6/src/gavd6_sjepa/research_directions/temporal_gait/contracts.py:69) fingerprints code, notebooks, launchers and dependencies, but excludes `protocol/protocol.md` and the governing decision document. The analysis lock contains selected rules, not the complete reviewed protocol.
   
   **Consequence:** interpretation, scope or threshold-justification text can change after locking without invalidating `verify_run` or test access.  
   **Minimal fix:** retain and hash the governing protocol and analysis-decision snapshot.  
   **Falsification test:** modify the protocol after locking; test access must reject it or continue against an explicitly retained original snapshot.

**Nonblocking discrepancy:** notebook regeneration equality fails. An in-memory comparison identified markdown differences in notebook `00`, `03` and `04`, cell 3; executable cells were not the discrepancy. Regenerate before freezing. This may reflect ongoing edits rather than a settled implementation.

**Checks and limits:** inspected the checklist/contracts, current protocol/decisions, implementation, historical geometry/CE paths, evaluation reductions, source access and scheduler interfaces. The existing unittest run reported **47 tests: 25 passed, 21 errors from unavailable scratch directories, one notebook-equality failure**. This does not verify checkpoint resume, fresh-kernel execution or mocked submissions requiring scratch.

The inspected paths support separate context/teacher inputs, stopped teacher gradients, equal-example loss, uncapped bout indexing, paired target-defined scoring and train-only fitted readouts. They do not establish upstream extraction causality, real coverage, GPU behavior or utility.

Anchors were rechecked against working-tree SHA256 prefixes: contracts `69da3cd822e1`, workflow `eab2a635d1e9`, evaluation `009ee1a03513`; later edits require rechecking. No files were edited, assets located, jobs submitted or expensive training run. **Scientific empirical gate 3 remains pending real artifacts; a positive model result is not required.**