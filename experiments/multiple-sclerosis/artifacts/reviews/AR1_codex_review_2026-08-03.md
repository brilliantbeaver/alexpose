## Findings

- **P0 — Completed RF baseline is not reproducible across the two “paired” runners.**  
  [scripts_phase0_provenance.py:214](/Users/pmui/dev/alexpose/experiments/multiple-sclerosis/scripts_phase0_provenance.py:214), [scripts_r1_repaired.py:79](/Users/pmui/dev/alexpose/experiments/multiple-sclerosis/scripts_r1_repaired.py:79)  
  Same registry, features, seed 42, and 47 clips nevertheless produce five different RF predictions. Phase 0 pooled macro-F1 is `0.72499`; R1 reports `0.66657`. Phase 0 used Python 3.14.6, R1 Python 3.12.13, while neither manifest captures sklearn/ambient versions.  
  **Impact:** the comparator metric is environment-dependent and not reproducible; the claimed paired baseline is unstable.  
  **Correction:** freeze and record the complete environment and feature-matrix hash, or consume one locked RF OOF artifact.  
  **Test:** both runners must produce byte-identical RF OOF predictions and feature hashes.

- **P0 — The completed R1 result does not identify its producing code.**  
  [scripts_r1_repaired.py:199](/Users/pmui/dev/alexpose/experiments/multiple-sclerosis/scripts_r1_repaired.py:199), [scripts_phase0_provenance.py:286](/Users/pmui/dev/alexpose/experiments/multiple-sclerosis/scripts_phase0_provenance.py:286)  
  Manifests record HEAD `f21aa607…`, but nearly all reviewed repairs and all result artifacts are untracked, while `models.py` is modified. Checking out the recorded revision cannot reconstruct the run.  
  **Impact:** reported metrics and checkpoints are scientifically unverifiable from provenance.  
  **Correction:** require a clean committed tree, or record hashes/diffs for every execution file.  
  **Test:** fail manifest creation when dirty, or verify recorded code hashes against disk.

- **P1 — Resume does not restore the RNG used by the recorded MPS run.**  
  [train_v2.py:153](/Users/pmui/dev/alexpose/experiments/multiple-sclerosis/sjepa/train_v2.py:153), [train_v2.py:197](/Users/pmui/dev/alexpose/experiments/multiple-sclerosis/sjepa/train_v2.py:197), [train_v2.py:249](/Users/pmui/dev/alexpose/experiments/multiple-sclerosis/sjepa/train_v2.py:249)  
  `torch.get_rng_state()` explicitly captures only the CPU generator. `random_view` draws on `x.device`; the completed R1 manifest says `device=mps`. No `torch.mps.get_rng_state()` or CUDA states are saved.  
  **Impact:** MPS/CUDA resume diverges at the first augmentation.  
  **Correction:** save/restore device-specific RNG states.  
  **Test:** interrupted versus uninterrupted equality on each supported accelerator, including predictions and parameters.

- **P1 — Schedule state is not saved; the resume test supplies hidden external knowledge.**  
  [train_v2.py:124](/Users/pmui/dev/alexpose/experiments/multiple-sclerosis/sjepa/train_v2.py:124), [train_v2.py:136](/Users/pmui/dev/alexpose/experiments/multiple-sclerosis/sjepa/train_v2.py:136), [test_train_v2.py:112](/Users/pmui/dev/alexpose/experiments/multiple-sclerosis/sjepa/tests/test_train_v2.py:112)  
  The checkpoint contains CE, optimizer, step, CPU RNG, mask RNG, and data RNG—but not `schedule_updates`, warmup, seed, mask parameters, or device RNG. The test manually gives both segments horizon 8. That is valid for a pre-planned stop, but it does not establish the broader “resume an interrupted run” claim or default API behavior.  
  **Impact:** an ordinary first call with `total_updates=4`, followed by resume to 8, uses a different LR/EMA trajectory.  
  **Correction:** persist and validate the complete training specification.  
  **Test:** resume using only checkpoint state, with no repeated schedule arguments.

- **P2 — The strict path invokes the label-producing dataset getter.**  
  [train_v2.py:168](/Users/pmui/dev/alexpose/experiments/multiple-sclerosis/sjepa/train_v2.py:168)  
  `dataset[i][0]` first executes the complete `__getitem__`, including label construction, then discards element 1. No label affects the objective and no VICReg is used, so I found no mathematical label leakage—but “never reads y” is literally false.  
  **Impact:** the firewall depends on dataset behavior and permits label-dependent side effects.  
  **Correction:** retrieve windows from a label-free dataset/view.  
  **Test:** training must succeed with a dataset whose label accessor raises.

- **P2 — D1 permutation test does not test hidden target slots.**  
  [test_correctness.py:71](/Users/pmui/dev/alexpose/experiments/multiple-sclerosis/sjepa/tests/test_correctness.py:71)  
  It sets every token visible, then permutes `spatial_pos`; it proves general sensitivity to positional parameters, not correct identity at hidden slots or correct scattering. `randperm` can also theoretically be identity.  
  The implementation itself appears correct: direct probing gave hidden-position std `0.053–0.059`; zeroing both positional tables reduced it to approximately zero.  
  **Correction/test:** use a deterministic nonidentity permutation and inspect only hidden outputs; independently perturb spatial and temporal tags.

- **P2 — Diagnostic failures are swallowed and do not prevent a “COMPLETED” result.**  
  [train_v2.py:93](/Users/pmui/dev/alexpose/experiments/multiple-sclerosis/sjepa/train_v2.py:93), [scripts_r1_repaired.py:113](/Users/pmui/dev/alexpose/experiments/multiple-sclerosis/scripts_r1_repaired.py:113), [scripts_r1_repaired.py:210](/Users/pmui/dev/alexpose/experiments/multiple-sclerosis/scripts_r1_repaired.py:210)  
  Any SVD exception becomes `NaN`; R1 performs no finite-value gate before writing `COMPLETED.json`.  
  **Impact:** collapse diagnostics can silently become unusable while the run is certified successful.  
  **Correction:** raise or mark the run failed.  
  **Test:** inject an SVD failure and require non-completion.

## Verified

D1 position layout matches `t * V + v`; target indexing aligns. For the production 33-joint configuration, masks always retained context in sampled banks. One-visible-token attention was finite; I found no all-masked production row. Fold registry sources were disjoint, all 47 clips appeared exactly once in OOF, probe/scalers were fit train-only, and the embedding mask was fixed before test embedding. Shortcut-control scaling is train-only.

I could not execute pytest because the read-only environment provides no writable temporary directory. I used direct read-only probes instead. MPS-specific execution could not be run because MPS is unavailable in this environment. No files were modified.
