# Historical AMASS Core11 pilot-probe gate

`outputs/run1/seed-7_standard.pt` is an immutable historical training artifact. It was produced by commit `c2385c125d41698451486cc7cc30f1892d3ec773`, not by the current packaged implementation. The current package has no `standard` variant, so it must not be used to load this checkpoint.

The historical code is restored in a detached, clean worktree at:

```text
/Users/theodoremui/dev/alexpose-amass-core11-c2385c1
```

The source worktree is intentionally separate from the current repository. Do not modify it. The original checkpoint's expected SHA-256 is:

```text
4741bbbcdb4ba16cf0b75a798b4f7e1fea2c5aeccf41da17250822c3ebd7000b
```

## Verify before any feature extraction

From the current GAVD6 directory:

```bash
.venv/bin/python scripts/verify_historical_amass_checkpoint.py
```

The verifier fails closed unless all of the following match:

1. The historical worktree is detached at `c2385c1` and has no edits.
2. Both historical source files match the SHA-256 values stored in `run_config.json` and the checkpoint metadata.
3. The checkpoint SHA-256 matches its recorded immutable value before and after loading.
4. The historical `standard` model and projector load with `strict=True`.
5. The frozen EMA `target_encoder` produces finite features for a canonical Core11-shaped pilot input.

It writes a new sidecar at `outputs/run1/pilot_probe_c2385c1/compatibility_report.json`; it never writes the `.pt` checkpoint. Re-running does not overwrite that report unless `--overwrite-report` is supplied explicitly.

The local verifier may use a different PyTorch runtime than training. Its purpose is architecture and state-dict compatibility. Run the same verifier on HAIC with the original CUDA environment before extracting the real probe embeddings.

## Pilot-probe boundary

The verified output is suitable for **feature extraction only**, not a downstream performance claim. Use `model.target_encoder`, which is the saved frozen EMA encoder, and declare that choice before fitting a probe. Its pilot output has two branches, each with 176 tokens and 96 features for a 64-frame Core11 window.

A performance pilot additionally requires:

1. A labelled downstream task whose coordinates are transformed into the same Core11 body-frame contract.
2. A source- or identity-disjoint split chosen before fitting, with no downstream test identity exposed to representation training for an inductive claim.
3. Frozen encoder weights; tune the linear/ridge probe only inside the training portion.
4. A raw-coordinate baseline and a randomly initialized matched-encoder baseline.
5. The same feature pooling, split, tuning, and metric for all variants and, for a final comparison, all matched seeds.

Record this result as `pilot` and `historical-AMP-skip-confounded`: the standard checkpoint contains 248,495 successful AdamW updates out of 248,600 scheduler attempts. Do not use it alone to rank variants or make a final downstream-performance claim.
