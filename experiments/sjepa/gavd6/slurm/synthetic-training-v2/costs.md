# Costs, retries, and later runs

The managed launcher handles cost ledgers and phase configurations. You do not need to create `next-scope.py`, calculate `measured_gpu_hours`, or edit JSON between stages. This page describes the recovery commands and accounting rules behind the [four-step guide](README.md).

## What the budget means

`init --gpu-hours N` declares the total authorized GPU hours for the named scope. `--prior-gpu-hours P` declares costs already charged to that scope, including relevant setup and failed allocations. Both numbers are explicit. Alternatively, supply `--prior-ledger FILE` to carry earlier records forward.

Preparation reserves one one-hour H100 allocation. Source training reserves two sequential one-hour H100 allocations. Within each GPU stage, the Python guard has a 54-minute limit, leaving time for initialization and cleanup; the scheduler limit remains the outer bound. These limits are not promises that the chosen panel and recipe will finish within an hour.

Before moving to the next phase, the launcher reads each recorded job's top-level `sacct` allocation. It counts failed attempts and preparation, and adds one second for rounding on an allocated GPU job. It does not also count `.batch`, `.extern`, per-fit timers, or the overlapping preparation timers. A canceled pending job with no allocation has no GPU cost.

Submission requests `--no-requeue`; use explicit retries so each attempt has its own job ID. If accounting is delayed, a job is still active, or duplicate allocation records appear, the launcher waits for the discrepancy to be resolved instead of freezing an incomplete ledger. Ordinary CPU jobs consume no GPU budget. Any unexpected GPU allocation shown by accounting is retained in the cost history.

Configurations and prior ledgers are immutable once used. A preparation retry gets a new scope and `paired-XX` directory. Source retries keep the same source configuration and compatible checkpoints; allocation overhead missing from a Python attempt record is added once as a separate cost record.

This accounting covers the jobs recorded by this managed run plus the explicit prior ledger. Separately submitted setup jobs or other runs must be included in the prior total. Separate named runs do not share a live budget lock; do not spend one allowance concurrently through several runs.

## Retry a failed job

First inspect the state and printed log paths:

```bash
bash slurm/synthetic-training-v2/submit.sh status
```

Correct the actual error before retrying. A failed preparation can then be retried with:

```bash
bash slurm/synthetic-training-v2/submit.sh prepare --retry
```

This preserves the failed output, carries its cost forward, and writes to the next `paired-XX` directory. Inspect the new overlays before training.

If a source stage fails, later jobs may remain pending on its failed `afterok` dependency. Cancel those pending dependent job IDs shown by `status`, using `scancel JOB_ID ...`; do not cancel unrelated jobs. Wait until all recorded jobs have terminal states. Then:

```bash
bash slurm/synthetic-training-v2/submit.sh source --retry
```

The launcher checks run identity and every retained receipt, skips completed stages, and builds a new dependency chain for the remaining stages. It preserves every earlier job ID and compatible training checkpoints. Repeating a normal `source` submission without `--retry` is rejected.

If the cause requires a different scientific recipe, code, or source data, start a new named run. Do not remove receipts or edit a frozen configuration to force a resume.

## Recover an uncertain submission

A scheduler client can fail after Slurm accepted a job. The launcher records a unique job-name token before calling `sbatch` and blocks further submissions until that uncertainty is resolved.

Run `status` to see the token and recorded command. Inspect your queue and recent accounting:

```bash
squeue -u "$USER" -o '%.18i %.64j %.12T'
sacct -u "$USER" --format=JobID,JobName%64,State,ExitCode,ElapsedRaw
```

Find the exact `stv2-STAGE-TOKEN` job name. If Slurm accepted it, record its numeric ID:

```bash
# Replace 12345 with the matching accepted job.
bash slurm/synthetic-training-v2/submit.sh recover --job-id 12345
```

The helper verifies the job name and owner through accounting. If accounting has not caught up, wait and repeat the recovery command. If you have established that the scheduler rejected the command and no job was created, use:

```bash
bash slurm/synthetic-training-v2/submit.sh recover --not-submitted
```

Then run `status` and the appropriate retry command. An empty queue by itself does not prove that a job was never accepted; check accounting too.

## Carry costs into a later experiment

After all jobs are terminal, export the complete managed cost ledger:

```bash
bash slurm/synthetic-training-v2/submit.sh ledger \
  --output "$STV2_WORK/total-costs.json"
```

The export refuses to overwrite an existing file. Use a new filename if you have accumulated more attempts since an earlier export.

Initialize the next run with `--prior-ledger "$STV2_WORK/total-costs.json"` instead of `--prior-gpu-hours`. Its `--gpu-hours` value is the total allowance including those prior costs. Choose `--updates` and `--seeds` at initialization, based on measured throughput, before sourcing the new run's `session.env`.

The default comparison matches data and update budgets. The separate low-level `equal_total_compute` option remains bounded by its update cap: the flag alone does not establish equal actual compute. Do not describe that comparison as matched compute unless the saved timings support it. Changes to endpoints, objective controls, or claim scope require a declared protocol and an appropriate new experiment.

## Existing manual runs

The managed launcher starts fresh named runs and does not adopt or rewrite earlier manually assembled scopes. Keep their configurations, ledgers, and logs. Include their measured costs when initializing a managed run. The low-level `run.py`, `prepare.py`, and `submit.py` remain available for those existing workflows, but phase bookkeeping is now handled by `submit.sh` for new runs.
