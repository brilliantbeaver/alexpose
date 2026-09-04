# Compute contract for candidate screening

## Local anchor

The user supplied a conservative anchor of 3 hours for 100 epochs of AMASS joint-embedding predictive architecture training. A joint-embedding predictive architecture, or JEPA, predicts hidden internal vectors instead of reconstructing coordinates.

The executed run record is `outputs/repaired-jepa-seed7-v2/summary.csv`. It reports 7,326 seconds, or 2.04 hours, for 100 epochs. The launcher `slurm/train-amass-core11-full.sbatch` requests one H100. No distributed launcher appears in that job. Candidate estimates therefore use the more conservative user anchor:

`1 H100 x 3 hours x epochs / 100 = 0.03 x epochs H100-hours per JEPA-equivalent run`.

Examples:

- 25 epochs: `1 x 3 x 25 / 100 = 0.75 H100-hours`.
- 100 epochs across three seeds: `1 x 3 x 1 x 3 = 9 H100-hours`.
- Four matched 100-epoch arms across three seeds: `1 x 3 x 4 x 3 = 36 H100-hours`.

The eight available H100s can run independent arms or seeds concurrently. Do not divide GPU-hours by eight. Divide only wall time for jobs that actually run in parallel. Do not assume linear distributed scaling for one model without evidence.

## External adaptation anchors

- GoalForce trains one Wan2.2 high-noise ControlNet for under 48 hours on four A100 80 GB GPUs. Its published upper bound is `4 x 48 = 192 A100-hours`. H100 time is unknown.
- Masked Visual Actions trains two Wan2.2 low-rank adaptations for four days on eight H200 GPUs. Its reported cost is `8 x 96 = 768 H200-hours`. H100 time is unknown.
- GaitDynamics trains its small diffusion model for 30 hours on one RTX A6000. Reusing its released checkpoint avoids this cost.
- SC3-Eval trains for 2.2 days on 32 GB200 GPUs. Its reported cost is `32 x 52.8 = 1,689.6 GB200-hours`, so reproduction is outside this program.

These hardware types are not interchangeable units. Candidate screening reports the published hardware-hour figure and does not invent a conversion to H100-hours.

## Screening rule

A 72-hour first result must use frozen inference, a small head, or a short adapter run. Any full Wan adaptation needs a reduced pilot with a predeclared stop rule. A candidate fails feasibility if its only decisive test requires reproducing a foundation-model training run or an unavailable checkpoint.
