# Notebook 17: pretraining performance review

Reviewed on 8 September 2026. The main blocker was a **CPU-only PyTorch wheel**
despite an NVIDIA RTX 4090 Laptop GPU. The working CUDA kernel is now
**GAVD5 CUDA (PyTorch 2.13)**, backed by `.venv-cuda/Scripts/python.exe` in the
repository root. The original `.venv` and its running kernels remain intact.

## 1. Select and verify the kernel

Select **GAVD5 CUDA (PyTorch 2.13)** in both Notebooks 17 and 18. Run from the
configuration cell and check the hardware card: CUDA must resolve successfully.
The inspected machine has 16,376 MiB GPU memory, driver 596.08, an i9-13980HX
(24 cores/32 logical processors), and approximately 64 GiB system RAM.

To reproduce the installation from the repository root:

```powershell
.\neurips-laterality\scripts\setup_cuda_kernel.ps1
```

The script creates a separate environment, pins `torch==2.13.0` from the
official CUDA 13.0 wheel index, copies the other dependency versions from the
existing environment, verifies a CUDA matrix multiplication, and registers
the kernel. PyTorch publishes separate CPU and CUDA wheel installation
commands; an installed NVIDIA driver alone does not select a CUDA wheel.
[Official PyTorch installation commands](https://pytorch.org/get-started/previous-versions/)

The verified build is `2.13.0+cu130`. Real `DEVICE="auto"` training now refuses
an accidental CPU run when NVIDIA hardware is detected but CUDA is unavailable.
Use `DEVICE="cpu"` explicitly for an intentional CPU pilot.

## 2. Choose the numerical mode before running the grid

Keep `PRECISION="fp32"` for the reference numerical mode. For the faster tested
CUDA mode, set `PRECISION="bf16"` in **both** notebooks, or set
`LATERALITY_MOTION_PRECISION=bf16` before starting their kernels.

BF16 autocast applies to transformer and projector computation. Parameters,
optimizer state, EMA teachers, target centers, sharpened softmax and loss
reductions remain FP32. The VICReg covariance calculation also stays FP32.
Frozen evaluation is always FP32. Native BF16 support is required; there is
no silent CPU, FP16 or emulated-BF16 fallback. These explicit precision
boundaries follow PyTorch's autocast model of choosing precision per operation.
[PyTorch AMP documentation](https://docs.pytorch.org/docs/2.14/accelerator/amp.html)

Numerical mode is part of training and evaluation compatibility. BF16 and FP32
jobs cannot satisfy one another's cache identities. FP32 matrix multiplication
uses `highest` within the call, and restores the previous policy afterward.
Cross-device or mixed-precision bitwise equivalence is not assumed.
[PyTorch CUDA numerical policies](https://docs.pytorch.org/docs/2.14/notes/cuda.html)

### FP8 and 8-bit quantization decision

The registered grid intentionally accepts only `PRECISION="fp32"` and
`PRECISION="bf16"`. Do not enter `"fp8"`, `"int8"`, or an 8-bit optimizer in
the Notebook 17 configuration: the runner rejects unsupported precision names
before training. The options solve different problems and need different
validation.

| Option | Decision for the current grid | Reason |
|---|---|---|
| FP8 matrix training | Future isolated benchmark only | Ada hardware can execute FP8, but the current native-Windows stack has no FP8 training package and the measured model is small |
| 8-bit AdamW state | Do not use | The three motion arms could save only about 12 MiB of moment state, versus about 684 MiB measured peak allocation; fused AdamW is already active |
| INT8 weights/activations | Inference or deployment study only | It does not accelerate this gradient-based pretraining loop and would change the frozen representation being evaluated |
| BF16 compute with FP32 state/reductions | Recommended accelerated mode | It is implemented, identity-separated, tested for exact resume, and measured on this machine |

The RTX 4090 Laptop GPU is Ada compute capability 8.9, whose fourth-generation
Tensor Cores include FP8 support. Hardware support alone is insufficient.
FP8 training also needs quantized linear modules, scale/amax state, compatible
shapes, checkpoint support, and accuracy validation.
[NVIDIA Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)

The two practical FP8 software routes do not fit the registered run today:

- NVIDIA Transformer Engine lists Linux x86-64 as a prerequisite, while this
  workflow currently runs native Windows. Its FP8 layers maintain scaling and
  amax history, and supported linear dimensions must be divisible by 16. The
  model's initial 12-to-96 patch projection would therefore remain outside that
  path unless the architecture or representation changed.
  [Transformer Engine installation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html),
  [FP8 training requirements](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- TorchAO can convert eligible `Linear` modules to Float8Linear, but its official
  examples use `torch.compile` for competitive performance and demonstrate the
  gains on much larger transformer workloads. This Windows notebook has not yet
  established a compilation benefit. The BF16 profile is led by efficient
  attention backward/forward kernels, which a linear-only conversion would not
  automatically replace.
  [TorchAO float8 pretraining](https://docs.pytorch.org/ao/stable/eager_tutorials/pretraining.html)

The trainable online model and projector contain about 719,424 parameters per
arm. FP32 Adam moments occupy about 5.49 MiB per arm. Even the theoretical
three-arm saving from reducing both moments from 32 to 8 bits is only about
12.35 MiB; excluding small tensors leaves about 12.02 MiB. That is roughly 1.8%
of the measured BF16 peak and does not address the profiled attention cost.
Eight-bit optimizers are most useful when optimizer state is a material memory
constraint. They also retain small or sensitive tensors at higher precision by
default.
[bitsandbytes 8-bit optimizer guidance](https://huggingface.co/docs/bitsandbytes/optimizers)

Reconsider FP8 only as a new numerical experiment. A candidate must have its
own environment and artifact identity, record format and scale-state recipe,
preserve FP32 teacher EMA, centers, temperature-softmax and VICReg reductions,
and pass interruption/resume validation. Benchmark at least 100 post-warmup
paired updates against BF16, including preparation and checkpoints. Proceed to
a longer pilot only if it gives a repeatable end-to-end gain of at least 15%,
has no non-finite values or underflow alarms, and retains acceptable loss and
feature-variance trajectories. Any eventual held-out comparison must repeat all
five folds and seeds and report FP8 as a separate numerical condition.

Keep the registered batch size, model width/depth and 1,200 updates unchanged.
Changing batch size changes exposure and the VICReg covariance estimate.
Mask replacement still encodes the full token grid; a high mask ratio does
not make this encoder a visible-token-only MAE.

## 3. What now saves time

| Path | Implemented behavior | Experimental boundary |
|---|---|---|
| Fold inputs | Reuse a training-only device tensor bank across seeds and experiments; release it at the next fold | Outer-test rows never enter this bank |
| Motion scores | Memoize robust medians and MAMP logits by clip content and policy | Random masks still use the original independent RNG streams |
| Target selection | Prepare integer indices once; use fixed-shape gathers instead of CUDA boolean indexing | Ragged targets retain per-clip loss weighting |
| Optimizer/teacher | Fused CUDA AdamW and multi-tensor EMA operations | Same betas, momentum, clipping and separate arm states |
| Numerical acceleration | Optional BF16 transformer/projector operations | Separate cache identity; FP32 reductions/evaluation |
| Recovery | Pass `RESUME_INTERVAL=100` to every real job | Checked model, optimizer, projector and history at a shared arm boundary |
| Frozen evaluation | Reuse resident evaluation inputs, reduce features on-device, avoid redundant validation/boolean-selection synchronization | Inputs checked once and content-checked before reuse; FP32 features |
| Readouts | Pass newly trained encoders directly to evaluation; fit identical initial/direct-pose controls once per job | Same training-source ridge selection |
| Complete reports | Reuse content-checked per-job and pooled tables, including bootstrap intervals | Corrupt, incompatible or partial artifacts fail closed |

The optimizer loop already uses resident data, `zero_grad(set_to_none=True)`,
inference mode for evaluation, and buffered scalar diagnostics. A DataLoader
or pinned-memory workers would add little to this small resident dataset.
The chosen optimizations target measured work and synchronization, consistent
with the official tuning guide.
[PyTorch performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html)

`torch.compile`, activation checkpointing and concurrent GPU jobs are not
enabled. Compilation needs a measured amortization benefit on the actual
Windows stack; activation checkpointing trades compute for memory that this
model does not currently need. GPU-memory occupancy is not a throughput target.
FP8 and 8-bit optimizer packages are not installed in the CUDA environment.

The BF16 profiler recorded `aten::_efficient_attention_forward/backward`,
CUTLASS BF16 attention and Tensor Core GEMMs. Let SDPA select a compatible
kernel with the missing-token masks intact. Do not remove validity masks to
force a particular attention implementation.
[PyTorch SDPA backend selection](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)

## 4. Measured throughput and reproducible probes

These are short, real-GAVD fold-0/seed-42 probes with batch 20, width 96,
four encoder layers, two predictor layers, four heads and three motion arms.

| Measurement | Median seconds per paired update | Peak allocated CUDA memory |
|---|---:|---:|
| Existing CPU path, 4 updates | 3.853 | — |
| Existing CUDA FP32 path, 4 updates | 0.208 | 1,081 MiB |
| Optimized CUDA FP32, 12 updates | 0.177 | 1,084 MiB |
| Optimized CUDA BF16, 12 updates | 0.096 | 684 MiB |
| BF16 follow-up, 40 updates | 0.099 | 686 MiB |

The optimized 12-update measurements exclude two warmup updates and the final
snapshot boundary. The initial four-update checks used three intervals after
the first update, including the final snapshot. Approximate CPU-to-CUDA
speedups therefore describe short throughput checks, not an end-to-end
speedup guarantee. Thermal state, notebook activity, preparation, checkpoint
I/O, readouts and ridge fitting affect total time. Profiling adds overhead.

A separate 60-batch real-data mask probe measured 5.246 seconds without score
reuse, 2.895 seconds with a cold cache, and 1.919 seconds with a warm cache.
All masks **and coverage metadata were exactly equal**. The cache contained
690 entries (two policies for 345 distinct sampled training clips).

Reproduce a bounded probe from the repository root; no scientific checkpoint
is written:

```powershell
.venv-cuda/Scripts/python.exe neurips-laterality/scripts/benchmark_motion_pretraining.py --precision fp32 --output work/pretraining-performance/recheck-fp32.json
.venv-cuda/Scripts/python.exe neurips-laterality/scripts/benchmark_motion_pretraining.py --precision bf16 --output work/pretraining-performance/recheck-bf16.json
.venv-cuda/Scripts/python.exe neurips-laterality/scripts/benchmark_motion_pretraining.py --precision bf16 --steps 8 --profile --output work/pretraining-performance/recheck-profile.json
```

Notebook 17 includes the same optional 12-update probe through
`notebook_progress.py`. Leave `RUN_BENCHMARK=False` for routine Run All.
Reports and the original profiler trace are retained locally in
`work/pretraining-performance/`; a compact inventory is in
[the measurement summary](figures/motion_pretraining_performance_summary.json).

## 5. Validation and interpretation

Validation covers exact cached/uncached mask equality, matching reference
losses and gradients for equal and ragged target counts, teacher gradient
isolation, cache separation by numerical mode, rejection of stale evaluation
inputs, and restoration of numerical settings after failure. Interrupted CPU
and CUDA BF16 jobs resumed to exactly the uninterrupted final model weights.

A real GPU integration pilot trained all five motion/region encoders for
12 updates each on 436 training clips, evaluated all 189 held-out clips
(7,560 prediction rows), and reopened the checked report without training or
repeating its bootstrap. Fresh synthetic notebooks and real-input notebook
preflights provide separate execution checks.

The full 125-encoder, five-fold/five-seed, 1,200-update real grid has **not**
been run as part of this performance review. Finite pilots establish software
operation, not equivalent BF16 convergence or improved movement prediction.
Run the declared full grid in Notebook 17 and interpret its held-out outputs
in Notebook 18 using the existing source-balanced protocol.
