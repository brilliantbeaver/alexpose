"""Editable GAVD tutorial: paired pretraining over five source folds and seeds."""
from nbformat.v4 import new_notebook
from .motion_results_20260908 import add_saved_result_interpretation
from .masking_shared import (md, code, setup_cell, data_instructions, configuration_cell, inputs_cell, plan_cell)


def build_notebook():
    notebook = new_notebook(cells=[
        md('''
        # 17 — Train the motion and structure JEPA grid on real GAVD

        This is the training continuation of notebooks 15 and 16. The default
        data path uses the same **625 GAVD clips, 93 source videos, five outer
        folds and seeds 42--46** as the existing real-data investigations.
        Every experiment is repeated over all 25 train/test combinations.

        Set `RUN_TRAINING = True` in the configuration cell to execute the grid,
        or set `LATERALITY_RESEARCH_RUN_REAL=1` before starting the kernel, as in
        Notebook 12. With the switch off, Run All still prepares/verifies real
        inputs, displays the full workload and checks saved-job availability.
        This keeps the expensive training decision visible before optimization.
        Saved outputs are consumed directly by Notebook 18 in a fresh kernel.
        '''),
        setup_cell(), data_instructions(), configuration_cell(), inputs_cell(),
        md('''
        ## 2. Freeze the questions and training recipe

        | Experiment | Paired arms | Real grid |
        |---|---|---:|
        | Motion | Uniform, MAMP code convention, robust motion mixture | 3 × 5 folds × 5 seeds = 75 encoders |
        | Regions | Connected six-landmark half-window region, count-matched uniform | 2 × 5 folds × 5 seeds = 50 encoders |

        Together these are **50 paired jobs, 125 encoders and 150,000 optimizer
        updates**. An update trains every arm in its job on identical source
        draws and geometric views. The two uniform controls have different
        target budgets and remain separate. To conduct the declared trajectory
        or completion follow-up, add its experiment name here and in Notebook 18.
        A fold/seed subset is a pilot and is labeled as such.

        Like Notebook 12, use the starting recipe retained from Notebook 08:
        1,200 updates, batch 20, width 96, four encoder layers, two predictor
        layers and four attention heads. The tracked summary provides the
        settings without requiring old local checkpoints. Read its complete
        configuration below; do not inherit tiny teaching-model defaults.
        '''),
        plan_cell(),
        md('''
        ### Hardware, numerical, and cache preflight

        The requested device is only a preference until the **current notebook
        kernel's PyTorch build** resolves it. `DEVICE="auto"` chooses CUDA when
        this kernel has CUDA support, Apple MPS when available, and otherwise
        CPU. Seeing an NVIDIA adapter in Task Manager or `nvidia-smi` is not
        sufficient: a CPU-only PyTorch wheel still resolves `auto` to CPU. The
        next cell reports both views of the machine. If it detects NVIDIA but
        `torch.cuda.is_available()` is false, install a CUDA-enabled PyTorch
        build in the environment used by this kernel, restart the kernel, and
        rerun from the configuration cell. An explicit `DEVICE="cuda"` fails
        instead of silently falling back.

        **Windows setup for this workspace:** the review found an RTX 4090
        Laptop GPU (16 GiB) behind a CPU-only PyTorch kernel. A separate kernel,
        **GAVD5 CUDA (PyTorch 2.13)**, was installed and verified with an actual
        CUDA matrix multiplication. Select it in the notebook's kernel picker,
        then run from the top. To recreate it from the repository root:

        ```powershell
        .\\neurips-laterality\\scripts\\setup_cuda_kernel.ps1
        ```

        The script uses the official PyTorch CUDA 13.0 wheel and preserves the
        existing environment's other package versions. It does not restart your
        running kernels. Real training with `DEVICE="auto"` now stops when it
        detects NVIDIA hardware but cannot use CUDA; an intentional CPU pilot
        must use `DEVICE="cpu"` explicitly.

        Real training keeps only the current outer-training fold, its validated
        target masks, and its sampling schedule resident on the selected device;
        outer-test tensors remain sealed. Shared geometric views are generated
        on that device, and CUDA uses fused AdamW. Jobs run serially on one GPU
        so independent fold/seed models do not contend for memory.

        Content-keyed motion scores are reused across clip draws and seeds.
        These deterministic scores never include target labels. Every stochastic
        mask still uses its original seed, step and clip-offset stream. Fixed-size
        target gathers avoid CUDA synchronization from boolean indexing, and
        multi-tensor teacher EMA updates reduce small kernel launches. The grid
        reuses the training tensor bank until the outer fold changes.

        `PRECISION="fp32"` retains the reference numerical mode. For faster CUDA
        training, set **`PRECISION="bf16"`** in the configuration cell and use
        the same value in Notebook 18. Native BF16 support is checked before
        training. Transformer/projector matrix operations use autocast while
        model weights, AdamW states, teacher EMA, centers, sharpened softmax,
        variance/covariance losses and frozen evaluation remain FP32. BF16 has
        a **different training/cache identity**; its results must be reported as
        a separate numerical mode. A finite pilot does not establish identical
        long-run convergence.

        FP32 matrix reductions explicitly use the highest-precision policy;
        the caller's previous setting is restored afterward. `torch.compile`
        stays disabled: the native Windows environment and this small model
        need a separate compilation benchmark before paying compilation costs.
        PyTorch selects a compatible attention backend with the validity masks
        intact. The measured BF16 run used memory-efficient CUDA attention and
        Tensor Core GEMMs. Do not remove missing-token masks to force FlashAttention.

        FP8, INT8 and 8-bit AdamW are deliberately absent from the configuration.
        This Ada GPU has FP8-capable Tensor Cores, but the native-Windows CUDA
        environment has no FP8 training package, and FP8 needs explicit scaling
        state, compatible layers, resume support and a separate validation grid.
        Eight-bit Adam would save only about 12 MiB across the three motion arms
        against roughly 684 MiB measured BF16 peak allocation. INT8 is relevant
        to a later deployment benchmark, not this gradient-based pretraining or
        FP32 frozen-feature evaluation. See the performance review's
        [precision decision](docs/MOTION_PRETRAINING_PERFORMANCE.md#fp8-and-8-bit-quantization-decision).

        There are three distinct kinds of reuse:

        | Layer | What it saves | When it may count |
        |---|---|---|
        | Complete training cache | All encoder arms, histories, schedules, and controls | Only after identity, inventory, hashes, shapes, and finite values validate |
        | Fold tensors and clip scores | Host-to-device copies and repeated motion medians | In memory, within the current training fold; content changes invalidate scores |
        | Paired resume candidate | Model, projector, optimizer, and history at one shared update boundary | Only when periodic resume is enabled and its checksum and full state validate; never by itself a completed result |
        | Evaluation/grid cache | Frozen features, readouts, diagnostics, and pooled tables | Only after its training identity and table contents validate |

        `RESUME_INTERVAL=100` is connected to every real training job by default.
        A shared boundary stores every arm's model, projector, optimizer and
        history, with a checksum. At most the updates after the last saved
        boundary need repeating. Final checkpoints remain separate. Set zero
        to disable periodic recovery; it does not change the numerical identity.
        Per-job readout caches store reduced tables; resident evaluation inputs
        and features are transient. Identical initial/direct-pose readouts are
        fitted once per job, and a validated pooled report reuses its bootstrap.

        The inventory below prints the exact expected paths. A cache candidate's
        mere existence is never evidence. Changing data, folds, seeds, model or
        mask code, backend, PyTorch runtime, or numerical mode intentionally
        produces a different content identity rather than overwriting old work.
        Explicit synthetic mode remains a separate **one-fold, one-seed,
        one-update CPU software check** and produces no GAVD evidence.
        '''),
        code('''
        import shutil
        import subprocess
        import torch
        from laterality_extensions.masked_learning import configure_learning_runtime
        from laterality_extensions.motion_runtime import motion_numerical_policy

        def visible_nvidia_adapters():
            """Report NVIDIA hardware independently of PyTorch, without a shell."""
            executable = shutil.which("nvidia-smi")
            if executable is None:
                return pd.DataFrame(columns=["adapter", "memory_mib", "driver", "compute_capability"])
            command = [executable,
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits"]
            try:
                completed = subprocess.run(command, capture_output=True, text=True,
                    timeout=10, check=True,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            except (OSError, subprocess.SubprocessError):
                return pd.DataFrame(columns=["adapter", "memory_mib", "driver", "compute_capability"])
            rows = []
            for line in completed.stdout.splitlines():
                values = [value.strip() for value in line.split(",")]
                if len(values) == 4:
                    rows.append(dict(zip(
                        ("adapter", "memory_mib", "driver", "compute_capability"), values)))
            return pd.DataFrame(rows)

        # Synthetic plans deliberately replace any requested accelerator with CPU.
        effective_request = str(plan["settings"]["device"])
        runtime_error = None
        try:
            resolved_runtime = configure_learning_runtime(effective_request)
        except (RuntimeError, ValueError) as error:
            runtime_error = error
            resolved_runtime = {"device": "unavailable", "cpu_threads": None, "accelerated": False}

        nvidia_adapters = visible_nvidia_adapters()
        torch_cuda_ready = bool(torch.cuda.is_available())
        if not nvidia_adapters.empty and not torch_cuda_ready:
            accelerator_note = (
                "NVIDIA hardware is visible, but this kernel's PyTorch build cannot use CUDA. "
                "Install a CUDA-enabled PyTorch build in this kernel environment and restart."
            )
        elif torch_cuda_ready:
            accelerator_note = "CUDA is available to this notebook kernel."
        elif resolved_runtime["device"] == "mps":
            accelerator_note = "Apple MPS is available to this notebook kernel."
        else:
            accelerator_note = "This plan will use the CPU."

        runtime_card = pd.DataFrame([
            {"check": "data mode", "value": DATA_MODE},
            {"check": "user-requested device", "value": DEVICE},
            {"check": "effective plan request", "value": effective_request},
            {"check": "resolved training device", "value": resolved_runtime["device"]},
            {"check": "PyTorch", "value": torch.__version__},
            {"check": "PyTorch CUDA runtime", "value": torch.version.cuda or "none (CPU-only build)"},
            {"check": "cuDNN", "value": torch.backends.cudnn.version() or "unavailable"},
            {"check": "FP32 matmul policy", "value": torch.get_float32_matmul_precision()},
            {"check": "training precision", "value": plan["execution"]["precision"]},
            {"check": "periodic paired resume", "value": plan["execution"]["resume_interval"]},
            {"check": "kernel executable", "value": sys.executable},
            {"check": "torch.compile", "value": "disabled by this workflow"},
            {"check": "real training enabled", "value": RUN_TRAINING},
            {"check": "readiness", "value": accelerator_note},
        ])
        display(runtime_card.style.hide(axis="index"))
        if not nvidia_adapters.empty:
            display(nvidia_adapters.style.hide(axis="index").set_caption(
                "NVIDIA adapters visible to the operating system"))
        if runtime_error is not None:
            raise RuntimeError(
                "CUDA preflight failed before training. "
                f"Requested device: {effective_request!r}. {runtime_error} "
                "A running notebook cannot change its Python environment: select "
                "'GAVD5 CUDA (PyTorch 2.13)' in the kernel picker, restart the kernel, "
                "and Run All from the top."
            ) from runtime_error
        print("Effective numerical contract:", motion_numerical_policy(
            resolved_runtime["device"], plan["execution"]["precision"]))
        '''),
        md('''
        ### Measure a bounded pilot before the complete grid

        The optional cell runs **12 paired motion updates on real fold 0** using
        batch 20, width 96 and the declared four/two-layer model. It excludes
        warmup and the final weight-copy boundary from the steady-state timing.
        It writes a timing report, not a scientific checkpoint. Keep it disabled
        during routine Run All, or set `RUN_BENCHMARK=True` to inspect throughput
        and peak allocated GPU memory through the same progress wrapper.

        On this laptop, short measurements gave about 3.85 seconds per paired
        CPU update, 0.177 seconds with optimized CUDA FP32 and 0.096 seconds with
        CUDA BF16. These are throughput checks, not full-grid runtime promises;
        preparation, checkpoint writes, frozen readouts and CPU ridge fitting
        add time. The full mask schedule also occupies memory beyond this short
        pilot. See [the performance review](docs/MOTION_PRETRAINING_PERFORMANCE.md)
        for the measurements, profiler evidence and reproduction commands.

        Keep the scientific batch size at 20. A larger batch changes the number
        of clips seen and the VICReg covariance estimate. GPU memory occupancy
        alone does not measure speed. For this resident dataset, adding a
        DataLoader, pinned-memory workers or a second transfer stream would add
        machinery to an optimization loop that already needs no input transfer.
        '''),
        code('''
        from notebook_progress import run_notebook_task
        from scripts.benchmark_motion_pretraining import benchmark
        RUN_BENCHMARK = os.getenv("LATERALITY_MOTION_BENCHMARK", "0") == "1"
        benchmark_progress = NotebookTaskProgress("Real GAVD throughput pilot", "stage")
        timing_report = run_notebook_task(benchmark, progress=benchmark_progress,
            label="Twelve paired motion updates; warmup excluded from throughput",
            enabled=RUN_BENCHMARK and DATA_MODE == "gavd",
            device=resolved_runtime["device"], precision=plan["execution"]["precision"],
            steps=12, output=OUTPUT_ROOT / "performance" / f"{plan['execution']['precision']}.json")
        if timing_report is not None:
            display(pd.Series({key: value for key, value in timing_report.items()
                               if key not in ("identity", "history")}))
        '''),
        code('''
        workload = plan["workload"].groupby(["experiment", "fold"], sort=False).size().unstack(0)
        ax = workload.plot.bar(figsize=(8, 3.5), title=f"{DATA_MODE.upper()}: encoders across declared seeds")
        ax.set(xlabel="Outer fold", ylabel="Encoders")
        ax.figure.tight_layout(); display(ax.figure); plt.close(ax.figure)
        checkpoint_progress = NotebookTaskProgress("Training checkpoint inspection", "stage")
        jobs_before = grid_status_with_progress(plan, inputs, progress=checkpoint_progress)
        display(jobs_before)

        from laterality_extensions.comparative_training import _resume_checksum_path, _resume_path
        cache_rows = []
        for job in jobs_before.itertuples(index=False):
            training_directory = Path(job.training_directory)
            resume_path = _resume_path(training_directory)
            checksum_path = _resume_checksum_path(resume_path)
            resume_files = (resume_path.is_file(), checksum_path.is_file())
            resume_state = (
                "paired candidate present; validate before resume"
                if all(resume_files) else
                "incomplete candidate; fail closed" if any(resume_files) else
                "absent"
            )
            cache_rows.append({
                "experiment": job.experiment,
                "fold": job.fold,
                "seed": job.seed,
                "complete_cache": (
                    "validated complete" if str(job.training_status).startswith("complete") else "missing"
                ),
                "resume_cache": resume_state,
                "training_directory": str(training_directory),
                "resume_path": str(resume_path),
                "resume_checksum_path": str(checksum_path),
            })
        cache_inventory = pd.DataFrame(cache_rows)
        cache_summary = (cache_inventory.groupby(
            ["complete_cache", "resume_cache"], dropna=False).size()
            .rename("paired_jobs").reset_index())
        display(cache_summary)
        display(cache_inventory)
        print("Exact cache inventory is retained in `cache_inventory`; output root:",
              Path(plan["output_dir"]).resolve())
        '''),
        md('''
        ## 3. Trace one fold through the information boundaries

        1. Take only the current fold's training videos for self-supervised
           optimization. Sample videos uniformly and clips within videos; many
           clips from one recording must not dominate the source schedule.
        2. Initialize the online encoder, predictor and regularizer projector
           once per job. Before the first optimizer update, save an exact model
           copy as the matched initial control, then copy the same starting
           parameters into every mask arm. Seeds are explicit, and each fold/seed
           starts a new model. The saved control stays unchanged while the other
           copies are trained.
        3. Use separate streams for clip exposure, shared geometric views and
           arm-specific mask draws. Validate every scheduled mask before the
           first optimizer update. The mask sampler never reads the target label.
        4. Feed visible context to the online encoder/predictor and full valid
           input to the teacher. Optimize centered teacher-feature cross-entropy
           plus the shared unmasked feature-variation regularizer. Average target
           loss within each clip, then average clips, even with ragged masks.
        5. Keep the teacher gradient-free; update its parameters by EMA after
           each online optimizer step. Save online, teacher, predictor, target
           center, initial features, schedules, policies and training histories.
        6. Freeze the encoder. Fit preprocessing and ridge readout using only
           training sources, then predict every outer-test clip once for this
           seed/arm. Test outcomes cannot choose masks, checkpoints or penalties.

        The online and teacher encoders are also related within each trained
        model. They start with identical weights. Gradient descent updates the
        online encoder directly, and an exponential moving average of its weights
        updates the gradient-free teacher after each optimizer step. Notebook 18
        therefore compares the saved starting encoder with both descendants of
        that same initialization: the final online encoder and the final teacher.

        The objective retains AdamW betas (0.9, 0.95), constant learning rate,
        temperatures 0.06/0.10, center momentum 0.9, gradient clipping at 1,
        rotations up to eight degrees and translations up to 0.03 prepared units.
        All arms retain the same 33-landmark input, twelve-landmark regularizer
        pool and five bilateral readout pairs. These remaining anatomical choices
        are part of the gait adaptation, not learned discoveries.
        '''),
        code('''
        from laterality_extensions.masked_learning import LearningSettings
        first_fold, first_seed = FOLDS[0], SEEDS[0]
        data = inputs["datasets"][first_fold]
        local = replace(LearningSettings(**plan["settings"]), fold=first_fold, seed=first_seed)
        rows = data.train_rows[:local.batch_size]
        display(pd.DataFrame({"sequence_id": data.sequence_ids[rows], "source_id": data.source_ids[rows], "role": "train"}))
        preview = []
        for experiment in EXPERIMENTS:
            arms = {name: StudyArm(**spec) for name, spec in plan["arms"][experiment].items()}
            masks, _ = paired_study_masks(data, rows, local, arms)
            for name, mask in masks.items():
                preview.append({"experiment": experiment, "condition": name,
                    "fold": first_fold, "seed": first_seed,
                    "smallest_target_count": int(mask.sum((1, 2)).min()),
                    "largest_target_count": int(mask.sum((1, 2)).max())})
        display(pd.DataFrame(preview))
        print("Preview only. The training runner validates the complete sampled schedule for every job.")
        '''),
        md('''
        ## 4. Execute or reuse the complete declared grid

        Confirm the printed mode, folds, seeds, arm definitions, settings and
        output directory, then enable the configuration switch to train.
        The updating progress display shows the fold, seed, experiment, mask
        preflight and optimizer update count. It then identifies frozen-feature
        encoding, ridge fitting, predictor diagnostics and cached-table checks.
        Training-only tensors and masks remain resident, shared augmentations are
        generated on the selected CPU/CUDA/MPS device, and accelerator diagnostics
        are transferred back only at bounded reporting or checkpoint boundaries.
        Runtime depends on the backend; the hardware card records what this kernel
        can actually use rather than assuming that `DEVICE="auto"` means GPU.

        A complete compatible job is reused by content identity. When periodic
        paired resume is configured, an interrupted job may continue only from a
        checksum-validated shared-arm optimizer boundary; otherwise it restarts
        from its seed. A resume or incomplete staging directory is not completed
        evidence. Per-job
        readout tables are cached separately, so changing evaluation can reuse
        trained encoders when their training identity is unchanged.

        The default artifact root is `artifacts/motion_structured`. Each training
        job has a manifest and checkpoint files under its experiment/digest.
        Each evaluation has its own digest under `evaluations`. A complete grid
        adds a table index, source membership and pooled summaries under `grids`.
        '''),
        code('''
        # Synthetic mode is an explicitly requested one-update software check.
        training_progress = NotebookTaskProgress("Motion and structure JEPA grid", "stage")
        result = run_gavd_grid_with_progress(plan, inputs, progress=training_progress,
            enabled=RUN_TRAINING or DATA_MODE == "synthetic")
        print(result["status"])
        display(result["jobs"])
        if result["status"] == "Complete":
            print("Complete grid report:", result["directory"])
            display(result["per_seed"][["experiment", "condition", "representation", "seed",
                "r2", "mae", "evaluated_clips", "evaluated_sources"]])
        else:
            print("Enable RUN_TRAINING in the configuration cell, rerun that cell, then this cell.")
        '''),
        md('''
        ## 5. Interpret the outcome at the correct level

        Finite losses show optimization is functioning. An arm's own teacher
        changes during learning, so a lower own-teacher loss cannot rank useful
        movement information across arms. Notebook 18 evaluates trained online,
        EMA teacher and initial encoders with the same frozen readouts, alongside
        direct-pose features and the training mean.

        It pools all five held-out folds **within each seed** before computing
        source-balanced R² and MAE. Fold R² values must not be averaged. Every
        video receives equal total weight, regardless of its clip count. Seed
        variation reflects training randomness on reused data, not extra subjects.

        A motion-sensitive readout that helps trained and initial features equally
        supports a readout explanation. A repeated trained-over-initial gain
        supports useful pretraining. If neither gains, inspect timing preparation
        and the objective before broad mask mixtures. This cohort has already
        informed the hypotheses, so new results remain development evidence.
        Continue with [18](18_motion_information_and_readout.ipynb), using the
        same mode, experiment list, fold/seed scope, device and output root.
        '''),
    ])
    return add_saved_result_interpretation(notebook, 17)
