"""Cell constructors for the isolated notebooks 15–18."""
from textwrap import dedent
from nbformat.v4 import new_code_cell, new_markdown_cell


def md(text):
    return new_markdown_cell(dedent(text).strip())


def code(text):
    return new_code_cell(dedent(text).strip())


def setup_cell():
    return code('''
        from pathlib import Path
        from dataclasses import asdict, replace
        import os
        import sys
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display
        from matplotlib_inline.backend_inline import set_matplotlib_formats

        def locate_suite():
            for parent in (Path.cwd(), *Path.cwd().parents):
                for candidate in (parent, parent / "neurips-laterality"):
                    if (candidate / "laterality_extensions/motion_structured_masks.py").is_file():
                        return candidate.resolve()
            raise FileNotFoundError("Run from the research project directory.")

        SUITE_ROOT = locate_suite()
        if str(SUITE_ROOT) not in sys.path:
            sys.path.insert(0, str(SUITE_ROOT))
        from laterality_extensions.motion_structured_masks import (
            StudyArm, mamp_logits, sample_study_mask, study_arms,
            paired_study_masks, context_cue_audit,
        )
        from laterality_extensions.comparative_masks import motion_scores
        from notebook_progress import (
            NotebookTaskProgress, run_notebook_task, study_inputs_with_progress,
            audit_training_masks_with_progress, grid_status_with_progress,
            run_gavd_grid_with_progress, collect_gavd_grid_with_progress,
            evaluate_retained_motion_with_progress,
        )
        set_matplotlib_formats("svg", "png")
        pd.set_option("display.precision", 3)
    ''')


def data_instructions():
    return md('''
        ## 1. Load the real GAVD cohort and declare the full split/seed grid

        Run cells in order in a Python kernel with the project's dependencies.
        Long tasks use the shared `notebook_progress.py` wrapper: one updating
        display shows the current stage, fold/seed, elapsed time and estimated
        remaining time. Mask batches and optimizer updates appear within their
        active stage. ETA adjusts as stages finish; their costs differ. Cached,
        disabled, missing-input and failed tasks receive explicit status labels.
        `DATA_MODE="gavd"` is the default. The helper below follows the same
        preparation and source splitting functions as notebooks 01 and 02:

        1. Verify an existing paper-profile cohort and split manifest by their
           content hashes. If absent, read the local GAVD pose archives and
           official annotations, apply the existing QC and target rules, and
           create those two artifacts. The first run takes longer.
        2. The protocol fixes 642 pose archives and 666 annotations. If the local
           cache contains later additions, recover the original extraction
           generations only when their inventory **exactly** matches the locked
           count and SHA-256. Store verified copies under the paper artifact
           root. The original cache stays intact. A mismatch stops with an
           actionable error; it cannot silently switch to generated data.
        3. The reference QC result is 625 clips from 93 source videos. Inputs have
           shape `[clips, 64, 33, 3]`; four prepared steps form each of 16 tokens
           per landmark. Naturally missing observations remain marked invalid.
           The target is a coordinate-derived bilateral movement contrast, not
           the dataset's condition annotation.
        4. Reuse the five video-disjoint outer folds. Seeds 42--46 change
           initialization, source draws, augmentations and masks; they repeat
           the **same** train/test partitions. A video is held out exactly once
           per seed. There are 25 fold/seed combinations, not 25 independent
           datasets. Video separation does not establish subject separation.

        The printed census makes train and test membership visible. Source IDs
        and sequence IDs are retained in `inputs["memberships"]`. The reference
        train/test clip counts are 436/189, 443/182, 553/72, 548/77 and 520/105.
        Unequal clip counts are expected because entire videos stay together.

        A small generated-data path remains available only through the explicit
        `LATERALITY_MOTION_DATA_MODE=synthetic` software-check setting. It prints
        its reduced scope and uses a separate artifact directory. It provides
        no GAVD results. See [the run guide](docs/MOTION_GAVD_WORKFLOW.md) for
        paths, environment settings, recovery and a notebook-by-notebook walkthrough.
    ''')


def configuration_cell():
    return code('''
        from laterality_extensions.motion_gavd import gavd_plan, readout_contrasts
        DATA_MODE = os.getenv("LATERALITY_MOTION_DATA_MODE", "gavd")
        FOLDS = (0, 1, 2, 3, 4)
        SEEDS = (42, 43, 44, 45, 46)
        EXPERIMENTS = tuple(os.getenv("LATERALITY_MOTION_EXPERIMENTS", "motion,regions").split(","))
        CREATE_MISSING_INPUTS = True
        DEVICE = os.getenv("LATERALITY_DEVICE", "auto")
        # Same explicit training switch as Notebook 12; also accept the study-specific alias.
        RUN_TRAINING = os.getenv("LATERALITY_MOTION_RUN_REAL",
                                os.getenv("LATERALITY_RESEARCH_RUN_REAL", "0")) == "1"
        if DATA_MODE == "synthetic":
            FOLDS, SEEDS = (0,), (42,)
            print("EXPLICIT SYNTHETIC SOFTWARE CHECK: one fold/seed, one update, no GAVD evidence")
        OUTPUT_ROOT = Path(os.getenv("LATERALITY_MOTION_OUTPUT_ROOT", str(SUITE_ROOT / "artifacts" /
            ("motion_structured" if DATA_MODE == "gavd" else "motion_structured_synthetic"))))
        print(f"Mode={DATA_MODE}; folds={FOLDS}; seeds={SEEDS}; device={DEVICE}")
        print(f"Training enabled={RUN_TRAINING}; outputs={OUTPUT_ROOT}")
    ''')


def inputs_cell():
    return code('''
        input_progress = NotebookTaskProgress("Dataset preparation and source splits", "stage")
        inputs = study_inputs_with_progress(mode=DATA_MODE, folds=FOLDS, seeds=SEEDS,
            create_missing=CREATE_MISSING_INPUTS, progress=input_progress)
        display(inputs["census"])
        assert inputs["census"].source_overlap.eq(0).all()
        display(inputs["memberships"].head(8))
        if DATA_MODE == "gavd":
            cohort = inputs["cohort"]
            display(cohort.table.groupby("condition").agg(
                accepted_clips=("sequence_id", "size"), source_videos=("video_id", "nunique")))
            display(pd.Series({key: cohort.attrition[key] for key in
                ("input_sequences", "accepted_sequences", "accepted_sources", "excluded_sequences")}))
            print("Cohort:", cohort.cohort_digest)
            print("Split:", inputs["splits"]["split_digest"])
            print("Artifacts:", inputs["context"].artifact_root)
    ''')


def plan_cell():
    return code('''
        plan = gavd_plan(inputs, experiments=EXPERIMENTS, device=DEVICE, output_dir=OUTPUT_ROOT)
        display(pd.Series({key: plan[key] for key in
            ("scope", "training_runs", "optimizer_updates", "recipe_source", "output_dir")}))
        display(pd.Series(plan["settings"], name="Declared training settings"))
        display(plan["workload"].groupby(["experiment", "fold", "seed"], sort=False).agg(
            encoders=("condition", "size"), optimizer_updates=("updates", "sum")))
        display(pd.DataFrame([{"experiment": e, **arm} for e, arms in plan["arms"].items()
                              for arm in arms.values()]))
    ''')
