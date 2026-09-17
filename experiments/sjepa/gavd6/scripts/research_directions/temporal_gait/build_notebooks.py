"""Build nine deterministic, output-free temporal-gait teaching notebooks."""
from __future__ import annotations

import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat

ROOT = Path(__file__).resolve().parents[3]
DESTINATION = ROOT / "notebooks/temporal_gait"

PROGRAMS = [
    ("00_inventory_and_split.ipynb", "Inventory and independent sources", ("inventory",),
     "Which complete walking bouts are explicitly supplied, and which sources may each stage use?",
     "The input manifests name exact videos, pose exports, full-bout bounds, source groups, exposure and split roles. "
     "Inventory must not locate alternative data or infer participant identities. The exposure audit explicitly identifies historical-overlap sources. "
     "Existing exposed sources cannot become a fresh test.",
     "Inspect source/bout counts, explicit exclusions, frozen configuration and task mapping. A manifest is not proof of readable media.",
     "This stage establishes a data contract. It supplies no model accuracy or clinical evidence."),
    ("01_full_bout_timing_and_windows.ipynb", "Complete bouts, clocks and support", ("prepare",),
     "Can every allowed bout produce correctly timed windows without changing the measurement?",
     "For issue time b, use [b−2.56,b) and 64 bin-left queries at 25 Hz. The last query is b−0.04. "
     "Retain actual source times and age, sample throughout complete bouts, and keep SSL eligibility separate from endpoint support.",
     "Inspect duration/window coverage, short bouts, timestamp failures, source-balanced counts and pose provenance. "
     "Future coordinates and missingness never decide context values or inference queries.",
     "The current input milestone requires explicit compatible full-bout pose exports. It does not silently extract poses from video or reuse the legacy clip cache."),
    ("02_information_and_baseline_audit.ipynb", "Measure what the observations support", ("audit",),
     "Do simple past-only predictors and the declared target have adequate common support?",
     "Audit persistence, robust velocity, periodic extrapolation and direct pose prediction on identical sampled histories. "
     "The historical signed speed ratio is a separate deterministic observation statistic; uniform duration changes approximately cancel.",
     "Inspect valid bilateral pairs, prefix scale, temporal coverage, failures and baseline outputs. Source-video effects carry more information than a window count.",
     "Synthetic fixtures validate execution and arithmetic. They do not establish a gait-model benefit or clinical endpoint."),
    ("03_time_faithful_masked_jepa.ipynb", "A controlled masked JEPA comparison", ("masked",),
     "Does time-faithful preparation help while model capacity and exposure remain controlled?",
     "Compare masked_index and masked on identical raw windows. The first mechanism comparison uses explicitly audited historical-overlap sources "
     "and keeps width 96, four encoder layers, two predictor layers, four-sample patches and the same recipe. "
     "Full-allowed cohort expansion, clock channels and two-sample patches are subsequent named comparisons in separate run roots.",
     "Inspect the selected global task ID, seed, realized masks, context/target support, initialization/online/teacher checkpoints and learning curves.",
     "Prepared-feature masking is not raw-observation withholding. This bidirectional-prefix objective is not future-only pretraining; lower teacher loss is not the success metric."),
    ("04_causal_future_jepa.ipynb", "Predict future features from a fixed prefix", ("future",),
     "Does a future-feature objective improve observable forecasts beyond the strongest simple comparator?",
     "Query horizons 0.25, 0.50 and 1.00 seconds from issue time, with 0.50 primary. Encode future target intervals separately from context. "
     "An 80 ms latent interval ending at a horizon is distinct from the single coordinate endpoint within 20 ms of its original-time query.",
     "Compare future, future_wrong_source and future_wrong_time tasks. Inspect per-horizon support, target mismatch provenance, "
     "EMA health and unchanged predictions when strictly future observations are altered.",
     "Future support, target timestamps and validity are teacher/loss/scoring information only. Predictions must use fixed requested horizons even when future targets are missing."),
    ("05_optional_dense_and_video_transfer.ipynb", "Optional dense and RGB questions", ("extensions", "cache-video"),
     "Which additional question is justified after the baseline mechanism is understood?",
     "Dense/context supervision and intermediate-layer prediction are a factorial proposal. A released frozen RGB encoder is a separate "
     "pose-only/RGB-only/late-fusion comparator with past-only crops and explicit checkpoint provenance.",
     "Read the returned extension status. The current milestone explicitly gates E3 and video execution until their implementations and assets are verified.",
     "gated_not_implemented is an honest unsupported branch, not an empirical failure or a successful model run. No checkpoint is downloaded and no stand-in is substituted."),
    ("06_development_comparison.ipynb", "Compare on independent development sources", ("evaluate",),
     "Which comparison survives matched fitting access, support and source-balanced evaluation?",
     "Readouts and direct forecasters fit on the same training-source pool. Development selects their settings. "
     "Evaluate context-feature and predicted-future-feature decoding separately; observed-future features are privileged diagnostics.",
     "Inspect all expected phase/arm/seed members, per-window predictions, per-video reductions, paired effects, and the saved expansion decision.",
     "A failed or missing array member makes the comparison incomplete. Stop is a legitimate decision; never average only successful jobs or open test after a failed utility gate."),
    ("07_locked_calibration_and_test.ipynb", "Freeze the analysis before test access", ("calibrate", "test"),
     "Are the selected models, readouts and measurement choices fixed before untouched data are opened?",
     "The default action is calibrate: freeze the analysis and perform only declared optional uncertainty calibration. "
     "The test action is a separate explicit choice. It may not fit a new decoder, select a model, or change the primary comparator.",
     "Inspect the saved lock, complete expected tasks and test-open receipt. Calibration is not an exclusive training set for JEPA readouts.",
     "If no untouched eligible sources exist, report development-only evidence. A technical test rerun preserves the original lock and records the repair."),
    ("08_aggregate_and_claim_audit.ipynb", "Rebuild results and audit claims", ("aggregate",),
     "Can every reported number be rebuilt from complete saved predictions?",
     "For the primary metric, score observed paired knee/ankle/heel/foot-tip points, normalize by prefix body length, "
     "then average window→bout→video→equal videos. Duplicate/known-person groups are bootstrap clusters.",
     "Inspect per-seed and paired source effects, confidence intervals, exclusions, coverage, compute and artifact completeness. "
     "Keep development, test, synthetic and unimplemented status visible in every report.",
     "Five seeds and many overlapping windows do not create new independent people. Pose-coordinate forecasts are not validated clinical outcomes."),
]

SETUP = """
from pathlib import Path
import json
import os
import sys
from IPython.display import Markdown, display

project = os.environ.get("GAVD6_ROOT")
if project is None:
    project = next((str(p) for p in [Path.cwd(), *Path.cwd().parents]
                    if (p / "src/gavd6_sjepa").is_dir()), None)
if project is None:
    raise FileNotFoundError("Set GAVD6_ROOT to the gavd6 checkout.")
PROJECT_ROOT = Path(project).expanduser().resolve()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from gavd6_sjepa.research_directions.temporal_gait.config import RunConfig
from gavd6_sjepa.research_directions.temporal_gait import workflow

cfg = RunConfig.from_env()
cfg.validate(check_input_paths=False)
phase = os.environ.get("TG_PHASE", "pilot")
execute = os.environ.get("TG_EXECUTE") == "1"
task_id = int(os.environ["TG_TASK_ID"]) if "TG_TASK_ID" in os.environ else None
print(f"Mode: {cfg.mode}; phase: {phase}; execute: {execute}; root: {cfg.root}")
if cfg.mode == "synthetic":
    display(Markdown("**SYNTHETIC SOFTWARE FIXTURE — no real GAVD result.**"))
if not execute:
    display(Markdown("**PLAN ONLY.** Set TG_EXECUTE=1 only to run this selected stage."))
"""


def cell(kind, source):
    source = dedent(source).strip()
    value = (nbformat.v4.new_markdown_cell(source) if kind == "markdown"
             else nbformat.v4.new_code_cell(source))
    value.id = hashlib.sha256((kind + source).encode()).hexdigest()[:16]
    return value


def render_all():
    result = {}
    for name, title, stages, question, design, inspect, limits in PROGRAMS:
        action = f'''
stage = os.environ.get("TG_STAGE", {stages[0]!r})
if stage not in {stages!r}:
    raise ValueError(f"This notebook cannot execute stage {{stage}}")
if execute:
    if stage in ("masked", "future") and task_id is None:
        raise ValueError("Choose one TG_TASK_ID from the immutable phase task plan.")
    result = workflow.run_stage(cfg, stage, task_id=task_id,
                                role="test" if stage == "test" else "development",
                                phase=phase)
    if result.get("status") in {{"incomplete", "failed", "error"}}:
        print(json.dumps(result, indent=2, default=str))
        raise RuntimeError(f"Stage did not complete: {{result['status']}}; inspect retained artifacts")
else:
    result = {{"status": "plan_only", "stage": stage, "mode": cfg.mode,
              "phase": phase, "task_id": task_id, "media_opened": False}}
'''
        cells = [cell("markdown", f"# {name[:2]} · {title}\n\n{question}"),
                 cell("markdown", "These notebooks call the tested package; they contain no separate training loop. "
                      "Real mode defaults to inspection. Explicit synthetic configuration plus TG_EXECUTE=1 runs software fixtures only.\n\n"
                      "[Notebook guide](README.md) · [HAIC runbook](../../docs/studies/temporal-gait/execution/haic.md)"),
                 cell("code", SETUP), cell("markdown", "## 1. The controlled question\n\n" + design),
                 cell("code", "print(json.dumps(cfg.to_dict(), indent=2, default=str))\n"
                      "tasks = workflow.plan_tasks(cfg, phase)\nprint(json.dumps(tasks, indent=2, default=str))"),
                 cell("markdown", "## 2. Execute only the selected stage\n\n"
                      "The configured task ID refers to one immutable arm/seed row. A normal notebook render does not train. "
                      "Slurm and the command-line runner pass the same stage and phase."),
                 cell("code", action),
                 cell("markdown", "## 3. Inspect retained evidence\n\n" + inspect),
                 cell("code", "print(json.dumps(result, indent=2, default=str))\n"
                      "print('This record describes execution/artifacts; scientific conclusions require the saved comparison.')"),
                 cell("markdown", "## 4. Interpretation and next decision\n\n" + limits)]
        notebook = nbformat.v4.new_notebook(cells=cells)
        notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"},
                             "temporal_gait": {"stages": list(stages), "source": "build_notebooks.py"}}
        result[name] = notebook
    return result


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, notebook in render_all().items():
        nbformat.write(notebook, DESTINATION / name)
    print(f"Built {len(PROGRAMS)} output-free notebooks in {DESTINATION}")


if __name__ == "__main__":
    main()
