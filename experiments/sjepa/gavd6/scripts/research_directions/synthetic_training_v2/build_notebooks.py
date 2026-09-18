#!/usr/bin/env python3
"""Build deterministic, output-free study notebooks from one canonical source."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from textwrap import dedent

import nbformat

ROOT = Path(__file__).resolve().parents[3]
DESTINATION = ROOT / "notebooks" / "synthetic_training_v2"

# A fresh kernel can execute any stage once its artifact prerequisites exist.
# Resolve the supplied config before changing directory so relative paths remain explicit.
SETUP = dedent('''\
    import os
    import sys
    from pathlib import Path
    from IPython.display import Image, display

    if not os.environ.get("STV2_CONFIG"):
        raise RuntimeError("Set STV2_CONFIG to an explicit study JSON configuration before execution.")
    config_path = Path(os.environ["STV2_CONFIG"]).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Study configuration does not exist: {config_path}")
    search_root = Path(os.environ.get("GAVD6_ROOT", Path.cwd())).expanduser().resolve()
    PROJECT_ROOT = next((path for path in (search_root, *search_root.parents)
                         if (path / "src/gavd6_sjepa").is_dir() and (path / "pyproject.toml").is_file()), None)
    if PROJECT_ROOT is None:
        raise RuntimeError("Run inside the repository or set GAVD6_ROOT to its root.")
    if str(PROJECT_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
    os.chdir(PROJECT_ROOT)
    os.environ["STV2_CONFIG"] = str(config_path)
    from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig
    from gavd6_sjepa.research_directions.synthetic_training_v2.workflow import run_stage

    cfg = RunConfig.load(os.environ["STV2_CONFIG"])
    display({"run_id": cfg.run_id, "mode": cfg.mode, "device": cfg.device,
             "artifact_root": str(cfg.root), "confirmation": "closed"})
    if cfg.mode == "fixture":
        print("CPU software fixture: method ordering is not empirical evidence.")
''')


def md(source):
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source):
    return nbformat.v4.new_code_cell(dedent(source).strip(), execution_count=None, outputs=[])


def stage_notebook(filename, title, stages, question, inputs, computation, cells,
                   outputs, interpretation, next_gate):
    notebook = nbformat.v4.new_notebook(cells=[
        md(f"# {title}\n\n## Question\n\n{question}"),
        md(f"## Inputs\n\n{inputs}\n\n"
           "Use an explicit `STV2_CONFIG` JSON file and a unique run ID. "
           "The setup resolves relative artifact paths from the repository root. "
           "See the [execution guide](README.md), [development protocol]"
           "(../../docs/studies/synthetic-training-v2/protocol.md) and [literature ledger]"
           "(../../docs/studies/synthetic-training-v2/literature.md)."),
        code(SETUP),
        md(f"## Computation\n\n{computation}\n\n"
           "`run_stage` implements the computation in the study modules. "
           "It checks prerequisite receipts and returns the saved result on an unchanged rerun; "
           "a changed configuration or code identity requires a new run ID."),
        *[code(cell) for cell in cells],
        md(f"## Outputs and checks\n\n{outputs}\n\n"
           "Stage receipts under `receipts/` record elapsed time and hashes of produced artifacts. "
           "Inspect the saved files for full diagnostics; the display above is deliberately brief."),
        md(f"## Interpretation\n\n{interpretation}\n\n"
           "A completed fixture checks software behavior. Scientific gates use `pass`, `fail` or "
           "`insufficient_evidence`; fixture success cannot make a scientific gate pass."),
        md(f"## Next gate\n\n{next_gate}"),
    ], metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "synthetic_training_v2": {
            "stages": list(stages), "config_environment": "STV2_CONFIG",
            "canonical_output_free": True, "opens_confirmation": False,
        },
    })
    for index, cell in enumerate(notebook.cells):
        identity = f"{filename}\n{index}\n{cell.cell_type}\n{cell.source}"
        cell.id = hashlib.sha256(identity.encode()).hexdigest()[:16]
    nbformat.validate(notebook)
    return notebook


def fit_summary(variable):
    return f'''\
display({{"evidence_status": {variable}["evidence_status"],
         "resource_contrast": {variable}["resource_contrast"],
         "equal_compute_claim": {variable}["equal_compute_claim"]}})
display([{{key: row.get(key) for key in
          ("arm", "seed", "status", "optimizer_updates", "elapsed_seconds", "termination")}}
         for row in {variable}["fits"]])'''


def render_all():
    """Return canonical notebooks in dependency order without executing study code."""
    specs = [
        dict(filename="00_artifact_audit.ipynb", title="00 — Reconstruct the retained pilot", stages=("audit",),
             question="What do the retained pilot artifacts actually establish, and which historical stops remain binding?",
             inputs="Retained source rows, selector decisions, historical reports and the preservation manifest. No new model is fitted.",
             computation="Reconstruct counts, matched comparisons and retrospective oracle opportunities from retained artifacts. Verify preserved files before proceeding.",
             cells=['''\
audit = run_stage(cfg, "audit", repo_root=PROJECT_ROOT)
display({"evidence_status": audit["evidence_status"], "counts": audit["counts"]})
display(audit["table"])
display({key: audit["oracle"].get(key) for key in
         ("shared_scene_error", "student_specific_error", "additional_relative_reduction_percent")})'''],
             outputs="Read `audit/pilot-audit.json` and `audit/pilot-table.csv`. Check duplicate keys, matched counts, source units and the reconstruction tolerance. A discrepancy stops interpretation of the historical pilot.",
             interpretation="The roughly 0.0406% extra student-specific oracle headroom is retrospective potential on an exposed development panel. It is not achieved personalization and does not justify reopening the branch. The audit may reconstruct source aggregates even when later stages are fixtures.",
             next_gate="Proceed to [01 — paired data](01_paired_data.ipynb) after the retained-artifact checks pass. Preserve the existing future-feature, accessibility and personalization stops."),
        dict(filename="01_paired_data.ipynb", title="01 — Validate paired tracks and their provenance", stages=("data",),
             question="Are noisy inputs and projected reference tracks aligned without leaking reference information into inference?",
             inputs="The audit receipt and either an explicitly configured analytic fixture or an audited source TrackBundle. Source preparation, licenses, rendering and extraction are separate prerequisites.",
             computation="Validate the body-12 schema, target validity, observed-input flags, physical timestamps, canonical person groups and held-extractor exposure. Save the validated bundle, achieved counts and an input contact sheet.",
             cells=['''\
data = run_stage(cfg, "data", repo_root=PROJECT_ROOT)
display(data)
display(Image(filename=str(cfg.root / "data/track-contact-sheet.png")))'''],
             outputs="Inspect `data/achieved-size.json`, `data/bundle/` and `data/track-contact-sheet.png`. A contact sheet is a quick input audit, not the independent landmark-overlay or anatomical-convention review required before large source fitting. Keep nuisance pairs, aliases and overlapping source windows together.",
             interpretation="Projected SMPL-H centers are approximate synthetic targets. The fixture verifies plumbing and grouping, but cannot establish pose-estimator robustness, anatomical accuracy or a held-family transfer claim. Missing inputs can retain valid restoration targets.",
             next_gate="Record independent landmark and source-exposure audits before a source screen. Continue to [02 — image adaptation](02_image_adaptation.ipynb) to record the supporting branch, then [03 — information ladder](03_information_ladder.ipynb)."),
        dict(filename="02_image_adaptation.ipynb", title="02 — Record the separate image-adaptation gate", stages=("adaptation",),
             question="Does synthetic image adaptation add value beyond augmented-COCO replay without harming retention?",
             inputs="The validated-data receipt. An empirical answer additionally needs the unchanged, replay, augmented-COCO, matched-synthetic and pooled-synthetic image-model runs.",
             computation="Record Gate A's current evidence status. This stage currently saves the pending branch and its required comparisons; it does not perform image-model optimization.",
             cells=['''\
adaptation = run_stage(cfg, "adaptation", repo_root=PROJECT_ROOT)
display(adaptation)'''],
             outputs="The `adaptation` receipt states the missing experiment. Future evidence must include gradients, changed model outputs, matched evaluation and retention, with rendering and extraction costs recorded.",
             interpretation="A successful temporal refiner does not prove image adaptation. PoseSyn and related synthetic-pair work already motivate synthetic supervision, so improvements must be compared against the strongest available augmentation control.",
             next_gate="Keep Gate A separate from the temporal Gate B. Proceed to [03 — information ladder](03_information_ladder.ipynb); do not interpret this pending receipt as a trained image model."),
        dict(filename="03_information_ladder.ipynb", title="03 — Establish the unchanged and filtering controls", stages=("information",),
             question="How much accuracy and motion preservation can simple temporal processing provide under the same information and latency?",
             inputs="The validated bundle and development partition. Every method observes the same 64 samples at 25 Hz, approximately 2.52 seconds from first to last sample.",
             computation="Save unchanged predictions and the predeclared filter strengths 0, 1 and 2 frames for each configured seed. Use input coordinates, confidence, observed flags and timestamps only; preserve target records separately for scoring.",
             cells=['''\
information = run_stage(cfg, "information", repo_root=PROJECT_ROOT)
display(information)'''],
             outputs="Check `predictions/unchanged-*.npz`, `predictions/filter*-*.npz` and their record sidecars. Predictions retain physical timestamps and evaluation masks so subsequent tables can be rebuilt. Cadence remains unsupported when the window lacks adequate cycles.",
             interpretation="Smoother trajectories can erase real amplitude or shift events. SmoothNet and SynSP establish temporal refinement as prior work; coordinate accuracy alone cannot establish that gait motion survived. These are offline observed-window controls, not forecasting.",
             next_gate="Proceed to [04 — coordinate versus JEPA](04_coordinate_vs_jepa.ipynb). Keep all predeclared filtering strengths in the tradeoff plot and select nothing using confirmation data."),
        dict(filename="04_coordinate_vs_jepa.ipynb", title="04 — Fit practical and objective-matched controls", stages=("direct", "jepa"),
             question="Does clean-target latent pretraining add value beyond matched coordinate learning and practical temporal denoising?",
             inputs="The information-stage receipt, the fixed train/development groups and an explicit run scope. CPU fixtures use small models and two-update phase checks. Source CUDA execution requires the pinned HAIC runtime and authorized, measured all-stage compute scope.",
             computation="Fit the configured practical and initialized arms first, then coordinate reconstruction, ordinary JEPA, paired JEPA and shuffled-pair controls. Each representation receives its own matched readout. The training modules own masks, loss support, EMA, optimization, checkpoints and predictions.",
             cells=['direct = run_stage(cfg, "direct", repo_root=PROJECT_ROOT)\n' + fit_summary("direct"),
                    'jepa = run_stage(cfg, "jepa", repo_root=PROJECT_ROOT)\n' + fit_summary("jepa")],
             outputs="Inspect per-arm `fits/<arm>-<seed>/training.json`, checkpoints and saved development predictions. Check actual updates, teacher lag, skipped loss support, input dependence, cost and trainable capacity. Displayed fit status describes execution, not superiority. No notebook submits a GPU job.",
             interpretation="The SmoothNet-style 2D MLP is a practical comparator, not an exact paper reproduction. The per-frame control shares whole-window input normalization, so it tests added coordinate history conditional on that transform; it is not an absolute no-history baseline. PoseBERT and MotionBERT already use corrupted or partial pose reconstruction; S-JEPA motivates latent prediction. The strongest competing explanation is clean-target supervision or ordinary denoising. Paired versus coordinate is the objective contrast; ordinary JEPA changes supervision source. Equal steps do not establish equal compute.",
             next_gate="Proceed to [05 — evaluation](05_evaluation.ipynb). A fixture cannot choose finalists; decisive source comparisons require prespecified repeated seeds, adequate motion support and independently calibrated margins."),
        dict(filename="05_evaluation.ipynb", title="05 — Reconstruct accuracy and motion evidence", stages=("evaluate",),
             question="Does a method reduce coordinate error while preserving supported displacement, amplitude, timing and anatomical sides under person-level uncertainty?",
             inputs="Saved development predictions from unchanged, filtering, practical and representation arms, plus separate target validity, visibility, timestamps, evaluation scales and group identifiers.",
             computation="Recompute per-window endpoints and balanced person-level summaries, paired cluster uncertainty and nuisance strata. Plot visible coordinate error against fixed-lag displacement error. Report each extractor and seed separately. Here amplitude is the RMS of demeaned horizontal left-minus-right ankle separation, normalized by the reference-box diagonal. Event timing measures its positive local maxima in seconds; these are operational projected-trajectory events, not heel strikes or clinical events.",
             cells=['''\
evaluation = run_stage(cfg, "evaluate", repo_root=PROJECT_ROOT)
display({key: evaluation[key] for key in
         ("evidence_status", "windows", "extractors", "seeds")})
display({"sampling_groups": evaluation["independent_people"],
         "group_kind": "analytic fixture IDs" if cfg.mode == "fixture" else "audited canonical people"})
display(evaluation["gates"])
display(Image(filename=str(cfg.root / "evaluation/accuracy-preservation.png")))'''],
             outputs="Inspect `evaluation/per-window.csv`, `per-person-balanced-summary.csv`, `nuisance-strata.csv`, contrast artifacts, `gates.json` and the tradeoff image. The displayed window count is metric rows across methods; it is not the number of independent samples. Check endpoint support counts and missing metrics before comparing methods.",
             interpretation="The current workflow leaves Gate B insufficient until repeated-seed preservation, clean-retention and calibrated-margin adjudication exist. Independent real temporal references are also pending. Paired intervals resample people with motions nested, condition on the fitted model, and exclude training/selection uncertainty. In fixtures they check arithmetic on analytic groups, not population uncertainty. Low latent loss, high rank or a clean-looking curve cannot establish useful gait measurement. No time warping is allowed during scoring.",
             next_gate="Proceed to [06 — optional gates](06_optional_gates.ipynb). A failed JEPA gate stops that expansion while a useful direct denoiser may proceed independently. Do not convert missing motion evidence into a passing gate."),
        dict(filename="06_optional_gates.ipynb", title="06 — Keep optional branches evidence-gated", stages=("optional",),
             question="Is there independent evidence and allocated scope to reopen personalization or add video features?",
             inputs="The retained-artifact audit and current optional-branch evidence. The historical personalization panel is exposed; a fresh development panel would be needed to reopen it.",
             computation="Record the closed personalization branch and the pending video branch. This stage does not train either extension.",
             cells=['''\
optional = run_stage(cfg, "optional", repo_root=PROJECT_ROOT)
display(optional)'''],
             outputs="The `optional` receipt records branch status, skipped execution and reasons. Preserve the distinction between a historical failed opportunity gate and an untested video hypothesis.",
             interpretation="GaitForeMer's small clinical setting and silhouette recognition systems such as GaitJEPA do not validate restoration of 2D clinical motion here. Additional modalities need incremental-benefit evidence under matched scope; their availability alone does not justify expansion.",
             next_gate="Proceed to [07 — development snapshot](07_development_snapshot.ipynb). Optional branches remain closed or pending until their own development evidence and authorized scope exist."),
        dict(filename="07_development_snapshot.ipynb", title="07 — Save a development snapshot; keep confirmation closed", stages=("freeze",),
             question="Can the current development evidence be recorded reproducibly without opening independent confirmation?",
             inputs="The evaluation receipt, run identity and saved prediction files. Independent real-development references, exposure audits and calibrated confirmation margins are still required.",
             computation="Call the workflow's `freeze` stage to save a development snapshot. Despite that internal stage name, this operation does not freeze a qualified confirmation protocol or access protected confirmation labels.",
             cells=['''\
snapshot = run_stage(cfg, "freeze", repo_root=PROJECT_ROOT)
display({key: snapshot[key] for key in ("status", "confirmation_opened", "reason")})
display({"development_snapshot": str(cfg.root / "development-snapshot.json")})'''],
             outputs="Inspect `development-snapshot.json` for run identity and prediction hashes. Confirm `confirmation_opened` is false. The file records the current insufficient-evidence decision; it is not a substitute for an independently reviewed, immutable confirmation lock.",
             interpretation="Unknown exposure and missing independent temporal references prevent confirmation claims. A separate explicit operation is required after references, margins, group definitions and methods are qualified; that confirmation operation is unavailable in this notebook suite.",
             next_gate="Proceed to [08 — evidence report](08_evidence_report.ipynb). Finish the pending development and independent review work before any future request to open confirmation."),
        dict(filename="08_evidence_report.ipynb", title="08 — Rebuild the evidence report from artifacts", stages=("report",),
             question="Can every reported result and limitation be traced to retained inputs, saved predictions, receipts and gate decisions?",
             inputs="Evaluation, adaptation, optional and development-snapshot receipts from this run. No notebook state from earlier kernels is required.",
             computation="Reconstruct the report from saved per-window predictions and group records. Include branch decisions, evidence origin, cost limits and the strongest competing explanation.",
             cells=['''\
report = run_stage(cfg, "report", repo_root=PROJECT_ROOT)
display({"report": report["report"], "evidence_status": report["evidence_status"]})
display(report["gates"])'''],
             outputs="Open the reported `report.md` path alongside `identity.json`, `effective-config.json`, `environment.json`, stage receipts and per-fit cost records. Full tables remain in artifacts rather than the canonical notebook. Independently compare the generated report against saved predictions and unresolved review findings.",
             interpretation="A fixture report is executable software evidence, not synthetic-to-real validation or a confirmation result. The narrow hypotheses concern motion-preserving extractor-shift evaluation and paired JEPA's added value over matched coordinate learning. Novelty depends on an empirical result beyond the closest prior work. Temporal refinement, synthetic pairing and masked pose reconstruction already exist.",
             next_gate="Independent review must inspect implementation and saved artifacts, then record dispositions. Source data, runtime, anatomical references, real temporal annotations, repeated finalists and calibrated margins remain separate empirical prerequisites. See the [protocol review](../../docs/studies/synthetic-training-v2/protocol-review.md)."),
    ]
    return {spec["filename"]: stage_notebook(**spec) for spec in specs}


def write_notebooks(destination=DESTINATION):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename, notebook in render_all().items():
        path = destination / filename
        path.write_text(nbformat.writes(notebook) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    args = parser.parse_args()
    for path in write_notebooks(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
