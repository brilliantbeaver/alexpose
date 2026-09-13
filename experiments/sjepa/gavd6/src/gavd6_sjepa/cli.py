"""Lazy, discoverable command router for the GAVD6 research workspace."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Callable, Sequence


@dataclass(frozen=True)
class CommandSpec:
    handler: str
    description: str
    accepts_options: bool = True


COMMANDS: dict[tuple[str, str], CommandSpec] = {
    ("amass", "inventory"): CommandSpec(
        "gavd6_sjepa.data_foundations.amass_inventory:main",
        "Inventory raw AMASS pose archives.",
        accepts_options=False,
    ),
    ("amass", "finalize-subjects"): CommandSpec(
        "gavd6_sjepa.data_foundations.amass_subjects:main",
        "Finalize the evidence-annotated AMASS subject registry.",
    ),
    ("amass", "convert"): CommandSpec(
        "gavd6_sjepa.data_foundations.amass_conversion:main",
        "Convert SMPL+H motion into validity-aware Core11 tensors.",
    ),
    ("amass", "train"): CommandSpec(
        "gavd6_sjepa.research_directions.reflection_equivariance.train_amass:main",
        "Train the matched AMASS Core11 JEPA variants.",
        accepts_options=False,
    ),
    ("amass", "verify-checkpoint"): CommandSpec(
        "gavd6_sjepa.archive.historical_checkpoints.amass_checkpoint:main",
        "Verify the frozen historical AMASS checkpoint contract.",
    ),
    ("gavd", "download"): CommandSpec(
        "gavd6_sjepa.data_foundations.video_download:main",
        "Download and validate unique GAVD source videos.",
    ),
    ("gavd", "convert-core11"): CommandSpec(
        "gavd6_sjepa.data_foundations.gavd_conversion:main",
        "Adapt GAVD MediaPipe poses into the Core11 contract.",
    ),
    ("gavd", "evaluate-core11"): CommandSpec(
        "gavd6_sjepa.research_directions.reflection_equivariance.gavd_probe:main",
        "Evaluate frozen AMASS representations on GAVD Core11.",
    ),
    ("gavd", "annotate-normal"): CommandSpec(
        "gavd6_sjepa.archive.gavd96_augmentation.clip_annotations:main",
        "Create the augmented-normal clip annotation contract.",
        accepts_options=False,
    ),
    ("gavd", "extract-augmented"): CommandSpec(
        "gavd6_sjepa.archive.gavd96_augmentation.pose_extraction:main",
        "Extract the selected augmented GAVD pose cohort.",
        accepts_options=False,
    ),
    ("gavd", "migrate-augmented"): CommandSpec(
        "gavd6_sjepa.archive.gavd96_augmentation.artifact_migration:main",
        "Validate and migrate legacy augmented-pose artifacts.",
    ),
    ("laterality", "build-manifest"): CommandSpec(
        "gavd6_sjepa.research_directions.latent_laterality.manifest:main",
        "Build persistent sequence-level corruption draws.",
    ),
    ("laterality", "benchmark"): CommandSpec(
        "gavd6_sjepa.research_directions.latent_laterality.benchmark:main",
        "Run the sequence-level benchmark eligibility gate.",
    ),
    ("laterality", "train-source-transfer"): CommandSpec(
        "gavd6_sjepa.research_directions.latent_laterality.source_training:main",
        "Train the source-transfer screening models.",
    ),
    ("laterality", "evaluate-source-transfer"): CommandSpec(
        "gavd6_sjepa.research_directions.latent_laterality.source_evaluation:main",
        "Evaluate the common source-transfer readout.",
    ),
    ("laterality", "train"): CommandSpec(
        "gavd6_sjepa.research_directions.latent_laterality.training:main",
        "Train the gated Semantic-Gauge JEPA comparison.",
    ),
    ("laterality", "evaluate"): CommandSpec(
        "gavd6_sjepa.research_directions.latent_laterality.evaluation:main",
        "Evaluate validation or sealed-test gauge readouts.",
    ),
    ("laterality", "smoke"): CommandSpec(
        "gavd6_sjepa.research_directions.latent_laterality.smoke:main",
        "Run a synthetic CPU smoke test of gauge training.",
    ),
    ("swap-probe", "run"): CommandSpec(
        "gavd6_sjepa.research_directions.reflection_equivariance.run_swap_probe:main",
        "Run the validation-only bilateral swap probe.",
    ),
    ("notebooks", "validate"): CommandSpec(
        "gavd6_sjepa.workspace_validation.notebooks:main",
        "Validate generated notebook syntax and startup paths.",
        accepts_options=False,
    ),
}

# Resolve experiment modules only when the selected command runs. Global and
# command-specific help stay usable without loading teacher/pose dependencies.
for _command, _handler, _description in (
    ("init-cached-run", "init_cached_main", "Initialize calibrated direct-v3 using a read-only parent teacher cache."),
    ("calibrate-repair", "calibrate_main", "Calibrate joint predictors on fixed synthetic source-held fixtures."),
    ("verify-repair", "verify_main", "Read-only reconstruction of repaired predictions and numerical reports."),
    ("init-run", "init_main", "Freeze the Future Innovation Experiment 0 protocol and input hashes."),
    ("build-cohort", "cohort_main", "Build label-blind full-GAVD window candidates."),
    ("extract-poses", "poses_main", "Extract aligned whole-body histories and freeze the 50-window cohort."),
    ("cache-teacher", "cache_main", "Cache frozen, causally isolated V-JEPA 2.1 features and targets."),
    ("audit-teacher", "audits_main", "Run repeated-input, causal-leakage, and pixel-edit validity audits."),
    ("run-gate", "fit_main", "Fit nested source-fold baselines and all residual/control heads."),
    ("score-gate", "score_main", "Score complete OOF predictions and paired source bootstraps."),
    ("build-report", "report_main", "Write the immutable Experiment 0 decision and report."),
    ("smoke", "smoke_main", "Exercise all model folds and reports with explicitly synthetic cached data."),
):
    COMMANDS[("future-innovation", _command)] = CommandSpec(
        f"gavd6_sjepa.research_directions.future_prediction.cli:{_handler}", _description
    )


def _package_version() -> str:
    try:
        return version("gavd6-sjepa")
    except PackageNotFoundError:
        return "development"


def _help(category: str | None = None) -> str:
    title = f"gavd6 {_package_version()}\n\nUsage: gavd6 <group> <command> [options]"
    rows = []
    for path, spec in sorted(COMMANDS.items()):
        if category is None or path[0] == category:
            rows.append(f"  {' '.join(path):38} {spec.description}")
    if category is not None and not rows:
        return f"Unknown command group: {category}"
    heading = "\n\nCommands:\n"
    suffix = "\n\nRun `gavd6 <group> <command> --help` for command-specific options."
    return title + heading + "\n".join(rows) + suffix


def _resolve_handler(reference: str) -> Callable[[], object]:
    module_name, function_name = reference.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments in (["--help"], ["-h"]):
        print(_help())
        return 0
    if arguments == ["--version"]:
        print(_package_version())
        return 0
    if len(arguments) == 2 and arguments[1] in {"--help", "-h"}:
        print(_help(arguments[0]))
        return 0
    if len(arguments) < 2:
        print(_help(arguments[0]), file=sys.stderr)
        return 2

    command_path = (arguments[0], arguments[1])
    spec = COMMANDS.get(command_path)
    if spec is None:
        print(f"Unknown command: {' '.join(command_path)}\n", file=sys.stderr)
        print(_help(command_path[0]), file=sys.stderr)
        return 2

    command_arguments = arguments[2:]
    if not spec.accepts_options and command_arguments:
        if command_arguments in (["--help"], ["-h"]):
            print(f"Usage: gavd6 {' '.join(command_path)}\n\n{spec.description}")
            print("\nThis command is configured through documented environment variables.")
            return 0
        print(
            f"Command `gavd6 {' '.join(command_path)}` does not accept arguments.",
            file=sys.stderr,
        )
        return 2

    handler = _resolve_handler(spec.handler)
    previous_argv = sys.argv
    sys.argv = ["gavd6 " + " ".join(command_path), *command_arguments]
    try:
        result = handler()
    finally:
        sys.argv = previous_argv
    return int(result) if result is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
