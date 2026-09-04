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
        "gavd6_sjepa.data_foundations.amass_inventory_generation:main",
        "Inventory raw AMASS pose archives.",
        accepts_options=False,
    ),
    ("amass", "finalize-subjects"): CommandSpec(
        "gavd6_sjepa.data_foundations.amass_subject_registry_finalization:main",
        "Finalize the evidence-annotated AMASS subject registry.",
    ),
    ("amass", "convert"): CommandSpec(
        "gavd6_sjepa.data_foundations.amass_core11_conversion_pipeline:main",
        "Convert SMPL+H motion into validity-aware Core11 tensors.",
    ),
    ("amass", "train"): CommandSpec(
        "gavd6_sjepa.research_directions.reflection_equivariance.amass_training_entrypoint:main",
        "Train the matched AMASS Core11 JEPA variants.",
        accepts_options=False,
    ),
    ("amass", "verify-checkpoint"): CommandSpec(
        "gavd6_sjepa.archive.historical_checkpoints.historical_amass_checkpoint_verification:main",
        "Verify the frozen historical AMASS checkpoint contract.",
    ),
    ("gavd", "download"): CommandSpec(
        "gavd6_sjepa.data_foundations.gavd_video_download_pipeline:main",
        "Download and validate unique GAVD source videos.",
    ),
    ("gavd", "convert-core11"): CommandSpec(
        "gavd6_sjepa.data_foundations.gavd_core11_conversion_pipeline:main",
        "Adapt GAVD MediaPipe poses into the Core11 contract.",
    ),
    ("gavd", "evaluate-core11"): CommandSpec(
        "gavd6_sjepa.research_directions.reflection_equivariance.gavd_core11_probe_entrypoint:main",
        "Evaluate frozen AMASS representations on GAVD Core11.",
    ),
    ("gavd", "annotate-normal"): CommandSpec(
        "gavd6_sjepa.archive.gavd96_augmentation.gavd_normal_clip_annotation:main",
        "Create the augmented-normal clip annotation contract.",
        accepts_options=False,
    ),
    ("gavd", "extract-augmented"): CommandSpec(
        "gavd6_sjepa.archive.gavd96_augmentation.gavd_augmented_pose_extraction:main",
        "Extract the selected augmented GAVD pose cohort.",
        accepts_options=False,
    ),
    ("gavd", "migrate-augmented"): CommandSpec(
        "gavd6_sjepa.archive.gavd96_augmentation.gavd_augmented_artifact_migration:main",
        "Validate and migrate legacy augmented-pose artifacts.",
    ),
    ("laterality", "build-manifest"): CommandSpec(
        "gavd6_sjepa.research_directions.latent_laterality.laterality_manifest_construction:main",
        "Build persistent sequence-level corruption draws.",
    ),
    ("laterality", "benchmark"): CommandSpec(
        "gavd6_sjepa.research_directions.latent_laterality.laterality_benchmark_entrypoint:main",
        "Run the sequence-level benchmark eligibility gate.",
    ),
    ("laterality", "train-source-transfer"): CommandSpec(
        "gavd6_sjepa.research_directions.latent_laterality.laterality_training_entrypoint:main",
        "Train the source-transfer screening models.",
    ),
    ("laterality", "evaluate-source-transfer"): CommandSpec(
        "gavd6_sjepa.research_directions.latent_laterality.source_transfer_evaluation_pipeline:main",
        "Evaluate the common source-transfer readout.",
    ),
    ("laterality", "train"): CommandSpec(
        "gavd6_sjepa.research_directions.latent_laterality.laterality_gauge_training_pipeline:main",
        "Train the gated Semantic-Gauge JEPA comparison.",
    ),
    ("laterality", "evaluate"): CommandSpec(
        "gavd6_sjepa.research_directions.latent_laterality.laterality_gauge_evaluation_pipeline:main",
        "Evaluate validation or sealed-test gauge readouts.",
    ),
    ("laterality", "smoke"): CommandSpec(
        "gavd6_sjepa.research_directions.latent_laterality.laterality_training_smoke_check:main",
        "Run a synthetic CPU smoke test of gauge training.",
    ),
    ("swap-probe", "run"): CommandSpec(
        "gavd6_sjepa.research_directions.reflection_equivariance.swap_probe_entrypoint:main",
        "Run the validation-only bilateral swap probe.",
    ),
    ("notebooks", "validate"): CommandSpec(
        "gavd6_sjepa.workspace_validation.notebook_startup_validation:main",
        "Validate generated notebook syntax and startup paths.",
        accepts_options=False,
    ),
}


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
