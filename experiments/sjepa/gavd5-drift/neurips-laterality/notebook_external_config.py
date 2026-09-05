"""Safe local configuration discovery for Notebook 06.

This module resolves path settings only. It cannot grant authorization and it
does not weaken the fail-closed validator in :mod:`laterality.external`.
Keeping notebook-only configuration outside the scientific package also avoids
invalidating model checkpoints when display or operator ergonomics change.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


EXTERNAL_MANIFEST_ENV = "LATERALITY_EXTERNAL_MANIFEST"
EXTERNAL_GOVERNANCE_ENV = "LATERALITY_EXTERNAL_GOVERNANCE"
EXTERNAL_POSE_ROOT_ENV = "LATERALITY_EXTERNAL_POSE_ROOT"
DISABLE_EXTERNAL_DOTENV_ENV = "LATERALITY_DISABLE_EXTERNAL_DOTENV"
EXTERNAL_SETTING_NAMES = (
    EXTERNAL_MANIFEST_ENV,
    EXTERNAL_GOVERNANCE_ENV,
    EXTERNAL_POSE_ROOT_ENV,
)


@dataclass(frozen=True)
class ExternalGateSettings:
    """Resolved external-gate paths without exposing them in notebook output."""

    manifest: Path | None
    governance: Path | None
    pose_root: Path | None
    sources: tuple[tuple[str, str], ...]
    dotenv_files_checked: tuple[Path, ...]
    dotenv_disabled: bool

    @property
    def missing_required(self) -> tuple[str, ...]:
        values = {
            EXTERNAL_MANIFEST_ENV: self.manifest,
            EXTERNAL_GOVERNANCE_ENV: self.governance,
        }
        return tuple(name for name, value in values.items() if value is None)

    @property
    def configured_required(self) -> tuple[str, ...]:
        values = {
            EXTERNAL_MANIFEST_ENV: self.manifest,
            EXTERNAL_GOVERNANCE_ENV: self.governance,
        }
        return tuple(name for name, value in values.items() if value is not None)

    @property
    def configuration_state(self) -> str:
        configured_count = len(self.configured_required)
        if configured_count == 0:
            return "not_configured"
        if configured_count < 2:
            return "incomplete"
        return "ready_for_validation"

    @property
    def ready_for_validation(self) -> bool:
        """Whether both required paths exist as settings, not whether they pass."""

        return self.configuration_state == "ready_for_validation"

    def redacted_summary(self) -> dict[str, object]:
        """Report configuration presence and origin, never path values."""

        source_by_name = dict(self.sources)
        values = {
            EXTERNAL_MANIFEST_ENV: self.manifest,
            EXTERNAL_GOVERNANCE_ENV: self.governance,
            EXTERNAL_POSE_ROOT_ENV: self.pose_root,
        }
        return {
            "settings": {
                name: {
                    "configured": value is not None,
                    "source": source_by_name.get(name),
                }
                for name, value in values.items()
            },
            "missing_required": list(self.missing_required),
            "configured_required": list(self.configured_required),
            "configuration_state": self.configuration_state,
            "dotenv_files_checked": [str(path) for path in self.dotenv_files_checked],
            "dotenv_disabled": self.dotenv_disabled,
            "paths_redacted": True,
        }


def _configured_path(value: object, *, base: Path) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    supplied = Path(text).expanduser()
    return (supplied if supplied.is_absolute() else base / supplied).resolve()


def load_external_gate_settings(
    *,
    environ: Mapping[str, str] | None = None,
    dotenv_paths: Sequence[str | Path] = (),
    working_directory: str | Path | None = None,
) -> ExternalGateSettings:
    """Load only the three external-gate settings from explicit sources.

    Process-environment values take precedence over optional dotenv files. A
    relative process value uses ``working_directory``; a relative dotenv value
    uses the directory containing that dotenv file. The function neither
    mutates ``os.environ`` nor interprets a configured path as approval.
    """

    environment = os.environ if environ is None else environ
    process_base = Path(working_directory or Path.cwd()).expanduser().resolve()
    dotenv_disabled = str(environment.get(DISABLE_EXTERNAL_DOTENV_ENV, "")).lower() in {
        "1",
        "true",
        "yes",
    }
    checked_files = (
        ()
        if dotenv_disabled
        else tuple(Path(path).expanduser().resolve() for path in dotenv_paths)
    )

    raw_values: dict[str, tuple[object, Path, str]] = {}
    for name in EXTERNAL_SETTING_NAMES:
        value = environment.get(name)
        if value is not None and str(value).strip():
            raw_values[name] = (value, process_base, "process environment")

    from dotenv import dotenv_values

    for dotenv_path in checked_files:
        if not dotenv_path.is_file():
            continue
        file_values = dotenv_values(dotenv_path)
        for name in EXTERNAL_SETTING_NAMES:
            if name in raw_values:
                continue
            value = file_values.get(name)
            if value is not None and str(value).strip():
                raw_values[name] = (value, dotenv_path.parent, str(dotenv_path))

    resolved = {
        name: _configured_path(raw_values[name][0], base=raw_values[name][1])
        if name in raw_values
        else None
        for name in EXTERNAL_SETTING_NAMES
    }
    sources = tuple(
        (name, raw_values[name][2])
        for name in EXTERNAL_SETTING_NAMES
        if name in raw_values
    )
    return ExternalGateSettings(
        manifest=resolved[EXTERNAL_MANIFEST_ENV],
        governance=resolved[EXTERNAL_GOVERNANCE_ENV],
        pose_root=resolved[EXTERNAL_POSE_ROOT_ENV],
        sources=sources,
        dotenv_files_checked=checked_files,
        dotenv_disabled=dotenv_disabled,
    )


def external_gate_figure(context, status: Mapping[str, object]):
    """Render the optional gate's tri-state result without changing model code."""

    import matplotlib.pyplot as plt

    gate_state = str(status.get("gate_state", ""))
    if gate_state == "contract_validated":
        color = "#2a9d8f"
        label = "MANIFEST PREREQUISITES VALIDATED — EVALUATION NOT RUN"
    elif gate_state == "not_configured":
        color = "#b7791f"
        label = "OPTIONAL EXTERNAL STUDY NOT CONFIGURED — NOT RUN"
    else:
        color = "#c44536"
        label = "EXTERNAL CONTRACT BLOCKED — EVALUATION NOT RUN"

    figure, axis = plt.subplots(figsize=(8.5, 2.2))
    axis.barh([0], [1], color=color)
    axis.text(
        0.5,
        0,
        label,
        ha="center",
        va="center",
        color="white",
        weight="bold",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title("Optional subject-indexed external gate")
    for spine in axis.spines.values():
        spine.set_visible(False)
    profile_label = (
        "PAPER PROFILE — EMPIRICAL OUTPUT"
        if context.is_paper
        else "SYNTHETIC SMOKE — NON-EVIDENTIARY"
    )
    figure.text(
        0.995,
        0.005,
        profile_label,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#264653" if context.is_paper else "#8b1e3f",
        weight="bold",
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.98))
    return figure


__all__ = [
    "DISABLE_EXTERNAL_DOTENV_ENV",
    "EXTERNAL_GOVERNANCE_ENV",
    "EXTERNAL_MANIFEST_ENV",
    "EXTERNAL_POSE_ROOT_ENV",
    "EXTERNAL_SETTING_NAMES",
    "ExternalGateSettings",
    "external_gate_figure",
    "load_external_gate_settings",
]
