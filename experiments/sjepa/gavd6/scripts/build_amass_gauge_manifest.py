#!/usr/bin/env python3
"""Generate persistent full-sequence AMASS-Gauge corruption draws."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

from gavd6_sjepa.latent_laterality import (
    SequenceGaugeConfig,
    generate_sequence_draw,
    sequence_gauge_config_json,
)


def _safe_tensor(root: Path, relative_text: str) -> Path:
    relative = PurePosixPath(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe tensor path: {relative_text!r}")
    path = root.joinpath(*relative.parts).resolve()
    path.relative_to(root.resolve())
    return path


def build(
    source_manifest: Path,
    tensor_root: Path,
    output_manifest: Path,
    *,
    draws: int,
    seed: int,
    config: SequenceGaugeConfig | None = None,
    chart_pairs: bool = False,
) -> pd.DataFrame:
    source = pd.read_csv(source_manifest)
    required = {
        "tensor_relative_path",
        "identity",
        "split",
        "canonical_frames",
        "coordinate_frame",
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Source manifest is missing {sorted(missing)}")
    if "status" in source:
        source = source.loc[
            source.status.isin({"converted", "skipped_valid_existing"})
        ].copy()
    if source.empty:
        raise ValueError("Source manifest contains no successfully converted sequence")
    if not source.tensor_relative_path.is_unique:
        raise ValueError("Source tensor paths must be unique")
    frame_names = source.coordinate_frame.astype(str)
    neutral = (
        frame_names.str.startswith("gauge-neutral-travel-v1")
        | frame_names.str.startswith("gauge_neutral_pelvis_travel")
        | (frame_names == "declared_image_space_unanchored")
    )
    if not neutral.all():
        raise ValueError(
            "Every sequence must use a declared gauge-neutral travel or image-space "
            "chart; legacy named-hip frames are diagnostic-only"
        )
    config = config or SequenceGaugeConfig()
    config_json = sequence_gauge_config_json(config)
    source = source.loc[
        pd.to_numeric(source.canonical_frames, errors="coerce") >= config.window_frames
    ].copy()
    if source.empty:
        raise ValueError("No successfully converted sequence is long enough for one window")
    rows = []
    for sequence_index, row in source.iterrows():
        path = _safe_tensor(tensor_root, str(row.tensor_relative_path))
        with np.load(path, allow_pickle=False) as archive:
            coordinates = np.asarray(archive["coordinates"], dtype=np.float32)
            valid = np.asarray(archive["valid"], dtype=bool)
            pelvis = (
                np.asarray(archive["pelvis_world_m"], dtype=np.float32)
                if "pelvis_world_m" in archive.files
                else None
            )
        sequence_id = str(row.get("motion_id", row.tensor_relative_path))
        # AMASS has repeated human-readable motion names.  Use the immutable
        # archive path to seed the corruption, while preserving the readable
        # name in the manifest for reporting.
        generation_key = str(row.tensor_relative_path)
        for corruption_draw in range(draws):
            draw, _ = generate_sequence_draw(
                coordinates,
                valid,
                sequence_id=generation_key,
                identity=str(row.identity),
                split=str(row.split),
                corruption_draw=corruption_draw,
                seed=seed,
                config=config,
                pelvis_world=pelvis,
            )
            chart_bits = (0, 1) if chart_pairs else (draw.latent_chart_bit,)
            pair_id = f"{row.tensor_relative_path}:draw-{corruption_draw}"
            for chart_bit in chart_bits:
                member = (
                    replace(
                        draw,
                        latent_chart_bit=int(chart_bit),
                        generator_version="sequence-gauge-v2-chart-paired",
                    )
                    if chart_pairs
                    else draw
                )
                payload = member.to_dict()
                record = {
                    "sequence_index": int(sequence_index),
                    "sequence_id": sequence_id,
                    "tensor_relative_path": str(row.tensor_relative_path),
                    "identity": str(row.identity),
                    "split": str(row.split),
                    "canonical_frames": int(row.canonical_frames),
                    "corruption_draw": corruption_draw,
                    "path_family": member.path_family,
                    "gauge_path_rle": json.dumps(payload["gauge_path_rle"]),
                    "switch_frames": json.dumps(payload["switch_frames"]),
                    "semantic_scope": member.semantic_scope,
                    "sensor_reflection_bit": member.sensor_reflection_bit,
                    "latent_chart_bit": member.latent_chart_bit,
                    "nuisance_boundary_frames": json.dumps(
                        payload["nuisance_boundary_frames"]
                    ),
                    "occlusion_seed": member.occlusion_seed,
                    "noise_seed": member.noise_seed,
                    "generator_version": member.generator_version,
                    "source_sha256": str(row.get("source_sha256", "")),
                }
                if chart_pairs:
                    # The complementary source and chart actions cancel:
                    # P^(path xor chart) P^chart z = P^path z.  Thus neither
                    # observed coordinates nor validity can reveal this label.
                    record.update(
                        {
                            "chart_pair_id": pair_id,
                            "source_chart_bit": int(chart_bit),
                            "sequence_gauge_config": config_json,
                        }
                    )
                rows.append(record)
    output = pd.DataFrame(rows)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    os.replace(temporary, output_manifest)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--chart-pairs",
        action="store_true",
        help="Emit complementary source/chart rows with identical observations.",
    )
    parser.add_argument(
        "--boundary-mode",
        choices=("interpolate", "gap"),
        default="interpolate",
        help="Apply the same boundary nuisance at true and pseudo events.",
    )
    parser.add_argument("--boundary-radius-frames", type=int, default=1)
    parser.add_argument(
        "--nuisance-selection",
        choices=("matched", "independent"),
        default="matched",
        help="Whether gap locations are conditioned on switches or sampled independently.",
    )
    args = parser.parse_args()
    if args.draws < 1:
        parser.error("--draws must be positive")
    if args.boundary_radius_frames < 0:
        parser.error("--boundary-radius-frames must be nonnegative")
    return args


def main() -> int:
    args = parse_args()
    output = build(
        args.source_manifest.resolve(),
        args.tensor_root.resolve(),
        args.output_manifest.resolve(),
        draws=args.draws,
        seed=args.seed,
        config=SequenceGaugeConfig(
            boundary_mode=args.boundary_mode,
            boundary_radius_frames=args.boundary_radius_frames,
            nuisance_selection=args.nuisance_selection,
        ),
        chart_pairs=args.chart_pairs,
    )
    print(f"Wrote {len(output):,} full-sequence corruption draws to {args.output_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
