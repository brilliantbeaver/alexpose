"""Immutable per-window teacher features and strict cache aggregation."""

from pathlib import Path

import numpy as np
import pandas as pd

from gavd6_sjepa.shared_infrastructure.artifact_io_operations import (
    atomic_write_dataframe_csv,
    sha256_file,
)

from .fi_cohort import load_cohort
from .fi_contracts import DIRECT_PROTOCOL, protocol_name, check_run, read_json, save_npz, stable_key, write_once_json
from .fi_nuisance_features import context_nuisance
from .fi_token_regions import fixed_projection, pool_context, pool_target, region_masks, union_box


def load_window(row):
    def array(key, name):
        with np.load(row[key], allow_pickle=False) as payload:
            return payload[name]

    return (
        array("frame_path", "video"),
        array("box_path", "source_boxes"),
        array("model_box_path", "model_boxes"),
        array("pose_path", "skeleton"),
    )


def cache_binding(root):
    root = Path(root)
    return stable_key(
        *[
            sha256_file(root / path)
            for path in (
                "config/run-contract.json",
                "config/cohort-contract.json",
                "config/teacher-contract.json",
                "config/nuisance-contract.json",
                "config/projection-contract.json",
            )
        ]
    )


def cache_teacher(root, adapter):
    root = Path(root)
    contract = check_run(root)
    direct = protocol_name(contract) == DIRECT_PROTOCOL
    cohort = load_cohort(root, verify_artifacts=True)
    projection_path = root / "config/projection-256.npy"
    if (root / "config/projection-contract.json").exists():
        saved = read_json(root / "config/projection-contract.json")
        if sha256_file(projection_path) != saved["sha256"]:
            raise ValueError("Projection checksum mismatch")
        # Reuse the frozen matrix, not a new QR decomposition whose final bits
        # may differ across otherwise compatible BLAS/NumPy versions.
        projection = np.load(projection_path, allow_pickle=False)
        if projection.shape != (768, 256) or not np.isfinite(projection).all():
            raise ValueError("Invalid frozen projection")
    else:
        projection = fixed_projection(768, seed=contract["projection_seed"])
        temporary = projection_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, projection)
        temporary.replace(projection_path)
    write_once_json(
        root / "config/projection-contract.json",
        {
            "sha256": sha256_file(projection_path),
            "seed": contract["projection_seed"],
            "input_dim": 768,
            "output_dim": 256,
            "scaling": "sqrt(768/256); canonical-sign orthogonal columns",
        },
    )
    binding = cache_binding(root)
    schema = None
    for row in cohort.to_dict("records"):
        path = root / "teacher-cache" / f"{row['window_id']}.npz"
        receipt = path.with_suffix(".json")
        if receipt.exists():
            saved = read_json(receipt)
            if saved["binding"] != binding or saved["sha256"] != sha256_file(path):
                raise ValueError("Existing cache entry has wrong lineage or content")
            continue
        video, boxes, model_boxes, skeleton = load_window(row)
        geometry_error = adapter.verify_geometry(video)
        context = pool_context(adapter.encode_past_context(video), model_boxes, allow_empty_background=direct)
        person, background = pool_target(adapter.encode_full_target(video), model_boxes, allow_empty_background=direct)
        nuisance, names, matching = context_nuisance(video, boxes, skeleton, row, allow_empty_background=direct)
        if direct:
            support = np.mean([region_masks(union_box(model_boxes[t:t + 2]),
                                           allow_empty_background=True)[1].mean() for t in range(0, 32, 2)])
            nuisance = np.r_[nuisance, support]
            names = [*names, "observed_background_token_fraction"]
        if schema is not None and schema != names:
            raise ValueError("Nuisance schema changed across windows")
        schema = names
        write_once_json(
            root / "config/nuisance-schema.json",
            {
                "columns": names,
                "context_embedding_columns": len(context),
                "context_pool_order": [
                    "context_global",
                    "context_person_last",
                    "context_background",
                ],
                "context_pool_width": 768,
            },
        )
        save_npz(
            path,
            baseline=np.r_[context, nuisance].astype(np.float32),
            skeleton=skeleton,
            person=(person @ projection).astype(np.float32),
            background=(background @ projection).astype(np.float32),
            matching=matching,
            window_id=np.array(row["window_id"]),
            binding=np.array(binding),
        )
        write_once_json(
            receipt,
            {
                "window_id": row["window_id"],
                "binding": binding,
                "sha256": sha256_file(path),
                "geometry_max_abs": geometry_error,
            },
        )
    entries = []
    for row in cohort.to_dict("records"):
        path = root / "teacher-cache" / f"{row['window_id']}.npz"
        receipt = read_json(path.with_suffix(".json"))
        entries.append({"window_id": row["window_id"], "path": str(path), **receipt})
    index_path = root / "manifests/cache-index.csv"
    if (root / "config/cache-contract.json").exists():
        load_cache(root)
        return
    atomic_write_dataframe_csv(index_path, pd.DataFrame(entries))
    write_once_json(
        root / "config/cache-contract.json",
        {
            "binding": binding,
            "index_sha256": sha256_file(index_path),
            "schema_sha256": sha256_file(root / "config/nuisance-schema.json"),
        },
    )
    load_cache(root)


def load_cache(root):
    root = Path(root)
    if read_json(root / "config/run-contract.json").get("protocol") == "direct-v3":
        from .fi_cache_reuse import load_reused_cache
        return load_reused_cache(root)
    cohort = load_cohort(root)
    contract = read_json(root / "config/cache-contract.json")
    binding = cache_binding(root)
    if (
        contract["binding"] != binding
        or contract["index_sha256"] != sha256_file(root / "manifests/cache-index.csv")
        or contract["schema_sha256"]
        != sha256_file(root / "config/nuisance-schema.json")
    ):
        raise ValueError("Cache configuration/index changed")
    if (
        sha256_file(root / "config/projection-256.npy")
        != read_json(root / "config/projection-contract.json")["sha256"]
    ):
        raise ValueError("Projection checksum mismatch")
    index = pd.read_csv(root / "manifests/cache-index.csv")
    if not index.window_id.is_unique or set(index.window_id) != set(cohort.window_id):
        raise ValueError("Missing or duplicated cache entries")
    if {p.stem for p in (root / "teacher-cache").glob("*.npz")} != set(
        cohort.window_id
    ):
        raise ValueError("Unexpected or missing teacher cache files")
    index = index.set_index("window_id")
    arrays = {
        name: []
        for name in ("baseline", "skeleton", "person", "background", "matching")
    }
    for window_id in cohort.window_id:
        row = index.loc[window_id]
        expected = root / "teacher-cache" / f"{window_id}.npz"
        if (
            Path(row.path).resolve() != expected.resolve()
            or sha256_file(expected) != row.sha256
            or row.binding != binding
        ):
            raise ValueError(f"Cache checksum/path/binding mismatch: {window_id}")
        with np.load(expected, allow_pickle=False) as payload:
            if (
                str(payload["window_id"]) != window_id
                or str(payload["binding"]) != binding
            ):
                raise ValueError("Cache identity mismatch")
            for name, entries in arrays.items():
                value = payload[name]
                if not np.isfinite(value).all():
                    raise ValueError("Non-finite cache array")
                entries.append(value)
    arrays = {name: np.stack(values) for name, values in arrays.items()}
    if arrays["skeleton"].shape != (len(cohort), 32, 33, 4) or any(
        arrays[name].shape != (len(cohort), 256) for name in ("person", "background")
    ):
        raise ValueError("Cache array shapes violate experiment contract")
    return cohort, arrays
