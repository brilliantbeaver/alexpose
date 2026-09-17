"""Uncapped full-bout indexing and physically separate context/target arrays."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .contracts import atomic_npz, atomic_json, read_json, digest
from .preprocessing import context_input, nearest_endpoint, signed_speed_contrast

CONTEXT_KEYS = ("context", "context_valid", "context_times", "context_age", "query_times")


@dataclass
class WindowDataset:
    arrays: dict
    records: list

    def __post_init__(self):
        if any(len(a) != len(self.records) for a in self.arrays.values()):
            raise ValueError("Dataset arrays/records row counts differ")
        if len({r["window_id"] for r in self.records}) != len(self.records):
            raise ValueError("Duplicate window identity")

    def __len__(self):
        return len(self.records)

    def subset(self, indices):
        ids = np.asarray(indices, dtype=int)
        return WindowDataset({k: v[ids] for k, v in self.arrays.items()}, [self.records[i] for i in ids])

    def save(self, path):
        path = Path(path)
        atomic_npz(path, **self.arrays)
        atomic_json(path.with_suffix(".json"), self.records)

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as raw:
            arrays = {k: raw[k] for k in raw.files}
        return cls(arrays, read_json(path.with_suffix(".json")))


def issue_times(start, end, cfg):
    """Full half-open bout retained; even no-future end windows remain SSL eligible."""
    first = start + cfg.prefix_seconds
    if first > end + 1e-9:
        return np.empty(0)
    issues = first + np.arange(int(np.floor((end - first) / cfg.window_stride_seconds + 1e-9)) + 1) * cfg.window_stride_seconds
    if end - issues[-1] > 1e-9:
        issues = np.append(issues, end)
    return issues


def make_window(pose, cfg, bout, issue):
    past = context_input(pose, cfg, issue)
    index = context_input(pose, cfg, issue, index_resized=True)
    if not past["context_valid"].any():
        raise ValueError("no_observed_context")
    h = len(cfg.horizons)
    future = np.zeros((h, 2, 33, 2), np.float32)
    future_valid = np.zeros((h, 2, 33), bool)
    future_times = np.full((h, 2, 33), np.nan)
    endpoint = np.zeros((h, 33, 2), np.float32)
    endpoint_valid = np.zeros((h, 33), bool)
    endpoint_times = np.full((h, 33), np.nan)
    for hi, horizon in enumerate(cfg.horizons):
        query = issue + horizon
        for j, target_time in enumerate((query - 1 / cfg.grid_hz, query)):
            # All pose arrays passed here are restricted to the full declared bout.
            coords, valid, times = nearest_endpoint(pose, target_time, issue, cfg.endpoint_tolerance_seconds,
                                                    interval=(query - cfg.target_interval_seconds, query))
            future[hi, j, valid] = (coords[valid] - past["origin"]) / past["scale"]
            future_valid[hi, j] = valid
            future_times[hi, j] = times
        coords, valid, times = nearest_endpoint(pose, query, issue, cfg.endpoint_tolerance_seconds)
        endpoint[hi, valid] = (coords[valid] - past["origin"]) / past["scale"]
        endpoint_valid[hi] = valid
        endpoint_times[hi] = times
    arrays = {k: past[k] for k in CONTEXT_KEYS}
    arrays.update({"index_" + k: index[k] for k in CONTEXT_KEYS})
    arrays.update(future=future, future_valid=future_valid, future_times=future_times, endpoint=endpoint,
                  endpoint_valid=endpoint_valid, endpoint_times=endpoint_times,
                  origin=past["origin"], scale=np.float64(past["scale"]), scale_valid=np.bool_(past["scale_valid"]))
    key = digest([bout["sequence_id"], bout["video_id"], issue, cfg.prefix_seconds, cfg.grid_hz, cfg.horizons, cfg.endpoint_tolerance_seconds])[:28]
    record = dict(window_id=key, video_id=bout["video_id"], sequence_id=bout["sequence_id"], group_id=bout["group_id"],
                  role=bout["role"], issue_time=float(issue), bout_start=bout["start_pts"], bout_end=bout["end_pts_exclusive"],
                  mode=cfg.mode, scale_reason=past["scale_reason"], context_start_s=float(issue - cfg.prefix_seconds),
                  last_query_s=float(issue - 1 / cfg.grid_hz), prefix_open_right=True,
                  target_queries=[float(issue + x) for x in cfg.horizons],
                  preprocessing="prefix-pts-pixels-v1", index_control="index-linear-prefix-v1_not_historical_replay")
    record["historical_laterality_member"] = bout.get("historical_laterality_member", False)
    raw_prefix = (pose.times >= issue - cfg.prefix_seconds) & (pose.times < issue)
    for name, xyz, ok, clock in (("raw", pose.coords[raw_prefix], pose.observed[raw_prefix], pose.times[raw_prefix]),
                                  ("timegrid", past["context"], past["context_valid"], past["query_times"]),
                                  ("indexgrid", index["context"], index["context_valid"], index["query_times"])):
        value = signed_speed_contrast(xyz, ok, clock)
        record["diagnostic_" + name + "_signed_speed_contrast"] = value if np.isfinite(value) else None
    return arrays, record


def prepare_bout(pose, cfg, bout):
    pose.validate()
    if np.any(pose.times < bout["start_pts"]) or np.any(pose.times >= bout["end_pts_exclusive"]):
        raise ValueError("Pose cache extends outside declared half-open bout")
    items, records, exclusions = [], [], []
    for issue in issue_times(bout["start_pts"], bout["end_pts_exclusive"], cfg):
        try:
            arrays, record = make_window(pose, cfg, bout, float(issue))
        except ValueError as e:
            if str(e) not in {"no_observed_context", "Empty half-open prefix"}:
                raise
            exclusions.append({"issue_time": float(issue), "reason": str(e)})
            continue
        items.append(arrays)
        records.append(record)
    dataset = WindowDataset({k: np.stack([r[k] for r in items]) for k in items[0]} if items else {}, records)
    intervals = sorted((r["context_start_s"], r["issue_time"]) for r in records)
    merged = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(hi, merged[-1][1])
        else:
            merged.append([lo, hi])
    seconds = bout["end_pts_exclusive"] - bout["start_pts"]
    report = {"sequence_id": bout["sequence_id"], "video_id": bout["video_id"], "group_id": bout["group_id"], "role": bout["role"],
              "available_seconds": seconds, "retained_original_frames": len(pose.times), "unique_windows": len(records),
              "covered_context_seconds": sum(hi - lo for lo, hi in merged), "window_exclusions": exclusions,
              "reason": "eligible" if records else "short_bout_or_no_observed_prefix", "vfr": bool(len(pose.times) > 2 and np.ptp(np.diff(pose.times)) > 1e-5),
              "ssl_windows": len(records), "primary_scale_supported_windows": int(dataset.arrays["scale_valid"].sum()) if records else 0}
    return dataset, report


def concatenate(datasets):
    datasets = [d for d in datasets if len(d)]
    if not datasets:
        return WindowDataset({}, [])
    return WindowDataset({k: np.concatenate([d.arrays[k] for d in datasets]) for k in datasets[0].arrays}, [r for d in datasets for r in d.records])
