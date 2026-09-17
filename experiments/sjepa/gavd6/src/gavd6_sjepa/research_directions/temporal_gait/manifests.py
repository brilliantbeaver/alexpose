"""Versioned exact-input inventory. No discovery, downloads or guessed roles."""
from pathlib import Path
import math
import re
from .contracts import read_json, digest

SCHEMA = "temporal-gait-manifest-v1"
ROLES = {"train", "development", "calibration", "test", "excluded"}


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def read_manifest(path, kind):
    data = read_json(path)
    if data.get("schema_version") != SCHEMA or data.get("kind") != kind or not isinstance(data.get("rows"), list):
        raise ValueError(f"Expected versioned {kind} JSON manifest: {path}")
    return data


def unique(rows, key):
    out = {}
    for r in rows:
        if not isinstance(r, dict) or not isinstance(r.get(key), str) or not r[key].strip() or r[key] in out:
            raise ValueError(f"Missing/duplicate {key}")
        out[r[key]] = dict(r)
    return out


def groups(videos, links):
    parent = {v: v for v in videos}
    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    known = {}
    for row in links:
        v = row.get("video_id")
        if v not in parent or not row.get("provenance"):
            raise ValueError("Unknown identity link or missing verification provenance")
        found = False
        for kind in ("participant_id", "duplicate_id"):
            identity = row.get(kind)
            if not identity:
                continue
            found = True
            key = (kind, str(identity))
            if key in known:
                a, b = find(v), find(known[key])
                parent[max(a, b)] = min(a, b)
            known[key] = v
        if not found:
            raise ValueError("Identity row requires verified participant_id or duplicate_id")
    components = {}
    for v in parent:
        components.setdefault(find(v), []).append(v)
    return {v: digest(sorted(members))[:24] for members in components.values() for v in members}


def inventory(cfg):
    cfg.validate(check_input_paths=True)
    if cfg.mode != "real":
        raise ValueError("Real manifest inventory may not synthesize input")
    docs = {name: read_manifest(getattr(cfg, name + "_manifest"), name) for name in ("video", "sequence", "pose", "identity", "exposure", "reservation", "split")}
    if docs["identity"].get("verified_links_complete") is not True:
        raise ValueError("Identity registry must explicitly attest verified links are complete (empty is permitted)")
    if docs["exposure"].get("historical_laterality_93_checked") is not True or not docs["exposure"].get("audit_provenance"):
        raise ValueError("Exposure audit must account explicitly for historical 93-source development cohort")
    videos = unique(docs["video"]["rows"], "video_id")
    seqs = unique(docs["sequence"]["rows"], "sequence_id")
    poses = unique(docs["pose"]["rows"], "sequence_id")
    split = unique(docs["split"]["rows"], "video_id")
    exposure = unique(docs["exposure"]["rows"], "video_id")
    reservation = unique(docs["reservation"]["rows"], "video_id")
    for name, mapping in (("split", split), ("exposure", exposure), ("reservation", reservation)):
        if set(mapping) != set(videos):
            raise ValueError(f"{name} must cover exactly the supplied video roster; no implicit fresh sources")
    if any(not valid_sha(v.get("sha256")) for v in videos.values()):
        raise ValueError("Video SHA256 must be 64 lowercase hexadecimal characters")
    # Explicit content digests establish exact duplicates even if the operator
    # accidentally omits the corresponding link. They never infer a person.
    content_links = [{"video_id": v, "duplicate_id": "content-sha256:" + row["sha256"],
                      "provenance": "explicit_manifest_content_digest"} for v, row in videos.items()]
    group = groups(videos, [*docs["identity"]["rows"], *content_links])
    for v, item in videos.items():
        if not Path(item.get("video_path", "")).is_absolute() or not valid_sha(item.get("sha256")):
            raise ValueError("Explicit full-video absolute path and content SHA256 required")
        if item.get("is_full_video") is not True:
            raise ValueError("A clipped cache cannot be represented as a full source video")
        role = split[v].get("role")
        if role not in ROLES or not split[v].get("reason"):
            raise ValueError("Every source needs explicit valid role and reason")
        state = exposure[v].get("status")
        if type(exposure[v].get("historical_laterality_member")) is not bool:
            raise ValueError("Exposure must explicitly audit historical_laterality_member for every source")
        if state not in {"certified_unexposed", "development_exposed", "test_evaluated", "unknown"} or not exposure[v].get("provenance"):
            raise ValueError("Explicit exposure status/provenance required")
        if role in {"test", "calibration"} and state != "certified_unexposed":
            raise ValueError("Exposed/unknown sources cannot enter an untouched test/calibration")
        if role in {"test", "calibration"} and exposure[v]["historical_laterality_member"]:
            raise ValueError("Historical laterality development source cannot become untouched test")
        if type(reservation[v].get("protected")) is not bool or not reservation[v].get("reason"):
            raise ValueError("Every reservation requires protected bool and reason")
        if reservation[v]["protected"] and role != "excluded":
            raise ValueError("Historical protected sources remain excluded in this protocol")
        item.update(role=role, group_id=group[v], reason=split[v]["reason"], exposure=state,
                    historical_laterality_member=exposure[v]["historical_laterality_member"])
    bygroup = {}
    for v in videos.values():
        bygroup.setdefault(v["group_id"], set()).add(v["role"])
    if any(len(r) != 1 for r in bygroup.values()):
        raise ValueError("Duplicate/known-person component crosses source partitions")
    bout_ledger = []
    for s, row in seqs.items():
        v = row.get("video_id")
        if v not in videos:
            raise ValueError("Bout references unknown full video")
        start, end = row.get("start_pts"), row.get("end_pts_exclusive")
        if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (start, end)) or start < 0 or end <= start:
            raise ValueError("Bout requires finite explicit PTS [start,end) boundaries")
        if row.get("boundary_convention") != "pts_half_open" or not row.get("annotation_provenance"):
            raise ValueError("No guessed index base or continuity; require complete PTS bout provenance")
        eligible = row.get("walking_status") == "complete_walking_bout" and videos[v]["role"] != "excluded"
        reason = "eligible_complete_bout" if eligible else ("protected_or_excluded_source" if videos[v]["role"] == "excluded" else "not_complete_walking_bout")
        if eligible and s not in poses:
            raise ValueError(f"No explicitly supplied complete-bout pose cache for {s}; no extractor fallback")
        row.update(role=videos[v]["role"], group_id=group[v], eligible=eligible, reason=reason,
                   historical_laterality_member=videos[v]["historical_laterality_member"])
        bout_ledger.append(row)
    if set(poses) - set(seqs):
        raise ValueError("Pose cache references undeclared bout")
    for s, p in poses.items():
        if not Path(p.get("pose_path", "")).is_absolute() or not valid_sha(p.get("sha256")) or not valid_sha(p.get("checkpoint_sha256")):
            raise ValueError("Pose requires exact path/SHA256")
        required = {"extractor", "extractor_version", "checkpoint_sha256", "tracking_provenance"}
        if any(not p.get(k) for k in required):
            raise ValueError("Pose extractor/checkpoint/tracking provenance required")
        if p.get("causal_preprocessing") is not True or p.get("coordinate_frame") != "full_frame_pixels" or p.get("joint_order") != "mediapipe33" or p.get("time_kind") != "video_pts" or p.get("complete_bout") is not True:
            raise ValueError("Noncausal/unaligned/partial pose cache incompatible with forecasting")
        if p.get("source_video_sha256") != videos[seqs[s]["video_id"]]["sha256"]:
            raise ValueError("Pose/full-video content identity mismatch")
    return {"schema_version": SCHEMA, "videos": list(videos.values()), "bouts": bout_ledger, "poses": poses,
            "groups": len(bygroup), "declared_videos": len(videos), "declared_bouts": len(seqs),
            "available_declared_seconds": sum(r["end_pts_exclusive"] - r["start_pts"] for r in seqs.values()),
            "test_status": "untouched_source_held_out_planned" if any(v["role"] == "test" for v in videos.values()) else "exploratory_development_only",
            "identity_limit": "Source is not a participant; only verified links were merged.",
            "media_access": "metadata_only; no video or pose contents opened"}
