#!/usr/bin/env python3
"""Generate the GAVD media READMEs from completed preparation and verification.

The independent verifier may legitimately report incomplete coverage when
YouTube sources cannot be retrieved. This generator retains every expected
sequence and its actual processing/verification state instead of hiding it.
Run only after preparation and independent verification have finished.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("cerebral palsy", "myopathic", "normal", "parkinsons", "stroke")


def cell(value):
    return re.sub(r"\s+", " ", str(value)).strip().replace("|", "\\|").replace("`", "'")


def concise_reason(value):
    """Keep download support boilerplate in JSON instead of wide Markdown cells."""
    value = re.sub(r"\x1b\[[0-9;]*m", "", str(value))
    if "private video" in value.lower():
        return "YouTube reports that this video is private."
    if "this video is unavailable" in value.lower():
        return "YouTube reports that this video is unavailable."
    short = re.search(r"Source has (\d+) frames; requires (\d+)", value)
    if short:
        return f"The downloaded video contains {int(short[1]):,} frames, but this sequence needs frame {int(short[2]):,}."
    for marker in (". If the owner", ". Use --cookies", ". See  https", " Use --list-formats"):
        value = value.split(marker, 1)[0]
    value = re.sub(r"ERROR:\s*(?:\[youtube\]\s*)?(?:[A-Za-z0-9_-]{11}:\s*)?", "", value)
    return cell(value)


def number(value):
    return f"{value:,}"


def size(value):
    return f"{value / (1024 ** 3):.3f} GiB ({value:,} bytes)"


def readable_time(value):
    return datetime.fromisoformat(value).astimezone(timezone.utc).strftime("%B %d, %Y, at %H:%M:%S UTC")


def sequence_result(row, check):
    if row["status"] == "completed" and check["status"] == "verified":
        return "Saved; all frames match"
    if row["status"] == "completed":
        return "Saved; frame checks need attention"
    if row["status"] == "blocked_source":
        return "Missing; source unavailable"
    if "Source has" in row.get("error", ""):
        return "Missing; source is too short"
    return "Missing; processing failed"


def file_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def link(label, target, readme):
    relative = os.path.relpath(file_path(target), readme.parent)
    return f"[{cell(label)}]({quote(relative, safe='/.-_')})"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dimensions(record):
    info = record.get("probe", {})
    return f"{info['width']}×{info['height']}" if "width" in info and "height" in info else "—"


def fps(record):
    info = record.get("probe", {})
    value = info.get("avg_frame_rate", info.get("r_frame_rate"))
    if not value:
        return "—"
    try:
        return f"{float(Fraction(value)):.6g} frames/s"
    except (ValueError, ZeroDivisionError):
        return cell(value)


def load_inputs(manifest_path, verification_path):
    manifest = json.loads(manifest_path.read_text())
    verification = json.loads(verification_path.read_text())
    if not manifest.get("completed_at"):
        raise ValueError("Preparation has not finished: completed_at is absent")
    if not verification.get("finished_at"):
        raise ValueError("Independent verification has not finished")
    if datetime.fromisoformat(verification["finished_at"]) < datetime.fromisoformat(manifest["completed_at"]):
        raise ValueError("Independent verification predates the finished preparation run")
    if tuple(manifest.get("categories", [])) != CATEGORIES or tuple(verification.get("categories", [])) != CATEGORIES:
        raise ValueError("Documentation requires all five requested categories")
    sequence_map = {row["sequence_id"]: row for row in manifest["sequences"]}
    verified_map = {row["sequence_id"]: row for row in verification["sequences"]}
    if len(sequence_map) != len(manifest["sequences"]) or len(verified_map) != len(verification["sequences"]):
        raise ValueError("Duplicate sequence identity in manifest or verification")
    if sequence_map.keys() != verified_map.keys():
        raise ValueError("Preparation and verification sequence inventories differ")
    for sequence_id, row in sequence_map.items():
        check = verified_map[sequence_id]
        for key in ("gait_pat", "video_id", "start_frame", "end_frame", "expected_frames"):
            if row[key] != check[key]:
                raise ValueError(f"Preparation and verification disagree on {sequence_id}/{key}")
        if row["annotation_sha256"] != check["csv_sha256"]:
            raise ValueError(f"CSV changed between preparation and verification: {sequence_id}")
        if file_path(row["output_path"]).resolve() != file_path(check["output"]).resolve():
            raise ValueError(f"Verification refers to a different output file: {sequence_id}")
        if row["status"] == "completed" and check["status"] == "verified":
            if row["bytes"] != check["bytes"] or row["probe"]["decoded_frames"] != check["decoded_frames"]:
                raise ValueError(f"Verified output metadata differs from preparation: {sequence_id}")
        if row["status"] == "pending":
            raise ValueError(f"Unprocessed sequence still pending: {sequence_id}")
    if any(row["status"] == "pending" for row in manifest["sources"]):
        raise ValueError("Unprocessed source still pending")
    source_keys = [(row["gait_pat"], row["video_id"]) for row in manifest["sources"]]
    check_source_keys = [(row["gait_pat"], row["video_id"]) for row in verification["sources"]]
    if len(source_keys) != len(set(source_keys)) or set(source_keys) != set(check_source_keys):
        raise ValueError("Preparation and verification source inventories differ")
    source_checks = {(row["gait_pat"], row["video_id"]): row for row in verification["sources"]}
    for source in manifest["sources"]:
        if source["status"] != "available":
            continue
        check = source_checks[(source["gait_pat"], source["video_id"])]
        if not check.get("path") or file_path(source["path"]).resolve() != file_path(check["path"]).resolve():
            raise ValueError(f"Verification refers to a different source file: {source['video_id']}")
        if check["status"] == "verified_decode":
            if any(source["probe"][key] != check[key] for key in ("width", "height", "decoded_frames")) or source["bytes"] != check["bytes"]:
                raise ValueError(f"Verified source metadata differs from preparation: {source['video_id']}")
    return manifest, verification, verified_map


def category_summary(manifest, checks, category=None):
    sources = [row for row in manifest["sources"] if category is None or row["gait_pat"] == category]
    sequences = [row for row in manifest["sequences"] if category is None or row["gait_pat"] == category]
    completed = [row for row in sequences if row["status"] == "completed"]
    verified = [row for row in sequences if checks[row["sequence_id"]]["status"] == "verified"]
    return dict(expected_sources=len(sources), available_sources=sum(row["status"] == "available" for row in sources),
                expected_sequences=len(sequences), completed_sequences=len(completed), verified_sequences=len(verified),
                missing_sequences=len(sequences) - len(completed),
                expected_frames=sum(row["expected_frames"] for row in sequences),
                completed_frames=sum(row["expected_frames"] for row in completed),
                verified_frames=sum(row["expected_frames"] for row in verified),
                source_bytes=sum(row.get("bytes", 0) for row in sources if row["status"] == "available"),
                clip_bytes=sum(row.get("bytes", 0) for row in completed))


def coverage_table(manifest, checks):
    lines = ["| Gait label | Source videos saved / requested | Clips saved / requested | Clips with every frame checked | Missing clips | Frames saved / requested | Frames that match the source |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for category in (*CATEGORIES, None):
        row = category_summary(manifest, checks, category)
        label = f"`{category}`" if category else "**Total**"
        lines.append(f"| {label} | {row['available_sources']} / {row['expected_sources']} | "
                     f"{row['completed_sequences']} / {row['expected_sequences']} | {row['verified_sequences']} | "
                     f"{row['missing_sequences']} | {number(row['completed_frames'])} / {number(row['expected_frames'])} | "
                     f"{number(row['verified_frames'])} |")
    return "\n".join(lines)


def source_inventory(manifest, checks, readme):
    lines = ["| Gait label | YouTube video and local source | Download result | Clips saved / checked / requested | Requested first–last frames | Frames in the source | Image size and playback rate |",
             "| --- | --- | --- | ---: | --- | ---: | --- |"]
    for source in sorted(manifest["sources"], key=lambda row: (row["gait_pat"], row["video_id"])):
        sequences = sorted([row for row in manifest["sequences"]
                            if row["gait_pat"] == source["gait_pat"] and row["video_id"] == source["video_id"]],
                           key=lambda row: (row["start_frame"], row["end_frame"]))
        completed = sum(row["status"] == "completed" for row in sequences)
        verified = sum(checks[row["sequence_id"]]["status"] == "verified" for row in sequences)
        video = f"[{source['video_id']}]({source['url']})"
        if source.get("path"):
            video += " / " + link("local original", source["path"], readme)
        status = "Saved locally" if source["status"] == "available" else "Unavailable on YouTube"
        if source["status"] == "failed":
            status = "Download failed"
        ranges = ", ".join(f"{row['start_frame']}–{row['end_frame']}" for row in sequences)
        lines.append(f"| `{source['gait_pat']}` | {video} | {cell(status)} | "
                     f"{completed} / {verified} / {len(sequences)} | {ranges} | "
                     f"{number(source.get('probe', {}).get('decoded_frames', 0)) if source.get('probe') else '—'} | "
                     f"{dimensions(source)}; {fps(source)} |")
    return "\n".join(lines)


def missing_table(manifest, checks, readme):
    incomplete = [row for row in manifest["sequences"]
                  if row["status"] != "completed" or checks[row["sequence_id"]]["status"] != "verified"]
    if not incomplete:
        return "None. Every requested sequence was completed and independently verified."
    lines = ["| Gait label | Sequence CSV | Requested video, frame range and length | Result | What prevented completion |",
             "| --- | --- | --- | --- | --- |"]
    for row in sorted(incomplete, key=lambda item: (item["gait_pat"], item["video_id"], item["start_frame"])):
        check = checks[row["sequence_id"]]
        reasons = [row["error"]] if row.get("error") else []
        reasons.extend(check.get("issues", []))
        if not reasons:
            reasons.append("The requested sequence video is missing." if check["status"] == "missing_output"
                           else "The sequence still needs a successful frame comparison.")
        clip = Path(row["output_path"]).name
        lines.append(f"| `{row['gait_pat']}` | {link(row['sequence_id'], row['annotation_csv'], readme)} | "
                     f"`{clip}`; {row['start_frame']}–{row['end_frame']}; {row['expected_frames']} | "
                     f"{sequence_result(row, check)} | {concise_reason('; '.join(reasons))} |")
    return "\n".join(lines)


def source_failures(manifest):
    failed = [row for row in manifest["sources"] if row["status"] != "available"]
    if not failed:
        return "None. All requested source IDs have a retained, decodable local source video."
    lines = ["| Gait label | YouTube video | Last download attempt | Download methods tried | YouTube's response |",
             "| --- | --- | --- | --- | --- |"]
    for source in sorted(failed, key=lambda row: (row["gait_pat"], row["video_id"])):
        attempts = source.get("attempts", [])
        latest = readable_time(attempts[-1]["finished_at"]) if attempts and attempts[-1].get("finished_at") else "Not recorded"
        names = {"default": "standard client", "embedded": "embedded player", "android": "Android client"}
        strategies = ", ".join(names.get(attempt["strategy"], attempt["strategy"].replace("_", " ")) for attempt in attempts)
        lines.append(f"| `{source['gait_pat']}` | [{source['video_id']}]({source['url']}) | "
                     f"{latest} | {cell(strategies)} | {concise_reason(source.get('error', source['status']))} |")
    return "\n".join(lines)


def complete_sequence_inventory(manifest, checks, readme):
    lines = ["| Gait label | Sequence CSV | Saved video or requested filename | First–last source frames | Length in frames | Frame extraction result | Reference review |",
             "| --- | --- | --- | ---: | ---: | --- | --- |"]
    for row in sorted(manifest["sequences"], key=lambda item: (item["gait_pat"], item["video_id"], item["start_frame"], item["sequence_id"])):
        label = Path(row["output_path"]).name
        clip = link(label, row["output_path"], readme) if row["status"] == "completed" else f"`{label}`"
        lines.append(f"| `{row['gait_pat']}` | {link(row['sequence_id'], row['annotation_csv'], readme)} | "
                     f"{clip} | {row['start_frame']}–{row['end_frame']} | {row['expected_frames']} | "
                     f"{sequence_result(row, checks[row['sequence_id']])} | "
                     f"{'Leave out of ML: video reference mismatch' if row.get('exclude_from_ml') else 'No specific warning recorded'} |")
    return "\n".join(lines)


def quality_section(manifest, readme):
    flagged = [row for row in manifest["sequences"] if row.get("quality_flags") or row.get("exclude_from_ml")]
    review = manifest.get("content_review", {})
    lines = [f"A separate visual review looked at still images from {review.get('reviewed_sources', 'the available')} saved source videos. "
             "For each source, it used the middle frame of the earliest completed clip. For the source that was too short to produce its requested clip, it used the middle frame of the available source video instead. "
             "Four additional source frames were examined for the video identified below. This review can reveal an obvious mismatch between a video and its annotations, but a single still cannot establish correct walking motion, person identity, bounding boxes or diagnosis throughout every clip."]
    if flagged:
        frames = sum(row["expected_frames"] for row in flagged)
        lines.extend(["", f"**{len(flagged)} saved clips, containing {frames:,} frames, should be left out of model training and evaluation until their video reference is resolved.** "
                      "They come from `normal/2LbERGI34lA`, where the requested ranges show a map introduction or wide street views rather than the near-full-height walking subject described by the annotations. "
                      "The clips remain saved so the discrepancy can be investigated, and their CSV files have not been changed. "
                      "These clips are separate from the missing clips: their requested frames were extracted correctly, but the video content appears inconsistent with the annotation reference.", "",
                      "| Gait label | YouTube video | Sequence CSV | Requested first–last frames | Finding and next step | Review image |",
                      "| --- | --- | --- | ---: | --- | --- |"])
        for row in sorted(flagged, key=lambda item: (item["gait_pat"], item["video_id"], item["start_frame"])):
            evidence = ", ".join(link("View the sampled frames", item, readme) for item in row.get("content_review_evidence", []))
            note = "The video content does not match the annotated walking subject. Leave this clip out of ML use until the reference is resolved."
            lines.append(f"| `{row['gait_pat']}` | `{row['video_id']}` | "
                         f"{link(row['sequence_id'], row['annotation_csv'], readme)} | "
                         f"{row['start_frame']}–{row['end_frame']} | {note} | {evidence} |")
        lines.extend(["", "The manifest records these decisions with `exclude_from_ml: true` and retains the review evidence. "
                      "No additional obvious mismatch was found in the other representative stills, but the limited review does not establish that every remaining annotation is correct."])
    else:
        lines.extend(["", "No specific video-reference warning was recorded. This does not establish the accuracy of every annotation."])
    return "\n".join(lines)


def build_sequence_readme(manifest, verification, checks, manifest_path, verification_path, readme):
    total = category_summary(manifest, checks)
    failed_sources = total["expected_sources"] - total["available_sources"]
    completed_unverified = sum(row["status"] == "completed" and checks[row["sequence_id"]]["status"] != "verified"
                               for row in manifest["sequences"])
    flagged = [row for row in manifest["sequences"] if row.get("exclude_from_ml")]
    content_exclusions = len(flagged)
    content_notice = (f"A separate review flagged **{content_exclusions} saved clips** because their video content appears inconsistent with the annotations. "
                      "Those clips should be left out of machine-learning datasets until the reference is resolved; they are listed separately from the missing videos below.\n"
                      if content_exclusions else "")
    generated = readable_time(datetime.now(timezone.utc).isoformat())
    annotation_root = file_path(manifest["annotation_root"])
    annotation_manifest_path = annotation_root / "source/manifest.json"
    annotation_manifest = json.loads(annotation_manifest_path.read_text())
    source_root = file_path(manifest["source_root"])
    dimension_mismatches, short_sources = [], []
    for source in manifest["sources"]:
        if source["status"] != "available":
            continue
        actual = source.get("probe", {})
        infos = source.get("annotation_video_infos", [source.get("annotation_video_info", {})])
        if any((actual.get("width"), actual.get("height")) != (info.get("width"), info.get("height")) for info in infos):
            dimension_mismatches.append(source)
        if not source.get("covers_all_annotations", actual.get("decoded_frames", 0) >= source["required_last_frame"]):
            short_sources.append(source)
    mismatch_lines = ["| Gait label | YouTube video | Image sizes recorded in the CSV | Downloaded image size |",
                      "| --- | --- | --- | --- |"]
    for source in dimension_mismatches:
        infos = source.get("annotation_video_infos", [source.get("annotation_video_info", {})])
        labels = ", ".join(f"{info.get('width')}×{info.get('height')}" for info in infos)
        mismatch_lines.append(f"| `{source['gait_pat']}` | `{source['video_id']}` | {labels} | {dimensions(source)} |")
    dimension_section = "\n".join(mismatch_lines) if dimension_mismatches else "All saved source videos have the image dimensions recorded in their annotation CSVs."
    short_details = "Every saved source contains enough frames for its requested sequences."
    if short_sources:
        paragraphs = []
        for row in short_sources:
            paragraphs.append(f"The saved video `{row['gait_pat']}/{row['video_id']}` contains {row['probe']['decoded_frames']:,} frames, "
                              f"while its annotations require frames through {row['required_last_frame']:,}. Its requested sequence therefore remains missing.")
            for alternative in row.get("alternative_format_checks", []):
                paragraphs.append(f"An additional download of format `{alternative['format_id']}` on {readable_time(alternative['checked_at'])} "
                                  f"also contained {int(alternative['probe']['streams'][0]['nb_read_frames']):,} frames. "
                                  "This second download did not provide the missing frames, so the sequence was not padded, repeated or otherwise invented.")
        short_details = "\n\n".join(paragraphs)
    metadata_audit = manifest.get("source_metadata_audit", {})
    metadata_note = ""
    if metadata_audit:
        aspect_sources = ", ".join(f"`{row['source_key']}`" for row in metadata_audit.get("aspect_ratio_differences", []))
        absent = ", ".join(f"`{key}`" for key in metadata_audit.get("missing_historical_metadata", []))
        metadata_note = (f"An additional check compared the file fingerprints and sizes of all {metadata_audit['available_sources']} saved sources, "
                         f"then compared {metadata_audit['historical_compared']} of them with older records of their YouTube formats. "
                         f"It found {len(metadata_audit.get('sha256_or_size_issues', []))} file fingerprint or size differences, "
                         f"{len(metadata_audit['fps_mismatches'])} playback-rate differences greater than 0.6 frames per second, and "
                         f"{len(metadata_audit['duration_mismatches'])} duration differences greater than the larger of two seconds or one percent. "
                         f"The {len(metadata_audit['missing_historical_metadata'])} sources without an older comparison record were {absent}. "
                         "These older records provide useful supporting evidence, but they do not prove that the videos are identical to the versions used when GAVD was annotated.\n\n"
                         f"The downloaded image proportions differ from the CSV image proportions by more than two percent for {aspect_sources}. "
                         "For these videos, inspect any cropping or borders before mapping annotation coordinates, because resizing alone may not align the images. "
                         "The detailed comparison is stored in the manifest and in `.review/source-metadata-audit.json`.\n")
    validation = manifest.get("additional_validation", {})
    validation_note = ""
    if validation:
        parts = []
        regression = validation.get("regression_tests", {})
        if regression:
            parts.append(f"The recorded local regression run passed {regression['passed']} tests.")
        integrity = validation.get("integrity", {})
        if integrity:
            parts.append(f"A separate integrity check examined all {integrity['checked_media_files']} saved source and clip files "
                         f"and found {integrity['sha256_or_size_mismatches']} differences in their file fingerprints or sizes.")
        opencv = validation.get("opencv_smoke_test", {})
        if opencv:
            parts.append(f"OpenCV {opencv['version']} also opened the example clip and read all {opencv['decoded_frames']} frames.")
        if validation.get("annotation_verification_report"):
            parts.append(f"The {link('separate annotation check', validation['annotation_verification_report'], readme)} "
                         "records the comparison with the pinned GAVD annotation snapshot.")
        validation_note = " ".join(parts)
    all_checks_text = ("All requested videos passed the frame checks." if verification["complete"] else
                       "The overall verification remains incomplete because some requested videos are missing.")
    unmatched_note = (f"There are also {completed_unverified} saved clips whose frame checks still need attention. "
                      if completed_unverified else "Every saved clip passed the frame checks. ")
    diagram = "!" + link("The example keeps frames 16 through 111, including both ends, to produce a 96-frame clip.",
                          ROOT / "scripts/assets/gavd-sequence-extraction.svg", readme)
    return f"""# GAVD sequence videos

This collection contains **{total['completed_sequences']:,} of the {total['expected_sequences']:,} requested sequence videos** and retains **{total['available_sources']:,} of the {total['expected_sources']:,} referenced YouTube sources**. A separate check confirmed that all **{total['verified_sequences']:,} checked clips contain exactly the requested frames**, for a total of **{total['verified_frames']:,} frames**. The collection covers the GAVD labels `cerebral palsy`, `myopathic`, `normal`, `parkinsons`, and `stroke`.

**{total['missing_sequences']:,} requested clips are still missing.** {failed_sources:,} YouTube sources could not be downloaded, and the available source described below is too short for its requested sequence. {unmatched_note}The full inventory preserves all requested sequences, including the ones that could not be produced.

{content_notice}
This report was generated on **{generated}** from the finished preparation record and independent verification report.

## What was saved

{coverage_table(manifest, checks)}

The saved clips occupy **{size(total['clip_bytes'])}**, and the retained full source videos occupy **{size(total['source_bytes'])}**. These figures count the media files only. They exclude CSVs, reports, temporary downloads and small files used to record processing history.

Frame totals add the length of each sequence, so a source frame can be counted more than once when requested ranges overlap. A source is counted once per YouTube ID. None of the IDs is shared between these five gait labels.

## How the requested frames were selected

Each sequence video preserves the full downloaded image and includes every frame from the first number listed in its CSV through the last number, including both ends. “Cropped” therefore means trimmed in time here; no person bounding box or other spatial crop was applied. All {total['expected_sequences']:,} requested CSVs were checked for consecutive positive frame numbers, consistent sequence and video identities, the correct gait label, and unique output filenames.

For example, {link('cljo2wwu7001a3n6ljmqm39l6.csv', annotation_root / 'normal/cljo2wwu7001a3n6ljmqm39l6.csv', readme)} lists frames **16 through 111** of `3KnFt8bH3tE`. Keeping both endpoints produces **111 − 16 + 1 = 96 frames**, saved as `normal/3KnFt8bH3tE-16-111.mp4`.

{diagram}

GAVD calls the first frame of a source video frame 1, while FFmpeg counts its first frame as position 0. The preparation script adjusts for that difference once, at extraction. It also resets the clip's playback start to zero while preserving the source frame order, image size and timing. No frames are added or dropped to change the playback rate, and the output omits audio, subtitles, data streams and source chapters.

<details>
<summary>Technical extraction and video format details</summary>

The one-based inclusive CSV range `[first_frame, last_frame]` becomes the zero-based half-open FFmpeg range `[first_frame-1, last_frame)`. The filter is:

```text
trim=start_frame=(first_frame-1):end_frame=last_frame,setpts=PTS-STARTPTS
```

The expected length is `last_frame-first_frame+1`. Each clip is an MP4 encoded with `libx264`, CRF 0, preset `fast`, `yuv420p` pixels and `-fps_mode passthrough`. The source's inherited metadata and chapters are removed with `-map_metadata -1` and `-map_chapters -1`. There is exactly one video stream.

CRF 0 preserves the decoded source pixels in the chosen `yuv420p` representation. It does not restore quality already lost before download. These files can be larger than ordinary compressed clips and can require H.264 High 4:4:4 Predictive profile support even though their pixels are `yuv420p`; FFmpeg-based readers support the files, and the example was also checked with OpenCV. Exact source and output measurements remain in the JSON reports.

</details>

## Clips whose video reference needs investigation

{quality_section(manifest, readme)}

## Missing videos and incomplete source material

The following YouTube responses came from actual download attempts during this task. Accessible titles or other public metadata were not treated as proof that the video itself could be downloaded. The {link('preparation record', manifest_path, readme)} keeps the attempt times and full error messages.

{source_failures(manifest)}

{short_details}

The table below lists every sequence that is missing or that still needs a successful frame comparison. A filename without a video link is the requested filename, not evidence that the file exists.

{missing_table(manifest, checks, readme)}

## How the saved frames were checked

The independent verifier reread the CSV files rather than trusting the preparation script's frame counts. It then opened every saved source and clip from beginning to end and compared each clip frame with the corresponding frame of its source. This checks the entire requested range, including the first and last frames, and would detect a clip shifted by one frame even if its length were correct.

The verifier also checked image dimensions, frame counts, playback timestamps and the absence of extra streams. It required the clip to begin at time zero, its frame times to increase in order, and its relative timing to stay within one millisecond of the source. Verification ran from **{readable_time(verification['started_at'])}** to **{readable_time(verification['finished_at'])}**. It found **{len(verification.get('unexpected_outputs', []))} unexpected MP4 files** in the requested folders.

{all_checks_text} The {link('independent verification report', verification_path, readme)} records the result for every sequence. The four video-reference warnings described above remain separate from this frame comparison: matching the downloaded frames does not establish that the annotations describe the right video.

The local tests deliberately include a correctly trimmed clip, a clip shifted by one frame, a clip missing its last requested frame, and a clip played at the wrong speed despite having the correct image frames. They also check missing sources, gaps in CSV frame numbers, duplicate output filenames and unexpected output videos. {validation_note}

<details>
<summary>Technical details of the frame comparison</summary>

FFmpeg produces an MD5 fingerprint for every decoded frame in the same `yuv420p` pixel format for both source and output. The ordered output fingerprints must exactly equal the source fingerprints in `[first_frame-1:last_frame]`. Full decoding must finish without an error, and there must be exactly one output video stream. These frame fingerprints serve a different purpose from the SHA256 file fingerprints below: they compare decoded images even though the source and sequence files use different encoding and container bytes.

</details>

## Image size, playback rate and annotation coordinates

The downloader selected the best available video at or below the annotation's recorded image height and with a stated rate of at most 30 frames per second. It did not resample the video to reach that rate. The CSV files do not record the frame rate of the historical video used for annotation, so the current download cannot by itself establish exact agreement with that earlier version. The verified frame ranges are exact relative to the retained source files.

For **{len(dimension_mismatches):,} saved source videos**, at least one CSV image size differs from the downloaded image size. Bounding boxes in the CSVs remain expressed in their original coordinate system. Before using them to crop or draw over a downloaded image, account for the actual image size and check whether cropping or borders have changed the framing. This preparation did not rewrite any annotation coordinates.

{metadata_note}

<details>
<summary>Downloaded image sizes that differ from their CSV records</summary>

All dimensions below are given as width × height in pixels.

{dimension_section}

</details>

## Folder layout and processing history

Full source videos are saved under `data/videos-gavd/<gait_pat>/`, and the trimmed sequence videos are saved under `data/videos-ms/gavd-sequences/<gait_pat>/`. Folder names preserve the CSV's label exactly, including the space in `cerebral palsy`. Grouping full sources by gait label follows the explicit request for this task, although the repository generally prefers a single shared source directory.

```text
data/videos-gavd/
├── README.md
├── manifest.json
└── <gait_pat>/<youtube-id>.<source-container>

data/videos-ms/gavd-sequences/
├── README.md
├── manifest.json
├── verification.json
└── <gait_pat>/<youtube-id>-<first-frame>-<last-frame>.mp4
```

The source folders did not exist when this task began, so all saved originals were acquired during this session. Some JSON records say `reused_canonical` because a later step reused a pilot download or a file saved earlier in the same session. That wording does not indicate an older collection. All completed sequence clips were created during this task.

Before reusing a source, the script read it completely and counted its frames. Before reusing a clip, it required matching source, annotation and output file fingerprints. Other pretrimmed datasets were not accepted as full source videos just because their filenames resembled YouTube IDs. Downloads were checked in a temporary location before being moved to their final filenames, and incomplete or invalid files were excluded from the totals. Some interrupted pilot records lack selected-format information; the measurements made directly from the saved files are the authoritative values for this collection.

Preparation ran from **{readable_time(manifest['started_at'])}** to **{readable_time(manifest['completed_at'])}**. It used Python **{manifest.get('tools', {}).get('python', 'not recorded')}**, yt-dlp **{manifest.get('tools', {}).get('yt_dlp', 'not recorded')}** and **{cell(manifest.get('tools', {}).get('ffmpeg', 'an unrecorded FFmpeg version'))}**. The Node.js version was **{manifest.get('tools', {}).get('node', 'not recorded')}**. The recovery download for `EHx08gSNfxQ` used Node.js with yt-dlp's `ejs:github` solver after a direct stream failed. No existing Python environment or dependency lockfile was changed.

Hidden `.provenance/` folders hold the small records used for safe reruns. Source folders may also contain temporary fragments under `.acquiring/` or preserved invalid files under `.invalid/`; none is counted as an available source or completed clip. The videos remain local and ignored by Git, while the repository allows selected README files and manifests to be tracked.

## Complete source inventory

The table below includes all **{total['expected_sources']:,} requested YouTube IDs**. A local source link appears only when that source was retained. The three clip counts mean saved clips, clips that passed the frame comparison, and requested clips. Both ends of each listed source-frame range are included. Image sizes and playback rates were measured from the saved source files.

{source_inventory(manifest, checks, readme)}

## Complete sequence inventory

This table includes all **{total['expected_sequences']:,} requested sequence CSVs**. Saved filenames link to their MP4 files, while missing filenames are shown without a video link. The reference-review column identifies the four clips that should be left out of machine-learning use pending investigation; an entry without a specific warning does not establish that all of its annotations are correct.

The {link('preparation record', manifest_path, readme)} provides the underlying CSV path and file fingerprint, source ID and fingerprint when available, requested frame range, expected length, processing outcome, and saved output size and fingerprint. The {link('verification report', verification_path, readme)} gives the independent frame-check result for each sequence.

{complete_sequence_inventory(manifest, checks, readme)}

## Annotation sources and file fingerprints

The annotation inventory is documented in {link('data/gavd/README.md', annotation_root / 'README.md', readme)}. Its {link('source manifest', annotation_manifest_path, readme)} pins commit `{annotation_manifest['commit']}` of the [upstream GAVD repository]({annotation_manifest['repository']}). Only the CSV files grouped by gait label were counted. The copies in `data/gavd/source/` contain the same annotations and were excluded to avoid double counting. No annotation records or pinned source files were changed while preparing these videos.

The reports record SHA256 values, which act as file fingerprints for detecting changes. Each annotation CSV, saved source video and completed clip has a recorded fingerprint. These values support integrity checks and repeatable processing, but they do not establish that a current YouTube video matches the historical annotation reference.

| Report | SHA256 file fingerprint |
| --- | --- |
| {link('Preparation record', manifest_path, readme)} | `{digest(manifest_path)}` |
| {link('Independent verification report', verification_path, readme)} | `{digest(verification_path)}` |

The {link('source-video report', source_root / 'README.md', readme)} and its {link('source manifest', source_root / 'manifest.json', readme)} describe the retained originals. Attribution and annotation terms are preserved in the {link('upstream README', annotation_root / 'source/UPSTREAM_README.md', readme)} and {link('annotation license', annotation_root / 'source/LICENSE', readme)}. Those annotation terms do not themselves establish rights to reuse each third-party YouTube video.

Gait labels remain the dataset's annotations rather than independently confirmed diagnoses. When dividing these clips into machine-learning training and evaluation sets, keep clips and windows from the same source YouTube ID together so that one original video does not appear in both sets.

## Reproducing or updating this collection

Run these commands separately from the repository root. The existing Python 3.12 environment supplies yt-dlp and loguru, while FFmpeg and FFprobe must be on `PATH`.

```bash
experiments/multiple-sclerosis/.venv/bin/python scripts/prepare_gavd_sequence_media.py --inventory-only
experiments/multiple-sclerosis/.venv/bin/python scripts/prepare_gavd_sequence_media.py --workers 3
python3 scripts/verify_gavd_sequence_media.py --jobs 3
python3 scripts/document_gavd_sequence_media.py
```

The preparation and verification commands return a nonzero exit code while requested media remain missing or invalid. Read their reports, then run the documentation command separately; that exit code should not be treated as proof that the successfully saved clips are absent. Rerunning preparation retries unavailable sources and reuses only compatible checked files. The verifier and report generator use Python's standard library.

The focused local tests need FFmpeg and FFprobe but do not access the network:

```bash
python3 -m unittest discover -s scripts -p 'test_*gavd_sequence_media.py' -v
```

To update this generated report, edit {link('scripts/document_gavd_sequence_media.py', ROOT / 'scripts/document_gavd_sequence_media.py', readme)} and regenerate it from finished preparation and verification reports. This keeps the written counts and inventories tied to the recorded results.
"""


def build_source_readme(manifest, verification, checks, readme, sequence_readme):
    total = category_summary(manifest, checks)
    content_exclusions = sum(bool(row.get("exclude_from_ml")) for row in manifest["sequences"])
    return f"""# GAVD source videos

This folder retains **{total['available_sources']:,} of the {total['expected_sources']:,} requested YouTube source videos**, grouped under the GAVD labels `cerebral palsy`, `myopathic`, `normal`, `parkinsons`, and `stroke`. The source files occupy **{size(total['source_bytes'])}**, and preparation finished on **{readable_time(manifest['completed_at'])}**.

These files preserve the full downloaded videos. The shorter sequence videos are stored separately under {link('data/videos-ms/gavd-sequences/', sequence_readme.parent, readme)}, where **{total['completed_sequences']:,} of {total['expected_sequences']:,} requested clips** were saved. A separate check compared every frame in **{total['verified_sequences']:,} clips**, totaling **{total['verified_frames']:,} frames**, with the corresponding source frames. The {link('sequence report', sequence_readme, readme)} explains the extraction, lists all missing clips and includes every requested sequence.

**{content_exclusions} saved clips have a video-reference warning and should be left out of machine-learning use until that reference is resolved.** They remain separate from the missing clips: their frames were extracted correctly, but their source content appears inconsistent with the annotations.

## What was saved

{coverage_table(manifest, checks)}

## Folder layout and file records

Each source is stored as `<gait_pat>/<youtube-id>.<source-container>`. The folders preserve the exact CSV label, including the space in `cerebral palsy`. This arrangement follows the request to group originals by gait label, although the repository generally prefers one shared source directory. The requested IDs are distinct across the five labels, so these category folders do not duplicate a source video.

The source folders did not exist at the beginning of this task. All retained originals were acquired during this session, even when a JSON record says `reused_canonical` because a later step reused a pilot download or a file saved earlier in the session. Some interrupted pilot records lack selected-format details; measurements made directly from the retained files are authoritative.

The {link('source manifest', readme.parent / 'manifest.json', readme)} records each source URL, download result, attempt times and errors, available format details, measured image size and playback rate, frame count, file size and SHA256. A SHA256 value is a file fingerprint that helps detect changes. Hidden `.provenance/` records support safe reruns, while temporary download fragments under `.acquiring/` and invalid files preserved under `.invalid/` are excluded from available-source counts. The video files stay local and ignored by Git.

The downloader chose the best available version at or below the image height recorded in the annotations and with a stated rate of at most 30 frames per second. It did not change the frame rate. A downloaded image may differ in size, framing or borders from the image used in the CSV, so annotation coordinates need to be checked before drawing bounding boxes or applying a spatial crop. The CSVs do not establish the original annotation-time frame rate. The {link('sequence report', sequence_readme, readme)} records the image-size differences, the comparison with older video-format records, and the source that was too short for its requested range.

## Clips whose video reference needs investigation

{quality_section(manifest, readme)}

## Sources that could not be downloaded

The following responses came from actual attempts to retrieve the video files. Their associated CSVs remain in the complete requested inventory.

{source_failures(manifest)}

## Complete source inventory

This table includes all **{total['expected_sources']:,} requested IDs**. Each requested range includes both its first and last source frames. The clip counts mean saved clips, clips whose frames were independently checked, and requested clips; the clips themselves live in the separate sequence folders.

{source_inventory(manifest, checks, readme)}

## Annotation sources and further details

The {link('GAVD annotation inventory', ROOT / 'data/gavd/README.md', readme)} documents the input CSVs and their source. The {link('upstream annotation license', ROOT / 'data/gavd/source/LICENSE', readme)} does not itself establish reuse rights for every third-party YouTube video. Gait labels remain dataset annotations rather than independently confirmed diagnoses. When building machine-learning evaluation sets, keep clips from the same original video together.

This report was generated by {link('scripts/document_gavd_sequence_media.py', ROOT / 'scripts/document_gavd_sequence_media.py', readme)} from the finished preparation record and the {link('independent frame verification report', sequence_readme.parent / 'verification.json', readme)}, which finished on **{readable_time(verification['finished_at'])}**. The {link('sequence report', sequence_readme, readme)} contains the full sequence inventory, file fingerprints, processing methods and commands for reproducing or updating the collection.
"""


def atomic_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        stream.write(value)
        temporary = Path(stream.name)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/videos-ms/gavd-sequences/manifest.json")
    parser.add_argument("--verification", type=Path, default=ROOT / "data/videos-ms/gavd-sequences/verification.json")
    args = parser.parse_args()
    manifest, verification, checks = load_inputs(args.manifest, args.verification)
    sequence_readme = file_path(manifest["output_root"]) / "README.md"
    source_readme = file_path(manifest["source_root"]) / "README.md"
    sequence_text = build_sequence_readme(manifest, verification, checks, args.manifest, args.verification, sequence_readme)
    source_text = build_source_readme(manifest, verification, checks, source_readme, sequence_readme)
    atomic_write(sequence_readme, sequence_text)
    atomic_write(source_readme, source_text)
    print(f"Wrote {sequence_readme} ({len(sequence_text.encode('utf-8')):,} bytes)")
    print(f"Wrote {source_readme} ({len(source_text.encode('utf-8')):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
