#!/usr/bin/env python3
"""Audit distinct source YouTube IDs using YouTube's public oEmbed endpoint.

HTTP 200 with valid YouTube video metadata is an existence proxy, not playback
confirmation. oEmbed failures receive a watch-page player-status fallback;
HTTP failures alone never establish unavailability.
No video files are downloaded. Results are cached for 24 hours by default.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request


METHOD = "youtube_oembed_v1"
SUCCESS = "public_metadata_available"
UNKNOWN = "inconclusive"
PLAYER_OK = "player_status_ok"
UNAVAILABLE = "unavailable_publicly"
RESTRICTED = "restricted"
STATUSES = (SUCCESS, PLAYER_OK, UNAVAILABLE, RESTRICTED, UNKNOWN)
WATCH_METHOD = "yt_initial_player_response_v2"


def classify_player_status(status):
    """Classify displayed reasons, never tracking strings or generic HTTP errors."""
    if status.get("status") == "OK":
        return PLAYER_OK
    texts = []
    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("reason", "content", "simpleText", "text", "title", "description") and isinstance(item, str):
                    texts.append(item)
                elif key == "messages" and isinstance(item, list):
                    texts.extend(x for x in item if isinstance(x, str))
                if isinstance(item, (dict, list)):
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
    collect(status)
    reason = " ".join(texts).lower()
    unavailable = ("removed", "deleted", "no longer available", "does not exist",
                   "terminated", "copyright", "isn't available anymore")
    restrictions = ("sign in", "login", "age-restricted", "age restricted", "confirm your age",
                    "country", "region", "bot", "consent")
    if "private video" in reason:
        return UNAVAILABLE
    if status.get("status") == "LOGIN_REQUIRED" or any(term in reason for term in restrictions):
        return RESTRICTED
    if any(term in reason for term in unavailable):
        return UNAVAILABLE
    return UNKNOWN


def check_watch_page(result, timeout):
    """Keep exact player evidence; only explicit reasons imply public unavailability."""
    evidence = {"checked_at_utc": utc_now(), "url": result["url"], "method": WATCH_METHOD}
    result["watch_page_evidence"] = evidence
    try:
        request = urllib.request.Request(result["url"], headers={
            "User-Agent": "Mozilla/5.0 (compatible; GAVD-availability-audit/1.0)"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read(3_000_001)
            evidence.update(http_status=response.status, final_url=response.url,
                            bytes_read=len(body), body_truncated=len(body) > 3_000_000,
                            body_sha256=hashlib.sha256(body).hexdigest())
        html = body.decode("utf-8", errors="replace")
        for match in re.finditer(r"(?:var\s+)?ytInitialPlayerResponse\s*=\s*", html):
            try:
                player, _ = json.JSONDecoder().raw_decode(html[match.end():])
            except ValueError:
                continue
            if not isinstance(player, dict) or not isinstance(player.get("playabilityStatus"), dict):
                continue
            details = player.get("videoDetails", {})
            if not isinstance(details, dict):
                evidence["error"] = "Player response has malformed videoDetails"
                continue
            if details.get("videoId", result["id"]) != result["id"]:
                evidence["error"] = "Player response has a different video ID"
                continue
            status = player["playabilityStatus"]
            evidence["playability_status"] = status
            result["status"] = classify_player_status(status)
            break
        else:
            evidence["error"] = "No parseable ytInitialPlayerResponse playabilityStatus found"
    except urllib.error.HTTPError as error:
        evidence.update(http_status=error.code, error=str(error),
                        response_body=error.read(65536).decode("utf-8", errors="replace"))
    except (OSError, ValueError) as error:
        evidence.update(http_status=None, error=f"{type(error).__name__}: {error}")
    result["checked_at_utc"] = evidence["checked_at_utc"]
    return result


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def video_memberships(dataset_dir):
    """Read all five annotation parts, combining duplicate IDs across rows."""
    memberships = defaultdict(set)
    for index in range(1, 6):
        path = dataset_dir / "source" / f"GAVD_Clinical_Annotations_{index}.csv"
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                video_id = row["id"]
                if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
                    raise ValueError(f"Unexpected YouTube ID in {path}: {video_id!r}")
                memberships[video_id].add(row["gait_pat"])
    return memberships


def check_video(video_id, timeout):
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    endpoint = "https://www.youtube.com/oembed?" + urllib.parse.urlencode({"url": watch_url, "format": "json"})
    attempts = []
    result = {"id": video_id, "url": watch_url, "endpoint": endpoint, "method": METHOD,
              "status": UNKNOWN, "attempts": attempts}
    for number in range(1, 4):
        attempt = {"attempt": number, "checked_at_utc": utc_now()}
        retry = False
        try:
            request = urllib.request.Request(endpoint, headers={
                "User-Agent": "Mozilla/5.0 (compatible; GAVD-availability-audit/1.0)",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read(65537)
                attempt.update(http_status=response.status, final_url=response.url,
                               content_type=response.headers.get("Content-Type"),
                               response_body=body[:65536].decode("utf-8", errors="replace"),
                               body_truncated=len(body) > 65536)
            metadata = json.loads(body)
            if (attempt["http_status"] == 200 and isinstance(metadata, dict)
                    and metadata.get("provider_name") == "YouTube" and metadata.get("type") == "video"
                    and metadata.get("title") and not metadata.get("error")):
                result.update(status=SUCCESS, metadata=metadata)
            else:
                attempt["error"] = "Response is not successful YouTube video metadata"
        except urllib.error.HTTPError as error:
            attempt.update(http_status=error.code, error=str(error),
                           response_body=error.read(65536).decode("utf-8", errors="replace"))
            retry = error.code == 429 or error.code >= 500
        except (OSError, ValueError) as error:
            attempt.update(http_status=None, error=f"{type(error).__name__}: {error}")
            retry = True
        attempts.append(attempt)
        if not retry or number == 3:
            break
        time.sleep(number)
    result["checked_at_utc"] = attempts[-1]["checked_at_utc"]
    if result["status"] == UNKNOWN:
        check_watch_page(result, timeout)
    return result


def reusable_results(output, cache_hours):
    if not output.exists() or cache_hours <= 0:
        return {}
    try:
        old = json.loads(output.read_text(encoding="utf-8"))
        cutoff = datetime.now(timezone.utc) - timedelta(hours=cache_hours)
        if old.get("method", {}).get("name") != METHOD:
            return {}
        return {item["id"]: item for item in old["videos"]
                if datetime.fromisoformat(item["checked_at_utc"]) >= cutoff
                and item.get("method") == METHOD and item.get("attempts")
                and item.get("status") in STATUSES
                and (item["status"] == SUCCESS or item.get("watch_page_evidence", {}).get("method") == WATCH_METHOD)}
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def audit(dataset_dir, output, workers=8, timeout=20, cache_hours=24):
    started = utc_now()
    memberships = video_memberships(dataset_dir)
    cached = reusable_results(output, cache_hours)
    results = {video_id: cached[video_id] for video_id in memberships if video_id in cached}
    pending = sorted(set(memberships) - set(results))
    print(f"Auditing {len(memberships)} unique IDs: {len(results)} cached, {len(pending)} network checks", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(check_video, video_id, timeout): video_id for video_id in pending}
        for future in as_completed(futures):
            result = future.result()
            results[result["id"]] = result
            if len(results) % 25 == 0 or len(results) == len(memberships):
                print(f"Checked {len(results)}/{len(memberships)}: {dict(Counter(x['status'] for x in results.values()))}", flush=True)
    videos = []
    for video_id in sorted(memberships):
        result = dict(results[video_id])
        result["gait_patterns"] = sorted(memberships[video_id])
        videos.append(result)
    patterns = sorted(set.union(*memberships.values()))
    summary = []
    for pattern in patterns:
        counts = Counter(item["status"] for item in videos if pattern in item["gait_patterns"])
        summary.append({"gait_pat": pattern, "unique_video_ids": sum(counts.values()),
                        "public_availability_evidence": counts[SUCCESS] + counts[PLAYER_OK],
                        **{status: counts[status] for status in STATUSES}})
    manifest = json.loads((dataset_dir / "source" / "manifest.json").read_text(encoding="utf-8"))
    payload = {
        "audit_started_at_utc": started, "audit_completed_at_utc": utc_now(),
        "source_commit": manifest["commit"],
        "method": {"name": METHOD, "endpoint": "https://www.youtube.com/oembed",
                   "success_definition": "HTTP 200 JSON with provider_name=YouTube, type=video and a nonempty title",
                   "watch_page_fallback": "For oEmbed failures only, parse ytInitialPlayerResponse JSON from the direct YouTube watch page. OK => player_status_ok; explicit private/removed/deleted/copyright reasons => unavailable_publicly; login/age/region/bot restrictions => restricted; generic errors remain inconclusive.",
                   "limitations": [
                       "Public metadata availability is an existence proxy; playback, regional access, age restrictions, and downloadability are untested.",
                       "HTTP 401, 403, 404, other errors, malformed responses and network failures alone are inconclusive, not proof of deletion or invalidity.",
                       "public_availability_evidence combines public_metadata_available and player_status_ok; neither downloads or plays the video. Unavailable_publicly includes private videos and does not necessarily mean deleted.",
                       "Checks are time- and location-dependent; per-video timestamps are authoritative when cached results are reused.",
                       "Each unique ID is checked once per audit, with up to three attempts for transient errors; no videos are downloaded.",
                       "A video can have several gait patterns, so per-pattern unique counts must not be summed for a global total.",
                   ]},
        "summary": {"unique_video_ids": len(videos),
                    "public_availability_evidence": sum(x["status"] in (SUCCESS, PLAYER_OK) for x in videos),
                    **{status: sum(x["status"] == status for x in videos) for status in STATUSES}, "playback_confirmed": 0,
                    "cached_results_used": len(memberships) - len(pending)},
        "by_gait_pattern": summary, "videos": videos,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent, prefix=".gavd-availability-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(json.dumps(payload["summary"], indent=2), flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, help="Default: video_availability.json in dataset directory")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent requests, 1–16 (default: 8)")
    parser.add_argument("--timeout", type=float, default=20, help="Per-request timeout seconds (default: 20)")
    parser.add_argument("--cache-hours", type=float, default=24, help="Reuse younger results; zero forces fresh checks")
    args = parser.parse_args()
    if not 1 <= args.workers <= 16 or args.timeout <= 0 or args.cache_hours < 0:
        parser.error("Workers must be 1–16, timeout positive, and cache-hours nonnegative")
    try:
        audit(args.dataset_dir, args.output or args.dataset_dir / "video_availability.json",
              args.workers, args.timeout, args.cache_hours)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Availability audit failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
