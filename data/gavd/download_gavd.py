#!/usr/bin/env python3
"""Restore the five pinned upstream CSVs using source/manifest.json.

Only CSVs are downloaded. Keep the checked-in manifest with the source cache;
it also records the upstream data inventory and retrieval date. Valid files are
reused. Each replacement is downloaded and hash-verified before atomic rename.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request


COMMIT = "a87859c881603443f200bcd640663d2c3d7a8136"
REPOSITORY = "https://github.com/Rahmyyy/GAVD"
BLOBS = (
    "b4c31789853e758e213438dc68d354075c98480b",
    "93294863acc9c83185d935d102139439f8ab3dea",
    "fb260580c2c086962c95c329ec42d4f1e410378c",
    "f196b723cb06f57cec8edfbf4fb8fe9e5c36ade6",
    "0f4d5e390fddc24d1e2895018ed48f66e8b51cff",
)


def load_manifest(source_dir):
    """Validate paths, URLs, and Git identities against this pinned snapshot."""
    with (source_dir / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("repository") != REPOSITORY or manifest.get("commit") != COMMIT:
        raise ValueError("Manifest repository/revision does not match the pinned snapshot")
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != 5:
        raise ValueError("Manifest must contain exactly five source CSVs")
    expected_names = {f"GAVD_Clinical_Annotations_{i}.csv" for i in range(1, 6)}
    if {entry.get("name") for entry in entries} != expected_names:
        raise ValueError("Manifest contains missing, duplicate, or unexpected CSV filenames")
    for entry in entries:
        name = entry["name"]
        index = int(name.removesuffix(".csv").rsplit("_", 1)[1]) - 1
        expected_url = f"https://raw.githubusercontent.com/Rahmyyy/GAVD/{COMMIT}/data/{name}"
        if entry.get("url") != expected_url or entry.get("git_blob_sha") != BLOBS[index]:
            raise ValueError(f"Untrusted URL or Git identity for {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
            raise ValueError(f"Invalid SHA-256 for {name}")
        if type(entry.get("bytes")) is not int or entry["bytes"] <= 0:
            raise ValueError(f"Invalid byte count for {name}")
        if entry.get("rows") != (91624 if index == 0 else 91623):
            raise ValueError(f"Invalid row count for {name}")
    return sorted(entries, key=lambda entry: entry["name"])


def matches(path, entry):
    """Check exact bytes with SHA-256 and Git's length-prefixed blob SHA-1."""
    if not path.is_file() or path.stat().st_size != entry["bytes"]:
        return False
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f"blob {entry['bytes']}\0".encode("ascii"))
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(chunk)
            blob.update(chunk)
    return sha256.hexdigest() == entry["sha256"] and blob.hexdigest() == entry["git_blob_sha"]


def restore(source_dir):
    entries = load_manifest(source_dir)
    for entry in entries:
        target = source_dir / entry["name"]
        if target.is_symlink():
            raise ValueError(f"Refusing symlink source path: {target}")
        if matches(target, entry):
            print(f"Verified cached {entry['name']}")
            continue
        for attempt in range(1, 4):
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(prefix=".gavd-download-", dir=source_dir, delete=False) as stream:
                    temporary = Path(stream.name)
                    request = urllib.request.Request(entry["url"], headers={"User-Agent": "alexpose-gavd-sync/1"})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        size = 0
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            size += len(chunk)
                            if size > entry["bytes"]:
                                raise ValueError("Response exceeds the pinned source byte count")
                            stream.write(chunk)
                if not matches(temporary, entry):
                    raise ValueError("Downloaded CSV does not match the pinned hashes/byte count")
                os.replace(temporary, target)
                print(f"Downloaded and verified {entry['name']}")
                break
            except (OSError, ValueError, urllib.error.URLError) as error:
                if attempt == 3:
                    raise RuntimeError(f"Could not restore {entry['name']} after three attempts: {error}") from error
                print(f"Retrying {entry['name']} ({attempt}/3): {error}", file=sys.stderr)
                time.sleep(attempt)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent / "source",
                        help="CSV cache containing manifest.json (default: source beside this script)")
    args = parser.parse_args()
    try:
        restore(args.source_dir.resolve())
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, AttributeError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
