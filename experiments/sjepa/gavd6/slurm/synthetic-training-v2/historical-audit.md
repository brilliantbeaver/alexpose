# Repair a historical-file audit failure

Use this page when Step 2 of the [HAIC guide](README.md) reports missing historical files or SHA256 mismatches. This check runs before `reconstruct_pilot`; a preservation exception means the arithmetic reconstruction has not run yet. It does not diagnose CUDA, model training, or the AMASS assets.

The original error, `Pre-existing artifacts changed`, combined two different cases: a file did not exist, or its contents did not match the recorded SHA256. The updated checker separates missing, mismatched, and unreadable files. Every recorded file must still match; the repair does not change the manifest or skip evidence.

**1. Identify what is missing or different on HAIC.**

With the updated source and scripts copied to HAIC, run:

```bash
"$STV2_PYTHON" scripts/research_directions/synthetic_training_v2/check_history.py --repo "$STV2_ROOT" --json
```

This is read-only. It exits with status 1 while any file is missing, mismatched, or unreadable. The JSON includes full expected and actual hashes for mismatches.

If you have not updated the code yet, this standard-library check works with the existing manifest. On **HAIC**:

```bash
python3 - <<'PY'
import hashlib, json, os
from collections import Counter
from pathlib import Path
root = Path(os.environ['STV2_ROOT'])
manifest = root / 'docs/studies/synthetic-training-v2/preservation-manifest.json'
counts = Counter()
for name, expected in json.loads(manifest.read_text())['files'].items():
    path = root / name
    try:
        if not path.is_file():
            status = 'MISSING_OR_NOT_FILE'
        else:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            status = 'MATCH' if actual == expected else 'HASH_MISMATCH'
    except OSError:
        status = 'UNREADABLE'
    counts[status] += 1
    if status != 'MATCH':
        print(status, name)
print(dict(counts))
PY
```

Interpret the result:

| Result | Meaning and action |
| --- | --- |
| Missing historical files | Transfer the retained snapshot. `notebook_runs/` is ignored by Git, so pulling code does not fetch these files. |
| Hash mismatches | An existing file differs from the recorded snapshot. Back it up before replacing it with a verified copy. `rsync --ignore-existing` leaves this case unresolved. |
| Unreadable files | Correct the file access problem before concluding the contents changed. |
| Only `.DS_Store` differs | The original manifest also records this Finder metadata file. It is not scientific evidence, but the current preservation check still requires its recorded bytes. Restore the verified snapshot; do not rewrite the manifest to hide differences in other files. |

A list containing all 24 `notebook_runs/synthetic-training/` entries is consistent with a missing or incomplete historical transfer. It does not by itself prove that all 24 files were edited or corrupted.

**2. Verify the Mac copies and make an exact transfer list.**

On your **Mac**, from the local gavd6 checkout:

```bash
cd /Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6
export STV2_HISTORY_LIST=$(mktemp /tmp/stv2-history-files.XXXXXX)

python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
root = Path.cwd()
manifest = root / 'docs/studies/synthetic-training-v2/preservation-manifest.json'
saved = json.loads(manifest.read_text())['files']
names = [name for name in saved if name.startswith('notebook_runs/synthetic-training/')]
if not names:
    raise SystemExit('No historical files listed; check the selected checkout and manifest.')
for name in names:
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != saved[name]:
        raise SystemExit(f'Local source does not match the recorded snapshot: {name}')
Path(os.environ['STV2_HISTORY_LIST']).write_text('\n'.join(names) + '\n')
print(f'Verified {len(names)} files. Transfer list: {os.environ["STV2_HISTORY_LIST"]}')
PY
```

**Continue only when every source file verifies.** This generates the list from the existing hashes; it does not regenerate hashes. If verification fails locally, locate the matching retained snapshot before transferring anything. The current list contains 24 files, about 7.9 MB in total.

**3. Preview a transfer that preserves differing HAIC copies.**

Continue in the same **Mac** shell. These commands may ask for your normal HAIC interactive authentication:

```bash
haic_root="tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6"
history_stamp=$(date -u +%Y%m%dT%H%M%SZ)
haic_backup="/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training-v2/history-backup-$history_stamp"

ssh tedmui@haic.stanford.edu \
  'mkdir -p /hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/notebook_runs/synthetic-training'

rsync -avR --checksum --dry-run --itemize-changes \
  --backup --backup-dir="$haic_backup" \
  --files-from="$STV2_HISTORY_LIST" ./ "$haic_root/"
```

Inspect the listed changes. This transfer is restricted to the 24 manifest-listed historical files. `--checksum` detects differing contents even when size and modification time agree. `--backup` retains replaced destination files under the unique HAIC backup directory. Unlisted files remain in place; there is no `--delete`.

**4. Transfer, then rerun the audit.**

After reviewing the preview, on your **Mac**:

```bash
rsync -avR --checksum --itemize-changes \
  --backup --backup-dir="$haic_backup" \
  --files-from="$STV2_HISTORY_LIST" ./ "$haic_root/"
printf 'Replaced HAIC copies, if any, were retained under: %s\n' "$haic_backup"
```

On **HAIC**, rerun the original check. This also works before installing the improved diagnostic:

```bash
"$STV2_PYTHON" - <<'PY'
import os
from pathlib import Path
from gavd6_sjepa.research_directions.synthetic_training_v2.audit import reconstruct_pilot
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import verify_preservation
root = Path(os.environ['STV2_ROOT'])
print(verify_preservation(root))
report = reconstruct_pilot(root)
print(report['counts'])
print('HISTORICAL_AUDIT_PASSED')
PY
```

With the updated diagnostic, the equivalent command is:

```bash
"$STV2_PYTHON" scripts/research_directions/synthetic_training_v2/check_history.py --reconstruct
```

Require preservation to pass for all 72 files, followed by `HISTORICAL_AUDIT_PASSED`. The retained pilot reconstructs 2,304 source rows, 6,912 selector decisions, and 144 selector configurations. Repeat `submit.sh check` in Step 2 of the main guide, then proceed to preparation. No GPU allocation is needed for this repair.

If files outside `notebook_runs/synthetic-training/` fail, this transfer intentionally does not replace them. Review those paths separately against the same manifest. If a historical mismatch remains after the transfer, compare the Mac and HAIC manifest versions and the full expected/actual hashes before attempting another repair.
