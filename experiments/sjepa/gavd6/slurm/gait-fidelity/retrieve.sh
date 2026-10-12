#!/usr/bin/env bash
# Download a run for inspection without changing its remote contents.
set -euo pipefail
[[ $# == 2 ]] || { echo 'Usage: bash slurm/gait-fidelity/retrieve.sh HOST:/absolute/WORK LOCAL_DIRECTORY' >&2; exit 2; }
remote="$1"; local_directory="$2"
[[ "$remote" =~ ^([[:alnum:]_][[:alnum:]_.-]*@)?[[:alnum:]][[:alnum:].-]*:/[[:alnum:]_./-]+$ ]] || {
  echo 'Use HOST:/absolute/WORK with plain path characters.' >&2; exit 2;
}
case "/${remote#*:}/" in *'/../'*|*'/./'*) echo 'Use a normalized remote work path.' >&2; exit 2 ;; esac
mkdir -p "$local_directory"
# Keep predictions, manifests, reports and viewer assets. Large optimizer/model
# snapshots remain on HAIC; omit these exclusions for a full reproducibility copy.
rsync -a --partial --progress --exclude='*.pt' --exclude='*.pth' \
  "${remote%/}/" "${local_directory%/}/"
printf '\nDownloaded analysis bundle to %s\nModel checkpoints remain on HAIC.\n' "$local_directory"
