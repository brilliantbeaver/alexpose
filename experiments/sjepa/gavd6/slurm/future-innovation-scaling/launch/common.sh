#!/usr/bin/env bash
# README-compatible environment; the old jobs receive their internal aliases.
set -euo pipefail
: "${GAVD6_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd -P)}"
: "${FI_RUN_ROOT:?Set FI_RUN_ROOT to a new source-learning-curve output directory}"
export GAVD6_ROOT="$(cd "$GAVD6_ROOT" && pwd -P)"
for name in FI_RUN_ROOT FI_PARENT_ROOT; do
  value="${!name:-}"
  [[ -n "$value" ]] || continue
  [[ "$value" == /* ]] || value="$GAVD6_ROOT/$value"
  export "$name=$value"
done
export GAVD_FULL_ROOT="${GAVD_FULL_ROOT:-$GAVD6_ROOT/data/gavd_full}"
export FI_ANNOTATION_ROOT="${FI_ANNOTATION_ROOT:-$GAVD_FULL_ROOT/annotations/GAVD/data}"
export FI_VIDEO_ROOT="${FI_VIDEO_ROOT:-$GAVD_FULL_ROOT/youtube/all}"
export FI_PYTHON="${FI_PYTHON:-${FI_ENVIRONMENT:-$GAVD6_ROOT/.venv}/bin/python}"
export FI_SCALING_ROOT="$FI_RUN_ROOT"
export PYTHONUNBUFFERED=1
cd "$GAVD6_ROOT"
