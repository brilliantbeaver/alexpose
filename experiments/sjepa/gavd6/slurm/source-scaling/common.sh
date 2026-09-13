#!/usr/bin/env bash
# README-compatible environment; the old jobs receive their internal aliases.
set -euo pipefail
: "${GAVD6_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
export GAVD6_ROOT="$(cd "$GAVD6_ROOT" && pwd -P)"
export FI_PYTHON="${FI_PYTHON:-${FI_ENVIRONMENT:-$GAVD6_ROOT/.venv}/bin/python}"
export PYTHONPATH="$GAVD6_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
# ID selection is optional; explicit existing root invocations remain supported.
if [[ -n "${FI_RUN_ID:-}" ]]; then
  select_args=(select "$FI_RUN_ID" --new-study future-innovation)
  [[ -z "${FI_RUN_ROOT:-}" ]] || select_args+=(--root "$FI_RUN_ROOT")
  export FI_RUN_ROOT
  FI_RUN_ROOT="$("$FI_PYTHON" -m gavd6_sjepa.shared_infrastructure.result_catalog \
    --project "$GAVD6_ROOT" "${select_args[@]}")"
fi
if [[ -n "${FI_PARENT_ID:-}" ]]; then
  select_args=(select "$FI_PARENT_ID")
  [[ -z "${FI_PARENT_ROOT:-}" ]] || select_args+=(--root "$FI_PARENT_ROOT")
  export FI_PARENT_ROOT
  FI_PARENT_ROOT="$("$FI_PYTHON" -m gavd6_sjepa.shared_infrastructure.result_catalog \
    --project "$GAVD6_ROOT" "${select_args[@]}")"
fi
: "${FI_RUN_ROOT:?Set FI_RUN_ID or FI_RUN_ROOT for the source-learning-curve study}"
for name in FI_RUN_ROOT FI_PARENT_ROOT; do
  value="${!name:-}"
  [[ -n "$value" ]] || continue
  [[ "$value" == /* ]] || value="$GAVD6_ROOT/$value"
  export "$name=$value"
done
export GAVD_FULL_ROOT="${GAVD_FULL_ROOT:-$GAVD6_ROOT/data/gavd_full}"
export FI_ANNOTATION_ROOT="${FI_ANNOTATION_ROOT:-$GAVD_FULL_ROOT/annotations/GAVD/data}"
export FI_VIDEO_ROOT="${FI_VIDEO_ROOT:-$GAVD_FULL_ROOT/youtube/all}"
export FI_SCALING_ROOT="$FI_RUN_ROOT"
export FI_INSPECTION_ROOT="${FI_INSPECTION_ROOT:-$GAVD6_ROOT/outputs/inspections/${FI_RUN_ID:-${FI_RUN_ROOT##*/}}}"
[[ "$FI_INSPECTION_ROOT" == /* ]] || export FI_INSPECTION_ROOT="$GAVD6_ROOT/$FI_INSPECTION_ROOT"
export PYTHONUNBUFFERED=1
cd "$GAVD6_ROOT"
printf 'Selected study %s: %s\n' "${FI_RUN_ID:-${FI_RUN_ROOT##*/}}" "$FI_RUN_ROOT"
[[ -z "${FI_PARENT_ROOT:-}" ]] || printf 'Parent: %s\n' "$FI_PARENT_ROOT"
