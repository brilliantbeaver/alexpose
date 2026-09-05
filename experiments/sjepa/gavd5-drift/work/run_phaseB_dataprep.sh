#!/bin/zsh
cd /Users/pmui/dev/alexpose/experiments/sjepa/gavd5-drift
PY=.venv/bin/python
LOG=work/artifacts/phaseB_dataprep.log
echo "=== Phase B data-prep started $(date) ===" > $LOG
run_nb () {
  echo "" >> $LOG; echo ">>> executing $1 $(date)" >> $LOG
  $PY -m nbconvert --to notebook --execute "$1" \
      --output-dir work/nb_executed --ExecutePreprocessor.timeout=-1 \
      >> $LOG 2>&1
  rc=$?
  echo ">>> $1 exit code $rc $(date)" >> $LOG
  return $rc
}
run_nb 01_gavd_manifest_and_youtube.ipynb || { echo "STOP: download stage failed" >> $LOG; exit 1; }
cached=$(find work/youtube -name '*.mp4' | wc -l | tr -d ' ')
echo ">>> cached videos after download: $cached / 96" >> $LOG
run_nb 02_extract_and_watch_skeletons.ipynb || { echo "STOP: pose extraction failed" >> $LOG; exit 1; }
poses=$(find work/artifacts/real/poses -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo ">>> extracted pose sequences: $poses / 645" >> $LOG
echo "=== Phase B data-prep finished $(date) ===" >> $LOG
