#!/bin/zsh
cd /Users/pmui/dev/alexpose/experiments/sjepa/gavd5-drift
PY=.venv/bin/python; LOG=work/artifacts/verify_0708.log; OUT=work/nb_executed
echo "=== verify 07/08 started $(date) ===" > $LOG
for nb in 07_temporal_readout_diagnostic.ipynb 08_normal_anchor_drift_and_consolidation.ipynb; do
  echo ">>> $nb $(date)" >> $LOG
  $PY -m nbconvert --to notebook --execute "$nb" --output-dir "$OUT" --ExecutePreprocessor.timeout=-1 >> $LOG 2>&1
  echo ">>> $nb exit code $? $(date)" >> $LOG
done
echo "=== done $(date) ===" >> $LOG
