#!/bin/zsh
# Waits for the detached inductive training (PID passed as $1) to exit and all 15
# feature caches to be present, then runs inductive_probe.py to emit the 3 result JSONs.
# Harness-tracked so the main session is notified on completion.
set -u
cd /Users/pmui/dev/alexpose/experiments/sjepa/gavd5-drift
PID="${1:?usage: probe_watch.sh <training_pid>}"
FEAT=work/artifacts/real/inductive/features
LOG=work/logs/inductive_probe_watch.log
: > "$LOG"
echo "[watch $(date '+%F %T')] waiting for training PID $PID + 15 feature caches" >> "$LOG"
while kill -0 "$PID" 2>/dev/null; do sleep 180; done
echo "[watch $(date '+%F %T')] training PID $PID has exited" >> "$LOG"
sleep 5
NFEAT=$(ls -1 "$FEAT"/*.npz 2>/dev/null | wc -l | tr -d ' ')
NENC=$(ls -1 work/artifacts/real/inductive/encoders/*.pt 2>/dev/null | wc -l | tr -d ' ')
echo "[watch $(date '+%F %T')] encoders=$NENC/15  features=$NFEAT/15" >> "$LOG"
if [ "$NFEAT" != "15" ]; then
  echo "[watch $(date '+%F %T')] INCOMPLETE -> not running probe. Tail of train log:" >> "$LOG"
  tail -n 40 work/logs/inductive_train.log >> "$LOG" 2>&1
  exit 2
fi
echo "[watch $(date '+%F %T')] all 15 caches present -> running inductive_probe.py" >> "$LOG"
.venv/bin/python work/experiments/inductive_probe.py >> "$LOG" 2>&1
RC=$?
echo "[watch $(date '+%F %T')] inductive_probe.py exit code = $RC" >> "$LOG"
for j in idea5_inductive_result.json idea9_equivariant_readout_inductive_result.json idea9_equivariant_encoder_inductive_result.json; do
  if [ -f "work/artifacts/real/$j" ]; then echo "[watch] wrote $j" >> "$LOG"; else echo "[watch] MISSING $j" >> "$LOG"; fi
done
exit $RC
