#!/bin/zsh
# Phase B training auto-chain: train the 5-stage SJEPA curriculum on the full
# 642/94 corpus, run all downstream diagnostic notebooks, regenerate figures.
cd /Users/pmui/dev/alexpose/experiments/sjepa/gavd5-drift
PY=.venv/bin/python
LOG=work/artifacts/phaseB_training.log
OUT=work/nb_executed
mkdir -p "$OUT"
echo "=== Phase B training started $(date) ===" > $LOG

poses=$(find work/artifacts/real/poses -name '*.npz' 2>/dev/null | wc -l | tr -d ' ')
echo ">>> pose sequences available: $poses / 642" >> $LOG
if [ "$poses" -lt 642 ]; then
  echo "STOP: extraction incomplete ($poses/642); not launching training." >> $LOG
  exit 2
fi

run_nb () {  # $1 notebook, $2 = "gate" (stop chain on failure) or "soft"
  echo "" >> $LOG; echo ">>> executing $1 $(date)" >> $LOG
  $PY -m nbconvert --to notebook --execute "$1" \
      --output-dir "$OUT" --ExecutePreprocessor.timeout=-1 >> $LOG 2>&1
  rc=$?
  echo ">>> $1 exit code $rc $(date)" >> $LOG
  if [ "$rc" -ne 0 ] && [ "$2" = "gate" ]; then
    echo "STOP: gating notebook $1 failed (rc=$rc)." >> $LOG
    exit 1
  fi
  return $rc
}

# --- Training (gate: everything downstream depends on the checkpoint) ---
run_nb 04_pretrain_sjepa_on_normal.ipynb gate
echo ">>> training checkpoints:" >> $LOG
ls -1 work/artifacts/real/*.pt 2>/dev/null >> $LOG

# --- Downstream diagnostics (soft: run all, record failures, keep going) ---
for nb in 05_inspect_latent_motion.ipynb \
          nb_05a_signed_laterality_probe.ipynb \
          nb_05b_reflection_reach_and_futures.ipynb \
          06_capstone_health_condition_classifiers.ipynb \
          07_temporal_readout_diagnostic.ipynb \
          08_normal_anchor_drift_and_consolidation.ipynb \
          09_predictive_surprise_world_model.ipynb; do
  run_nb "$nb" soft
done

# --- Figure regeneration (soft) ---
echo "" >> $LOG; echo ">>> regenerating figures $(date)" >> $LOG
for script in docs/make_figures.py docs/make_brainbody_figures.py \
              docs/make_evolution_figures.py docs/make_downstream_probe_figure.py \
              docs/make_loss_flow.py slides/make_current_result_figures.py; do
  if [ -f "$script" ]; then
    echo ">>> $script" >> $LOG
    $PY "$script" >> $LOG 2>&1
    echo ">>> $script exit code $? " >> $LOG
  fi
done

echo "" >> $LOG
echo "=== Phase B training finished $(date) ===" >> $LOG
echo ">>> SUMMARY (exit codes):" >> $LOG
grep 'exit code' $LOG >> $LOG
