#!/bin/zsh
cd /Users/pmui/dev/alexpose/experiments/sjepa/gavd5-drift
PY=.venv/bin/python
LOG=work/artifacts/phaseB_downstream.log
OUT=work/nb_executed
echo "=== Phase B downstream re-run started $(date) ===" > $LOG
run_nb () {
  echo "" >> $LOG; echo ">>> executing $1 $(date)" >> $LOG
  $PY -m nbconvert --to notebook --execute "$1" --output-dir "$OUT" --ExecutePreprocessor.timeout=-1 >> $LOG 2>&1
  echo ">>> $1 exit code $? $(date)" >> $LOG
}
for nb in 05_inspect_latent_motion.ipynb nb_05a_signed_laterality_probe.ipynb \
          nb_05b_reflection_reach_and_futures.ipynb 06_capstone_health_condition_classifiers.ipynb \
          07_temporal_readout_diagnostic.ipynb 08_normal_anchor_drift_and_consolidation.ipynb \
          09_predictive_surprise_world_model.ipynb; do
  run_nb "$nb"
done
echo "" >> $LOG; echo ">>> figures $(date)" >> $LOG
for s in docs/make_figures.py docs/make_evolution_figures.py docs/make_brainbody_figures.py \
         docs/make_downstream_probe_figure.py docs/make_loss_flow.py slides/make_current_result_figures.py; do
  $PY "$s" >> $LOG 2>&1; echo ">>> $s exit code $?" >> $LOG
done
echo "=== downstream re-run finished $(date) ===" >> $LOG
echo ">>> SUMMARY:" >> $LOG; grep 'exit code' $LOG >> $LOG
