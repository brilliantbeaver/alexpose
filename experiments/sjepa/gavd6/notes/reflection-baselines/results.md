# Current evidence: fixed-reflection baselines

The available run, `outputs/repaired-jepa-seed7-v2`, is evidence that the AMASS
training scaffold runs. Its capacity table places all four current variants at
about 822k trainable parameters (maximum spread 0.043%).

The newest `standard_sjepa` artifact completed 236,170 updates in 7,326 seconds
with a selected checkpoint at epoch 95. Its validation features have nonzero
variance and effective rank, so it does not show the obvious collapsed-feature
failure mode. The code's smoke suite covers the standard control, capacity
matching, and commutation checks.

This is **not** an architecture ranking:

- the aggregate `summary.csv` currently contains only the newest standard-S-JEPA
  run;
- the current output contains one full seed per stored arm rather than a paired
  multi-seed study;
- `evaluate_test` is false in the saved full-run configuration; and
- no GAVD-v2, force, balance, real-gauge, or calibration endpoint is present.

Record this evidence as pretraining health and baseline readiness only.
