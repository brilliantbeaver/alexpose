# Reproducing the current downstream probes

This page covers the **non-augmented-normal, GAVD-only** checkpoint. It does not cover any file whose name contains `_augmented`.

## Required checkpoint

```text
file:        work/artifacts/real/sjepa_curriculum_final.pt
fingerprint: 7d13841aceac9eda843d43ca8434193e294d2fa10a48b6c6d21f6413a6e457e2
SHA-256:     64008d77689cefa4beb51a0dcf5ed6cae743454134c163e9087f66510af4e7ad
```

The checkpoint contains 626 coverage-valid sequence IDs and records that the augmented-normal cohort was not used.

## Rerun

From the `experiments/sjepa/gavd5-drift` experiment root, with the project environment active:

```sh
SJEPA_INCLUDE_AUGMENTED_NORMAL=0 \
  jupyter nbconvert \
  --to notebook \
  --execute neurips-brain-body/06_capstone_health_condition_classifiers.ipynb \
  --output-dir work/nb_executed \
  --ExecutePreprocessor.timeout=-1
```

Then verify the final checkpoint hash and regenerate the downstream figure:

```sh
shasum -a 256 work/artifacts/real/sjepa_curriculum_final.pt
MPLCONFIGDIR=cache/matplotlib \
  .venv/bin/python docs/make_downstream_probe_figure.py
```

## Verified current results

The stored split label `all_96_stratified_video_confounded` is a legacy name. In the current artifact it contains **all 626 coverage-valid rows**, split 438 for classifier fitting and 188 for classifier testing.

|Probe|Accuracy|Balanced accuracy|Macro-F1|Scope|
|---|---:|---:|---:|---|
|Five class, all 626 rows|0.9202|0.9001|0.8985|Sequence split; video-confounded; encoder-transductive|
|Five class, exact legacy 47/21 rows|0.8571|0.8800|0.8607|Sequence split; all 9 test videos overlap training|
|Missingness only, all 626 rows|0.4415|0.4269|0.3547|Same sequence split; no pose coordinates|
|Missingness only, exact 47/21 rows|0.3333|0.3638|0.3361|Same exact split|

One-versus-normal results on the all-row sequence split are 1.000 macro-F1 for Parkinson's, 1.000 for stroke, 0.985 for myopathic, and 0.850 for cerebral palsy. These are also transductive and video-confounded.

## Exposure audit

|Split|Train rows|Test rows|Train videos|Test videos|Shared videos|Test rows seen by encoder|
|---|---:|---:|---:|---:|---:|---:|
|All-row legacy-named split|438|188|88|69|64|188|
|Exact 47/21 split|47|21|12|9|9|21|

In the all-row split, 181 of 188 test sequences come from the 64 video IDs shared with classifier training. The encoder was trained with condition labels after Stage 0. Splitting only the Random Forest cannot remove that representation exposure.

## Stale grouped artifact

`lane_c_video_disjoint_metrics.csv` currently names `sjepa_curriculum_final_augmented.pt` and 159 rows. It is not a result for the current checkpoint. Do not use it in figures or tables.

A current classifier-video-grouped probe can be rerun after notebook 06 is updated to write a new lineage-checked artifact. Even then, it would remain encoder-transductive. A generalization estimate requires retraining the entire encoder inside each outer source-video fold.

The saved outputs inside the source `neurips-brain-body/06_capstone_health_condition_classifiers.ipynb` still display the retired augmented checkpoint and 159-row Lane C result. Its selection code is flag-guarded, but the embedded outputs are stale. Until the source notebook is cleared and rerun, use the current executed notebook under `work/nb_executed/` together with the hash-checked artifacts as the evidence record. The 82-feature handcrafted baseline in `exp5_comparison.csv` is also hard-coded by notebook 06 rather than recomputed, so do not claim a verified improvement over that baseline yet.

## Minimum audit after rerunning

1. Confirm `include_augmented_normal` is `false` in `classifier_contract.json`.
2. Confirm the embedding table has 626 rows and 384 latent columns.
3. Confirm every embedding has fingerprint `7d13841a...`.
4. Confirm the checkpoint file SHA-256 is `64008d77...`.
5. Run `make_brainbody_figures.py` and `make_downstream_probe_figure.py`; their exact artifact hashes, checkpoint hash, cohort checks, and split checks must all pass.
6. Treat changed numbers as a new run; do not copy old captions forward.
