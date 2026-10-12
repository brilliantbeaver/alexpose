**The v3 notebook completed its inventory successfully, but its recorded configuration is not ready to prepare synthetic training data.** All ten pose-model file checks pass. The three rendering inputs remain unset. This notebook establishes progress in asset configuration; it contains no model-loading, rendering, training, or accuracy result.

I inspected every cell and saved output in [00_question_and_assets.ipynb](00_question_and_assets.ipynb), compared v1/v2/v3, validated notebook structure and Python syntax, and traced the configuration, inventory, manifest readers, and immediate data-preparation dependencies. This review used the local checkout at commit `50d486fc8ee6423dee1e533fc9e49890512778db`. The notebook does not record its HAIC source commit, so implementation observations below refer to the current checkout. The three notebook code cells are identical to the current source notebook. I did not rerun the HAIC notebook or access its remote assets.

**The execution record is internally consistent.** There are eight cells: five Markdown cells and three code cells, executed in order with counts 1, 2, and 3. There are no saved error outputs. The notebook passes `nbformat.validate`; all three code cells parse.

| Recorded item | Value |
| --- | --- |
| Start | September 17, 2026, 18:42:03.589503 UTC |
| Finish | September 17, 2026, 18:42:51.437192 UTC |
| Total elapsed time | 47.85 seconds |
| Python | `/hai/scratch/tedmui/envs/synthetic-training/bin/python`, Python 3.11.16 |
| Run root | `/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training/pilot-01` |
| Printed settings | `context_kind=vjepa`, `device=cuda` |
| Inventory timing | 2.3 seconds |

The setup cell accounts for approximately 40.54 seconds; the student-display cell takes 0.07 seconds and the inventory cell 2.27 seconds. These timings measure setup and filesystem inspection, not GPU training or rendering throughput. Printing `cuda` and `vjepa` confirms configuration values only.

| Cells, counted from 1 | Function | Assessment |
| --- | --- | --- |
| 1–2 | State the question and stage order | Clearly distinguish response-informed teaching from extra-training gains. |
| 3 | Locate the checkout, import code, read configuration | Successful setup; the effective configuration is only partially displayed. |
| 4–5 | Explain the hypothesis and show student roles | Roles match the intended pilot; this does not load or update the students. |
| 6–7 | Explain and execute inventory | Reports missing dependencies without failing; large tables and paths are truncated. |
| 8 | Limit interpretation and point to notebook 01 | Scientific limits are appropriate; the copied navigation link is broken. |

**The immediate blocker is the rendering configuration.** The asset table contains 22 entries: 19 true and three false. Those counts should not be read as a readiness percentage because individual assets have different validation requirements.

| Asset group | Saved result | What the result establishes |
| --- | --- | --- |
| Five student configurations and five checkpoints | 10/10 true | All ten selected paths were regular files at execution time. |
| AMASS, body models, COCO, GAVD, reservation, context repository, context checkpoint, image checkpoint | 9/9 true | The nine configured paths existed. |
| `render_texture_dir` | Empty path; false | No texture directory was configured. |
| `render_background_dir` | Empty path; false | No background directory was configured. |
| `render_uv_path` | Empty path; false | No UV topology asset was configured. |

The three false entries mean **unset**, rather than merely a typo in a displayed nonempty path. They do not establish that suitable files are absent everywhere on HAIC.

[prepare_data](../../../src/gavd6_sjepa/research_directions/synthetic_training/workflow.py) calls `prepare_amass_library`, which constructs `TexturedBodyRenderer`. Its constructor rejects an empty UV path with `FileNotFoundError: A licensed compatible UV topology NPZ or OBJ is required`. I reproduced that constructor behavior locally without loading a model. Earlier configuration or data errors could also occur, but this configuration cannot complete rendering.

Set `ST_TEXTURE_DIR`, `ST_BACKGROUND_DIR`, and `ST_UV_PATH` to actual assets. The first two need usable PNG/JPG/JPEG images; the UV asset needs a supported OBJ/NPZ representation whose face indices exactly match the loaded SMPL-H mesh. The texture images must fit that UV layout. Creating empty directories does not satisfy those requirements. The [renderer](../../../src/gavd6_sjepa/research_directions/synthetic_training/rendering.py) checks topology during rendering, beyond what inventory checks.

**v3 resolves the student-path problem shown in the earlier attempts.** v1 and v2 each reported all ten student entries false and all three rendering entries false. v3 moves the student paths from locations under the gavd6 checkout to `/hai/scratch/tedmui/models/...`, and all ten become true. The rendering entries do not change. This supports a successful path correction; it does not establish when the model files were downloaded or whether they are compatible.

**Several limitations prevent treating inventory completion as permission to launch the full source chain.**

1. **Existence checks are deliberately shallow.** In [workflow.inventory](../../../src/gavd6_sjepa/research_directions/synthetic_training/workflow.py), generic assets use `Path.exists()` and student files use `Path.is_file()`. In isolated temporary fixtures, I confirmed that an empty texture directory, a directory supplied as the COCO annotation path, and a zero-byte student checkpoint all receive true flags. These are demonstrations of the checker, not claims about the actual HAIC files. Checkpoint deserialization, model/config agreement, COCO contents, image decoding, CUDA operations, and EGL remain untested by this notebook.

2. **The body-model row does not establish complete SMPL-H/DMPL availability.** `body_model_root=True` checks the root alone. Inventory does not report `dmpl_root` or required per-gender model files. [SMPLHBody](../../../src/gavd6_sjepa/research_directions/motion_preservation/body_geometry.py) loads those later and checks the expected body-model API and active DMPL components. Validate the actual files for the genders selected into the source library.

3. **`completed` is not an asset-readiness gate.** The executor marks successful cell execution as completed; inventory returns a table even when dependencies are missing. The [source launcher](../../../slurm/synthetic-training/submit.sh) chains stages with `afterok`, so a successful inventory job with missing assets can release notebook 01. Use the separate inventory and data phases for this setup, or add an explicit stage-specific readiness check before automating the full chain.

4. **The effective configuration is not preserved per attempt.** [RunConfig.from_env](../../../src/gavd6_sjepa/research_directions/synthetic_training/config.py) combines JSON and supported environment overrides. Inventory writes `config.json` only when it does not already exist. I reproduced a case where the displayed inventory uses updated settings while the existing saved JSON retains older settings. A subsequent process without the same `ST_CONFIG` and overrides can therefore revert. This notebook does not identify the selected configuration file or embed its full resolved settings, so it cannot establish whether that happened on HAIC. Preserve a separate effective-configuration snapshot alongside each attempt, including relevant environment choices and source revision; do not silently overwrite historical experiment configuration.

5. **The copied notebook loses important audit detail.** Both text and HTML tables truncate long paths, and AMASS/GAVD tables show only their first and last five rows. The inventory writes `asset_inventory.csv` in the remote run root, but that file is absent from this copied directory and is overwritten by later inventory runs. Full AMASS/GAVD availability tables are not persisted by this function. Save full per-attempt tables plus concise availability summaries. A hash column copied from an AMASS manifest is historical manifest metadata; this inventory does not recompute those hashes against the current files.

6. **All four local navigation links are broken.** Their targets still use six `../` components for the original output depth. In this directory they resolve under `/Users/theodoremui/dev/alexpose/`, outside the gavd6 checkout. The correct prefix here is `../../../`, for example `../../../notebooks/synthetic_training/01_prepare_source_data.ipynb`. This affects navigation, not the recorded execution. I preserved the original notebook unchanged.

**The manifest counts are useful inventory context, with limits on independence and eligibility.** The saved output reports 8,854 AMASS rows and 1,874 GAVD sequences. It does not report the total available or unavailable rows; seeing true flags in the ten displayed rows cannot establish availability for every row.

I independently inspected the current local manifest metadata. Its row counts agree with the saved output:

| Current local metadata | Motions/sequences | Distinct people/recordings |
| --- | --- | --- |
| AMASS train | 7,217 | 151 audited people |
| AMASS validation | 769 | 19 audited people |
| AMASS test | 868 | 19 audited people |
| GAVD total | 1,874 | 348 video IDs |

These are local manifest counts, not fresh remote availability measurements. The AMASS loader joins approved, nonexcluded identities to existing person splits. Its displayed `calibration`, `development`, and `final` roles come from the shared motion-preservation loader. Synthetic preparation later assigns its own support/context/reference pools using `original_split`; those displayed legacy role names are not the teaching pools.

Under the current pilot defaults of seed 17 and 64 frames at 15 fps, local duration metadata gives 3,355 sufficiently long support motions, 1,057 training-context motions, 1,230 training-reference motions, 277 validation-context motions, and 242 validation-reference motions before file-availability filtering. This suggests ample manifest-level candidates, but does not verify the selected HAIC clips or the undisplayed effective run settings.

GAVD rows are clips, not independent known participants. Multiple rows share a video ID, and the notebook correctly labels identity as unknown. Notebook 00 does not apply reservation eligibility or verify checked person crops, view groups, reference coverage, or human annotations. Those checks occur later in [annotations.py](../../../src/gavd6_sjepa/research_directions/synthetic_training/annotations.py). The existing GAVD gait annotations are not independent twelve-landmark reference coordinates. Human landmark completion is a later evaluation dependency, not a prerequisite for starting synthetic source preparation.

**The scientific framing is sound for an initial feasibility experiment, but no hypothesis is tested here.** The configured students are:

| Role | Students | Interpretation |
| --- | --- | --- |
| Fit | RTMPose-m, HRNet-W32 | Two source training students. |
| Select settings | RTMPose-s, HRNet-W48 | Two excluded checkpoints, sharing source architecture families. |
| Held architecture | ViTPose-base | One family excluded from source fitting and selection. |

The intended evidence ladder is appropriate: improvement over full-budget replay; useful variation across lessons/students; extra information from target prediction change beyond matched source-progress and snapshot controls; and transfer to independently evaluated real video on the held family. Loaded V-JEPA features and matched context comparisons would additionally be needed to assess their contribution. Merely configuring V-JEPA supports none of these conclusions.

Two fitting students and two validation students are a small pilot, and the 24 source scene settings reuse underlying motions. [source_domains](../../../src/gavd6_sjepa/research_directions/synthetic_training/rendering.py) explicitly uses the same domain combinations for fitting and validation; it does not implement an additional domain-combination holdout. Distinct rendered frames or settings should not be counted as independent learners. This is narrower than the proposal's language about excluded domain combinations and should be made explicit in later reporting. No leakage is demonstrated by this notebook; it simply supplies no empirical validation of the later selection or transfer claims.

**The next useful milestone is a verified source-data preparation run.** First configure the three rendering inputs and preserve the effective settings used by the job. Rerun inventory and retain complete paths, available/missing counts by data role, and the per-attempt asset table. Validate the required body/DMPL files and a small real render with landmark overlays, texture alignment, camera orientation, visibility, and readable backgrounds. In a GPU allocation, verify the released estimators and configured context encoders actually load, and that a supervised update runs. The existing `check_environment.py --require-cuda` can check package/CUDA operations, but does not load model weights or exercise the renderer.

Then run notebook 01 separately using the [HAIC guide](../../../slurm/synthetic-training/README.md), inspect its RGB/label/context outputs, and proceed to source trials only after those checks succeed. Reserve human reference preparation and scoring for the later GAVD stages.

Validation performed for this review: notebook format and code syntax passed; execution counts and error outputs were inspected; v1/v2/v3 code and asset changes were compared; all four copied navigation links were checked; isolated inventory/configuration/renderer behavior checks passed; and all nine existing `test_data.py` tests passed. Those tests cover data-role separation, unlabeled-context access, camera projection, UV topology, recording separation, and reference-scale handling. No notebook, source code, experiment configuration, or saved output was modified; this analysis file is the only added artifact.
