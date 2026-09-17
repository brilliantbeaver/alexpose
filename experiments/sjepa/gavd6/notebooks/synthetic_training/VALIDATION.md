# Local workflow validation

These checks exercise the implementation and launch interfaces. They do not establish a teaching effect or real pose accuracy.

Completed locally:

- All 35 study tests pass. They cover loss supervision and frozen-backbone behavior, independent branch restoration, camera/UV geometry, person/recording separation, selector information boundaries, replay rejection, and recording-level uncertainty.
- A CPU integration fixture runs the actual source-trial, selector-fitting, deployment, crossover, reporting, and annotation-evaluation functions. Only the external pretrained estimator is replaced by a small trainable fixture. Tests retain failed predictions, shuffle annotation rows and prediction-cache order, verify independent reference scales, and require the configured held student before scoring.
- The matched source-progress selector uses the full method's algorithm and hyperparameter, while a separately tuned source-progress comparator challenges practical performance. Crossover scoring retains every preselected student pair, including identical lesson choices.
- Static annotation-page JavaScript passed syntax and functional checks for image-coordinate clicks, visibility, resumable state, and twelve-row CSV exports. The dashed identity guide does not populate the independently drawn reference box.
- All nine generated notebooks validate as notebook format 4. Their 27 code cells parse, remain output-free, and have valid local reading links.
- Notebook 00 completed in a fresh kernel using the selected interpreter and the real repository manifests. It reported actual local asset availability without loading GPU models.
- An earlier kernel startup failure was retained as a separate failed notebook attempt. The successful retry preserved that file.
- All eleven Slurm phases passed `--dry-run`. The default source array contains four train/validation students and excludes the held ViTPose family. Deployment and crossover arrays include all five students.
- A request to submit the held student to source trials was rejected. The source pipeline contains no GAVD evaluation stage.
- The executor and builder compile. All shell launchers pass Bash syntax checks.

The local interpreter is the existing macOS project environment, not the proposed HAIC MMPose/CUDA environment. The official Linux Python 3.11 MMCV wheel and public model URLs were checked against their release sources, but the installation recipe and GPU model updates have not been executed on HAIC.

Still required with the actual assets: successful released pose-model and video-encoder loads, one supervised update, compatible textured EGL rendering, measured GPU throughput, independent annotation completion, and the source/real scientific experiments. No generated test fixture is reported as AMASS or GAVD evidence.

Run the focused study checks from the repository root:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests/synthetic_training -v
```

Independent reviews examined the model adapters, data and renderer, source/target information boundaries, annotation workflow, and Slurm integration. Corrections included SMPL front/back orientation, non-prefix AMASS windows, protected recording aliases, reviewer separation, preservation of failed predictions, shared context-cache writes, and the explicit matched source-progress and crossover measurements.
