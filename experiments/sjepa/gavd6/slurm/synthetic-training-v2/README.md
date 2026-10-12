# Synthetic training v2: HAIC experiments

**Your pilot and its three post-run checks have completed. Start with the expanded experiment below.** The later numbered steps retain the original pilot and diagnostic instructions for reference. This is the single operating guide; the saved `source-smoke-01/session.env` supplies your interpreter, checkout and pilot location. No CSV edits, notebook runs or new environment variables are needed.

## Run the expanded experiment now

The completed pilot suggests that simple calibration explains much of the coordinate improvement: the affine correction beats the neural methods on that endpoint, while paired JEPA slightly worsens displacement error. The expanded experiment tests whether these findings persist across more people, seeds and training budgets. It also measures movement and reference eligibility separately, so unsupported timing records cannot appear as successful preservation.

| Setting | Expanded experiment |
| --- | --- |
| People | 24 training and 8 development people; excludes every pilot person, original test people and known protected aliases |
| Motion | One recording per person; four nonoverlapping 64-frame windows, sampled at 25 Hz, starting at 5, 8, 11 and 14 seconds |
| Rendered conditions | Training: clean, blur, obstruction. Development also includes combined blur and obstruction. Original frontal camera retained. |
| Extractors | HRNet-W32 and RTMPose-m for fitting; ViTPose-Base held out of fitting and calibration |
| Training | 200 and 2,000 updates **per phase**, each from scratch with seeds 17, 29 and 43: six complete comparison runs |
| Methods | All eight learned arms, unchanged tracks and three fixed filters; pooled joint-offset and affine calibration fitted only to training data |
| Endpoints | Original coordinate/motion metrics; calibration, timing eligibility/coverage, trajectories and losses; exploratory hip/knee/ankle left–right vector errors and occluded-joint motion |

The expected bundle contains **576 training tracks and 384 development tracks**, derived from 128 physical windows. There are **8 independent development people**. Conditions, extractors and training seeds do not increase that sample size. The 200-versus-2,000 comparison also changes the cosine learning-rate schedule; it is a training-budget comparison from scratch, not continuation of one learning curve.

**A. From the local checkout on your Mac, copy the supplied installer to HAIC:**

```bash
scp artifacts/synthetic-training-v2/full-experiment-install-20260919.sh haic:~/stv2-full-install.sh
```

The installer is a self-contained artifact produced with this revision. It works without a Git commit, push or pull. If you have subsequently edited the expansion scripts, regenerate it before copying with `.venv/bin/python scripts/research_directions/synthetic_training_v2/package_full_experiment.py`. The builder does not submit jobs.

**B. In your HAIC shell, install and launch:**

```bash
(
set -euo pipefail
bash "$HOME/stv2-full-install.sh"
source "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training-v2/source-smoke-01/session.env"
bash "$STV2_ROOT/slurm/synthetic-training-v2/full-experiment.sh" launch --name full-01 --gpu-hours 12
)
```

The installer checks the required scientific revision and dependencies, backs up changed helper files, and installs the expansion and compatible diagnostic checker. It preserves scientific source files, the frozen protocol and pilot results. Installation stops if a conflicting experiment is active. **Do not pull or edit this checkout while the expansion is pending or running.**

The launch command authorizes **at most 12 additional H100 allocation-hours** and requests one H100, 8 CPUs and 96 GB RAM. It imports the pilot's recorded costs and any earlier expansion allocations, including failures, and enforces the 48-hour cumulative ceiling. A conservative projection from your measured pilot preparation and fitting times must fit the requested cap before submission. Twelve hours is a hard limit, not an expected runtime; the job releases its allocation when finished. CPU screening and analysis within this allocation count toward that limit.

Expect `FULL_EXPERIMENT_SUBMITTED`, one job ID and its log path. You may disconnect. The job automatically validates the allocated environment, screens the new people, prepares the paired tracks once, and executes each seed's short and long training comparisons. After each comparison it verifies predictions and receipts, runs all three post-run checks, adds the exploratory movement analyses, and writes an updated report. A repeated launch with `--name full-01` does not submit a duplicate.

Selection carries forward the original reservation records and requires the requested panel exactly. Missing assets, changed protected-identity records, insufficient eligible people or exhausted time produce an explicit failure; the job never silently substitutes a smaller experiment. Successful earlier comparisons remain readable. No automatic paid retries are submitted.

**C. Check progress and read available results:**

```bash
source "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training-v2/source-smoke-01/session.env" &&
bash "$STV2_ROOT/slurm/synthetic-training-v2/full-experiment.sh" status --name full-01
```

All new files are under `$STV2_ROOT/outputs/synthetic-training-v2/full-01/`:

| Path | Meaning |
| --- | --- |
| `request.json`, `submission.json`, `status.json`, `slurm-JOB_ID.out` | Frozen settings, allocation and progress/failure details |
| `inputs/screen.json` | Requested/achieved people, exclusions and every screening attempt |
| `panel-verification.json`, `paired/bundle/` | Verified panel counts and reusable paired tracks |
| `runs/updates-0200-seed-17/report.md` | First completed comparison; analogous paths for 2,000 updates and seeds 29/43 |
| `diagnostics/updates-0200-seed-17/` | Calibration, timing, trajectories, residuals, learning curves and exploratory movement checks for that comparison |
| `report.md`, `latest-summary.json`, `summaries/completed-06/` | Progress-labelled overview, latest immutable summary, and final six-run tables |

The top-level `report.md` appears after the first completed comparison and states how many of the six are complete. Final success requires **`FULL_EXPERIMENT_COMPLETE`** from status: all six analyses finished and Slurm reports `COMPLETED`, exit `0:0`. `worker_complete` alone still awaits that scheduler check. `allocation_failed` means inspect `status.json` and the log; it can coexist with usable completed comparisons. An uncertain submission retains its unique job name in `submission.json`; resolve it with Slurm before creating another run name. Earlier allocation costs are imported before a later expansion can start.

Interpret the original visible-reference metrics first, by clean/corrupted condition and extractor. The additional **all-valid synthetic** endpoint uses available synthetic references even for occluded joints. A separate **fixed window scale** uses the median reference-box diagonal to check whether framewise scaling changes amplitude or peaks. Both are labelled exploratory and leave the original masks, scales and metrics intact. Read timing error with oracle eligibility, missed/extra peaks and coverage. Left–right vector error measures the geometry between paired joints; it is not a clinical asymmetry measure.

This completes the planned core synthetic development comparisons. Inputs remain machine-screened, targets remain projected anatomical proxies, and the camera and short observation window are inherited from the pilot. Scientific Gate B remains `insufficient_evidence`; independent anatomical review and real-video temporal annotations are still required for real-transfer or confirmation claims. The installer, selection, scheduler control and complete analysis path are tested locally; H100 rendering/training of this larger panel has not been executed by the assistant.

## Original pilot and diagnostic commands

**Publication prerequisite for the original pilot's Git route:** Step 1 below installs only changes already committed and pushed or merged into `origin/main`. The self-contained expansion installer above does not require publication. Step 1 stops if the required scripts are absent from Git; it does not commit, push, stash or discard changes.

Your reported environment and historical checks passed; the audit had zero rows and all 189 reservation decisions were blank. Automation leaves those worksheets intact. It creates a small **machine-screened development experiment**, prepares data, trains all comparisons, evaluates them and verifies the results. It does not invent human reviews or declare unknown reservations cleared.

The first run uses two training people and two validation people, two windows each, with the saved seed/update recipe. It delivers preliminary numerical comparisons for all eight learned methods and four baselines. The expansion above adds people and repeated seeds; independent anatomical validation and real-video transfer remain separate research work.

**1. On HAIC: pull the published revision, before starting any jobs.**

```bash
source "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training-v2/source-smoke-01/session.env" &&
cd "$STV2_ROOT" &&
"$STV2_PYTHON" - <<'PY'
from contextlib import ExitStack
import fcntl, json, os, subprocess, sys
from pathlib import Path

work = Path(os.environ['STV2_WORK'])
def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()

with ExitStack() as locks:
    for name in ('automation-launch', 'automation', 'manage'):
        path = work / 'locks/process' / f'{name}.lock'
        path.parent.mkdir(parents=True, exist_ok=True)
        stream = locks.enter_context(path.open('a'))
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('Run/controller is active. Use Step 3; do not update its code.')
    state = json.loads((work / 'control.json').read_text())
    if state['jobs'] or state.get('pending_submission') or state.get('source_config'):
        raise SystemExit('This run has submission history. Preserve its code and use Step 3.')
    if git('branch', '--show-current') != 'main':
        raise SystemExit('Expected the HAIC main branch. No pull performed.')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise SystemExit('Tracked HAIC files have local changes. Preserve them; no pull performed.')
    subprocess.run(['git', 'pull', '--ff-only', 'origin', 'main'], check=True)
    if git('rev-parse', 'HEAD') != git('rev-parse', 'refs/remotes/origin/main'):
        raise SystemExit('Local main differs from origin/main. No experiment launched.')
    scripts = 'scripts/research_directions/synthetic_training_v2/'
    required = [scripts + name for name in (
        'automated_run.py', 'automated_inputs.py', 'review_motion.py', 'check_results.py')]
    required.append('slurm/synthetic-training-v2/automated-inputs.sbatch')
    for name in required:
        if not Path(name).is_file() or not git('ls-files', '--', name):
            raise SystemExit(f'Automation revision is not fully published/installed: {name}')
    subprocess.run([sys.executable, scripts + 'check_history.py', '--reconstruct'], check=True)
print('GIT_UPDATE_READY: published automation files and historical evidence checked. Continue to Step 2.')
PY
```

This runs `git pull --ff-only origin main` while holding this run's management locks. Use it only while no other experiment is using the same checkout; these locks do not cover other run directories. It checks the branch, tracked local changes, installed automation entry points and retained historical evidence. Continue only after **`GIT_UPDATE_READY`**. Diverged branches or unpublished code stop the block; nothing is reset or stashed. Normal Git authentication may be required.

Your earlier `Historical evidence: PASSED` means the retained evidence was already valid on HAIC; the command rechecks it. Git retrieves committed files only. It does not transfer ignored/untracked artifacts, environments, licensed bodies, downloaded checkpoints or the saved run session. Reuse those existing HAIC assets. Once any jobs are recorded, preserve this scientific code and use Step 3 for progress. Step 4 installs only separate diagnostic scripts; it does not pull a new scientific revision.

**2. On HAIC: start the complete experiment.**

```bash
source "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training-v2/source-smoke-01/session.env" &&
cd "$STV2_ROOT" &&
"$STV2_PYTHON" scripts/research_directions/synthetic_training_v2/automated_run.py launch
```

Expect `Controller started` and the path to `automation.log`. The controller runs in the background without prompts. Launching is not proof of GPU execution; Step 3 reports progress or failure. Repeating launch while active does not submit another experiment.

If this immediately prints **`AUTOMATED_RUN: [Errno 9] Bad file descriptor`**, run the [controller-lock repair](#repair-the-controller-lock) below. That block applies the fix and repeats launch/status; no repeat of Step 1 or environment installation is needed for this defect.

```text
environment/history checks
  → CPU motion screening and generated records
  → input/asset validation
  → H100 rendering and frozen pose extraction
  → all ten source stages
  → verified predictions, metrics and report
```

Selection uses the approved identity registry, preferring normal treadmill names and then walking names. Names only order candidates. Actual SMPL-H/DMPL reconstruction must pass declared finite-geometry, limb-size and alternating-leg-motion checks at 5.00–7.52 s and 8.00–10.52 s. These are unvalidated screening heuristics. All attempted candidates, thresholds, decisions and source hashes are saved. The search is bounded to 24 recordings and requires four distinct canonical people.

Original splits are preserved; original test people and known reserved identities/aliases are excluded. The configured reservation file and existing worksheets are read automatically. Missing external reservation or exposure knowledge remains `unknown`. Accepted windows carry `algorithm_screened_locomotion`; outputs carry `automated-source-screen`. Playback and overlays are saved, but no human review is claimed or required in this development mode. Scientific Gate B remains `insufficient_evidence` regardless of numerical ordering.

**3. On HAIC: check progress or obtain result paths.** Repeat whenever needed, including after reconnecting:

```bash
source "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training-v2/source-smoke-01/session.env" &&
"$STV2_PYTHON" "$STV2_ROOT/scripts/research_directions/synthetic_training_v2/automated_run.py" status
```

`PENDING`, `RUNNING` and `ACCOUNTING_PENDING` mean wait. Final success is `"status": "complete"` with `AUTOMATED_EXPERIMENT_COMPLETE` in the log. This requires successful managed allocations, all ten valid stage receipts, complete fits/predictions and metrics reconstructed from those predictions. A GPU-hour balance alone is not completion.

Initial output paths, relative to `$STV2_WORK`:

| File | Purpose |
| --- | --- |
| `automation.log` / `automation.json` | Progress, failure reason and completion summary |
| `inputs/automated-01/screen.json` | Selection policy, rejected candidates, people and evidence hashes |
| `inputs/automated-01/playback/review.html` | Standalone motion playback; optional inspection |
| `paired-01/overlay-*.png` | Rendered pose overlays; optional inspection |
| `source-01/report.md` | Method comparison, uncertainty, support and limitations |
| `source-01/evaluation/per-person-balanced-summary.csv` | Coordinate/motion metrics by method, extractor and seed |
| `source-01/evaluation/accuracy-preservation.png` | Accuracy/motion-error plot |
| `source-01/evaluation/gates.json` | Scientific decisions, distinct from execution success |

Retries use numbered input/preparation directories; status supplies authoritative paths. Optionally print the finished report:

```bash
cat "$STV2_WORK/source-01/report.md"
```

**4. On HAIC: run the three checks on the completed source results.**

Run this block once **after the diagnostic files have been published to `origin/main`**. It backs up existing diagnostic scripts and their result checker, installs those files, and submits one CPU job for all three checks. The original scientific code, protocol, controller, source results and GPU ledger stay unchanged. Do not repeat the installation while a diagnostic job is pending or running; submitted jobs verify their diagnostic code hashes.

```bash
(
set -euo pipefail
source "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training-v2/source-smoke-01/session.env"
cd "$STV2_ROOT"
git fetch origin main
stv2_diag_paths=(
  scripts/research_directions/synthetic_training_v2/diagnostics/__init__.py
  scripts/research_directions/synthetic_training_v2/diagnostics/calibration.py
  scripts/research_directions/synthetic_training_v2/diagnostics/timing.py
  scripts/research_directions/synthetic_training_v2/diagnostics/plots.py
  scripts/research_directions/synthetic_training_v2/postrun_checks.py
  scripts/research_directions/synthetic_training_v2/check_results.py
  slurm/synthetic-training-v2/postrun-checks.sh
  slurm/synthetic-training-v2/postrun-checks.sbatch
)
stv2_diag_prefix="$(git rev-parse --show-prefix)"
for stv2_diag_path in "${stv2_diag_paths[@]}"; do
  git cat-file -e "origin/main:$stv2_diag_prefix$stv2_diag_path"
done
stv2_diag_backup="$(mktemp -d "$STV2_WORK/diagnostics-install-backup-XXXXXXXX")"
for stv2_diag_path in "${stv2_diag_paths[@]}"; do
  if [[ -e "$stv2_diag_path" ]]; then
    mkdir -p "$stv2_diag_backup/$(dirname "$stv2_diag_path")"
    cp -a "$stv2_diag_path" "$stv2_diag_backup/$stv2_diag_path"
  fi
done
git restore --source=origin/main --worktree -- "${stv2_diag_paths[@]}"
bash slurm/synthetic-training-v2/postrun-checks.sh all
)
```

Expect `CPU diagnostics job: JOB_ID`, an output directory, and a log path. The job requests **4 CPUs, 16 GB RAM and a 30-minute limit, with no GPU**. It first verifies successful source jobs, immutable receipts and saved prediction metrics. Then it reads cached tracks and predictions; it does not render, extract poses or retrain neural models. Queue delay and CPU runtime depend on HAIC. This is additional CPU work, recorded separately from the original GPU budget.

Check it after submitting or reconnecting:

```bash
source "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training-v2/source-smoke-01/session.env" &&
bash "$STV2_ROOT/slurm/synthetic-training-v2/postrun-checks.sh" status
```

Success requires **`POSTRUN_CHECKS_COMPLETE`** and a scheduler job that finishes `COMPLETED` with `0:0`. Status prints the latest output directory, scheduler state and log tail. Each submission writes a fresh `$STV2_WORK/diagnostics/<timestamp>-<id>/`; the original `source-01/report.md` is unchanged. Open the new directory's **`report.md`** first.

| Check | What it does | Main artifacts in the diagnostic directory |
| --- | --- | --- |
| Calibration | Fits a pooled 24-parameter joint offset and 72-parameter ridge affine residual on training people and HRNet/RTMPose tracks. Applies the frozen corrections to development people, including held-family ViTPose, without fitting its labels. Compares unchanged, calibrated, initialized, coordinate, direct and paired JEPA. | `calibration-models.json`, `per-person-balanced-summary.csv`, `by-condition.csv` |
| Timing support | Scores references against themselves to establish attainable support. Separates invisible/invalid/weak references from missing predictions and peak-count mismatches. Adds ordered one-to-one matching within a fixed 0.12 s tolerance, with missed/extra counts and timing coverage. | `timing-summary.csv`, `timing-support.csv`, `timing-per-window.csv` |
| Motion and learning curves | Selects four windows by sorted metadata, alternating people before taking second windows. Plots both ankles' x/y trajectories for identical windows, with every extractor and clean/corrupted condition in separate files. Exports displacement, amplitude ratios, per-joint residuals and each fit's learning history. | `selected-windows.json`, `images/*.svg`, `images/*.png`, `per-joint-residuals.csv`, `training-summary.csv`, `training-history.csv` |

The ridge penalty is fixed at `0.001`; development labels never select it. Both calibrations use the same input-only window normalization as the neural models. The offset is constant in normalized coordinates, giving a time-constant pixel shift within each window and preserving raw displacement. The affine correction can change motion. Missing input joints remain missing, and the original coordinate/displacement metrics penalize them. Joint residual summaries are conditional on finite outputs and include missingness counts. Numeric tables cover every saved seed; trajectory figures use the first saved seed, labeled in each figure.

Read coordinate gains alongside motion fidelity. If offset calibration captures most of the neural gain while preserving displacement, temporal pretraining is unnecessary for that component of the gain; this does not establish a particular joint-convention mechanism. If the reference oracle has little timing support, zero model support is not by itself evidence of destroyed timing. If oracle support is adequate but prediction support is poor, inspect missed/extra peaks and trajectories. A low matched-event error is useful only with adequate coverage. Peaks are operational 2D ankle-separation maxima, not heel strikes. Framewise reference-box scaling can also alter the normalized separation signal. Training losses differ across objectives and cannot alone establish convergence.

`all` is the fastest complete route. After installation, use these alternatives **only to rerun a particular check**, rather than submitting them in addition to `all`:

```bash
bash "$STV2_ROOT/slurm/synthetic-training-v2/postrun-checks.sh" calibration
bash "$STV2_ROOT/slurm/synthetic-training-v2/postrun-checks.sh" timing
bash "$STV2_ROOT/slurm/synthetic-training-v2/postrun-checks.sh" curves
```

Each alternative is a separate CPU submission. Timing and curve checks also fit the inexpensive calibrations because those comparisons are needed. Repeating a submission creates a new attempt; `status` only inspects the latest attempt. If source verification fails, preserve the original artifacts and read the named mismatch; do not regenerate receipts or rerun paid source stages merely to make diagnostics pass. If a diagnostic job times out, runs out of memory or is canceled, status reports the scheduler failure even if its saved Python state still says `running`. An uncertain `sbatch` response is recorded in that attempt's `submission.json`; inspect its unique job name with the scheduler commands below before resubmitting. These post-hoc checks retain `automated-source-screen` evidence and cannot open a scientific gate or confirmation claim.

<a id="repair-extraction-status-verification"></a>

**If diagnostics fail on apparently identical `extraction_status` values.** The older checker compares a CSV string with a reconstructed dictionary. `100.0 %` means every cell in that metadata column differs by representation; it does not measure pose error. This fails before calibration, timing or plotting. Decode the dictionary with `ast.literal_eval` while preserving all comparisons. Do not edit the saved CSV, drop the column, loosen tolerances or regenerate source receipts.

After the failed diagnostic job has ended, run this block on HAIC. It backs up and repairs only the checker, verifies the existing source results on CPU, then submits a new diagnostic attempt. It works before the repair is published; an already repaired checker is left intact. Other verification failures stop the block before submission.

```bash
(
set -euo pipefail
source "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training-v2/source-smoke-01/session.env"
cd "$STV2_ROOT"
"$STV2_PYTHON" - <<'PY'
import os, shutil, tempfile
from pathlib import Path

script = Path('scripts/research_directions/synthetic_training_v2/check_results.py')
before = script.read_text()
old = 'saved = pd.read_csv(cfg.root / "evaluation/per-window.csv")'
converter = 'converters={"extraction_status": ast.literal_eval}'
if before.count(old) == 1 and converter not in before:
    after = before.replace(old, old[:-1] + ', ' + converter + ')', 1)
    if '\nimport ast\n' not in after:
        if after.count('\nimport argparse\n') != 1:
            raise SystemExit('Unexpected checker imports; no files changed.')
        after = after.replace('\nimport argparse\n', '\nimport argparse\nimport ast\n', 1)
    compile(after, str(script), 'exec')
    backup = Path(tempfile.mkdtemp(prefix='metric-checker-backup-', dir=os.environ['STV2_WORK'])) / script.name
    shutil.copy2(script, backup)
    script.write_text(after)
    print(f'Metadata decoding repaired. Backup: {backup}')
elif converter in before and '\nimport ast\n' in before:
    print('Metadata decoding repair already installed.')
else:
    raise SystemExit('Unexpected checker version; no files changed.')
PY
"$STV2_PYTHON" scripts/research_directions/synthetic_training_v2/check_results.py
bash slurm/synthetic-training-v2/postrun-checks.sh all
)
```

Expect `SOURCE_RESULTS_COMPLETE`, then a new CPU job ID. Use the Step 4 `status` command. The old failed attempt remains available; the new attempt must finish with `POSTRUN_CHECKS_COMPLETE` and Slurm `COMPLETED`, exit `0:0`.

**What trains and what the budget covers.**

The stages remain `audit → data → adaptation → information → direct → jepa → evaluate → optional → freeze → report`. RTMPose-m and HRNet-W32 supply fitting tracks; ViTPose-Base supplies an excluded-family development panel. Image estimators stay frozen. COCO training, V-JEPA2 and GAVD processing are outside this run.

| Comparisons | Default optimizer updates per seed |
| --- | --- |
| Direct denoiser, SmoothNet-style temporal MLP, static control | 400 end-to-end each |
| Initialized-encoder readout | 200 readout |
| Clean-coordinate, ordinary JEPA, paired JEPA, shuffled-pair JEPA | 200 pretraining + 200 readout each |
| Unchanged tracks; filter strengths 0, 1, 2 | No fitting |

These are the initializer defaults, seed 17 and `updates=200`; your saved `control.json` governs execution. The `adaptation`, `optional` and `freeze` stages report pending/closed branches; they do not train additional models or open confirmation data.

CPU screening requests 4 CPUs/16 GB for at most 20 minutes. Rendering requests one H100/8 CPUs/96 GB for at most one hour. Each of two fitting stages requests one H100/8 CPUs/64 GB for at most one hour; other stages use CPUs. The initial sequence needs room for three one-hour GPU allocations within your existing four-hour allowance. Allocated time, failed attempts and overhead count. The 54-minute fitting guard and scheduler limits are bounds, not runtime estimates. Queue availability, licensed-asset compatibility and HAIC throughput cannot be guaranteed by local tests.

**If execution stops.**

The successful path requires no manual data editing. Actual dependency or scheduler failures still need correction; automation stops with a reason instead of inventing results or repeatedly spending GPU time.

| State | Action |
| --- | --- |
| Step 2 immediately prints `AUTOMATED_RUN: [Errno 9] Bad file descriptor` | An older launcher's read-only lock can cause this error on Linux NFS. Run the [controller-lock repair](#repair-the-controller-lock). |
| Controller inactive while jobs are pending/running | Rerun Step 2; it resumes monitoring recorded jobs. |
| Environment, checkpoint, body asset or rendering error | Step 3 and `logs/` name the failing dependency. Repair that dependency. |
| `insufficient_candidates` | Read `screen.json` for reconstruction errors or failed rules. No locomotion approval is invented. |
| Source job failed | Only that chain's still-pending dependents are canceled. Failed outputs and allocation costs remain recorded. |
| Git update stops | The message identifies unpublished files, a different/dirty/diverged branch, or a started run. Preserve existing work; do not force-reset or bypass the stop. |
| Historical-file error | Git may not contain the named artifact. Before any jobs start, use the recovery transfer below; do not rewrite preservation hashes. |

<a id="repair-the-controller-lock"></a>

**Repair the controller lock on HAIC.** Older launchers open `locks/process/automation.lock` with `open('r')`, then request an exclusive lock. Linux NFS requires a writable descriptor for that operation ([Linux `flock` documentation](https://man7.org/linux/man-pages/man2/flock.2.html)). Step 1 creates the lock file, so the defect can affect the first launch.

Run this single block on HAIC. It backs up the launcher under `$STV2_WORK/controller-lock-backup-*`, changes `r` to `r+`, then launches and prints status. Repeating it is safe: an installed fix is left intact and duplicate-controller protection remains active. The `r+` mode preserves existing lock contents; do not delete the lock files.

```bash
(
set -e
source "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training-v2/source-smoke-01/session.env"
cd "$STV2_ROOT"
"$STV2_PYTHON" - <<'PY'
import os, shutil, tempfile
from pathlib import Path

script = Path('scripts/research_directions/synthetic_training_v2/automated_run.py')
before = script.read_text()
old = "stream = path.open('r')"
new = "stream = path.open('r+')"
if before.count(old) == 1 and new not in before:
    after = before.replace(old, new, 1)
    compile(after, str(script), 'exec')
    backup = Path(tempfile.mkdtemp(prefix='controller-lock-backup-', dir=os.environ['STV2_WORK'])) / script.name
    shutil.copy2(script, backup)
    script.write_text(after)
    print(f'Controller lock fixed. Backup: {backup}')
elif old not in before and before.count(new) == 1:
    print('Controller lock fix already installed.')
else:
    raise SystemExit('Unexpected launcher version; no files changed. Inspect its controller_running function.')
PY
"$STV2_PYTHON" scripts/research_directions/synthetic_training_v2/automated_run.py launch
"$STV2_PYTHON" scripts/research_directions/synthetic_training_v2/automated_run.py status
)
```

Expect `Controller lock fixed` or `Controller lock fix already installed`, followed by `Controller started` or an indication that the controller is already running or the experiment is complete. Continue monitoring with Step 3. This repair changes only the launcher and leaves it modified in Git; preserve the repair when updating Git after the experiment.

The abbreviated error alone does not prove which call failed. If launch still reports the error after this repair, run this block on HAIC to expose the full traceback:

```bash
source "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training-v2/source-smoke-01/session.env" &&
cd "$STV2_ROOT" &&
"$STV2_PYTHON" -c 'import sys; sys.path.insert(0, "scripts/research_directions/synthetic_training_v2"); import automated_run; automated_run.main(["launch"])'
```

For missing historical artifacts outside Git, this optional recovery command runs on your **Mac**, from the local checkout containing the published revision. It transfers only verified preservation-manifest files and backs up replacements; scientific code is not copied. Use it only before jobs start, then repeat Step 1 on HAIC.

```bash
cd /Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6 &&
bash slurm/synthetic-training-v2/sync-to-haic.sh \
  tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6 --history-only --apply
```

<a id="retry-a-failed-job"></a>
After resolving a failure, explicitly retry within the remaining budget:

```bash
"$STV2_PYTHON" "$STV2_ROOT/scripts/research_directions/synthetic_training_v2/automated_run.py" launch --retry
```

Valid completed stages are reused. Failed input/preparation attempts get new directories. Paid work is never retried automatically. Changing scientific code, source data or the recipe after preparation requires a new declared run.

<a id="recover-an-uncertain-submission"></a>
An uncertain `sbatch` response is the exceptional case that cannot safely be guessed. The controller preserves its unique job-name token. Inspect actual scheduler records:

```bash
squeue -u "$USER" -o '%.18i %.64j %.12T'
sacct -u "$USER" --format=JobID,JobName%64,State,ExitCode,ElapsedRaw
```

Use `bash "$STV2_ROOT/slurm/synthetic-training-v2/submit.sh" recover --job-id ACTUAL_ID` only for the accepted job matching that token, or `recover --not-submitted` only after establishing rejection. Then use the retry command. An empty queue alone does not prove rejection.

<a id="carry-costs-into-a-later-experiment"></a>
This guide resumes the initialized four-hour scope. It does not grant additional compute or automatically expand the study. Preserve the managed cost ledger and declare the next sample, seed and compute plan before expanding.
