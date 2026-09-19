# Synthetic training v2: automated HAIC run

**After this revision is published, run the three command blocks below on HAIC, in order.** This is the complete operating guide for your existing `source-smoke-01` run. There are no CSV edits, review prompts, notebook runs, overlay approvals or additional environment variables to supply. The saved session supplies the interpreter, asset paths, scheduler account and partition.

**Publication prerequisite:** the complete automation revision must first be committed and pushed or merged into `origin/main`. Editing this guide does not publish the code. At this revision's review, the automation changes were still uncommitted locally; a pull alone could not install them. Step 1 stops if the required scripts are absent from Git. It does not commit, push, stash or discard changes.

Your reported environment and historical checks passed; the audit had zero rows and all 189 reservation decisions were blank. Automation leaves those worksheets intact. It creates a small **machine-screened development experiment**, prepares data, trains all comparisons, evaluates them and verifies the results. It does not invent human reviews or declare unknown reservations cleared.

The first run uses two training people and two validation people, two windows each, with the saved seed/update recipe. It delivers preliminary numerical comparisons for all eight learned methods and four baselines. Independent anatomical validation, larger samples, repeated seeds and real-video transfer remain separate research work.

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

Your earlier `Historical evidence: PASSED` means the retained evidence was already valid on HAIC; the command rechecks it. Git retrieves committed files only. It does not transfer ignored/untracked artifacts, environments, licensed bodies, downloaded checkpoints or the saved run session. Reuse those existing HAIC assets. Once any jobs are recorded, keep the checkout unchanged; subsequent logins need only Step 3.

**2. On HAIC: start the complete experiment.**

```bash
source "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training-v2/source-smoke-01/session.env" &&
cd "$STV2_ROOT" &&
"$STV2_PYTHON" scripts/research_directions/synthetic_training_v2/automated_run.py launch
```

Expect `Controller started` and the path to `automation.log`. The controller runs in the background without prompts. Launching is not proof of GPU execution; Step 3 reports progress or failure. Repeating launch while active does not submit another experiment.

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
| Controller inactive while jobs are pending/running | Rerun Step 2; it resumes monitoring recorded jobs. |
| Environment, checkpoint, body asset or rendering error | Step 3 and `logs/` name the failing dependency. Repair that dependency. |
| `insufficient_candidates` | Read `screen.json` for reconstruction errors or failed rules. No locomotion approval is invented. |
| Source job failed | Only that chain's still-pending dependents are canceled. Failed outputs and allocation costs remain recorded. |
| Git update stops | The message identifies unpublished files, a different/dirty/diverged branch, or a started run. Preserve existing work; do not force-reset or bypass the stop. |
| Historical-file error | Git may not contain the named artifact. Before any jobs start, use the recovery transfer below; do not rewrite preservation hashes. |

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
