"""Build the three NeurIPS-laterality tutorial notebooks in neurips-brain-body.

Emits into neurips-brain-body (overwriting nb_05a; creating nb_05c, nb_05d):
  nb_05a_signed_laterality_probe.ipynb        -- E1 hardened emergent null
  nb_05c_reflection_equivariant_readout.ipynb -- E2 frame-averaged readout (built-in)
  nb_05d_reflection_equivariant_encoder.ipynb -- E3a frame-averaged encoder + E3b augmentation

Design contract (why this is trustworthy):
  * Each notebook's "run" cell invokes the EXACT validated standalone script in
    work/experiments/ via a subprocess, which regenerates the canonical JSON artifact.
    So a headless `nbconvert --execute` reproduces every number from scratch.
  * Each notebook's "verify" cell loads that JSON and asserts the paper's cited numbers
    against it (tolerance-checked), then renders the tables. Nothing is hand-typed as a
    "result": the displayed values come from the freshly-written artifact.
  * E3b LOADS the two pre-trained Stage-0 checkpoints (flip 0.0 / 0.5) and probes them;
    it does NOT re-run the ~15 min/arm MPS training. That training is provenance, in
    e3b_reflection_augmented_retrain.py; the checkpoints are on disk.

All results are TRANSDUCTIVE (internal validity only; the encoder was trained on the
evaluated rows). The source video (not clip, not person) is the independent unit. Dataset
condition folder labels (normal/parkinsons/stroke/myopathic/cerebralpalsy) are dataset
annotations, NOT diagnoses. No institutional ethics determination or completed data-use
review is yet on record; both must be resolved before any submission.
"""
from pathlib import Path
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
NOTEBOOK_DIR = EXPERIMENT_DIR / "neurips-brain-body"

RESPONSIBLE_USE = """\
> **Responsible-use notice.** All results in this notebook are **transductive** — the
> S-JEPA encoder was trained on the very sequences being evaluated — so they carry
> *internal validity only* and make **no** claim of generalization to new sources or
> people. The **source video** (not the clip, not the individual) is the independent unit
> of analysis. The dataset's condition folders (`normal`, `parkinsons`, `stroke`,
> `myopathic`, `cerebralpalsy`) are **dataset annotations, not diagnoses**. The dataset's
> official distribution provides annotations and public video URLs, not raw video; this
> analysis uses derived pose sequences, infers no identity, and redistributes no raw or
> identity-bearing frames. **No institutional ethics determination or completed data-use
> review is yet on record; both must be resolved before submission.**"""

RUN_PREAMBLE = """\
import json, os, subprocess, sys, textwrap
from pathlib import Path

def find_experiment_dir(start=None):
    candidates = []
    if os.getenv("ALEXPOSE_ROOT"):
        env_root = Path(os.environ["ALEXPOSE_ROOT"]).expanduser().resolve()
        candidates.extend([env_root, env_root / "experiments" / "sjepa" / "gavd5-drift"])
    start = Path(start or Path.cwd()).resolve()
    for base in [start, *start.parents]:
        candidates.extend([base, base / "experiments" / "sjepa" / "gavd5-drift"])
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "work" / "experiments").is_dir():
            return candidate
    raise FileNotFoundError(f"Cannot locate gavd5-drift from {start}; set ALEXPOSE_ROOT.")


EXPERIMENT_DIR = find_experiment_dir()
NOTEBOOK_DIR = EXPERIMENT_DIR / "neurips-brain-body"
PY = EXPERIMENT_DIR / ".venv" / "bin" / "python"
PY = str(PY if PY.exists() else sys.executable)   # fall back to the running kernel
ART = EXPERIMENT_DIR / "work" / "artifacts" / "real"

def run_experiment(script_relpath):
    \"\"\"Run a validated standalone experiment script; it regenerates its JSON artifact.\"\"\"
    script = EXPERIMENT_DIR / script_relpath
    print(f"running {script.name} with {PY} ...")
    proc = subprocess.run([PY, str(script)], cwd=str(EXPERIMENT_DIR),
                          capture_output=True, text=True)
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print("STDERR (tail):\\n", proc.stderr[-4000:])
        raise RuntimeError(f"{script.name} exited {proc.returncode}")
    return proc

def approx(a, b, tol):
    return abs(float(a) - float(b)) <= tol
"""


# ======================================================================= nb_05a (E1)
def build_nb_05a():
    nb = new_notebook()
    c = nb.cells
    c.append(new_markdown_cell(
        "# nb_05a — Signed-laterality decodability probe (E1, hardened)\n\n"
        "**Does a self-supervised skeleton world model learn that the body is bilaterally "
        "symmetric?** This notebook makes the question exact, audits a frozen **S-JEPA** "
        "against it, and finds a *robust informative null*: the symmetry does **not** emerge.\n\n"
        "This is experiment **E1** of the NeurIPS 2026 *Physical World AI* laterality package "
        "(`../neurips-laterality/docs/`). It **binds to the canonical checkpoint** "
        "`sjepa_curriculum_final.pt` (fingerprint `7d13841a…`) and hardens the original probe "
        "with: SVD-solver ridge, **repeated** source-disjoint CV (10 reshuffles → stability "
        "interval), an "
        "**alpha-sensitivity sweep**, an explicit **cohort decision** (626 modeled / 642 "
        "superset), and a **landmark-missingness control lane**.\n\n" + RESPONSIBLE_USE))

    c.append(new_markdown_cell(
        "## 1. Bilateral symmetry as a $\\mathbb{Z}/2$ group action\n\n"
        "Reflecting a skeleton across the sagittal plane **and** swapping left/right landmarks "
        "yields another valid skeleton. The mirror operator $M$ negates the $x$-coordinate of "
        "every joint and swaps **all sixteen** bilateral landmark pairs — a valid whole-body "
        "reflection. Then $M^2=I$, so $G=\\{I,M\\}\\cong\\mathbb{Z}/2$.\n\n"
        "**Signed laterality target.** Over the six bilateral pairs that carry gait laterality\n\n"
        "$$(11,12),\\ (23,24),\\ (25,26),\\ (27,28),\\ (29,30),\\ (31,32)$$\n\n"
        "(shoulders, hips, knees, ankles, heels, foot-indices), let $\\ell_k$ (resp. $r_k$) be the "
        "per-joint *temporal* standard deviation of the left (resp. right) landmark, summed over "
        "$x,y,z$. Define\n\n"
        "$$ y(x) \\;=\\; \\sum_k (\\ell_k - r_k). $$\n\n"
        "Mirroring swaps $\\ell_k \\leftrightarrow r_k$, so **by construction**\n\n"
        "$$ y(Mx) = -\\,y(x), \\qquad\\text{i.e.}\\qquad y(T_g\\,x)=\\rho(g)\\,y(x),"
        "\\quad \\rho(I)=+1,\\ \\rho(M)=-1. $$\n\n"
        "A world model *\"encodes bilateral symmetry\"* iff a read-out of its features reproduces "
        "this antisymmetry — a **mirror slope of $-1$** — while remaining **decodable** ($R^2$ "
        "well above an untrained floor). This is a *geometry-aware* probe of an *articulated, "
        "deformable* body whose target is a *proprioceptive* quantity (which side moves more) "
        "— the workshop's themes made concrete. See `../neurips-laterality/docs/figures/fig1_group_action.svg`."))

    c.append(new_markdown_cell(
        "## 2. The audit protocol\n\n"
        "**Five lanes** (`../neurips-laterality/docs/figures/fig2_audit_protocol.svg`):\n\n"
        "| Lane | Feature | Role |\n"
        "|------|---------|------|\n"
        "| **A — learned** | per-pair $[\\ell-r,\\ \\ell+r]$ token statistics from the frozen encoder | the thing under test |\n"
        "| **B — raw ceiling** | target regressed on raw coordinates ($R^2\\approx1$) | decodability sanity check |\n"
        "| **C — floor** | identical architecture, **random** weights | untrained baseline |\n"
        "| **D — pooled** | whole-body mean/std, side-blind | must stay low |\n"
        "| **E — missingness** | per-joint left/right valid-fraction | gauges how much of the target pure left/right *visibility* explains |\n\n"
        "**Estimator.** Source-video-disjoint `GroupKFold` on `video_id`; inner ridge-penalty "
        "selection over $\\alpha\\in\\mathrm{logspace}(-3,3,13)$; per-fold standardization; **SVD** "
        "solver (removes the ill-conditioning of the original run). We report **repeated** "
        "shuffled grouped CV over **10 reshuffles** as mean $\\pm$ a $t$-based **stability "
        "interval** ($t^\\*=2.262$, $\\mathrm{df}=9$). This interval measures sensitivity to the "
        "*partition* under the fixed 626/93 cohort — not population sampling — so interval overlap "
        "is a stability heuristic, **not** a significance test.\n\n"
        "**Pre-registered gates.** (i) A beats C by $\\ge 0.05\\ R^2$; (ii) A reaches $\\ge 80\\%$ "
        "of B; (iii) sign correct on $\\ge 75\\%$ of held-out sources. **Secondary geometry band:** "
        "mirror slope negative and within $[-1.25,-0.80]$.\n\n"
        "**Cohort decision.** The checkpoint's `sequence_ids` are exactly **626** sequences / "
        "**93** source videos — that is the **PRIMARY** (fully transductive) cohort. The **642** "
        "pose-available superset (adds 16 coverage-QC-dropped rows the encoder never saw) is a "
        "**robustness** view only, and it reproduces the pre-hardening canonical numbers "
        "bit-for-bit (a pipeline check)."))

    c.append(new_code_cell(RUN_PREAMBLE))
    c.append(new_code_cell(
        "# Re-run the hardened E1 probe end-to-end (regenerates the canonical JSON).\n"
        "run_experiment('work/experiments/e1_laterality_hardened.py')\n"
        "res = json.loads((ART / 'idea5_signed_laterality_result_hardened.json').read_text())\n"
        "prim = res['primary_cohort']; rob = res['robustness_cohort']\n"
        "print('fingerprint', res['fingerprint'][:12], '| primary',\n"
        "      prim['n_sequences'], 'seq /', prim['n_sources'], 'sources')"))

    c.append(new_code_cell(
        "# ---- Table 1: five lanes on the 626 PRIMARY cohort (single-partition + repeated-CV CI)\n"
        "ci = prim['repeated_cv_ci95']\n"
        "def row(name, key_single, key_ci):\n"
        "    s = prim['lanes'][key_single]['r2']\n"
        "    if key_ci and key_ci in ci:\n"
        "        d = ci[key_ci]; band = f\"{d['mean']:.3f} [{d['ci95_lo']:.3f}, {d['ci95_hi']:.3f}]\"\n"
        "    else:\n"
        "        band = '—'\n"
        "    print(f\"  {name:26s} single={s:+.3f}   repeated-CV={band}\")\n"
        "print('E1 laterality probe — 626 primary (canonical checkpoint)')\n"
        "row('A — learned',           'A_learned',     'A_learned')\n"
        "row('C — untrained floor',   'C_floor',       'C_floor')\n"
        "row('E — missingness-only',  'E_missingness', 'E_missingness')\n"
        "row('D — pooled (side-blind)','D_pooled',     'D_pooled')\n"
        "row('B — raw ceiling',       'B_raw_null',    None)\n"
        "print()\n"
        "print(f\"  mirror slope           {prim['mirror']['slope']:+.4f}  (flips={prim['mirror']['flips']})\")\n"
        "asc = ci['A_sign_consistency']\n"
        "print(f\"  sign consistency (A)   {asc['mean']:.3f} [{asc['ci95_lo']:.3f}, {asc['ci95_hi']:.3f}]\")\n"
        "print(f\"  gates: beats_floor={prim['beats_floor_by_0.05']}  reaches_80pct_B={prim['reaches_80pct_of_null']}  \"\n"
        "      f\"sign_ok={prim['sign_consistent_75pct']}\")\n"
        "print('  PRIMARY_VERDICT:', prim['PRIMARY_VERDICT'])"))

    c.append(new_code_cell(
        "# ---- Alpha-sensitivity sweep for Lane A (why the feature is weak/collinear)\n"
        "sweep = prim['alpha_sweep_A']\n"
        "print('alpha        R2(A)')\n"
        "for a, r2 in sweep.items():\n"
        "    print(f'  {a:>8s}  {r2:+.3f}')\n"
        "lo = min(float(v) for v in sweep.values()); hi = max(float(v) for v in sweep.values())\n"
        "print(f'\\n  A decodes only under heavy regularization: {lo:+.2f} -> {hi:+.2f}')"))

    c.append(new_code_cell(
        "# ---- Robustness cohort (642) reproduces the pre-hardening canonical numbers\n"
        "print('642 robustness:',\n"
        "      f\"A={rob['lanes']['A_learned']['r2']:.3f}  C={rob['lanes']['C_floor']['r2']:.3f}  \"\n"
        "      f\"slope={rob['mirror']['slope']:+.3f}  verdict={rob['PRIMARY_VERDICT']}\")\n"
        "print('  (single-partition A-C =',\n"
        "      f\"{rob['lanes']['A_learned']['r2'] - rob['lanes']['C_floor']['r2']:+.3f}; \"\n"
        "      'the lone gate-pass that does NOT survive cohort-matching + repeated CV)')"))

    c.append(new_code_cell(
        "# ---- Assert the paper's E1 numbers trace to this freshly-written artifact\n"
        "A = ci['A_learned']; C = ci['C_floor']; E = ci['E_missingness']; D = ci['D_pooled']\n"
        "assert approx(A['mean'], 0.198, 0.01), A\n"
        "assert approx(C['mean'], 0.245, 0.01), C\n"
        "assert approx(E['mean'], 0.202, 0.02), E\n"
        "assert approx(D['mean'], 0.101, 0.01), D\n"
        "assert approx(prim['mirror']['slope'], -0.703, 0.02), prim['mirror']['slope']\n"
        "assert A['mean'] < C['mean'], 'learned should NOT beat floor under repeated CV'\n"
        "assert not (prim['beats_floor_by_0.05'] and prim['reaches_80pct_of_null']\n"
        "            and prim['sign_consistent_75pct']), 'all three gates must fail'\n"
        "print('OK — E1 numbers match the paper; robust informative null confirmed.')"))

    c.append(new_markdown_cell(
        "## 3. Reading the result\n\n"
        "The learned lateral feature reaches only $R^2\\approx0.198$ under repeated CV — "
        "**below the untrained floor** $\\approx0.245$ **in mean**, with overlapping stability "
        "intervals (a partition-stability heuristic, not a significance test) — and a mirror slope "
        "of $\\approx-0.70$ instead of the required $-1$. **All three "
        "pre-registered gates fail.** The single-partition ordering $A>C$ *reverses* under "
        "repetition (the original notebook's lone \"win\" was a single partition near the top of A's "
        "spread), and the alpha sweep shows A decodes only under heavy regularization — consistent "
        "with a weak, collinear feature rather than a clean axis. The missingness lane $\\approx0.202$ "
        "has a mean close to A ($\\approx0.198$), flagging possible visibility confounding and "
        "corroborating the null.\n\n"
        "**Verdict: a robust informative null** — the world model did *not* learn bilateral "
        "symmetry as a decodable, sign-flipping axis. See `../neurips-laterality/docs/figures/fig3_e1_null.svg`. Next we ask "
        "whether the geometry can be **recovered by construction** — `nb_05c` (read-out) and "
        "`nb_05d` (encoder)."))
    return nb


# ======================================================================= nb_05c (E2)
def build_nb_05c():
    nb = new_notebook()
    c = nb.cells
    c.append(new_markdown_cell(
        "# nb_05c — Reflection-equivariant read-out, by construction (E2)\n\n"
        "**E1** established the informative null: the frozen S-JEPA does *not* encode the signed "
        "left–right axis as a decodable, sign-flipping quantity (learned $\\approx$ floor; mirror "
        "slope $\\approx-0.70$). **E2** asks the constructive question:\n\n"
        "> If we *impose* reflection-equivariance on the **read-out**, do we recover exact "
        "antisymmetry — and does the axis become more decodable from the *same* frozen tokens?\n\n"
        + RESPONSIBLE_USE))

    c.append(new_markdown_cell(
        "## 1. Frame averaging over $G=\\{I,M\\}$\n\n"
        "Let $A(x)$ be the per-pair laterality feature of the frozen encoder. Average over the "
        "order-2 group (Puny et al., 2022):\n\n"
        "$$ \\Phi(x) = A(x) - A(Mx)\\quad(\\text{antisymmetric}),\\qquad "
        "\\Psi(x) = A(x) + A(Mx)\\quad(\\text{symmetric}). $$\n\n"
        "Because $M^2=I$, $\\;\\Phi(Mx) = A(Mx)-A(x) = -\\Phi(x)$ **exactly**, and "
        "$\\Psi(Mx)=+\\Psi(x)$ **exactly**. A *linear* read-out $w^\\top\\Phi$ therefore satisfies "
        "$w^\\top\\Phi(Mx) = -\\,w^\\top\\Phi(x)$: **mirror slope $=-1$ for *any* weights $w$** — the "
        "antisymmetry is a property of the construction, not of training "
        "(`../neurips-laterality/docs/figures/fig4_construction.svg`).\n\n"
        "The target $y$ is itself exactly antisymmetric, so a **symmetric** read-out on $\\Psi$ "
        "*cannot* track its sign — a clean falsification control.\n\n"
        "**Isolating the cause.** To show any gain comes from the *constraint* and not from "
        "capacity or the extra mirror pass, we add two learnable heads on the frozen tokens:\n"
        "- `eq_mlp`: $s(x)=m(A)-m(A_{\\mathrm{mir}})$ with a **shared** MLP $m$ → exactly "
        "antisymmetric for any $m$;\n"
        "- `free_mlp`: an **untied** MLP on $[A;A_{\\mathrm{mir}}]$ with $\\ge$ the same capacity, "
        "seeing both passes but free to ignore the symmetry."))

    c.append(new_code_cell(RUN_PREAMBLE))
    c.append(new_code_cell(
        "# Re-run E2 end-to-end (regenerates idea9_equivariant_readout_result.json).\n"
        "run_experiment('work/experiments/e2_equivariant_readout.py')\n"
        "res = json.loads((ART / 'idea9_equivariant_readout_result.json').read_text())\n"
        "prim = res['primary_cohort']\n"
        "print('fingerprint', res['fingerprint'][:12], '| primary',\n"
        "      prim['n_sequences'], 'seq /', prim['n_sources'], 'sources')"))

    c.append(new_code_cell(
        "# ---- Table 2: read-out lanes on 626 primary (repeated-CV R2 + mirror slope)\n"
        "lanes, slopes, ci = prim['lanes'], prim['mirror_slopes'], prim['repeated_cv_ci95']\n"
        "def band(k):\n"
        "    if k in ci:\n"
        "        d = ci[k]; return f\"{d['mean']:.3f} [{d['ci95_lo']:.3f}, {d['ci95_hi']:.3f}]\"\n"
        "    return f\"{lanes[k]['r2']:.3f}\"\n"
        "print('E2 read-out — 626 primary')\n"
        "print(f\"  A  — free ridge            R2={band('A_free'):24s} slope={slopes['A_free']:+.4f}\")\n"
        "print(f\"  Phi— frame-avg, learned    R2={band('Phi_learned'):24s} slope={slopes['Phi_learned']:+.7f}\")\n"
        "print(f\"  Phi— frame-avg, untrained  R2={band('Phi_floor'):24s} slope={slopes['Phi_floor']:+.7f}\")\n"
        "print(f\"  Psi— symmetric part        R2={lanes['Psi_learned']['r2']:+.3f}                    slope={slopes['Psi_learned']:+.4f}\")\n"
        "print(f\"  B  — raw ceiling           R2={lanes['B_raw']['r2']:.3f}\")\n"
        "sc = ci['Phi_learned_sign_consistency']\n"
        "print(f\"\\n  Phi_learned sign consistency {sc['mean']:.3f} [{sc['ci95_lo']:.3f}, {sc['ci95_hi']:.3f}]\")"))

    c.append(new_code_cell(
        "# ---- Learnable-head controls: the gain is the CONSTRAINT, not capacity/extra pass\n"
        "h = prim['learnable_heads']\n"
        "print(f\"  eq_mlp   (shared m: s=m(A)-m(A_mir))  R2={h['eq_mlp']['r2']:+.3f}  slope={h['eq_mlp']['mirror_slope']:+.7f}\")\n"
        "print(f\"  free_mlp (untied, >=capacity)         R2={h['free_mlp']['r2']:+.3f}  slope={h['free_mlp']['mirror_slope']:+.3f}\")"))

    c.append(new_code_cell(
        "# ---- Assert the paper's E2 numbers trace to this freshly-written artifact\n"
        "Phi = ci['Phi_learned']; Afree = ci['A_free']; Pfloor = ci['Phi_floor']\n"
        "assert approx(Phi['mean'], 0.273, 0.01), Phi\n"
        "assert approx(Afree['mean'], 0.198, 0.01), Afree\n"
        "assert approx(slopes['Phi_learned'], -1.0, 1e-4), slopes['Phi_learned']\n"
        "assert approx(slopes['Psi_learned'], +1.0, 1e-4), slopes['Psi_learned']\n"
        "assert approx(lanes['Psi_learned']['r2'], 0.015, 0.01), lanes['Psi_learned']['r2']\n"
        "assert approx(h['free_mlp']['r2'], 0.047, 0.02), h['free_mlp']\n"
        "# (1) built-in beats free with DISJOINT stability intervals (partition-stability heuristic)\n"
        "assert Phi['ci95_lo'] > Afree['ci95_hi'], (Phi, Afree)\n"
        "# (2) learned edges the untrained floor but intervals OVERLAP -> learning benefit suggestive\n"
        "assert Phi['mean'] > Pfloor['mean'] and Phi['ci95_lo'] < Pfloor['ci95_hi'], (Phi, Pfloor)\n"
        "print('OK — E2 numbers match the paper; built-in beats free (disjoint intervals);')\n"
        "print('     the learning-vs-geometry gap is honestly suggestive (overlapping intervals).')"))

    c.append(new_markdown_cell(
        "## 2. Reading the result\n\n"
        "Two observations stand out. **(1) Built-in beats free**, with *disjoint* stability "
        "intervals ($\\Phi\\approx0.273$ vs $A\\approx0.198$; a partition-stability heuristic, not a "
        "significance test): imposing the geometry unlocks "
        "decodable structure the free read-out does not reach from the *same* frozen encoder "
        "($\\Phi$ additionally evaluates the mirrored pass $A(Mx)$). **(2) "
        "The controls are consistent with the antisymmetric constraint — not capacity or the extra "
        "pass — as the source of the gain (they do not isolate a causal decomposition).** The "
        "symmetric $\\Psi$ cannot "
        "predict an antisymmetric target ($\\approx0.015$, slope $+1$); the shared-map `eq_mlp` is "
        "exactly antisymmetric and a *linear* $\\Phi$ is already optimal (nonlinearity adds "
        "nothing); the untied `free_mlp` — $\\ge$ the same capacity, both passes — collapses to "
        "$\\approx0.047$ with slope $\\approx-0.43$.\n\n"
        "**Honesty.** $\\Phi_{\\text{learned}}\\approx0.273$ only *edges* $\\Phi_{\\text{floor}}"
        "\\approx0.219$ and their stability intervals **overlap**, so the *learning* benefit is "
        "suggestive, not "
        "decisive — frame-averaging a random encoder already reaches most of the absolute score — "
        "and $\\Phi$ stays far below the raw ceiling "
        "(sign consistency $\\approx0.57 < 0.75$). The recovered axis is real but partial, which "
        "motivates pushing symmetry into the **encoder** (`nb_05d`). See "
        "`../neurips-laterality/docs/figures/fig5_builtin_beats_emergent.svg`."))
    return nb


# ======================================================================= nb_05d (E3a + E3b)
def build_nb_05d():
    nb = new_notebook()
    c = nb.cells
    c.append(new_markdown_cell(
        "# nb_05d — Reflection-equivariant encoder: build-in vs augment (E3)\n\n"
        "E2 imposed the group action on the *read-out*. **E3** lifts it to the **encoder**, two "
        "ways:\n\n"
        "- **E3a — frame-averaged encoder (exact, zero-retrain).** Wrap the frozen encoder so the "
        "*whole latent field* transforms correctly under reflection.\n"
        "- **E3b — reflection augmentation (does it emerge?).** Retrain Stage-0 with a mirror "
        "augmentation and ask whether equivariance is *induced* (Benton et al., 2020).\n\n"
        + RESPONSIBLE_USE))

    c.append(new_markdown_cell(
        "## E3a. Frame-averaged encoder $E'(x)=\\tfrac12\\big(E(x)+\\sigma\\!\\cdot\\!E(Mx)\\big)$\n\n"
        "With $\\sigma$ the left/right **token** permutation, the wrapped encoder is **exactly "
        "token-level reflection-equivariant**, $E'(Mx)=\\sigma\\cdot E'(x)$, to machine precision "
        "and with **no retraining**. Consequently the laterality feature on $E'$ splits *exactly* "
        "into an antisymmetric block (the $\\ell-r$ channels; slope $-1$ for any nonzero read-out) "
        "and a symmetric block (the $\\ell+r$ channels). The antisymmetric block **corresponds "
        "to** E2's $\\Phi$: it is one half of $\\Phi$'s $\\ell-r$ sub-block (up to standardization), "
        "while $\\Phi$ additionally antisymmetrizes the $\\ell+r$ block — so E3a and E2 are two "
        "views of the same frame-averaging construction, not identical features.\n\n"
        "**Honest boundary.** A *free* ridge on the *full* $E'$ feature still gives slope "
        "$\\approx-0.77$: encoder equivariance alone does **not** force an antisymmetric *decoder* "
        "— the read-out must still select the antisymmetric part. That boundary is exactly what "
        "motivates E3b (try to *train* the symmetry in)."))

    c.append(new_code_cell(RUN_PREAMBLE))
    c.append(new_code_cell(
        "# Re-run E3a end-to-end (writes the top-level of idea9_equivariant_encoder_result.json).\n"
        "run_experiment('work/experiments/e3a_frame_averaging_wrapper.py')\n"
        "res = json.loads((ART / 'idea9_equivariant_encoder_result.json').read_text())\n"
        "prim = res['primary_cohort']\n"
        "ex = prim['exactness']; lanes = prim['lanes']; slopes = prim['mirror_slopes']; ci = prim['repeated_cv_ci95']\n"
        "print('E3a exactness (all should be 0.0 to machine precision):')\n"
        "for k, v in ex.items():\n"
        "    print(f'  {k:38s} {v:.2e}')\n"
        "def band(k):\n"
        "    d = ci[k]; return f\"{d['mean']:.3f} [{d['ci95_lo']:.3f}, {d['ci95_hi']:.3f}]\"\n"
        "print()\n"
        "print(f\"  A'_diff learned (antisym block)  R2={band('Aprime_diff_learned'):24s} slope={slopes['Aprime_diff_learned']:+.4f}\")\n"
        "print(f\"  A'_diff floor  (untrained)       R2={band('Aprime_diff_floor'):24s} slope={slopes['Aprime_diff_floor']:+.4f}\")\n"
        "print(f\"  A' free ridge (full feature)                                    slope={slopes['Aprime_free']:+.4f}  <- NOT -1\")"))

    c.append(new_code_cell(
        "# ---- Assert E3a exactness + that E3a's antisym block reproduces E2's Phi\n"
        "assert ex['token_equivariance_max_abs_err'] < 1e-4, ex\n"
        "assert ex['diff_block_antisymmetry_max_abs_err'] < 1e-4, ex\n"
        "assert ex['sum_block_symmetry_max_abs_err'] < 1e-4, ex\n"
        "assert approx(slopes['Aprime_diff_learned'], -1.0, 1e-3), slopes['Aprime_diff_learned']\n"
        "assert approx(ci['Aprime_diff_learned']['mean'], 0.253, 0.02), ci['Aprime_diff_learned']\n"
        "assert slopes['Aprime_free'] > -1.0 + 0.05, 'free ridge on full E-prime must NOT be -1'\n"
        "print('OK — E3a: exact token-level reflection-equivariance, zero retrain.')"))

    c.append(new_markdown_cell(
        "## E3b. Does reflection augmentation *induce* it? (a single-seed negative)\n\n"
        "We retrain **Stage-0** (the `normal`-annotated rows only, 270 sequences / 29 sources, "
        "300 epochs, single seed on MPS) with **one switch**: sample-level consistent reflection augmentation **on** "
        "($p=0.5$) vs **off** ($p=0.0$), identical trainer/seed/hardware. `canonical` is the "
        "original Stage-0 checkpoint, a fidelity cross-check.\n\n"
        "> **Provenance.** The ~15-minute-per-arm MPS training lives in "
        "`work/experiments/e3b_reflection_augmented_retrain.py` and produced the checkpoints "
        "`sjepa_normal_e3b_flip0p00.pt` / `…flip0p50.pt` (final JEPA $0.720$ / $0.807$). The cell "
        "below **loads** those checkpoints and probes them — it does **not** retrain — so the "
        "notebook stays fast and deterministic while every probe number is reproduced from "
        "scratch.\n\n"
        "**Primary metric** = the *free-readout mirror slope*: does augmentation move it toward "
        "$-1$? And does the signed axis clear the untrained floor?"))

    c.append(new_code_cell(
        "# Probe the three Stage-0 encoders + floor (loads pre-trained checkpoints; no retraining).\n"
        "run_experiment('work/experiments/e3b_probe_and_merge.py')\n"
        "res = json.loads((ART / 'idea9_equivariant_encoder_result.json').read_text())\n"
        "rb = res['retrain']; enc = rb['primary_cohort']['encoders']; tr = rb['training']\n"
        "print('training: arm_off finalJEPA', tr['arm_off']['final_jepa'],\n"
        "      '| arm_on finalJEPA', tr['arm_on']['final_jepa'])\n"
        "print('\\nE3b — 270 normal-annotated (transductive)   A_free R2 [stability interval]   free-readout slope')\n"
        "for name in ('arm_on', 'arm_off', 'canonical', 'floor_untrained'):\n"
        "    e = enc[name]; d = e['A_free_r2_ci95']\n"
        "    print(f\"  {name:16s} {d['mean']:+.3f} [{d['ci95_lo']:+.3f}, {d['ci95_hi']:+.3f}]   \"\n"
        "          f\"slope={e['A_free_mirror_slope']:+.3f}\")"))

    c.append(new_code_cell(
        "# ---- Assert the paper's E3b single-seed negative: augmentation does NOT buy equivariance\n"
        "on  = enc['arm_on']['A_free_r2_ci95']; floor = enc['floor_untrained']['A_free_r2_ci95']\n"
        "assert approx(on['mean'], 0.062, 0.02), on\n"
        "assert approx(floor['mean'], 0.078, 0.02), floor\n"
        "# augmented axis does NOT clear the untrained floor\n"
        "assert on['mean'] <= floor['ci95_hi'], (on, floor)\n"
        "# the free-readout slope does NOT move toward -1; the UNTRAINED floor is closest\n"
        "s_on = enc['arm_on']['A_free_mirror_slope']; s_floor = enc['floor_untrained']['A_free_mirror_slope']\n"
        "assert s_floor < s_on, (s_on, s_floor)  # floor slope (-0.82) closer to -1 than arm_on (-0.51)\n"
        "print('OK — E3b: single-seed negative. Augmentation changes the loss, not the geometry;')\n"
        "print('     free-readout slope does not track equivariance (untrained floor is closest to -1).')"))

    c.append(new_markdown_cell(
        "## Synthesis — the emergent-vs-built-in ladder\n\n"
        "The four experiments form a $2\\times2$ grid (`../neurips-laterality/docs/figures/fig6_ladder.svg`): symmetry in the "
        "*read-out* or the *encoder*, *hoped for* or *built in*.\n\n"
        "|              | Emergent (hope) | Built-in (construct) |\n"
        "|--------------|-----------------|----------------------|\n"
        "| **Read-out** | E1: slope $-0.70$, learned $\\approx$ floor (intervals overlap), gates fail | E2: slope $-1.0000$, $0.273>0.198$ (disjoint intervals) |\n"
        "| **Encoder**  | E3b: slope not toward $-1$, axis $\\le$ floor | E3a: token error $0.0$, exact split |\n\n"
        "In the audited system (a single checkpoint; a single-seed augmentation arm), standard "
        "**and** reflection-augmented self-supervision leave bilateral symmetry un-learned; frame "
        "averaging the read-out or encoder recovers it to machine precision with zero or minimal "
        "retraining — and that constructive guarantee is general, not tied to this checkpoint. For "
        "geometry-aware world models of articulated bodies, symmetry is cheap to *impose* and — at "
        "least here — not reliably obtained by *hoping* it emerges: **build the geometry in; "
        "don't hope it emerges.**\n\n"
        "*(E3b is a single-seed, `normal`-annotated proof-of-concept; see `../neurips-laterality/docs/"
        "paper.md` §9 for the full limitations.)*"))
    return nb


def main():
    targets = {
        "nb_05a_signed_laterality_probe.ipynb": build_nb_05a(),
        "nb_05c_reflection_equivariant_readout.ipynb": build_nb_05c(),
        "nb_05d_reflection_equivariant_encoder.ipynb": build_nb_05d(),
    }
    for name, nb in targets.items():
        nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
        nb.metadata["language_info"] = {"name": "python"}
        path = NOTEBOOK_DIR / name
        nbf.write(nb, str(path))
        print(f"wrote {path}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
