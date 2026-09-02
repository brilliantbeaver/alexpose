"""Builder for nb_09c_futures_and_reach.ipynb (Idea 9, futures + external reach).

Run:  python3 _build_nb_09c.py
Emits the notebook at the gavd root (parents[3]) beside nb_09a/nb_09b. Pure numpy/sklearn/matplotlib:
no torch, no checkpoint, no pose cache, so it runs anywhere in seconds. It is the 05b analogue for
Idea 9 and does three things 09a/09b cannot:

  1. Possible-futures simulator. Deterministically synthesizes the canonical futures of the Idea-9
     endpoint (does the antisymmetry-CONSTRAINED head beat the binding bar max(D, C)), scores each
     against the SAME hardened gates 09a uses, and draws their expected-shape panels. A pre-registration
     aid, not a result.
  2. Decision table. Maps every future to the exact verdict language Idea 9 pre-registers, including the
     two failure traps (nuisance control fires; the win does not beat the capacity-matched control).
  3. External multi-view reach scaffold. An honestly-stubbed, runnable loader interface for the
     non-clinical reflection-stability reach arm (CASIA-B / OU-MVLP-Pose), clearly marked reach-tier.

The hardened gates here MATCH nb_09a's verdict cell (binding bar max(D,C), beat-floor, capacity-matched
attribution control, permutation null, anatomically invariant nuisance control, y-quality gate, exact
wiring check), and SUPERSEDE the proposal-level gates in the idea-9 README/METHODOLOGY worked example
(see IMPLEMENTATION.md).
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[3] / "nb_09c_futures_and_reach.ipynb"

_CELL_N = [0]


def _next_id(prefix):
    _CELL_N[0] += 1
    return f"{prefix}{_CELL_N[0]:02d}"


def md(text):
    return {"cell_type": "markdown", "id": _next_id("md"), "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "id": _next_id("code"), "metadata": {},
            "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


CELLS = []

CELLS.append(md(r"""
# Notebook 09c: The possible futures of Idea 9, and the external reach arm

Notebook 09a runs Arm 1 (the zero-retrain antisymmetric readout) on the frozen `ea59fea0` checkpoint and
emits one verdict; nb_09b runs Arm 2 (the equivariance-coupled retrain) as an ablation ladder. This
notebook does the two things neither of those can:

1. **Possible-futures simulator.** Before the real numbers land it is worth knowing exactly what each
   outcome would look like and what claim it would license. We deterministically synthesize the canonical
   futures of the Idea-9 endpoint, score each against the SAME hardened gates nb_09a uses, and draw their
   expected-shape panels. This is a pre-registration aid, not a result.
2. **External multi-view reach scaffold.** The clinical claim on this cohort (n=18 source videos) is
   internal-validity only. The reflection-equivariance property itself has a NON-clinical reach test on
   public multi-view pose cohorts (CASIA-B, OU-MVLP-Pose): does the signed axis stay stable across camera
   views, and does a genuine left/right camera swap flip it. We lay down a runnable, honestly-stubbed
   loader interface for that arm and state precisely why it is reach-tier and non-clinical.

The gates here are copied to match nb_09a exactly: beat the binding bar `max(D_standard, C_floor)` by at
least 0.05 R-squared, beat the untrained floor `C` by at least 0.05, beat a capacity-matched control
`Ac` (the same head that adds the symmetric left-plus-right path) by at least 0.05, clear a source-label
permutation null at p < 0.05, keep the anatomically invariant nuisance control below 0.05 in absolute
value, and clear the y-quality gate. The exact wiring-swap slope -1 is a separate by-construction check
(nb_09a section 5), not a future. See `notes/ideas-claude/_shared_facts.md` for the single source of
truth. Folder labels are dataset annotations, not diagnoses; all results on this cohort are transductive.
"""))

# ------------------------------------------------------------------ 0. Gates
CELLS.append(md(r"""
## 0. The pre-registered gates (frozen, matched to nb_09a)

These constants are the entire decision rule for the primary endpoint, and they match nb_09a's verdict
cell so a future cannot be scored on a moved goalpost. Three are new relative to Idea 05 and are the
load-bearing hardening: the binding bar is the LARGER of the standard-encoder comparator D and the
untrained floor C (beating only the weaker one would overclaim); a source-label permutation null replaces
a fixed sign-consistency threshold (at n=18 sources with per-condition counts as low as 1, a fixed
per-source fraction is not meaningful); and a capacity-matched control Ac (the same head that ADDS the
symmetric left-plus-right path) must be beaten by the same 0.05 margin, so a win is attributable to the
antisymmetry CONSTRAINT rather than to the head's nonlinearity, width, or pair information.
"""))
CELLS.append(code(r"""
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42


def idea9_dir():
    '''Resolve the 09 proposal folder robustly regardless of the kernel cwd.'''
    for base in [Path.cwd(), *Path.cwd().parents]:
        cand = base / "notes" / "ideas-claude" / "09-reflection-equivariant-symmetry-axis"
        if cand.exists():
            return cand
        cand = base / "09-reflection-equivariant-symmetry-axis"
        if cand.exists():
            return cand
    return Path.cwd()


IDEA9_DIR = idea9_dir()
print(f"IDEA9_DIR: {IDEA9_DIR}")

# ---- pre-registered gates (match nb_09a section 10) ----
FLOOR_MARGIN = 0.05          # A' must beat the binding bar max(D,C), the floor C, AND Ac by >= this R2.
PERM_ALPHA = 0.05            # A' must clear its source-label permutation null at p < this.
NUISANCE_ABS = 0.05          # |E_pooled R2| must be < this ABSOLUTELY (no OR-clause).
Y_BETWEEN_MIN = 0.30         # y between-source variance fraction must be >= this to trust R2.
MIRROR_BAND = (-1.25, -0.8)  # measured anatomical-mirror slope inside this band counts as "flips".

print("Pre-registered decision rule (matches nb_09a):")
print(f"  beat binding bar max(D,C) by     >= {FLOOR_MARGIN} R2")
print(f"  beat untrained floor C by        >= {FLOOR_MARGIN} R2")
print(f"  beat capacity-matched Ac by      >= {FLOOR_MARGIN} R2  (attribution to antisymmetry)")
print(f"  A' permutation null              p  < {PERM_ALPHA}")
print(f"  |E_pooled nuisance R2|           <  {NUISANCE_ABS}  (absolute, anatomically invariant)")
print(f"  y between-source variance frac   >= {Y_BETWEEN_MIN}")
print(f"  anatomical-mirror slope band      = {MIRROR_BAND}  (measured, NOT the wiring -1)")
"""))

# ------------------------------------------------------------------ 1. Scoring function
CELLS.append(md(r"""
## 1. One scoring function, shared by every future

`score_future` takes the five gated lane R-squared values (A' antisymmetric head, Ac capacity-matched
control, C untrained floor, D standard `ea59fea0`, E anatomically invariant nuisance), the A'-permutation
p-value, the C-permutation p-value, the y between-source fraction, and the measured anatomical-mirror
slope, and returns the exact verdict language Idea 9 pre-registers. The primary verdict is a positive ONLY
if every gate passes together; otherwise it is an informative null. Two traps are explicit: the nuisance
control firing (E) withdraws the signed claim, and a win that does NOT beat the capacity-matched control
(Ac) is not attributable to the antisymmetry constraint. The C-permutation is REPORTED as floor
characterization only (a strong random floor just raises the binding bar); it is NOT a claim-withdrawing
gate, exactly as in nb_09a. The raw-coordinate ceiling B is descriptive only and is NOT a gate (it is
near-circular).
"""))
CELLS.append(code(r"""
def score_future(a_r2, ac_r2, c_r2, d_r2, e_r2, a_perm_p, c_perm_p, y_between_frac, mirror_slope):
    binding_bar = max(d_r2, c_r2)
    beats_binding = (a_r2 - binding_bar) >= FLOOR_MARGIN
    beats_floor = (a_r2 - c_r2) >= FLOOR_MARGIN
    # Attribution gate: A' must beat the capacity-matched control Ac (same head, adds the symmetric
    # left-plus-right path). A gap here is attributable to the antisymmetry CONSTRAINT, not the head's
    # nonlinearity, width, or pair information (which Ac holds identical).
    beats_capacity_matched = (a_r2 - ac_r2) >= FLOOR_MARGIN
    perm_ok = a_perm_p < PERM_ALPHA
    nuisance_ok = abs(e_r2) < NUISANCE_ABS
    y_ok = y_between_frac >= Y_BETWEEN_MIN
    # C's own permutation is REPORTED but is NOT a gate: a random floor that preserves genuine laterality
    # is a STRONG floor (it raises the binding bar), not a broken null. This matches nb_09a exactly.
    c_null_significant = c_perm_p < PERM_ALPHA
    primary_positive = bool(beats_binding and beats_floor and beats_capacity_matched and perm_ok
                            and nuisance_ok and y_ok)
    flips = bool(MIRROR_BAND[0] <= mirror_slope <= MIRROR_BAND[1])
    if not nuisance_ok:
        primary = "ARTIFACT (nuisance control fired; signed claim withdrawn)"
    elif not y_ok:
        primary = "UNINTERPRETABLE (y is noise-dominated at the source level)"
    elif (beats_binding and beats_floor and perm_ok) and not beats_capacity_matched:
        primary = "NOT ATTRIBUTABLE TO ANTISYMMETRY (does not beat the capacity-matched control)"
    elif primary_positive:
        primary = "ANTISYMMETRY BEATS BINDING BAR AND CAPACITY-MATCHED CONTROL"
    else:
        primary = "INFORMATIVE NULL (constrained head does not beat max(D,C) above the floor)"
    mirror = ("FLIPS (measured slope antisymmetric through the encoder)" if flips
              else "DOES NOT FLIP (measured slope not antisymmetric)")
    return {
        "beats_binding_bar": beats_binding, "beats_floor": beats_floor,
        "beats_capacity_matched": beats_capacity_matched,
        "perm_ok": perm_ok, "nuisance_ok": nuisance_ok, "y_ok": y_ok,
        "c_null_significant": c_null_significant, "binding_bar": float(binding_bar),
        "primary_positive": primary_positive, "flips": flips,
        "primary_verdict": primary, "mirror_verdict": mirror,
    }


# sanity: a clean positive (A'=0.42 beats binding bar max(D=-0.02,C=0.05)=0.05 AND Ac=0.10, perm passes).
_demo = score_future(0.42, 0.10, 0.05, -0.02, 0.01, 0.004, 0.60, 0.55, -1.0)
assert _demo["primary_positive"] and _demo["flips"] and _demo["beats_capacity_matched"], _demo
# sanity: a clean positive is UNAFFECTED by a significant C-null (C is descriptive, not a gate).
_demo_strongC = score_future(0.42, 0.10, 0.05, -0.02, 0.01, 0.004, 0.01, 0.55, -1.0)
assert _demo_strongC["primary_positive"] and _demo_strongC["c_null_significant"], _demo_strongC
# sanity: nuisance firing withdraws the claim even with a strong A'.
_art = score_future(0.47, 0.10, 0.06, -0.02, 0.46, 0.004, 0.60, 0.55, -0.9)
assert (not _art["primary_positive"]) and (not _art["nuisance_ok"]), _art
# sanity: a strong A' that does NOT beat the capacity-matched control Ac is not attributable (trap F5).
_notattr = score_future(0.30, 0.28, 0.05, -0.02, 0.01, 0.02, 0.60, 0.50, -0.7)
assert (not _notattr["primary_positive"]) and (not _notattr["beats_capacity_matched"]), _notattr
print("scoring self-check OK (clean positive passes; nuisance-fire and not-attributable traps withhold the "
      "claim; a significant C-null does NOT withdraw a clean positive).")
"""))

# ------------------------------------------------------------------ 2. The futures
CELLS.append(md(r"""
## 2. The canonical possible futures

Each future is a named, plausible state of the world with the lane numbers it would produce. They are
illustrative expected shapes, not measurements, and that includes the "informative null" future.

CORRECTION, and it matters. F3's lane values (A -0.10, Ac -0.05, C +0.147, D -0.187, E -0.014, mirror
slope -0.34) were originally sketched from the SUPERSEDED `d0acc262` bundle and were described here as
"the actual Idea 05 result". They are NOT. The authoritative Idea 05 result, on checkpoint `ea59fea0`,
is A learned -0.602, C floor -0.156, D pooled -0.131, B raw null 1.000, measured anatomical-mirror
slope -0.741, verdict INFORMATIVE NULL. F3's numbers are deliberately left as they are, because they
are decision-rule INPUTS chosen to exercise the null branch: substituting the authoritative values
would move the binding bar `max(D, C)` from +0.147 to -0.156, at which point A = -0.10 would CLEAR the
bar and F3 would stop being a null at all. So read F3 as a scenario, never as a measurement.

F3 was also the predicted Arm-1 outcome, on the reasoning that the constrained head, read off the SAME
frozen features, is unlikely to beat a binding bar the unconstrained probe already failed to beat. That
prediction is SUPERSEDED by the real run: Arm 1's actual verdict was
`ARTIFACT (side-agnostic nuisance control fired)`, because the side-blind lane E outscored the
antisymmetric treatment. An artifact is a WITHDRAWAL of the claim rather than an answer to it, which is
a weaker epistemic state than the null F3 anticipated.

- **F1 head-beats-bar positive.** The antisymmetry-constrained head decodes the signed axis above the
  binding bar `max(D, C)`, beats the capacity-matched control `Ac`, and clears its permutation null, and
  the measured anatomical mirror flips it. For Arm 1 this would say the constraint EXTRACTS a signed axis
  the unconstrained probe missed; for Arm 2 it would say the equivariance loss BUILT one. The strong result.
- **F2 decodable-but-non-flipping.** The head beats the bar and `Ac` and clears the null, but the measured
  anatomical-mirror slope is not in the flip band. Licenses the decodability claim, withholds the
  measured-equivariance claim (the exact wiring -1 still holds by construction; that is separate).
- **F3 informative null (scenario; see the correction above).** The constrained head does NOT clear the binding bar above
  the floor. Overturns the hope that antisymmetry-by-construction alone rescues a signed axis on the
  frozen encoder; a clean publishable negative and the expected Arm-1 outcome. This anchor also shows a
  significant C-permutation (the untrained floor is strong): that is REPORTED but does not change the
  verdict, because a strong floor is already handled by the binding bar.
- **F4 artifact.** The head looks strong BUT the anatomically invariant nuisance control (E) also
  "recovers" the axis, which is impossible for a genuinely signed quantity. The signed claim is withdrawn.
- **F5 not-attributable trap.** The head beats the binding bar and clears its null, BUT it does not beat
  the capacity-matched control `Ac` by the margin, so the win rides on the head's generic capacity
  (nonlinearity, width, pair information) rather than the antisymmetry CONSTRAINT. We report it as not
  attributable to antisymmetry rather than claiming a constraint win.
"""))
CELLS.append(code(r"""
FUTURES = [
    {"key": "F1_head_beats_bar", "title": "F1 head beats bar (positive)",
     "a_r2": 0.42, "ac_r2": 0.10, "c_r2": 0.05, "d_r2": -0.02, "e_r2": 0.01,
     "a_perm_p": 0.004, "c_perm_p": 0.60, "y_between_frac": 0.55, "mirror_slope": -1.02},
    {"key": "F2_decodable_non_flip", "title": "F2 decodable but non-flipping",
     "a_r2": 0.40, "ac_r2": 0.09, "c_r2": 0.05, "d_r2": -0.02, "e_r2": 0.01,
     "a_perm_p": 0.01, "c_perm_p": 0.55, "y_between_frac": 0.52, "mirror_slope": -0.30},
    {"key": "F3_informative_null", "title": "F3 informative null (Idea-05 anchor)",
     "a_r2": -0.10, "ac_r2": -0.05, "c_r2": 0.147, "d_r2": -0.187, "e_r2": -0.014,
     "a_perm_p": 0.55, "c_perm_p": 0.01, "y_between_frac": 0.48, "mirror_slope": -0.34},
    {"key": "F4_artifact", "title": "F4 artifact (nuisance control fires)",
     "a_r2": 0.47, "ac_r2": 0.11, "c_r2": 0.06, "d_r2": -0.02, "e_r2": 0.46,
     "a_perm_p": 0.004, "c_perm_p": 0.55, "y_between_frac": 0.52, "mirror_slope": -0.9},
    {"key": "F5_not_attributable", "title": "F5 not attributable (Ac not beaten)",
     "a_r2": 0.30, "ac_r2": 0.28, "c_r2": 0.05, "d_r2": -0.02, "e_r2": 0.01,
     "a_perm_p": 0.02, "c_perm_p": 0.60, "y_between_frac": 0.50, "mirror_slope": -0.7},
]

rows = []
for f in FUTURES:
    v = score_future(f["a_r2"], f["ac_r2"], f["c_r2"], f["d_r2"], f["e_r2"],
                     f["a_perm_p"], f["c_perm_p"], f["y_between_frac"], f["mirror_slope"])
    rows.append({
        "future": f["title"], "A'_r2": f["a_r2"], "Ac_r2": f["ac_r2"], "C_r2": f["c_r2"],
        "D_r2": f["d_r2"], "E_r2": f["e_r2"], "binding_bar": v["binding_bar"],
        "A'_perm_p": f["a_perm_p"], "C_perm_p": f["c_perm_p"],
        "y_between": f["y_between_frac"], "mirror_slope": f["mirror_slope"],
        "primary": v["primary_verdict"], "mirror": v["mirror_verdict"],
        "nuisance_ok": v["nuisance_ok"], "beats_capacity_matched": v["beats_capacity_matched"],
        "c_null_significant": v["c_null_significant"],
    })
futures_table = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", 60)
print(futures_table.to_string(index=False))
"""))

# ------------------------------------------------------------------ 3. Decision table
CELLS.append(md(r"""
## 3. The decision table: future -> licensed claim

This is the single artifact a reviewer should read first. It states, for every future, exactly which
claim is licensed and which is withheld. F4 (nuisance fires) and F5 (does not beat the capacity-matched
control) are the two traps: in both, a strong-looking Lane A' is NOT enough on its own.
"""))
CELLS.append(code(r"""
CLAIMS = {
    "F1 head beats bar (positive)": (
        "LICENSED: the antisymmetry-constrained head decodes the signed axis above the binding bar "
        "max(standard-ea59fea0, untrained-floor), beats the capacity-matched control Ac, AND clears its "
        "permutation null, and the MEASURED anatomical mirror flips it. Arm 1: the constraint extracts a "
        "signed axis the unconstrained probe missed. Arm 2: the equivariance loss built one. (The exact "
        "wiring -1 is a separate by-construction check, always true.)"),
    "F2 decodable but non-flipping": (
        "LICENSED: signed axis is decodable above the binding bar and the capacity-matched control. "
        "WITHHELD: the MEASURED anatomical equivariance; the encoding decodes side but the through-encoder "
        "mirror slope is not in the flip band. The exact wiring-swap -1 still holds by construction."),
    "F3 informative null (Idea-05 anchor)": (
        "LICENSED (negative): antisymmetry-by-construction alone does NOT lift the frozen encoder above the "
        "binding bar. Matches Idea 05's measured result and is the expected Arm-1 outcome; a clean "
        "publishable negative for the representation audit. The significant C-permutation here is REPORTED "
        "as floor characterization (the untrained floor is strong, which just raises the binding bar) and "
        "does not change the verdict. Motivates Arm 2 (retrain), which is the only arm that can change the "
        "encoder."),
    "F4 artifact (nuisance control fires)": (
        "WITHHELD: the signed claim is withdrawn. The anatomically invariant nuisance control recovered the "
        "'axis', which a genuinely signed quantity cannot allow, so Lane A' reflects a magnitude/acquisition "
        "artifact, not a signed quantity."),
    "F5 not attributable (Ac not beaten)": (
        "WITHHELD (not attributable): Lane A' beats the binding bar and clears its null, but it does NOT "
        "beat the capacity-matched control Ac by the margin. The win rides on the head's generic capacity "
        "(nonlinearity, width, pair information), which Ac holds identical, rather than on the antisymmetry "
        "CONSTRAINT. Report as not attributable to antisymmetry rather than a constraint win."),
}
for _, r in futures_table.iterrows():
    print(f"### {r['future']}")
    print(f"  primary            : {r['primary']}")
    print(f"  mirror             : {r['mirror']}")
    print(f"  nuisance ok        : {r['nuisance_ok']}")
    print(f"  beats capacity ctrl: {r['beats_capacity_matched']}")
    print(f"  C-null significant : {r['c_null_significant']}  (floor characterization only)")
    print(f"  claim              : {CLAIMS[r['future']]}")
    print()
"""))

# ------------------------------------------------------------------ 4. Expected-shape panels
CELLS.append(md(r"""
## 4. Expected-shape panels for each future

For each future we synthesize a point cloud that reproduces its Lane A' R-squared and its measured
anatomical-mirror slope, so the reader sees the SHAPE each outcome makes on the two decisive plots
(decodability scatter and mirror scatter). These are simulated to the target statistics; they are not
data. The mirror panel also draws the exact wiring-swap line y = -x, which is true by construction for
every future and is therefore NOT what distinguishes them; the measured cloud is.
"""))
CELLS.append(code(r"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def simulate_scatter(r2, n=40, seed=0):
    '''Return (y_true, y_pred) with approximately the requested held-out R2.'''
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1, n)
    if r2 <= 0:
        pred = rng.normal(0, 1, n)
    else:
        noise_var = (1 - r2) / max(r2, 1e-6)
        pred = y + rng.normal(0, np.sqrt(noise_var), n)
    return y, pred


def simulate_mirror(slope, n=40, seed=0):
    rng = np.random.default_rng(seed)
    orig = rng.normal(0, 1, n)
    mir = slope * orig + rng.normal(0, 0.12, n)
    return orig, mir


ncol = len(FUTURES)
fig, axes = plt.subplots(2, ncol, figsize=(3.6 * ncol, 8))
for col, f in enumerate(FUTURES):
    seed = RANDOM_SEED + col
    yt, yp = simulate_scatter(f["a_r2"], seed=seed)
    ax = axes[0, col]
    ax.scatter(yt, yp, s=24, c="#e07a4b", edgecolors="#a44c26", linewidths=0.5)
    lim = 3.2
    ax.plot([-lim, lim], [-lim, lim], "--", color="#5f9e7e", linewidth=1.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    bar = max(f["d_r2"], f["c_r2"])
    ax.set_title(f"{f['title']}\nA' R2~{f['a_r2']:.2f}  bar max(D,C)~{bar:.2f}", fontsize=8)
    if col == 0:
        ax.set_ylabel("decoded signed scalar")
    ax.set_xlabel("ground-truth signed target")
    orig, mir = simulate_mirror(f["mirror_slope"], seed=seed + 100)
    ax2 = axes[1, col]
    mlim = 3.2
    xs = np.linspace(-mlim, mlim, 40)
    ax2.plot(xs, -xs, "-", color="#2f6f99", lw=1.6, label="wiring -1 (exact)")
    ax2.scatter(orig, mir, s=24, c="#e07a4b", edgecolors="#a44c26", linewidths=0.5, label="measured")
    ax2.axhline(0, color="#c4cdd8", lw=0.8); ax2.axvline(0, color="#c4cdd8", lw=0.8)
    ax2.set_xlim(-mlim, mlim); ax2.set_ylim(-mlim, mlim)
    flips = "FLIPS" if MIRROR_BAND[0] <= f["mirror_slope"] <= MIRROR_BAND[1] else "no flip"
    ax2.set_title(f"measured slope~{f['mirror_slope']:+.2f} ({flips})", fontsize=8)
    if col == 0:
        ax2.set_ylabel("decoded on mirrored input")
        ax2.legend(loc="upper right", fontsize=6)
    ax2.set_xlabel("decoded on original input")

fig.suptitle("Idea 9 possible futures (simulated expected shapes, NOT data). "
             "Blue line: exact wiring -1 (always true). Orange: measured anatomical mirror.", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
OUT = IDEA9_DIR / "images"
try:
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "idea9_possible_futures.png"
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")
except Exception as exc:
    print(f"(figure not saved to images/: {exc})")
plt.show()
"""))

# ------------------------------------------------------------------ 5. External reach scaffold
CELLS.append(md(r"""
## 5. External multi-view reach arm (non-clinical scaffold)

The clinical claim on this cohort is internal-validity only: n=18 canonical source videos, transductive
encoder, monocular capture. A simultaneously clinical + skeleton + participant-disjoint public cohort does
not exist, so there is NO honest skeleton-level clinical transfer test. What DOES exist is a non-clinical
reach test for the reflection-equivariance property itself, on public multi-view pose cohorts:

- **CASIA-B** (Yu, Tan, Tan 2006): 124 subjects, 11 camera views (0 to 180 degrees). Non-clinical gait.
- **OU-MVLP-Pose** (Takemura et al. 2018): about 10,000 subjects, multiple views, pose keypoints released.

The reach question is symmetry-specific and view-specific, not clinical:

1. **View stability.** Is the signed axis decoded from one view consistent with the same subject decoded
   from a nearby view, i.e. a property of the gait rather than the camera?
2. **Genuine mirror.** A true left-versus-right camera swap (a real physical reflection, not a synthetic
   x-negation) should flip the signed axis. CASIA-B's symmetric view angles around 90 degrees give a
   real-world analogue of the anatomical mirror, and let us MEASURE the flip slope on real reflections.

This cell defines the loader interface and the two metrics, then runs them on a small synthetic
multi-view fixture so the scaffold is exercised. Wiring real downloads is deliberately left as a marked
TODO: these are large licensed datasets and this pass does not fetch them.
"""))
CELLS.append(code(r"""
from dataclasses import dataclass


@dataclass
class MultiViewClip:
    subject_id: str
    view_deg: float
    coords: np.ndarray       # [T, 33, 3] normalized pose
    cohort: str = "synthetic"


def load_casia_b(root=None):
    '''TODO(real): parse a CASIA-B pose export at `root` into a MultiViewClip list.
    Not wired in this pass (large licensed dataset). Returns [] so the scaffold degrades cleanly.'''
    if root is None:
        return []
    raise NotImplementedError("Real CASIA-B loading is a marked TODO; provide a parser at the call site.")


def load_ou_mvlp_pose(root=None):
    '''TODO(real): parse OU-MVLP-Pose keypoints at `root`. Not wired in this pass.'''
    if root is None:
        return []
    raise NotImplementedError("Real OU-MVLP-Pose loading is a marked TODO.")


LEFT_RIGHT_PAIRS = [(11, 12), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]


def signed_axis(coords):
    total = 0.0
    for li, ri in LEFT_RIGHT_PAIRS:
        total += coords[:, li, :].std(axis=0).sum() - coords[:, ri, :].std(axis=0).sum()
    return float(total)


def synthetic_multiview_fixture(n_subjects=6, views=(54.0, 90.0, 126.0), frames=40, seed=RANDOM_SEED):
    '''Tiny multi-view fixture: each subject has a fixed signed lean visible from every view; the view
    rotates the pose in the x-z plane so the signed x-excursion is genuinely view-dependent.'''
    rng = np.random.default_rng(seed)
    clips = []
    for s in range(n_subjects):
        lean = 1.0 if s % 2 == 0 else -1.0
        phase = np.linspace(0, 4 * np.pi, frames, endpoint=False)
        base = np.zeros((frames, 33, 3), dtype=np.float32)
        for j in range(33):
            base[:, j, :] = rng.normal(0, 0.04, 3)[None, :] + 0.03 * np.sin(phase + j)[:, None]
        for li, ri in LEFT_RIGHT_PAIRS:
            base[:, li, 0] += 0.05 * lean * np.sin(phase)
            base[:, ri, 0] -= 0.05 * lean * np.sin(phase)
        for v in views:
            theta = np.deg2rad(v - 90.0)
            rot = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
            coords = base @ rot.T
            clips.append(MultiViewClip(subject_id=f"subj{s:02d}", view_deg=float(v), coords=coords.astype(np.float32)))
    return clips


def genuine_mirror_view(coords):
    '''A real left/right reflection: negate the x-axis and swap the six L/R pairs (as a camera on the
    opposite side would see). Used to MEASURE the flip slope on real reflections when a cohort is wired.'''
    out = coords.copy()
    out[:, :, 0] = -out[:, :, 0]
    for li, ri in LEFT_RIGHT_PAIRS:
        out[:, [li, ri], :] = out[:, [ri, li], :]
    return out


clips = synthetic_multiview_fixture()
print(f"external reach scaffold exercised on {len(clips)} synthetic multi-view clips "
      f"({len({c.subject_id for c in clips})} subjects x {len({c.view_deg for c in clips})} views).")

df = pd.DataFrame([{"subject": c.subject_id, "view": c.view_deg, "axis": signed_axis(c.coords)} for c in clips])
wide = df.pivot(index="subject", columns="view", values="axis")
views_sorted = sorted(df["view"].unique())
stability = float("nan")
if len(views_sorted) >= 2:
    v_lo, v_hi = views_sorted[0], views_sorted[-1]
    stability = float(np.corrcoef(wide[v_lo], wide[v_hi])[0, 1])
    print(f"view-stability corr(axis@{v_lo:.0f}deg, axis@{v_hi:.0f}deg) = {stability:+.3f}  (expect strong positive)")

# genuine-mirror flip check on the fixture (measured, not by construction): axis(mirror(x)) vs axis(x)
frontal = [c for c in clips if c.view_deg == 90.0]
ax_o = np.array([signed_axis(c.coords) for c in frontal])
ax_m = np.array([signed_axis(genuine_mirror_view(c.coords)) for c in frontal])
mirror_slope = float(np.polyfit(ax_o, ax_m, 1)[0]) if len(ax_o) >= 2 else float("nan")
print(f"genuine-mirror flip slope on fixture = {mirror_slope:+.3f}  (expect near -1 for this raw axis)")
print("\nNOTE: synthetic fixture only. Real CASIA-B / OU-MVLP-Pose loading is a marked TODO. "
      "This arm is NON-CLINICAL and reach-tier; it tests the reflection property, not any diagnosis.")
"""))

# ------------------------------------------------------------------ 6. Persist bundle
CELLS.append(md(r"""
## 6. Persist the futures + decision bundle

We write the futures table, the decision map, the gates, and the reach-scaffold status to JSON next to
the notebook so the methodology document and README can cite the exact simulated shapes, and so a reader
can diff the eventual real `idea9_antisymmetric_readout_result.json` (from 09a) and
`idea9_equivariant_retrain_result.json` (from 09b) against the canonical futures here.
"""))
CELLS.append(code(r"""
import json
bundle = {
    "notebook": "nb_09c_futures_and_reach",
    "gates": {"floor_margin": FLOOR_MARGIN, "perm_alpha": PERM_ALPHA,
              "nuisance_abs": NUISANCE_ABS, "y_between_min": Y_BETWEEN_MIN,
              "mirror_band": list(MIRROR_BAND)},
    "futures": rows,
    "decision": {r["future"]: {"primary": r["primary"], "mirror": r["mirror"],
                               "nuisance_ok": bool(r["nuisance_ok"]),
                               "beats_capacity_matched": bool(r["beats_capacity_matched"]),
                               "c_null_significant": bool(r["c_null_significant"]),
                               "claim": CLAIMS[r["future"]]}
                 for r in rows},
    "external_reach": {
        "cohorts": ["CASIA-B (Yu 2006)", "OU-MVLP-Pose (Takemura 2018)"],
        "status": "scaffold only; real loaders are marked TODO",
        "tier": "non-clinical reach; tests reflection stability + genuine-mirror flip, not diagnosis",
        "fixture_view_stability_corr": stability,
        "fixture_genuine_mirror_slope": mirror_slope,
    },
    "supersedes": "The proposal-level gates in the idea-9 README/METHODOLOGY worked example are superseded "
                  "by these hardened gates (binding bar max(D,C), permutation nulls, absolute nuisance "
                  "control, y-quality gate). See IMPLEMENTATION.md.",
    "notes": "Simulated expected shapes, NOT data. The clinical claim on this cohort is internal-validity "
             "only. The exact wiring-swap slope -1 is a by-construction check (nb_09a section 5), not a "
             "future. Folder labels are dataset annotations, not diagnoses; all results transductive.",
}
out = IDEA9_DIR / "idea9_futures_bundle.json"
try:
    out.write_text(json.dumps(bundle, indent=2))
    print(f"wrote {out}")
except Exception as exc:
    print(f"(bundle not written: {exc})")
print(json.dumps({"gates": bundle["gates"], "n_futures": len(rows)}, indent=2))
"""))

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (gavd3-sjepa)", "language": "python", "name": "gavd3-sjepa"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NB_PATH.write_text(json.dumps(nb, indent=1))
print(f"wrote {NB_PATH}  ({len(CELLS)} cells)")
