"""Builder for nb_05b_reflection_reach_and_futures.ipynb.

Run:  python3 _build_nb_05b.py
Emits the notebook in the gavd5-draft root beside nb_05a. Pure numpy/sklearn/matplotlib: no torch,
no checkpoint, no pose cache. It (1) simulates the four canonical possible futures of Idea 5 against
the pre-registered margins and draws their expected-shape panels, (2) renders one decision table that
maps each future to the exact verdict language the proposal pre-registers, and (3) lays down an
honest, runnable scaffold for the external multi-view mirror-consistency reach arm (CASIA-B /
OU-MVLP-Pose), clearly marked as non-clinical and not yet wired to real downloads.
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[3] / "nb_05b_reflection_reach_and_futures.ipynb"

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
# Notebook 05b: Reflection reach and the possible futures of Idea 5

Notebook 05a runs the single decisive probe on the frozen `d0acc262` checkpoint and emits one verdict.
This notebook does the two things 05a cannot:

1. **Possible-futures simulator.** Before the real numbers land, it is worth knowing exactly what each
   outcome would look like and what claim it would license. We deterministically synthesize the four
   canonical futures of the signed-laterality probe (clean-flip positive, decodable-but-non-flipping,
   informative null, artifact), score each against the SAME pre-registered margins 05a uses, and draw
   their expected-shape panels. This is a pre-registration aid, not a result.

2. **External multi-view reach scaffold.** The clinical claim on gavd5-draft (n=18 source videos) is
   internal-validity only. The reflection-equivariance idea has a NON-clinical reach test on public
   multi-view pose cohorts (CASIA-B, OU-MVLP-Pose): does the signed axis stay stable across camera
   views, and does a genuine left/right camera swap flip it. We lay down a runnable, honestly-stubbed
   loader interface for that arm and state precisely why it is reach-tier and non-clinical.

Every margin here is copied verbatim from the proposal: beat the untrained floor by at least 0.05
R-squared, reach at least 80 percent of the raw-coordinate-null R-squared, decoded sign correct on at
least 75 percent of held-out sources, and a mirror slope inside the band [-1.25, -0.8] to count as a
flip. See `notes/ideas-claude/_shared_facts.md` for the single source of truth. Folder labels are
dataset annotations, not diagnoses; all gavd5-draft readouts are transductive.
"""))

# ------------------------------------------------------------------ 0. Margins
CELLS.append(md(r"""
## 0. The pre-registered margins (frozen, copied from the proposal)

These four constants are the entire decision rule. Nothing downstream may change them; the simulator
and the real notebook both import the same numbers so a future cannot be scored on a moved goalpost.
"""))
CELLS.append(code(r"""
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42


def idea5_dir():
    '''Resolve the 05 proposal folder robustly regardless of the kernel cwd.'''
    for base in [Path.cwd(), *Path.cwd().parents]:
        cand = base / "notes" / "ideas-claude" / "05-signed-laterality-decodability"
        if cand.exists():
            return cand
        cand = base / "05-signed-laterality-decodability"
        if cand.exists():
            return cand
    return Path.cwd()


IDEA5_DIR = idea5_dir()
print(f"IDEA5_DIR: {IDEA5_DIR}")

# ---- pre-registered margins (proposal "The decisive experiment") ----
FLOOR_MARGIN = 0.05          # Lane A must beat Lane C (untrained floor) by >= this R2.
NULL_FRACTION = 0.80         # Lane A must reach >= this fraction of Lane B (raw-null) R2.
SIGN_CONSISTENCY = 0.75      # decoded sign correct on >= this fraction of held-out sources.
MIRROR_BAND = (-1.25, -0.8)  # mirror slope inside this band counts as "flips".

print("Pre-registered decision rule:")
print(f"  beat untrained floor by         >= {FLOOR_MARGIN} R2")
print(f"  reach fraction of raw null      >= {NULL_FRACTION}")
print(f"  decoded-sign correct on sources >= {SIGN_CONSISTENCY}")
print(f"  mirror slope band for 'flips'    = {MIRROR_BAND}")
"""))

# ------------------------------------------------------------------ 1. Scoring function
CELLS.append(md(r"""
## 1. One scoring function, shared by every future

`score_future` takes the four lane R-squared values, the fraction of held-out sources with the correct
decoded sign, and the mirror slope, and returns the exact verdict language the proposal pre-registers.
The primary verdict is a positive ("signed axis present above raw") ONLY if all three primary checks
pass together; otherwise it is an informative null. The mirror verdict is reported separately as the
secondary mechanism endpoint. The nuisance control (Lane D) must not recover a signed axis.
"""))
CELLS.append(code(r"""
def score_future(a_r2, b_r2, c_r2, d_r2, sign_frac, mirror_slope):
    beats_floor = (a_r2 - c_r2) >= FLOOR_MARGIN
    reaches_null = a_r2 >= NULL_FRACTION * max(b_r2, 1e-9)
    sign_ok = sign_frac >= SIGN_CONSISTENCY
    primary_positive = bool(beats_floor and reaches_null and sign_ok)
    flips = bool(MIRROR_BAND[0] <= mirror_slope <= MIRROR_BAND[1])
    # Lane D must stay near zero (side-agnostic by construction).
    d_control_ok = bool(abs(d_r2) < 0.05 or d_r2 < 0.5 * max(b_r2, 1e-9))
    if primary_positive:
        primary = "SIGNED AXIS PRESENT ABOVE RAW"
    else:
        primary = "INFORMATIVE NULL (signed axis not linearly present above raw)"
    if flips:
        mirror = "FLIPS (encoding is antisymmetric under the mirror)"
    else:
        mirror = "DOES NOT FLIP (encoding is non-antisymmetric)"
    return {
        "beats_floor": beats_floor, "reaches_null": reaches_null, "sign_ok": sign_ok,
        "primary_positive": primary_positive, "flips": flips, "d_control_ok": d_control_ok,
        "primary_verdict": primary, "mirror_verdict": mirror,
    }


# sanity: the worked example from the proposal (A=0.42, B=0.50, C=0.05, sign 0.75) is a primary pass
_demo = score_future(0.42, 0.50, 0.05, -0.30, 0.75, -1.0)
assert _demo["primary_positive"] and _demo["flips"], _demo
print("scoring self-check OK (proposal worked example scores as a primary pass, mirror flips).")
"""))

# ------------------------------------------------------------------ 2. The four futures
CELLS.append(md(r"""
## 2. The four canonical possible futures

Each future is a named, plausible state of the world with the lane numbers it would produce. They are
illustrative expected shapes, not measurements. The point is that the decision rule is total: every
future maps to an unambiguous, pre-registered claim.

- **F1 clean-flip positive.** The learned encoder carries the signed axis competitively with raw
  coordinates and the mirror cleanly negates it. This is the strong result: reflection-equivariance is
  learned, and the lateralized-vs-symmetric separation (stroke / hemiplegic-CP / early-PD vs myopathy)
  has a mechanism-faithful substrate.
- **F2 decodable-but-non-flipping.** The signed axis is linearly present above raw, but the mirror does
  not cleanly flip it. The encoding is informative yet non-antisymmetric: it decodes side without
  respecting the reflection symmetry. Licenses the decodability claim, withholds the equivariance claim.
- **F3 informative null.** The learned features do not clear the floor-plus-null bar. Raw coordinates
  decode the signed axis but the frozen S-JEPA tokens do not add it linearly. This overturns the
  hypothesis that the checkpoint has organized a laterality axis, and it is a publishable negative for a
  representation audit.
- **F4 artifact.** Lane A looks strong BUT the mean/std-pooled nuisance control (Lane D) also "recovers"
  the axis, which is impossible for a genuinely signed quantity. The apparent signal is a magnitude or
  acquisition artifact and the signed claim is withdrawn.
"""))
CELLS.append(code(r"""
FUTURES = [
    {"key": "F1_clean_flip_positive",
     "title": "F1 clean-flip positive",
     "a_r2": 0.44, "b_r2": 0.50, "c_r2": 0.06, "d_r2": -0.02, "sign_frac": 0.83, "mirror_slope": -1.02},
    {"key": "F2_decodable_non_flip",
     "title": "F2 decodable but non-flipping",
     "a_r2": 0.41, "b_r2": 0.50, "c_r2": 0.05, "d_r2": 0.01, "sign_frac": 0.80, "mirror_slope": -0.25},
    {"key": "F3_informative_null",
     "title": "F3 informative null",
     "a_r2": 0.12, "b_r2": 0.48, "c_r2": 0.08, "d_r2": 0.00, "sign_frac": 0.55, "mirror_slope": -0.6},
    {"key": "F4_artifact",
     "title": "F4 artifact (nuisance control fires)",
     "a_r2": 0.47, "b_r2": 0.50, "c_r2": 0.06, "d_r2": 0.46, "sign_frac": 0.82, "mirror_slope": -0.9},
]

rows = []
for f in FUTURES:
    v = score_future(f["a_r2"], f["b_r2"], f["c_r2"], f["d_r2"], f["sign_frac"], f["mirror_slope"])
    rows.append({
        "future": f["title"], "A_r2": f["a_r2"], "B_r2": f["b_r2"], "C_r2": f["c_r2"], "D_r2": f["d_r2"],
        "sign_frac": f["sign_frac"], "mirror_slope": f["mirror_slope"],
        "primary": v["primary_verdict"], "mirror": v["mirror_verdict"],
        "D_control_ok": v["d_control_ok"],
    })
futures_table = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", 60)
print(futures_table.to_string(index=False))
"""))

# ------------------------------------------------------------------ 3. Decision table
CELLS.append(md(r"""
## 3. The decision table: future -> licensed claim

This is the single artifact a reviewer should read first. It states, for every future, exactly which
claim is licensed and which is withheld. Note F4: when the nuisance control fires, the primary "signed"
claim is withdrawn regardless of how strong Lane A looked, because a mean/std pooling is side-agnostic
by construction and cannot carry a genuinely signed quantity.
"""))
CELLS.append(code(r"""
CLAIMS = {
    "F1 clean-flip positive": (
        "LICENSED: the frozen encoder linearly carries the signed laterality axis competitively with "
        "raw coordinates AND the encoding is antisymmetric under the anatomical mirror. Supports the "
        "reflection-equivariant symmetry-axis reading (Idea 9 substrate confirmed)."),
    "F2 decodable but non-flipping": (
        "LICENSED: signed axis is linearly present above raw coordinates. WITHHELD: reflection "
        "equivariance; the encoding decodes side without respecting the mirror symmetry."),
    "F3 informative null": (
        "LICENSED (negative): the frozen S-JEPA tokens do NOT add the signed axis above raw "
        "coordinates. Overturns the hypothesis that the checkpoint organized a laterality axis; a "
        "clean publishable negative for the representation audit."),
    "F4 artifact (nuisance control fires)": (
        "WITHHELD: the signed claim is withdrawn. The side-agnostic mean/std control recovered the "
        "'axis', so Lane A reflects a magnitude/acquisition artifact, not a signed quantity."),
}
decision = futures_table[["future", "primary", "mirror", "D_control_ok"]].copy()
decision["licensed_claim"] = decision["future"].map(CLAIMS)
for _, r in decision.iterrows():
    print(f"### {r['future']}")
    print(f"  primary : {r['primary']}")
    print(f"  mirror  : {r['mirror']}")
    print(f"  D control ok: {r['D_control_ok']}")
    print(f"  claim   : {r['licensed_claim']}")
    print()
"""))

# ------------------------------------------------------------------ 4. Expected-shape panels
CELLS.append(md(r"""
## 4. Expected-shape panels for each future

For each future we synthesize a point cloud that reproduces its lane R-squared and mirror slope, so the
reader sees the SHAPE each outcome makes on the two decisive plots (decodability scatter and mirror
scatter). These are simulated to the target statistics; they are not data. The panels mirror the two
SVG mockups in the proposal's `images/` folder.
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
        pred = rng.normal(0, 1, n)  # no relationship; R2 near 0 or negative
    else:
        noise_var = (1 - r2) / max(r2, 1e-6)
        pred = y + rng.normal(0, np.sqrt(noise_var), n)
    return y, pred


def simulate_mirror(slope, n=40, seed=0):
    rng = np.random.default_rng(seed)
    orig = rng.normal(0, 1, n)
    mir = slope * orig + rng.normal(0, 0.12, n)
    return orig, mir


fig, axes = plt.subplots(2, 4, figsize=(17, 8))
for col, f in enumerate(FUTURES):
    seed = RANDOM_SEED + col
    # top row: decodability (Lane A)
    yt, yp = simulate_scatter(f["a_r2"], seed=seed)
    ax = axes[0, col]
    ax.scatter(yt, yp, s=26, c="#e07a4b", edgecolors="#a44c26", linewidths=0.5)
    lim = 3.2
    ax.plot([-lim, lim], [-lim, lim], "--", color="#5f9e7e", linewidth=1.3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_title(f"{f['title']}\nLane A R2~{f['a_r2']:.2f} (null B~{f['b_r2']:.2f}, floor C~{f['c_r2']:.2f})", fontsize=9)
    if col == 0:
        ax.set_ylabel("decoded signed scalar")
    ax.set_xlabel("ground-truth signed target")
    # bottom row: mirror
    orig, mir = simulate_mirror(f["mirror_slope"], seed=seed + 100)
    ax2 = axes[1, col]
    ax2.scatter(orig, mir, s=26, c="#2f6f99", edgecolors="#1f4a68", linewidths=0.5)
    mlim = 3.2
    ax2.plot([-mlim, mlim], [mlim, -mlim], "--", color="#5f9e7e", linewidth=1.3)
    ax2.axhline(0, color="#c4cdd8", lw=0.8); ax2.axvline(0, color="#c4cdd8", lw=0.8)
    ax2.set_xlim(-mlim, mlim); ax2.set_ylim(-mlim, mlim)
    flips = "FLIPS" if MIRROR_BAND[0] <= f["mirror_slope"] <= MIRROR_BAND[1] else "no flip"
    ax2.set_title(f"mirror slope~{f['mirror_slope']:+.2f} ({flips})", fontsize=9)
    if col == 0:
        ax2.set_ylabel("decoded on mirrored input")
    ax2.set_xlabel("decoded on original input")

fig.suptitle("Idea 5 possible futures (simulated expected shapes, NOT data). Green dashes: top y=x, bottom y=-x.", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.96])
OUT = IDEA5_DIR / "images"
try:
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "idea5_possible_futures.png"
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")
except Exception as exc:
    print(f"(figure not saved to images/: {exc})")
plt.show()
"""))

# ------------------------------------------------------------------ 5. External reach scaffold
CELLS.append(md(r"""
## 5. External multi-view reach arm (non-clinical scaffold)

The clinical claim on gavd5-draft is internal-validity only: n=18 canonical source videos, transductive
encoder, monocular capture. A simultaneously clinical + skeleton + participant-disjoint public cohort
does not exist, so there is NO honest skeleton-level clinical transfer test. What DOES exist is a
non-clinical reach test for the reflection-equivariance property itself, on public multi-view pose
cohorts:

- **CASIA-B** (Yu, Tan, Tan 2006): 124 subjects, 11 camera views (0 to 180 degrees). Non-clinical gait.
- **OU-MVLP-Pose** (Takemura et al. 2018): ~10,000 subjects, multiple views, pose keypoints released.

The reach question is symmetry-specific and view-specific, not clinical:

1. **View stability.** Is the signed axis decoded from one view consistent with the same subject decoded
   from a nearby view (small change), i.e. is it a property of the gait rather than the camera?
2. **Genuine mirror.** A true left-versus-right camera swap (a real physical reflection, not a synthetic
   x-negation) should flip the signed axis. CASIA-B's symmetric view angles around 90 degrees give a
   real-world analogue of the anatomical mirror.

This cell defines the loader interface and the two metrics, then runs them on a small synthetic
multi-view fixture so the scaffold is exercised. Wiring real downloads is deliberately left as a marked
TODO: these are large licensed datasets and this pass does not fetch them.
"""))
CELLS.append(code(r"""
from dataclasses import dataclass, field


@dataclass
class MultiViewClip:
    subject_id: str
    view_deg: float          # camera azimuth in degrees
    coords: np.ndarray       # [T, 33, 3] normalized pose
    cohort: str = "synthetic"


def load_casia_b(root=None):
    '''TODO(real): parse CASIA-B pose export at `root` into MultiViewClip list.
    Not wired in this pass (large licensed dataset). Returns [] so the scaffold degrades cleanly.'''
    if root is None:
        return []
    raise NotImplementedError("Real CASIA-B loading is a marked TODO; provide a parser at call site.")


def load_ou_mvlp_pose(root=None):
    '''TODO(real): parse OU-MVLP-Pose keypoints at `root`. Not wired in this pass.'''
    if root is None:
        return []
    raise NotImplementedError("Real OU-MVLP-Pose loading is a marked TODO.")


def synthetic_multiview_fixture(n_subjects=6, views=(54.0, 90.0, 126.0), frames=40, seed=RANDOM_SEED):
    '''A tiny multi-view fixture: each subject has a fixed signed lean visible from every view; the
    view rotates the pose in the x-z plane so the signed x-excursion is genuinely view-dependent.'''
    rng = np.random.default_rng(seed)
    clips = []
    LEFT_RIGHT = [(11, 12), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
    for s in range(n_subjects):
        lean = 1.0 if s % 2 == 0 else -1.0
        phase = np.linspace(0, 4 * np.pi, frames, endpoint=False)
        base = np.zeros((frames, 33, 3), dtype=np.float32)
        for j in range(33):
            base[:, j, :] = rng.normal(0, 0.04, 3)[None, :] + 0.03 * np.sin(phase + j)[:, None]
        for li, ri in LEFT_RIGHT:
            base[:, li, 0] += 0.05 * lean * np.sin(phase)
            base[:, ri, 0] -= 0.05 * lean * np.sin(phase)
        for v in views:
            theta = np.deg2rad(v - 90.0)  # 90 deg is the frontal reference
            rot = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
            coords = base @ rot.T
            clips.append(MultiViewClip(subject_id=f"subj{s:02d}", view_deg=float(v), coords=coords.astype(np.float32)))
    return clips


LEFT_RIGHT_PAIRS = [(11, 12), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]


def signed_axis(coords):
    total = 0.0
    for li, ri in LEFT_RIGHT_PAIRS:
        total += coords[:, li, :].std(axis=0).sum() - coords[:, ri, :].std(axis=0).sum()
    return float(total)


clips = synthetic_multiview_fixture()
print(f"external reach scaffold exercised on {len(clips)} synthetic multi-view clips "
      f"({len({c.subject_id for c in clips})} subjects x {len({c.view_deg for c in clips})} views).")

# Metric 1: view stability = corr of the per-subject signed axis across two nearby views.
df = pd.DataFrame([{"subject": c.subject_id, "view": c.view_deg, "axis": signed_axis(c.coords)} for c in clips])
wide = df.pivot(index="subject", columns="view", values="axis")
views_sorted = sorted(df["view"].unique())
if len(views_sorted) >= 2:
    v_lo, v_hi = views_sorted[0], views_sorted[-1]
    stability = float(np.corrcoef(wide[v_lo], wide[v_hi])[0, 1])
    print(f"view-stability corr(axis@{v_lo:.0f}deg, axis@{v_hi:.0f}deg) = {stability:+.3f}  (expect strong positive)")
print("\nNOTE: synthetic fixture only. Real CASIA-B / OU-MVLP-Pose loading is a marked TODO. "
      "This arm is NON-CLINICAL and reach-tier; it tests the reflection property, not any diagnosis.")
"""))

# ------------------------------------------------------------------ 6. Persist futures bundle
CELLS.append(md(r"""
## 6. Persist the futures + decision bundle

We write the futures table and the decision map to JSON next to the notebook so the methodology
document and README can cite the exact simulated shapes, and so a reader can diff the eventual real
`idea5_signed_laterality_result.json` (from 05a) against the four canonical futures here.
"""))
CELLS.append(code(r"""
import json
bundle = {
    "notebook": "nb_05b_reflection_reach_and_futures",
    "margins": {"floor_margin": FLOOR_MARGIN, "null_fraction": NULL_FRACTION,
                "sign_consistency": SIGN_CONSISTENCY, "mirror_band": list(MIRROR_BAND)},
    "futures": rows,
    "decision": {r["future"]: {"primary": r["primary"], "mirror": r["mirror"],
                               "D_control_ok": bool(r["D_control_ok"]), "claim": CLAIMS[r["future"]]}
                 for r in rows},
    "external_reach": {
        "cohorts": ["CASIA-B (Yu 2006)", "OU-MVLP-Pose (Takemura 2018)"],
        "status": "scaffold only; real loaders are marked TODO",
        "tier": "non-clinical reach; tests reflection stability + genuine-mirror flip, not diagnosis",
    },
    "notes": "Simulated expected shapes, NOT data. gavd5-draft clinical claim is internal-validity only.",
}
out = IDEA5_DIR / "idea5_futures_bundle.json"
try:
    out.write_text(json.dumps(bundle, indent=2))
    print(f"wrote {out}")
except Exception as exc:
    print(f"(bundle not written: {exc})")
print(json.dumps({"margins": bundle["margins"], "n_futures": len(rows)}, indent=2))
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
