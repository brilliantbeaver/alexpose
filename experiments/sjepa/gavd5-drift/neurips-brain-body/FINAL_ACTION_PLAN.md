# Final Action Plan: Paper Regeneration with Canonical Dataset

> **Archived planning snapshot (2026-09-03).** This predates the current canonical paper draft. Use [the workshop readiness guide](docs/neurips-brain-body.md) for current status; shared artifacts referenced below live in `../work/`.

**Date:** 2026-09-03  
**Decision:** Regenerate on **Canonical Dataset** (626 sequences / 93 videos, GAVD only)  
**Status:** Ready to execute

---

## Current State

### Diagnostics Completed ✅
- [x] Identified dataset-artifact mismatch (augmented vs canonical)
- [x] Extracted all canonical numbers from artifacts
- [x] Run comprehensive adversarial review
- [x] Created detailed planning documents (3 files)
- [x] Identified critical blockers and high-priority fixes

### Canonical Numbers Extracted ✅
All extracted from `../work/artifacts/real/`:

| Section | Metric | Canonical Value |
|---|---|---|
| Abstract | Drift curve | 0.700 → 0.502 → 0.396 → 0.297 |
| §4.1 | Reproduction gap | 4.51×10⁻⁷ |
| §4.2 | Margin ablation (G1) | anchor=0.3221, std=0.3407 |
| §4.2 | Margin ablation (G0) | anchor=0.3269, std=0.4669 |
| §4.3 | AnchorGuard final anchor | 0.4786 |
| §4.3 | AnchorGuard feature std | 0.2516 |
| §4.3 | Binary F1 baseline | 0.6718 |
| §4.3 | Binary F1 AnchorGuard | 0.9708 |
| §4.3 | Five-class F1 baseline | 0.3766 |
| §4.3 | Five-class F1 AnchorGuard | 0.8172 |
| §5.1 | Future cosine (h=2) | CP=0.293, myo=0.305, normal=0.365, PD=0.402, stroke=0.377 |
| §5.1 | Copy-last baseline | 0.90-0.94 across conditions |
| §5.1 | Spatial ceiling | 0.5152 |
| §5.2 | Laterality R² (learned) | 0.2409 |
| §5.2 | Laterality R² (floor) | 0.1900 |
| §5.2 | RankMe (learned) | 2.863 |
| §5.2 | RankMe (untrained) | 2.760 |

### What Has NOT Been Updated Yet ❌
- Paper text (docs/bbfm2026_paper_draft.md)
- Figures (Figures 2-6)
- Appendix A (verified numbers)
- Any documentation in docs/ folder

---

## Work Breakdown: Phase-by-Phase

### Phase 1: Update Paper Text (§1-6 + Appendices)
**Files to edit:** `docs/bbfm2026_paper_draft.md`

**Changes required:**

#### Abstract (lines 8-11)
- [ ] Change: "159 sequences, 35 videos" → "626 sequences, 93 videos"
- [ ] Change: "0.954 to 0.594" → "0.700 to 0.297"
- [ ] Change: "0.893 vs 0.849" → "0.971 vs 0.672"
- [ ] Change: "honest negative" → "retention-plasticity trade-off" (more accurate)

#### §1.4 Contributions (lines 41-48)
- [ ] Update drift curve numbers: "0.954 → 0.839 → 0.707 → 0.594" → "0.700 → 0.502 → 0.396 → 0.297"
- [ ] Update margin ablation: "0.954 with it, 0.976 without" → "0.3221 with it, 0.3269 without"
- [ ] Update AnchorGuard: "0.538" → "0.479"
- [ ] Update downstream: "0.893 vs 0.849; five-class within margin" → "0.971 vs 0.672; five-class 0.817 vs 0.377"

#### §3.1 Data and Provenance (lines 71-75)
- [ ] **CRITICAL DELETE**: Remove entire "Added YouTube" section
- [ ] Rewrite corpus to: "The canonical GAVD corpus is 642 sequences from 94 source videos... After validity filtering, 626 sequences from 93 videos remain. All results use canonical GAVD only."
- [ ] **DELETE** the "pending regeneration" note entirely
- [ ] Remove discussion of "provenance asymmetry" (no longer applicable)

#### §4.1 Results: Drift (lines 117-119)
- [ ] Update: "0.9540 / 0.8389 / 0.7066 / 0.5942" → "0.7002 / 0.5021 / 0.3962 / 0.2966"
- [ ] Update: "max gap 4.7×10⁻⁷" → "max gap 4.51×10⁻⁷"

#### §4.2 Margin Ablation (lines 127-132)
- [ ] Update table: G1 anchor 0.9543 → 0.3221
- [ ] Update table: G0 anchor 0.9763 → 0.3269
- [ ] Update table: G1 centroid 0.7408 → 0.9015
- [ ] Update table: G0 centroid 0.8080 → 0.9407
- [ ] Update table: G1 std 0.4302 → 0.3407
- [ ] Update table: G0 std 0.4789 → 0.4669
- [ ] Update narrative: "Δ ≈ 0.022" → check actual delta (≈0.0048)

#### §4.3 AnchorGuard (lines 135-154)
- [ ] Update: "0.777 / 0.655 / 0.579 / 0.538" → "0.5416 / 0.4933 / 0.4829 / 0.4786"
- [ ] Update: "std ends at 0.342" → "std ends at 0.2516"
- [ ] Update gates table:
  - G1: "0.538" → "0.4786" (still fails)
  - G2: "0.342" → "0.2516" (still fails)
  - G3: "0.597 vs 0.622" → "0.8172 vs 0.3766" (now PASSES by large margin)
  - G4: "0.893 vs 0.849" → "0.9708 vs 0.6718" (now PASSES by large margin)
- [ ] Update downstream probe table:
  - Baseline: "0.849 / 0.849" → "0.6718 / 0.3766"
  - AnchorGuard: "0.893 / 0.893" → "0.9708 / 0.8172"

#### §4.4 Controls (lines 156-159)
- [ ] Update: "0.448" missingness accuracy (if canonical value differs, update)
- [ ] Remove or simplify "provenance split" discussion (no longer applicable)

#### §5.1 Forecasting (lines 168-174)
- [ ] Update future cosine values: "CP 0.233, stroke 0.296, myopathic 0.322, normal 0.352, PD 0.442" → "CP 0.293, stroke 0.377, myopathic 0.305, normal 0.365, PD 0.402"
- [ ] Update baseline ceiling: "0.547" → "0.5152"
- [ ] Update copy-last baseline: "0.88-0.95" (check if still accurate)
- [ ] Update surprise AUROC section with canonical values

#### §5.2 Laterality (lines 176-182)
- [ ] Update: "R² = −0.187" → "R² = 0.2409" (DIFFERENT SIGN!)
- [ ] Update: "floor +0.147" → "floor 0.1900"
- [ ] Update: "RankMe 3.48 > untrained 2.74" → "RankMe 2.863 > untrained 2.760"
- [ ] **IMPORTANT**: Reframe finding — laterality R² is POSITIVE now, not negative!

#### §6 Discussion (lines 188-197)
- [ ] May need minor updates depending on how gate failures change narrative
- [ ] Verify "why repair failed" section still makes sense with new numbers

#### Appendix A (lines 208-227)
- [ ] Replace entire "Verified numbers and provenance" table with canonical values:
  - Anchor drift curve: 0.7002 / 0.5021 / 0.3962 / 0.2966
  - Drift reproduction gap: 4.51×10⁻⁷
  - Margin ablation: [new values]
  - AnchorGuard results: [new values]
  - Downstream probes: [new values]
  - All other metrics: [update if changed]
- [ ] Update checkpoint fingerprint to canonical: `7d13841aceac9eda843d43ca8434193e294d2fa10a48b6c6d21f6413a6e457e2`
- [ ] Update experiment fingerprint accordingly
- [ ] Update corpus information: 626 sequences, 93 videos, canonical GAVD

#### Appendix B (lines 230-233)
- [ ] Remove reference to "added YouTube clips"
- [ ] Reframe: "The canonical set is 626 sequences / 93 videos. Future work extends to multiple sites via nested leave-source-videos-out retraining."

---

### Phase 2: Critical Experiments (Must Run)

**Experiment 1: Binary F1 Residualized Against Missingness**
- [ ] Load baseline and AnchorGuard classifiers
- [ ] Load pose missingness features
- [ ] Rank-residualize both predictions against missingness
- [ ] Compute macro-F1 on residualized predictions
- [ ] Report: "Baseline residualized F1: X, AnchorGuard residualized F1: Y"
- [ ] **Blocker decision**: Does improvement survive? If not, flag in paper.

**Experiment 2: AnchorGuard λ Sweep**
- [ ] Load canonical Stage-0 checkpoint
- [ ] Train AnchorGuard with λ ∈ {0.1, 0.5, 1.0, 2.0}
- [ ] Record for each: stage-4 anchor, feature std, binary F1, five-class F1
- [ ] Create table comparing results
- [ ] **Decision**: Does λ=0.5 remain best choice, or does larger λ work better?

**Experiment 3: Laterality Sign-Flip Check**
- [ ] Load laterality readout (learned, R²=0.2409)
- [ ] Flip sign: −1 × predictions
- [ ] Compute R² on flipped predictions
- [ ] **Result**: If flipped R² ≈ 0.24, confirm "anticorrelated"; if still ≈ 0.24, confirm "present"

---

### Phase 3: Update Figures

**Figure 2: Drift Curves**
- [ ] Update red canonical curve: 0.700 → 0.502 → 0.396 → 0.297
- [ ] Update blue AnchorGuard curve: 0.542 → 0.493 → 0.483 → 0.479
- [ ] **ADD**: Stage 0 reference point on left (visual baseline)
- [ ] **ADD**: Gate threshold line at 0.85
- [ ] Update caption

**Figures 3-6**
- [ ] Verify all numbers still match
- [ ] Regenerate if any numbers changed significantly
- [ ] Update captions

---

### Phase 4: Final Verification

**Artifact Cross-Check**
- [ ] Every numeric value in Appendix A matches artifact files
- [ ] All gate results (G1-G4) recorded and consistent
- [ ] All three experiments completed and results documented

**Readability Check**
- [ ] No "pending regeneration" notes remain
- [ ] Consistent language (transductive, single-site, etc.)
- [ ] Within 5-page limit
- [ ] No overclaiming

**Reproducibility Check**
- [ ] Full SHA-256 checkpoint hash documented
- [ ] Experiment fingerprint documented
- [ ] All artifact file paths traceable
- [ ] Gate thresholds and decisions transparent

---

## Critical Issue: Laterality R² Sign Change

**IMPORTANT FINDING**: The canonical results show laterality R² = **0.2409** (positive), not −0.187 (negative).

This is a **completely different finding** than claimed in the augmented paper!

**Implication**: 
- Paper claimed: "discarded the signed axis" (R² negative)
- Canonical shows: Model learned some of the asymmetry (R² positive)

**Action needed:**
1. Run sign-flip check to confirm it's truly positive
2. Reframe §5.2 narrative: "The model captures some signed asymmetry but underperforms raw kinematics"
3. Update Contributions to reflect this

---

## Gate Results: Major Change

**Canonical gate results differ significantly from paper claims:**

| Gate | Paper Claim | Canonical | Change |
|---|---|---|---|
| G1 (anchor ≥ 0.85) | FAIL (0.538) | FAIL (0.479) | Same verdict, bigger failure |
| G2 (std ≥ 0.35) | FAIL (0.342) | FAIL (0.252) | Same verdict, bigger failure |
| G3 (five-class non-inf) | PASS (0.597 vs 0.622) | **PASS** (0.817 vs 0.377) | HUGE improvement |
| G4 (binary non-inf) | PASS (0.893 vs 0.849) | **PASS** (0.971 vs 0.672) | HUGE improvement |

**Narrative shift**: 
- **Retention fails harder** (G1, G2)
- **But plasticity improves dramatically** (G3, G4)
- The trade-off is clearer and stronger

---

## Effort Estimate

| Phase | Task | Time |
|---|---|---|
| 1 | Update paper text | 2-3 hrs |
| 2 | Run 3 experiments | 1-2 hrs |
| 3 | Update figures | 1 hr |
| 4 | Final verification | 30 min |
| **Total** | | **5-7 hrs** |

---

## Success Criteria: Paper Ready When...

- [x] Dataset is canonical (626 / 93)
- [x] All numbers extracted and verified
- [ ] Paper text fully updated
- [ ] 3 critical experiments completed
- [ ] Figures regenerated
- [ ] All gates re-checked
- [ ] Laterality finding reframed
- [ ] No "pending" notes remain
- [ ] SHA-256 documented
- [ ] Appendix A updated
- [ ] Within 5 pages
- [ ] No overclaiming

---

## Next Steps: Your Decision

**I am ready to execute all phases.** But I need your confirmation on:

1. **Proceed with full regeneration?** (Phases 1-4)
2. **Run the 3 critical experiments?** (Phase 2)
3. **Timeline preference?** (Do it all now, or staged?)

All planning is complete. The canonical numbers are extracted. The changes are identified. I just need your go-ahead to start making edits.

**Estimated time to completion: 5-7 hours of focused work.**

---

## Files Ready for Modification

- `docs/bbfm2026_paper_draft.md` (PAPER)
- Figures (regeneration needed if numbers changed)
- Appendix A (full table replacement)

**All changes will be systematic, tracked, and verified against artifacts before final commit.**
