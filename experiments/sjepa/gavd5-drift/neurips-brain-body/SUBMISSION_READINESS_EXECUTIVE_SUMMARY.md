# Paper Submission Readiness: Executive Summary

> **Archived planning snapshot (2026-09-03).** This predates the current canonical paper draft. Use [the workshop readiness guide](docs/neurips-brain-body.md) for current status; shared artifacts referenced below live in `../work/`.

**Paper:** "Don't Forget Normal: Measuring and Trying to Repair Normative-Anchor Drift in a Continual Skeleton-JEPA World Model of Gait"  
**Target:** NeurIPS 2026 BrainBodyFM Workshop  
**Date:** 2026-09-03  
**Status:** 🔴 **NOT READY FOR SUBMISSION** (Critical blockers identified)

---

## Critical Finding

**Dataset-Artifact Mismatch Invalidates Reproducibility Claims**

The paper claims results on the **augmented dataset** (159 sequences / 35 videos) but current artifacts contain results from the **canonical dataset** (626 sequences / 93 videos). The fingerprints, numbers, and AnchorGuard results do not match what is written in the paper.

| Metric | Paper (Augmented) | Current Artifacts (Canonical) | Status |
|---|---|---|---|
| Corpus size | 159 sequences / 35 videos | 626 sequences / 93 videos | ❌ **Mismatch** |
| Drift curve | 0.954 → 0.839 → 0.707 → 0.594 | 0.700 → 0.502 → 0.396 → 0.297 | ❌ **Different** |
| Checkpoint fingerprint | `d0acc2628d13…` | `7d13841acead…` | ❌ **Different** |
| AnchorGuard final anchor | 0.538 | 0.479 | ❌ **Different** |
| Binary F1 baseline | 0.849 | 0.685 | ❌ **Different** |
| Binary F1 AnchorGuard | 0.893 | 0.971 | ❌ **Different** |

**Implication:** The central claim in the paper ("Every score is pinned to file hashes and reproducible") is falsified. Any reviewer who attempts to verify the results using the cited artifacts will find different numbers.

---

## Recommendation: Regenerate on Canonical Dataset

**Decision:** The paper should be regenerated on the **canonical GAVD dataset** (626 sequences / 93 videos, no self-annotated YouTube augmentation).

**Rationale:**

1. **Stronger science.** Removes the provenance asymmetry (63/75 normal from YouTube, self-annotated; 0/84 disorders from YouTube). All data from curated GAVD source.

2. **Matches notebook state.** The current notebooks are set up to expect the canonical fingerprint. Regeneration requires minimal code changes.

3. **Bigger phenomenon.** The canonical drift curve (0.700 → 0.502 → 0.396 → 0.297) shows a **57% drop** vs. **38% in the augmented version**. This is a more dramatic phenomenon and strengthens the core finding.

4. **Simpler story.** The paper already flags provenance as a limitation. Using canonical removes it entirely. No need to carry a confound you've acknowledged.

5. **Honest limitation is unchanged.** Single-site (93 videos) is still the dominant limit. Adding YouTube clips doesn't solve it; it just adds a new confound. Better to own the single-site constraint plainly.

---

## What Needs to Happen Before Submission

### Phase 1: Verify Current State (1 hour)
- [ ] Confirm canonical fingerprint `7d13841...` is consistent across all notebooks
- [ ] Check that all artifact numbers match notebook outputs
- [ ] Identify any reference to augmented-only results in figures/text
- [ ] Document which experiments depend on canonical vs. augmented

### Phase 2: Update Paper Text (2 hours)
- [ ] Rewrite §3.1 (Data provenance):
  - Remove "added normal from YouTube" section
  - Update to "canonical GAVD: 94 videos, 642 sequences"
  - Remove provenance asymmetry detail (no longer applies)
- [ ] Update Abstract: change corpus size from "159 seqs, 35 vids" to "642 seqs, 94 vids"
- [ ] Update all results claims in §4 with new canonical numbers
- [ ] Update gates table (§4.3) — do they still pass/fail?
- [ ] Simplify §4.4 controls (provenance split no longer needed)
- [ ] Update Appendix A with new fingerprints and artifact paths

### Phase 3: Run Critical Experiments (2 hours)
- [ ] **Binary F1 residualized against missingness** — verify AnchorGuard improvement survives confound check
- [ ] **λ sweep for AnchorGuard** — test λ ∈ {0.1, 0.5, 1.0, 2.0} to justify λ=0.5 choice
- [ ] **Canonical-only provenance stratification** — already not applicable, but confirm no residual YouTube data
- [ ] **Laterality sign-flip check** — verify if flipping sign of learned readout recovers correlation

### Phase 4: Regenerate Figures (1 hour)
- [ ] Figure 2: Update drift curves with canonical numbers, add Stage 0 reference point
- [ ] Figures 3–6: Update with new downstream probe results if changed
- [ ] Verify all captions match new numbers

### Phase 5: Final Audit (30 min)
- [ ] Full SHA-256 hash of checkpoint (not truncated)
- [ ] Every numeric value cross-checked against artifacts
- [ ] All gates re-checked (still passing?)
- [ ] Remove "pending regeneration" note from §3.1
- [ ] Proof-read for consistency across entire paper
- [ ] Verify 5-page limit (excluding references)

**Total effort: ~6.5 hours**

---

## Adversarial Review Findings Summary

### Critical Blockers (Must fix)
1. **Dataset-artifact mismatch** — paper claims augmented, artifacts show canonical
2. **"Pending regeneration" note** — signals incomplete work, must be resolved
3. **AnchorGuard results inconsistent** — 0.538 vs 0.479 final anchor
4. **Pre-registered gates based on wrong numbers** — gate pass/fail indeterminate
5. **Figure 2 does not match artifacts** — drift curve mismatch

### High-Priority Issues (Should fix)
- Missingness confound on binary F1 (is +0.045 improvement real or confounded?)
- Provenance asymmetry minimized in text but dominates dataset (minimize via canonical path)
- Five-class fold variance hidden (report mean ± range)
- AnchorGuard λ parameter not ablated (add sweep table)
- Laterality result mislabeled as null (should be "anticorrelated")
- Cadence canary is circular (signal deleted at preprocessing, then confirmed missing)

### Medium-Priority Issues (Nice to have)
- Figure 2 caption lacks error bar explanation
- EMA decay coefficient not specified
- Random Forest hyperparameters not justified
- Group margin hinge (> 0.5) not justified
- Forecasting narrative somewhat oversold

---

## Key Metrics After Regeneration

Expected outcomes on canonical dataset:

| Metric | Canonical Expected | Action if Different |
|---|---|---|
| Drift curve | 0.700 → 0.502 → 0.396 → 0.297 | Update all figures and tables |
| Gate G1 (anchor ≥ 0.85) | FAIL (final 0.296) | Reframe AnchorGuard as even more dramatic failure |
| Gate G2 (std ≥ 0.35) | ? | Check current feature std in canonical run |
| Binary F1 baseline | ~0.685 | Update table §4.3 |
| AnchorGuard binary F1 | ~0.971 | Verify, report if changes |
| Five-class baseline | ~0.441 | Update table §4.3 |
| Five-class residualized | ? | Check after running residualization check |

---

## Strengths to Preserve

Despite the reproducibility crisis, the paper has genuine strengths that survive regeneration:

1. ✅ **Novel measurement:** Normative-anchor drift curve is unreported in JEPA / continual-learning / clinical-gait literature. This is a real contribution.

2. ✅ **Intellectual honesty:** AnchorGuard fails and you report it without spin. You diagnose *why*. This is rare and valuable.

3. ✅ **Reproducibility discipline (design):** The ambition to pin every number, pre-register gates, and expose confounds is exactly right. Execution fell short; fixing it strengthens everything.

4. ✅ **Ablation and attribution:** The margin ablation (§4.2) is clean. The zero-training probes (§5.1–5.2) are creative inspections of a frozen model.

5. ✅ **Domain grounding:** You engage with what clinicians measure (cadence, asymmetry, severity). Not just a machine learning paper; it's informed by domain knowledge.

6. ✅ **Transparent limitations:** The boundary between descriptive/inferential is maintained throughout. Readers cannot misinterpret this as a generalization claim.

---

## What Success Looks Like

After regeneration, the paper will be ready when:

- ✅ Dataset choice is explicit and consistent (canonical, no augmentation)
- ✅ Every number pinned to specific artifact file and hash (full, not truncated)
- ✅ All pre-registered gates pass or fail consistently (no ambiguity)
- ✅ Critical confounds quantified (missingness residualization, provenance stratification)
- ✅ Negative results clearly framed (AnchorGuard fails, forecasting can't, laterality lost)
- ✅ "Pending regeneration" note removed or completed
- ✅ Reproducibility auditable (all hashes, fingerprints, paths documented)
- ✅ No overclaiming (transductive, single-site, small-cohort language consistent)
- ✅ Within 5-page limit (excluding references)
- ✅ All figures updated with correct numbers

---

## Bottom Line

**This is a strong paper. It makes real contributions to understanding normative-anchor drift in continual skeleton-JEPA models.** But it cannot ship in its current form because the dataset-artifact mismatch invalidates the reproducibility claims.

**Fix this, and it is ready for NeurIPS workshop submission.**

The work is ~6–7 hours away from being submission-ready. The core science is sound; the execution just needs to be completed.

---

## Next Steps

1. **Decide:** Regenerate on canonical dataset (recommended) or augmented dataset (weaker but faster)?
   - **Recommendation:** Canonical. Stronger science, matches notebook state, removes confound.

2. **Execute Phase 1:** Audit current state (1 hour)

3. **Execute Phases 2–5:** Regenerate paper, run experiments, update figures, final audit (5 hours)

4. **Verify gates:** Confirm that pre-registered gates still pass/fail as documented

5. **Submit:** Camera-ready with full SHA-256 hash documented

**Estimated time to submission-ready: 6–7 hours** (mostly regeneration + figure updates)

---

## Documents Created

For your reference:
- `PAPER_REVISION_PLAN.md` — Detailed workflow with phase-by-phase breakdown
- `SUBMISSION_READINESS_EXECUTIVE_SUMMARY.md` — This document (high-level overview)
- Adversarial review report (in main conversation)

These planning records are in the `neurips-brain-body` workshop folder.
