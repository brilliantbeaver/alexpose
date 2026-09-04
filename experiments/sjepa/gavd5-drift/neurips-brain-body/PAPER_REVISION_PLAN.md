# Paper Revision & Submission Readiness Plan

> **Archived planning snapshot (2026-09-03).** This predates the current canonical paper draft. Use [the workshop readiness guide](docs/neurips-brain-body.md) for current status; shared artifacts referenced below live in `../work/`.

**Date:** 2026-09-03  
**Paper:** "Don't Forget Normal: Measuring and Trying to Repair Normative-Anchor Drift in a Continual Skeleton-JEPA World Model of Gait"  
**Target:** NeurIPS 2026 BrainBodyFM Workshop

---

## Critical Issue Discovered

### Dataset Version Mismatch
The paper claims results on the **augmented dataset** (159 sequences / 35 videos) with drift curve **0.954 → 0.839 → 0.707 → 0.594**.

However, current artifacts in `../work/artifacts/real/` contain results from the **canonical dataset** (626 sequences / 93 videos) with drift curve **0.700 → 0.502 → 0.396 → 0.297**.

**Paper's intended dataset:**
- Canonical GAVD: 30 videos, 284 normal + 75 disorder sequences
- Added YouTube: 17 videos, 63 self-annotated normal sequences  
- **Total: 35 videos, 159 sequences (75 normal + 84 disorder)**
- Fingerprint: `d0acc2628d134959d8b91e96d5112fc3bed560fe8feb9569e5b13b11a8b614d1`

**Current artifacts:**
- Canonical GAVD only: 93 videos, 626 sequences (284 normal + 342 disorder)
- Fingerprint: `7d13841aceac9eda843d43ca8434193e294d2fa10a48b6c6d21f6413a6e457e2`

**Resolution options:**

#### Option A: Regenerate paper with Canonical dataset (RECOMMENDED)
- **Pros:** Removes provenance confound (all data from curated GAVD source); larger sample size; matches current notebooks
- **Cons:** Requires rewriting entire Results section; new numbers should be checked against paper's claims
- **Time:** ~4 hours to regenerate results + update paper + verify
- **Recommendation:** This is the stronger paper (no self-annotated data)

#### Option B: Regenerate paper with Augmented dataset  
- **Pros:** Matches paper as written; smaller pivot
- **Cons:** Keeps self-annotated normal (provenance concern that paper already flags); requires re-running notebooks on augmented data
- **Time:** ~2 hours to source augmented checkpoint + run notebooks + verify
- **Recommendation:** Faster but weaker scientifically

#### Option C: Hybrid - Acknowledge both datasets
- Report canonical as primary, augmented as sensitivity check
- Larger rewrite; beyond scope for workshop deadline

---

## Adversarial Review Findings

### Critical Blockers (must fix before submission)

1. **SHA-256 checkpoint hash truncated** (§A)
   - Paper says "pin in camera-ready" but provides `2aa20dd4ac92…`
   - **Action:** Restore full hash before final submission

2. **Missingness confound check incomplete** (§5.1, §4.3)
   - Surprise probe residualized against missingness (ρ=0.497, p=3.2e-11)
   - But AnchorGuard's binary F1 improvement (+0.045) NOT residualized
   - **Action:** Check if binary improvement survives missingness residualization
   - **Risk:** If confounded, "does not hurt" claim weakens

3. **Five-class fold variance hidden** (§3.7, §4.3)
   - Only 2 folds; no variance reported
   - Gate passes by 0.025 on gate threshold of 0.05
   - **Action:** Report five-class as mean ± fold-wise range
   - **Risk:** Hidden variance undermines transparency

### High-Priority Issues (should fix)

4. **AnchorGuard λ parameter not ablated** (§4.3)
   - Only tested λ=0.5
   - Why not 0.1, 1.0, 2.0?
   - **Action:** Add λ sweep table (4 rows)

5. **Provenance asymmetry not quantified** (§3.1, §4.4)
   - 63/75 normal from added path; 0/84 disorders
   - Are results robust to canonical-only normal?
   - **Action:** Report canonical-only stratification in appendix

6. **Laterality result mislabeled as null** (§5.2)
   - R² = −0.187 is negative R², not null (R² ≈ 0)
   - Suggests anticorrelated, not missing signal
   - **Action:** Reframe and check if sign-flip recovers correlation

### Medium-Priority (nice to have)

7. Add to Related Work: connection between forecasting capability and clinical task formulation
8. Clarify notation for c₀ in §1.3 (formal def in §3.4, introduce early)
9. Figure 2: add Stage 0 reference point for visual baseline

---

## Regeneration Workflow

### Path Forward: **Option A (Canonical Dataset)**

This is recommended because:
1. Removes the provenance asymmetry the paper already highlights as a limit
2. All data from curated GAVD source → stronger scientific claim
3. Matches current notebook state (they're set up for canonical)
4. Larger N reduces single-site cohort weakness

### Steps:

#### Phase 1: Verify Current State (1 hour)
- [ ] Confirm canonical fingerprint `7d13841...` is consistent across all notebooks
- [ ] Validate all artifact numbers match notebook outputs
- [ ] Document which notebooks/figures depend on canonical vs. augmented
- [ ] Identify any reference to augmented-only results

#### Phase 2: Update Paper Text (2 hours)
- [ ] Rewrite §3.1 Data provenance:
  - Remove "added normal from YouTube" section
  - Update to "canonical GAVD only: 94 videos, 642 sequences"
  - Simplify provenance story (no asymmetry)
- [ ] Update Abstract:
  - Change "159 sequences, 35 videos" → "642 sequences, 94 videos"
  - Keep drift curve but update numbers if changed
- [ ] Update all results claims in §4 with canonical numbers
- [ ] Update all gate results in Table §4.3
- [ ] Simplify §4.4 controls (missingness-only still exists, no provenance split needed)
- [ ] Update Appendix A with new fingerprints and artifact lineage

#### Phase 3: Verify Artifacts & Fix Critical Issues (2 hours)
- [ ] Run new SHA-256 hash of canonical checkpoint
- [ ] Check binary F1 residualized against missingness (blocker #2)
- [ ] Extract five-class fold-wise values and report variance (blocker #3)
- [ ] Add λ sweep for AnchorGuard (high-priority #4)
- [ ] Verify all figure numbers match new results
- [ ] Regenerate Figures 2, 3, 4, 5, 6 if numbers changed

#### Phase 4: Fix Identified Issues (1 hour)
- [ ] Correct laterality phrasing from "null" → "anticorrelated"
- [ ] Update Related Work connection to forecasting
- [ ] Clarify c₀ notation in §1.3
- [ ] Fix Figure 2 to include Stage 0 reference

#### Phase 5: Final Audit (30 min)
- [ ] Verify every number in paper ↔ artifact file
- [ ] Check gate results remain valid (no new failures)
- [ ] Proof-read all changes
- [ ] Ensure boundary between descriptive/inferential maintained
- [ ] Full SHA-256 hash documented

**Total time: ~6.5 hours**

---

## Key Results to Verify

After regeneration, confirm:

| Metric | Expected (Canonical) | Paper Claimed (Augmented) | Current Artifact |
|---|---|---|---|
| Corpus size | 642 seqs / 94 vids | 159 seqs / 35 vids | 626 seqs / 93 vids ✗ |
| Drift curve | ? | 0.954→0.839→0.707→0.594 | 0.700→0.502→0.396→0.297 |
| Binary F1 baseline | ? | 0.849 | 0.685 |
| AnchorGuard binary | ? | 0.893 | 0.971 |
| Five-class baseline | ? | 0.622 | 0.441 |
| AnchorGuard 5-class | ? | 0.597 | 0.866 |

---

## Blockers Requiring Experiments

### Must Run Before Submission:

1. **Binary macro-F1 residualized against missingness**  
   - Compute: residual F1 after rank-residualizing classifier output against detector missingness
   - Required for: Validates "does not hurt" claim robust to confound
   - Effort: 1 notebook cell

2. **λ sweep for AnchorGuard**  
   - Test: λ ∈ {0.1, 0.5, 1.0, 2.0}
   - Report: Stage-4 anchor and binary F1 for each
   - Required for: Justifies λ=0.5 choice vs. higher values
   - Effort: ~30 min training + analysis

3. **Canonical-only provenance stratification**  
   - Train/eval on canonical GAVD only (no added YouTube normal)
   - Report: Drift curve, binary F1, five-class F1
   - Required for: Quantify provenance confound impact
   - Effort: ~1 hour training if using existing code

### Nice to Have (if time):

4. Five-class fold-wise variance (already in artifacts, just needs reporting)
5. Laterality sign-flip check (diagnostic, 1 cell)

---

## Submission Checklist

- [ ] Dataset version resolved (canonical or augmented, documented)
- [ ] All numbers in paper verified against artifact files
- [ ] SHA-256 hash documented (full, not truncated)
- [ ] Missingness residualization check passed
- [ ] Five-class fold variance reported
- [ ] λ sweep added (confirms λ=0.5 choice)
- [ ] Provenance stratification in appendix
- [ ] Laterality reframed (anticorrelated not null)
- [ ] Figure 2 includes Stage 0 reference
- [ ] All gates still passing with new numbers
- [ ] Transductive/descriptive disclaimer on every result
- [ ] Related Work connection to forecasting added
- [ ] Notation c₀ clarified in §1.3
- [ ] Full proof-read for consistency
- [ ] 5-page limit met (excluding references)

---

## Success Criteria

Paper is ready to submit when:
1. ✅ Dataset choice is explicit and consistent throughout
2. ✅ Every number pinned to specific artifact file and hash
3. ✅ All gates passing (or failure explained and pre-registered)
4. ✅ Critical confounds (missingness, provenance) quantified
5. ✅ Negative/null results clearly framed (AnchorGuard failure, forecasting ceiling, laterality loss)
6. ✅ Boundary between descriptive and inferential maintained on every page
7. ✅ No overclaiming (transductive, single-site, small-cohort language consistent)
8. ✅ Reproducibility auditable (all hashes, fingerprints, artifact paths documented)

---

## Recommended: Make Decision on Dataset Now

**Recommend: Regenerate with Canonical (Option A)**

Reason: Removes confound, stronger science, matches current notebook state, and canonical GAVD is curated source. The paper's core contribution (drift measurement + mechanism) is not weaker with canonical; it's actually clearer.

The augmented dataset adds 63 YouTube clips that are self-annotated and never clinically reviewed — this is precisely the provenance asymmetry the paper flags as a limit. Using canonical removes this asymmetry entirely.

**Action:** Proceed with Phase 1 audit + Phase 2-5 regeneration on canonical fingerprint `7d13841...`.
