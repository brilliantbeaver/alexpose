# Paper Regeneration Checklist

> **Archived planning snapshot (2026-09-03).** This predates the current canonical paper draft. Use [the workshop readiness guide](docs/neurips-brain-body.md) for current status; shared artifacts referenced below live in `../work/`.

**Decision:** Regenerate on **Canonical Dataset** (626 sequences / 93 videos, GAVD only)

---

## Phase 1: Verify Current State ✓ Required

- [ ] List all artifacts in `../work/artifacts/real/` and verify timestamps are recent (Sep 3 2026)
- [ ] Confirm fingerprint `7d13841aceac9eda843d43ca8434193e294d2fa10a48b6c6d21f6413a6e457e2` appears in:
  - [ ] `classifier_contract.json`
  - [ ] `temporal_readout_results.json`
  - [ ] `idea5_signed_laterality_result.json`
- [ ] Verify `curriculum_stage_summary.csv` exists and shows canonical drift curve
- [ ] Check `anchor_guard_results.json` exists (and has which fingerprint?)
- [ ] Verify `predictive_surprise_results.json` corpus shows 626 sequences
- [ ] Identify any artifact files that reference augmented data (should not exist)
- [ ] Document the mapping: which notebook produces which artifact file

---

## Phase 2: Update Paper Text ✓ Required

### §3.1 Data Provenance
- [ ] Remove the entire "Added YouTube" paragraph
- [ ] Rewrite corpus description to read:
  ```
  The canonical GAVD corpus is 642 sequences from 94 source videos: normal 284 (30 videos), 
  Parkinson's 42 (9), stroke 75 (18), myopathic 183 (28), cerebral palsy 58 (9). 
  After validity filtering (landmark coverage > 0.45), this yields 626 sequences from 93 videos.
  All results use this canonical set; no augmentation with external data.
  ```
- [ ] Delete the "pending regeneration" note or replace it with completion statement

### Abstract
- [ ] Find and replace: "159 sequences, 35 videos" → "626 sequences, 93 videos"
- [ ] Find and replace: "normal 0.954 to 0.594" → "[NEW DRIFT NUMBERS]"
- [ ] Verify the four-disorder curriculum explanation still applies (it does)

### §4 Results
- [ ] Update Fig. 2 caption with new drift curve numbers
- [ ] Update §4.1 "Drift reproduction" paragraph with new canonical numbers
- [ ] Update Table in §4.2 (margin ablation) — these numbers likely don't change (retraining from Stage-0), but verify
- [ ] Update §4.3 AnchorGuard results:
  - [ ] New stage-end anchors (should be lower than paper claims)
  - [ ] New feature std (should be lower than paper claims)
  - [ ] New gates table — do G1, G2, G3, G4 still pass/fail?
  - [ ] New downstream probe metrics (binary and five-class)
- [ ] Update §4.4 controls — remove provenance split (no longer applicable)

### §5 Zero-training Probes
- [ ] Verify numbers in §5.1 (surprise AUROC, copy-last baseline) match `predictive_surprise_results.json`
- [ ] Verify numbers in §5.2 (readout sweep, laterality) match `temporal_readout_results.json` and `idea5_signed_laterality_result.json`
- [ ] Update figures if numbers changed

### Appendix A: Verified Numbers
- [ ] Replace entire table with new canonical numbers
- [ ] Update all artifact file references
- [ ] Update checkpoint fingerprint (full hash, not truncated)
- [ ] Update experiment fingerprint
- [ ] Update all row values to match canonical artifacts

### Appendix B: Scaling the Cohort
- [ ] Update the note: "The canonical set is 626 sequences / 93 videos, providing a firm single-site foundation. Future work should extend to multiple sites via nested leave-source-videos-out retraining."
- [ ] Remove any reference to "added YouTube clips" as a future path (not applicable for canonical)

---

## Phase 3: Run Critical Experiments ✓ Required

### Experiment 1: Binary F1 Residualized Against Missingness
**Purpose:** Verify that AnchorGuard's binary F1 improvement (+0.045 claimed, or different value in canonical) survives residualization against detector missingness.

```python
# Pseudocode:
baseline_predictions = random_forest.predict(baseline_embeddings)
anchorguard_predictions = random_forest.predict(anchorguard_embeddings)
missingness_features = pose_missingness.values  # already computed

# Rank-residualize
baseline_resid = rank_residualize(baseline_predictions, missingness_features)
anchorguard_resid = rank_residualize(anchorguard_predictions, missingness_features)

# Compute macro-F1 on residualized predictions
baseline_f1_resid = macro_f1(baseline_resid, ground_truth)
anchorguard_f1_resid = macro_f1(anchorguard_resid, ground_truth)
```

- [ ] Run residualization check on both baseline and AnchorGuard binary probes
- [ ] Report results in paper: "Binary improvement survives residualization (residualized baseline X, AnchorGuard Y)"
- [ ] If improvement flips sign or vanishes, flag as confound-driven and note in limitations

### Experiment 2: AnchorGuard λ Sweep
**Purpose:** Justify λ=0.5 choice by testing nearby values.

```python
# Test lambda in {0.1, 0.5, 1.0, 2.0}
for lambda_val in [0.1, 0.5, 1.0, 2.0]:
    # Retrain AnchorGuard from canonical Stage-0, λ=lambda_val
    train_stage_1_4_with_anchor_guard(lambda_val)
    
    # Record: stage-end anchor, feature std, binary F1, five-class F1
    results[lambda_val] = {
        'stage_4_anchor': x,
        'stage_4_std': y,
        'binary_f1': z,
        'five_class_f1': w,
    }
```

- [ ] Run λ-sweep (4 configurations, ~30 min training total)
- [ ] Create table: λ vs. (stage-4 anchor, feature std, binary F1, five-class F1)
- [ ] Include in paper as appendix or main results depending on which λ works best
- [ ] Update text: "We tested λ ∈ {0.1, 0.5, 1.0, 2.0} and selected λ=0.5 as a balance between retention and plasticity."

### Experiment 3: Laterality Sign-Flip Check
**Purpose:** Determine if R² = −0.187 is due to anticorrelation or random noise.

```python
# Learned readout gave R² = -0.187
learned_predictions = ridge_model.predict(embeddings)

# Check: if we flip the sign, do we get +R²?
flipped_predictions = -1 * learned_predictions
flipped_r2 = r2_score(ground_truth_asymmetry, flipped_predictions)
```

- [ ] Compute flipped_r2
- [ ] If flipped_r2 ≈ +0.187: model learned asymmetry but inverted it → report as "anticorrelated"
- [ ] If flipped_r2 still ≈ −0.187: signal genuinely absent → confirm "discarded"
- [ ] Update §5.2 with finding

---

## Phase 4: Regenerate Figures ✓ Required

### Figure 2: Drift Curves
- [ ] Plot canonical red curve (0.700 → 0.502 → 0.396 → 0.297)
- [ ] Plot canonical blue AnchorGuard curve (update from new experiments)
- [ ] **Add Stage 0 reference point** at left edge (0.954 or canonical equivalent)
- [ ] Add vertical dashed line at gate threshold 0.85
- [ ] Update caption with new numbers
- [ ] Verify visual matches narrative ("drift is steady and dramatic")

### Figures 3–6
- [ ] Figure 3: Check if consolidation signal visualization still applies
- [ ] Figure 4: Forecasting concept — should be unchanged (conceptual)
- [ ] Figure 5: Forecasting results by condition — update numbers if changed
- [ ] Figure 6: Readout sweep — update if numbers changed
- [ ] Regenerate all from code (not manual edits)
- [ ] Verify figure numbers match paper references

---

## Phase 5: Final Audit ✓ Required

### Artifact Verification
- [ ] Checkpoint SHA-256: Compute full hash (40+ chars, not truncated)
  ```bash
  sha256sum ../work/artifacts/real/curriculum_stage_*.pt
  ```
- [ ] Verify every numeric value in Appendix A matches artifact files
  - [ ] Drift curve (4 values)
  - [ ] Drift reproduction gap
  - [ ] Margin ablation (6 values)
  - [ ] AnchorGuard (stage anchors, feature std)
  - [ ] Gates (4 results)
  - [ ] Downstream probes (6 values)
  - [ ] All other tables

### Gate Validation
- [ ] G1: Stage-4 anchor ≥ 0.85? (Expected: fail, but verify exact value)
- [ ] G2: Feature std ≥ 0.35? (Expected: fail, but verify exact value)
- [ ] G3: Five-class macro-F1 within 0.05 of baseline? (Check new value)
- [ ] G4: Binary macro-F1 within 0.05 of baseline? (Check new value)
- [ ] Record pass/fail for each

### Text Consistency
- [ ] Search paper for all numbers and cross-check against artifacts
- [ ] Verify "transductive," "single-site," "cohort," "descriptive" language consistent
- [ ] Remove any reference to augmented data or YouTube clips
- [ ] Confirm 5-page limit (excluding references and appendices)

### Final Proof-Read
- [ ] Typo sweep (use spell-checker)
- [ ] Citation check (all references cited are in references.bib)
- [ ] Notation consistency (e.g., c₀ always formatted the same way)
- [ ] Figure caption completeness (all figures explained, no missing details)
- [ ] Consistency of condition names (Parkinson's vs PD, etc.)

---

## Experiments & Expected Outcomes

### Critical Experiments

| Experiment | Status | Expected | Action if Different |
|---|---|---|---|
| Binary F1 residualized | [ ] TODO | Improvement survives confound | Flag as limitation if not |
| λ sweep | [ ] TODO | λ ≈ 0.5 is reasonable choice | Update paper if better value found |
| Laterality sign-flip | [ ] TODO | Flipped R² ≈ +0.18 (anticorrelated) | Reframe if signal is genuinely absent |

### Numbers to Verify in Artifacts

| Metric | Expected (Canonical) | Actual | Match? |
|---|---|---|---|
| Drift curve, stage 1 | 0.700 | ? | [ ] |
| Drift curve, stage 4 | 0.297 | ? | [ ] |
| AnchorGuard final anchor | ~0.48 | ? | [ ] |
| Binary F1 baseline | ~0.685 | ? | [ ] |
| Binary F1 AnchorGuard | ~0.971 | ? | [ ] |
| Five-class baseline | ~0.441 | ? | [ ] |
| Five-class AnchorGuard | ~0.866 | ? | [ ] |
| Missingness-only accuracy | 0.448 | ? | [ ] |

---

## Submission Readiness Gates

Before final submission, confirm:

- [ ] **Dataset resolution:** Canonical (no augmentation) chosen and documented
- [ ] **No "pending regeneration" note:** All results finalized or note updated to reflect completion
- [ ] **All gates checked:** G1, G2, G3, G4 results recorded and consistent
- [ ] **Critical experiments run:** Residualization check, λ-sweep, sign-flip check completed
- [ ] **Figures updated:** All 6 figures regenerated with correct numbers
- [ ] **Artifact verification:** Every table value cross-checked against files
- [ ] **SHA-256 documented:** Full checkpoint hash (not truncated) in Appendix A
- [ ] **Paper length:** Within 5 pages (excluding references/appendices)
- [ ] **Reproducibility auditable:** Fingerprints, paths, file hashes all documented
- [ ] **No overclaiming:** Transductive/single-site/small-cohort qualifiers on every result

---

## Timeline

| Phase | Task | Est. Time | Status |
|---|---|---|---|
| 1 | Verify current state | 1 hour | [ ] |
| 2 | Update paper text | 2 hours | [ ] |
| 3 | Run experiments | 2 hours | [ ] |
| 4 | Regenerate figures | 1 hour | [ ] |
| 5 | Final audit | 30 min | [ ] |
| **Total** | | **6.5 hours** | |

---

## Success Criteria

After regeneration is complete, the paper will be ready when:

✅ Dataset is canonical, no augmentation  
✅ All numbers verified against artifacts  
✅ All gates pass or fail consistently  
✅ Critical confounds quantified  
✅ Figures updated with correct numbers  
✅ SHA-256 hash documented (full, not truncated)  
✅ "Pending regeneration" note removed  
✅ No overclaiming in text  
✅ Reproducibility auditable  

---

## Notes

- Keep the canonical checkpoint frozen; do not retrain unless absolutely necessary
- AnchorGuard experiments can use fresh seeds (not tied to canonical fingerprint)
- Prioritize the residualization check (blocker for "does not hurt" claim)
- λ-sweep is high-priority (justifies design choice)
- Sign-flip check is medium-priority (clarifies mechanism)

---

**Ready to begin?** Start with Phase 1 audit. Expected to be submission-ready in ~7 hours of focused work.
