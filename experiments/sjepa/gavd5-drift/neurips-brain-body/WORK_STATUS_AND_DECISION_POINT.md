# Work Status & Decision Point

> **Archived planning snapshot (2026-09-03).** This predates the current canonical paper draft. Use [the workshop readiness guide](docs/neurips-brain-body.md) for current status; shared artifacts referenced below live in `../work/`.

**Date:** 2026-09-03, 14:00 UTC  
**Paper:** "Don't Forget Normal: Measuring and Trying to Repair Normative-Anchor Drift in a Continual Skeleton-JEPA World Model of Gait"  
**Target:** NeurIPS 2026 BrainBodyFM Workshop

---

## What Has Been Completed ✅

### Diagnostic Phase (Complete)
- ✅ Identified critical dataset-artifact mismatch
- ✅ Extracted all canonical results from `../work/artifacts/real/`
- ✅ Run comprehensive adversarial review (detailed findings in conversation)
- ✅ Created 4 detailed planning documents:
  1. `PAPER_REVISION_PLAN.md` — Full workflow
  2. `SUBMISSION_READINESS_EXECUTIVE_SUMMARY.md` — High-level overview
  3. `REGENERATION_CHECKLIST.md` — Step-by-step checklist
  4. `FINAL_ACTION_PLAN.md` — Exact execution plan (THIS FILE)

### Analysis Completed
- ✅ All canonical numbers extracted and validated
- ✅ Critical issues identified (dataset, gates, laterality)
- ✅ All blockers and high-priority fixes documented
- ✅ Effort estimate: 5-7 hours to completion

---

## What Has NOT Been Done ❌

### Implementation Phase (Pending)
- ❌ Paper text updated (docs/bbfm2026_paper_draft.md)
- ❌ Figures regenerated (Figures 2-6)
- ❌ Critical experiments run (3 experiments needed)
- ❌ Appendix A updated with canonical numbers
- ❌ Final verification and audit

---

## The Decision Point

You have three options:

### Option 1: Full Execution (Recommended)
**I systematically execute all phases (1-4) right now:**
- Update paper text with all canonical numbers
- Run 3 critical experiments (residualization, λ-sweep, sign-flip)
- Regenerate figures
- Final verification
- **Result**: Paper is submission-ready in ~7 hours
- **Status**: All changes tracked, all numbers verified against artifacts

### Option 2: Staged Execution
**You want to review/approve changes incrementally:**
- Phase 1: Update paper text (you review)
- Phase 2: Run experiments (you review results)
- Phase 3: Update figures (you review)
- Phase 4: Final audit (you approve)
- **Result**: Same outcome, more checkpoints for you
- **Status**: Takes ~8 hours total (more overhead)

### Option 3: Manual Review First
**You want to review the canonical numbers before I touch anything:**
- I provide the mapping of all changes needed
- You verify the numbers are correct
- You give final approval
- Then I execute all phases
- **Result**: Extra verification step, then full execution
- **Status**: Takes ~8 hours (extra review time)

---

## Critical Finding: Laterality R² Sign Flip

⚠️ **IMPORTANT**: The canonical dataset shows laterality R² = **0.2409** (positive), not −0.187 (negative) as claimed in the augmented paper.

**This is a completely different finding:**
- **Paper claimed**: Model discarded signed axis (R² negative)
- **Canonical shows**: Model learned some asymmetry but underperforms raw kinematics (R² positive)

**Action required**: 
1. Run sign-flip confirmation experiment
2. Reframe §5.2 in paper
3. Update Contributions section

**This is why running the 3 experiments is critical.**

---

## Gate Results: Narrative Shift

Canonical results show **different gate pass/fail pattern**:

| Gate | Verdict | Canonical | Implication |
|---|---|---|---|
| G1 (anchor ≥ 0.85) | FAIL | 0.479 | Retention fails harder |
| G2 (std ≥ 0.35) | FAIL | 0.252 | Collapse worse |
| G3 (five-class) | **PASS** | 0.817 vs 0.377 | HUGE plasticity gain |
| G4 (binary) | **PASS** | 0.971 vs 0.672 | HUGE plasticity gain |

**Narrative evolution:**
- Canonical data shows **sharper retention-plasticity trade-off**
- AnchorGuard fails harder at retention (G1, G2)
- But succeeds more dramatically at plasticity (G3, G4)
- **The honest negative is stronger and clearer**

---

## What I Need From You

**Decision**: Which option above (1, 2, or 3)?

Once you decide, I will:
1. Execute all phases systematically
2. Track every change in the paper
3. Verify every number against artifacts
4. Provide complete audit trail
5. Mark paper as submission-ready

---

## Files Ready for Your Review

All planning records are in this `neurips-brain-body` workshop folder:

```
FINAL_ACTION_PLAN.md ← Exact execution steps
PAPER_REVISION_PLAN.md ← Comprehensive workflow
SUBMISSION_READINESS_EXECUTIVE_SUMMARY.md ← High-level overview
REGENERATION_CHECKLIST.md ← Detailed checklist
```

---

## Timeline

- **Option 1 (Full execution now)**: ~7 hours → done today
- **Option 2 (Staged)**: ~8 hours over multiple checkpoints
- **Option 3 (Manual review first)**: ~8-9 hours with extra review layer

All assume continuous work. Can be split across multiple sessions.

---

## Key Documents Created This Session

1. ✅ `PAPER_REVISION_PLAN.md` — Complete regeneration workflow
2. ✅ `SUBMISSION_READINESS_EXECUTIVE_SUMMARY.md` — Executive brief
3. ✅ `REGENERATION_CHECKLIST.md` — Step-by-step checklist
4. ✅ `FINAL_ACTION_PLAN.md` — Exact execution plan with all canonical numbers
5. ✅ `WORK_STATUS_AND_DECISION_POINT.md` — This document

---

## Bottom Line

**The hard work is done.** All diagnostic, planning, and analysis work is complete. The canonical numbers are extracted. The changes are identified. The experiments to run are specified.

**What's left is execution** — a straightforward, tracked process of updating the paper text, running 3 confirmation experiments, regenerating figures, and doing final verification.

**I'm ready to go. What's your call?**

Choose Option 1, 2, or 3 above, and I'll proceed immediately.
