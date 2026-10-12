# ICLR 2027 submission sprint

Prepared 24 September 2026. The author confirmed that a genuine abstract was registered and that no manuscript exists, and supplied its title and abstract. This package is a writing starter, not a verified submission-ready paper.

## The paper to write

**Recommended title: retain the registered title.** Evaluating Feature Prediction for 2D Pose Trajectory Restoration with Paired Synthetic Supervision

**Central question:** Does movement-aware predictive pretraining improve the response recovered from a frozen encoder across downstream objectives, and what happens to the underlying trajectories?

**Contribution:** A controlled empirical audit separating pretraining, readout supervision, movement-response fidelity, geometric fidelity, and prediction reliability. The new loss is a controlled experimental intervention, not a demonstrated general solution.

This preserves the registered question: what does reference-feature prediction add beyond direct coordinate training and untrained/shuffled-feature controls? The newer experiments expand its movement-preservation evaluation. The main argument should lead with that question; the delta auxiliary and readout interaction are explanatory follow-up experiments.

## Alignment with the registered abstract

The registered abstract described an initial eight-person treadmill evaluation, three pose estimators, one training run per neural method, 0.20-second coordinate changes, and exploratory ankle-separation variation/peaks. The current gait-fidelity study evaluates a different declared movement measurement on the retained 14-person development cohort with three seeds. These are different stages and cannot be silently combined into one sample or one endpoint.

Keep the shared question, paired synthetic supervision, frozen encoder/readout design, direct baseline, and initialized/shuffled controls. Update the abstract to the completed expanded evaluation. Include the earlier pilot only if its exact artifacts and protocol are verified and its distinct sample and measurement definitions are clear; otherwise use its observations as internal motivation rather than extra empirical evidence. The ankle-peak and 0.20-second claims should not be carried into the final paper without their own verified result tables.

The manuscript is still a development study. More participants and seeds than the initial pilot improve its evidence but do not create an untouched confirmation set or real clinical validation.

The current evidence is the user's rounded 16-row follow-up table, completed-run status, the locally retained core evaluation, and implementation/protocol files. The follow-up's person-level exports, confidence intervals, failure decomposition, and feature-diagnostic results have not yet been inspected locally. Statements about those results remain blocked on evidence, not on writing.

Start with [paper-draft.md](paper-draft.md). Its TODO markers are mandatory evidence gaps. The abstract is provisional and reports descriptive means only.

## Submission facts checked against official sources

- Full paper deadline: 25 September 2026, 23:59 AoE, equivalent to **26 September, 04:59 Pacific daylight time**. The experiment's 25 September 08:00 Pacific cutoff is a different deadline.
- The abstract deadline was 18 September. The author reports meeting it. Preserve the substantive connection to the genuine registered abstract; titles/abstracts can be updated before the full-paper deadline. New authors cannot be added or removed after the abstract deadline.
- Main text: at most nine pages in the official ICLR 2027 style. References and appendices are outside that limit.
- Main paper and supplement must be anonymous. Raw run JSON contains personal paths and should not be uploaded unchanged.
- A separate AI-use disclosure is required and is outside the page limit. Disclose the actual assistance with hypotheses, experiment design/implementation, interpretation, and writing. Do not claim human checks that have not occurred.
- Check the submission's reciprocal-reviewer requirement, with the exemption for author groups without eligible reviewers, and the submission-form fields.

Sources: [Author Guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines), [Call for Papers](https://iclr.cc/Conferences/2027/CallForPapers), [AI Policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors). Verified 24 September 2026; verify the actual registered OpenReview record before upload.

## Claims and evidence gates

| Claim | Present evidence | Required check before final wording |
| --- | --- | --- |
| Delta beats endpoint under change readout | Response difference +0.3731 degrees, endpoint minus delta | Saved paired person/seed interval; per-seed effects; failure contributions |
| The response comparison reverses under base readout | Base difference -0.7121 degrees; interaction +1.0852 degrees | Saved interaction interval; same encoder identity and matched person/seed support |
| Change supervision degrades geometric fidelity | Both waveform and all-joint NLE means worsen in all eight reported families | Paired person effects and intervals; common support; coefficient and update schedule explicitly bounded |
| Direct/base is the strongest practical baseline | Lowest four error means and highest direction accuracy | Reconstruct means from complete exports; distinguish end-to-end training from frozen readouts |
| A representation preserves movement information | Not established by the summary table | Training-only feature diagnostics, reference response curves, zero-response benchmark, failure analysis |
| Results generalize to independent people or real patients | Not established | No such claim in this submission |

Eight families are correlated experiments on the same development population, not eight independent replications. Three seeds do not increase the participant count. Secondary comparisons are unadjusted descriptive analyses. A non-significant primary result is not equivalence and need not invalidate a carefully bounded audit paper.

## First two hours: collect and verify the existing evidence

From the Mac repository root, an authenticated terminal can copy the small result files below. These commands read the remote run and write a separate local export; they do not launch training. The author should handle HAIC's interactive authentication.

```bash
mkdir -p outputs/gait-fidelity/jepa-response-02-export
rsync -av --files-from=docs/studies/gait-fidelity/manuscript/results-files.txt \
  tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/outputs/gait-fidelity/jepa-response-02/ \
  outputs/gait-fidelity/jepa-response-02-export/
```

Keep exported paths and manifests private. If a listed file is missing, inspect the receipt and report before treating the run as finalized. On HAIC the normal verifier reconstructs results from retained predictions; run it in an appropriate CPU allocation as documented in `slurm/gait-fidelity/START_RESPONSE_02.md`. Copying only these summary files cannot independently verify all upstream hashes.

Collect the diagnostics worker's `diagnostics-summary.json` and referenced `diagnostics.json` files separately if feature-accessibility claims are being considered. Their attempt path is recorded in the run ledger; do not assume a job or attempt identifier.

The absolute minimum analysis before a final claim is:

1. Reconstruct the supplied means from the follow-up `per-person.csv`; check all method/person/seed cells and sample counts.
2. Read the primary interval and the base/readout interaction interval. Preserve the registered primary even if it is inconclusive.
3. Decompose the primary response contrast into successful-error and failure contributions. Compare to zero response.
4. Inspect response curves and held-dose/observation strata for attenuation and reversals. State exploratory status for newly selected analyses.
5. Document development reuse, split identities, exact training population, source balance, frozen settings, and technical reference limitations.

## One-day schedule

| Hours from now | Deliverable | Stop rule |
| --- | --- | --- |
| 0–2 | Verified result exports; claim/evidence table; registered abstract checked | If results cannot be verified, do not replace missing evidence with confident prose |
| 2–5 | Primary/interaction intervals, failure and zero-response checks, person/seed consistency | No new sweep or selection of a more favorable primary |
| 5–8 | Three empirical figures and one design schematic | Use an all-method or fixed-rule display, not selected favorable examples |
| 8–14 | Complete nine-page main-text draft in official LaTeX template | End with a complete readable draft; avoid an oversized methods tutorial |
| 14–18 | Appendix, precise configuration, verified bibliography, anonymous reproducibility materials | No unverified citations or claims of releasing restricted data |
| 18–21 | Scientific audit of claims, denominators, plots, abstract, and limitations | Narrow unsupported claims; do not change evidence after seeing which version reads better |
| 21–24 | Compile, inspect all pages, verify submission fields and final PDF | Upload a complete version with buffer; reopen the uploaded PDF |

The owner performs the OpenReview submission. This starter does not upload anything or communicate with coauthors or administrators.

## Nine-page allocation

| Content | Pages |
| --- | ---: |
| Abstract, introduction, contribution | 1.25 |
| Related work | 0.50 |
| Measurement and data construction | 1.25 |
| Models, matched interventions, training protocol | 1.50 |
| Experimental design and uncertainty | 1.00 |
| Results, interpretation, and three empirical figures | 2.50 |
| Limitations and conclusion | 1.00 |
| **Total** | **9.00** |

Figure 1: movement/observation crossing plus shared-encoder readout comparison. Figure 2: base-to-change trajectories in response-versus-waveform error, including direct/base and all eight families. Figure 3: primary/base interaction with paired uncertainty and person-level effects. Figure 4: response curves and failure/success contribution comparison, replacing one results figure or sharing its space. The descriptive preview in this directory can guide layout, but needs the verified exports and uncertainty before being treated as final evidence.

## Reviewer objections to answer directly

- **Could the loss weight explain the tradeoff?** Yes, the current result concerns the frozen inherited coefficient and training procedure. A general impossibility claim would be unsupported. Report the coefficient and gradient calibration; disclose the absence of a readout-weight sweep.
- **Is this just synthetic data with few people?** The development sample contains 14 people in the retained core export, heavily concentrated in one source. Report exact child coverage and training counts from the bound manifest. Limit the conclusion to this benchmark.
- **Does delta supervision add novel mathematics?** No novelty claim should rest on the difference loss or its algebraic identity. Derivative supervision and predictive skeleton representations have precedents; the potential contribution is controlled empirical knowledge.
- **Could failures or collapsed responses explain the gain?** Answer with the existing decomposition, zero-response baseline and slopes before promoting a mechanism.
- **Why not use the strongest baseline?** Include direct/base prominently; it leads the reported means. Explain the different training path instead of hiding it.
- **Was the test set used to develop the method?** This is reused development data. Present the core-to-follow-up chronology and describe secondary inference accordingly.

The weakest points cannot be repaired by stronger adjectives: small independent sample, development reuse, restricted synthetic scope, untested hyperparameter robustness, and uncertain effect size. A clear bounded empirical paper may still have value. ICLR's [reviewer guidance](https://iclr.cc/Conferences/2027/ReviewerGuidelines) explicitly evaluates new knowledge rather than requiring a state-of-the-art result, but that is not a prediction of acceptance.

## Evidence-dependent conclusion

If the primary interval is positive and success-case improvement remains, report a conditional delta benefit. If it crosses zero, report an unresolved primary effect and the bounded fidelity/readout findings. If the gain is failure-driven, describe improved reliability. If the interaction interval is broad, call its sign reversal descriptive. If reconstruction fails or the geometric tradeoff is an implementation artifact, resolve it before submission or defer.
