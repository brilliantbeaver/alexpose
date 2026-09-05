# Laterality execution plan and runbook

This plan keeps computational validity, statistical interpretation, and governance as separate gates. Passing one does not waive another.

## Gate 1 — Frozen protocol and governance

- Review `PROTOCOL.md` and `config/protocol.json` before a paper run.
- Obtain an institutional ethics determination and completed data-use and derived-pose release reviews.
- Record dated internal references and scopes in `governance/status.json`; public availability is not authorization.
- Never place raw video, frames, URLs, or identity-bearing material in this suite or its artifacts.

Status: the computational protocol is frozen; the checked-in governance record deliberately remains unresolved and submission-blocking.

## Gate 2 — Read-only real-data audit

Run:

```bash
.venv/bin/python neurips-laterality/scripts/audit_real_data.py
```

The command validates the locked annotation/pose inventories, extraction provenance, source mapping, paired-valid target invariants, QC attrition, and source-level outer/inner split feasibility. It writes no artifacts and creates no empirical result.

Verified on 2026-09-04: 666 annotation files, 642 derived-pose archives, 625 eligible sequences, 93 eligible source videos, five source-disjoint outer folds, and four source-disjoint inner readout folds per outer fold. Re-run it after any authorized data change; an inventory change should fail until the protocol is deliberately revised and versioned.

## Gate 3 — Software and notebook integration

Run:

```bash
.venv/bin/python neurips-laterality/scripts/verify_suite.py --execute-smoke
```

This executes the adversarial contract tests and all seven notebooks in independent fresh kernels with temporary synthetic data. Smoke scores are marked synthetic and cannot become paper evidence.

Status: re-run after every protocol or implementation change; the verifier is the authoritative current count.

To inspect a fully rendered synthetic walkthrough without storing output in the canonical notebooks, run:

```bash
.venv/bin/python neurips-laterality/scripts/execute_notebooks.py --profile smoke
```

Open the timestamped copies under `neurips-laterality/executed/`. Inline figures are profile-watermarked and use aggregate derived-pose diagnostics only. Each saved notebook carries an `inline_output_audit` metadata record; execution fails unless all code cells ran, at least one PNG figure and one separate result payload are embedded, and no error output exists. Do not copy smoke plots into a paper or remove their non-evidentiary label.

## Gate 4 — Locked paper computation

Set `LATERALITY_PROFILE=paper`, choose a durable non-public artifact directory, and run notebooks 00 through 05 in numeric order. The complete registered run is 5 outer folds × 5 optimization seeds × 2 training variants = 50 fold-local encoders. Do not use subset overrides for final reporting. By default, artifacts are placed beneath a protocol-digest directory; never point a revised protocol at an older artifact directory.

Notebook 05 will fail on missing or stale lineage. Its scientific estimands compute metrics and squared mirror errors within each checkpoint before averaging registered seeds. Mean-prediction ensemble metrics are secondary and labeled. Direct token errors compare each learned checkpoint with its paired initial encoder before any read-out. Source-bootstrap intervals remain conditional on fixed cross-fitted checkpoints. A null or adverse diagnostic remains reportable and must not trigger protocol tuning.

Status: pre-v2.1 partial artifacts, if present, remain preserved but are invalid for the enhanced protocol. No v2.1 paper result is claimed until all folds, seeds, and variants complete.

## Gate 5 — Interpretation and submission

The primary estimand is mean registered-seed, single-checkpoint performance for the `learned_single_free` read-out from vanilla training on held-out GAVD source videos. A native probe-level symmetry statement requires absolute predictive utility, low per-checkpoint output error, and improvement over the paired initial encoder. A strict checkpoint representation statement separately requires low direct `E(Mx)` versus `S E(x)` token error and improvement over initialization. Their conjunction supports only training-induced symmetry conditional on the pose schema and architecture.

Constructed oddness, reflection augmentation, odd/even parity, free/zero-origin read-outs, visibility, acquisition, extraction version, dataset annotation, combined nuisance, paired initial encoders, global pooling, and target-component self-consistency are explicit controls or secondary analyses. Measured-control success does not eliminate unmeasured confounding.

Submission requires a complete paper run, exact artifact lineage, and a resolved governance record. Even then, the supported claim is only post-development within-GAVD cross-validated held-out-video performance. Condition folders are annotations, the source video is the independent unit, and no diagnosis, clinical validity, prevalence, treatment, or unseen-person claim is supported.

## Optional external subject gate

Notebook 06 validates prerequisites for a separately reviewed, custodian-indexed dataset with persistent subject IDs. It does not run an external model evaluation. An unseen-person claim remains blocked until that evaluation is implemented, trained without test-subject access, executed, and reported under external-dataset-specific governance.

The fresh-checkout status is intentionally `not configured / not run`, and the
preflight progress should end at 100%; the optional external study does not
block the internal GAVD workflow. To reach the narrower
`manifest contract validated; evaluation not run` state, follow
`docs/EXTERNAL_EVALUATION_GATE.md`: obtain the external dataset's own three
resolved review records, obtain custodian-supplied pseudonymous subject IDs,
create a subject-disjoint `BlazePose33` manifest with non-empty train and test
partitions, and configure its local paths. Do not substitute GAVD identifiers,
infer identity, or copy placeholder references into a completed record.
