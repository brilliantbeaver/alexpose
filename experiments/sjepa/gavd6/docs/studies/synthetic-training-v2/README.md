# Paired synthetic 2D pose restoration

This study asks whether paired synthetic supervision improves pose restoration while preserving motion, and whether latent JEPA adds value beyond matched coordinate learning. The proposal, executed analyses, manuscript drafts and future research plans are organized separately below.

Start with the [illustrated proposal](proposal/README.md) for the introduction, methodology and planned experiments. For executed evidence, start with the [complete seed-17 diagnostic analysis](results/seed17-complete-analysis-20260919/README.md). For writing, start with the [manuscript guide](manuscript/README.md).

## Directory guide

| Location | Contents |
| --- | --- |
| [proposal/](proposal/README.md) | Study design and conceptual figures; earlier rendering and figure-review snapshots are retained alongside them. |
| [results/](results/README.md) | Dated analysis bundles, each with its report, tables, figures, recomputation script and verification records. |
| [manuscript/](manuscript/README.md) | Abstract versions 01–10, revision history, evidence audits, reviews, scores and the shared pipeline illustration. |
| [research/](research/README.md) | Paper-development plans, statistical and dataset audits, diagnostic gallery and links to follow-on studies. |
| [development/](development/README.md) | Historical implementation, fixture and workflow reviews, software verification and the file-organization record. |

The follow-on [gait fidelity study](../gait-fidelity/README.md) has its own research home for the proposal and full writeup on preserving side-specific gait measurements under restoration.

## Core study documents

- [Development protocol](protocol.md): data contracts, comparisons, endpoints and decision rules.
- [Protocol review](protocol-review.md) and [review dispositions](review-dispositions.md): findings and their recorded resolutions.
- [Artifact audit](artifact-audit.md): retained evidence from the predecessor synthetic-training study.
- [Literature ledger](literature.md): related work and claim boundaries.
- [Preservation manifest](preservation-manifest.json): hashes of protected predecessor-study files.

These core documents retain their established paths. In particular, runtime tools consume the protocol and preservation manifest directly. The protocol's contents and the preservation manifest are unchanged by the directory organization.

## Execution and provenance

Use the [notebook guide](../../../notebooks/synthetic_training_v2/README.md) and [HAIC runbook](../../../slurm/synthetic-training-v2/README.md) for execution. The [development delivery report](development/report.md) describes the earlier implementation and fixture milestone; it is not the current empirical-results summary.

The proposal README is the current design document. Its [HTML rendering](proposal/proposal.html) and figure reviews are earlier snapshots. Likewise, dated analysis and manuscript verification records describe the files reviewed at their recorded times. The [organization record](development/organization-20260920.json) records the 20 September relocation paths and hashes before and after link and script-path updates. The [gait fidelity relocation record](../gait-fidelity/records/history/relocation-20260921.json) documents its subsequent move into a separate study. Saved numerical tables, JSON evidence and figure files are preserved byte-for-byte.

Historical synthetic/temporal/future-feature studies retain their results and STOPs. This namespace is a local restoration experiment, not an official “S-JEPA v2” release. Source arrays and executed notebooks are retained under unique local output identities and remain excluded from Git by the existing output policy.
