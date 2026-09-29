# Human authorship and AI disclosure audit

**Status: human attestation received; the earlier pending status is superseded.** On 26 September 2026, the user confirmed that human authors originated the research ideas, wrote the initial manuscript drafts, made the final edits, verified the full text/results/references, and approved the final version. Those completion statements may now be reported on the basis of that attestation. This audit does not independently establish the provenance of earlier drafts or reproduce the human verification.

Read the complete current `paper-v08.tex` and `appendix.tex`, both bundle instruction files, both archive builders, and both ZIP README entries. The archive check additionally confirmed that both bundled manuscript entries byte-match the canonical manuscript and the Overleaf README matches `OVERLEAF.md`. No manuscript or packaging file was edited by this audit.

## Current official policy

The [ICLR 2027 AI author policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors), fetched on 26 September 2026, requires disclosure in both the paper and submission form. The paper section is mandatory and excluded from the page limit. Authors retain responsibility for accuracy, including AI-assisted material. Conceptual frameworks, hypothesis refinement, methodology feedback, implementation, and interpretation are among required disclosure categories; manuscript drafting/editing, literature assistance, figures, and software creation are recommended categories. The example's assertion of completed review is an example to substantiate, not a reason to invent verification. The policy does not require humans to have written every intermediate draft. [Author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines) separately require anonymous main and supplementary material.

## Claims already present

| Location | Current assertion | Assessment after attestation |
|---|---|---|
| `paper-v08.tex:163` | AI inspected source/notebooks/exports; helped formulate and critique framing; checked literature/instructions; drafted and revised text; created plotting, analysis, and build code; inspected numerical/rendered outputs. Separate assistant reviews covered contribution, methods, statistics, writing, and figures. | Retain these concrete assistance categories. Human-origin research ideas and initial drafts can coexist with later AI framing and drafting assistance. |
| `paper-v08.tex:163` | Assistants did not recruit participants, collect source data, run new source training experiments, or provide independent empirical validation. | Keep scoped to assistance during this manuscript preparation. Do not extrapolate to an unsupported claim that no AI ever helped the earlier study code or data pipeline. |
| `paper-v08.tex:163` | Human authors remain responsible for future verification; the document explicitly does not assert that verification occurred. | The explicit non-attestation is stale and should be replaced with the now-confirmed completed human actions. Responsibility itself remains valid. |
| Main text `:113`; appendix `:84`, `:200`, `:239` | Quantities/exports are verified against receipts, ledgers, hashes, and executable checks; notebook execution has stated limits. | These are technical evidence checks, not implicit claims of human-only verification. No contradiction with the updated attribution. Preserve their limits. |
| `README.md:54`; source builder `:32–33` and bundled source README `:28–29` | Human verification is an author responsibility, separate from technical package completeness. | Not literally a denial of completion, but update for a consistent current status while keeping submission responsibilities distinct. |
| `README.md:22`; `OVERLEAF.md:15`, `:20`, `:55–57` | Codex review says “approve”; reference PDF and ZIP are verified through local compilation. | Clearly assistant/technical review, not human approval. No new human-only attribution is warranted. |

No passage in the complete appendix claims that humans alone generated the ideas, wrote the text, or approved the work. The appendix's claims of completed fits and incomplete notebook execution concern experimental/software evidence, not authorship.

## Accurate disclosure structure

Use three distinct components:

1. **Human contributions, grounded in the user's attestation:** human authors originated the research ideas and wrote the initial drafts. They reviewed AI-assisted revisions, made the final edits, verified the manuscript text, reported results, and references, and approved the final manuscript.
2. **Observed AI assistance:** preserve the existing task-specific disclosure, including framing, drafting/revision, result interpretation, analysis/plotting/build code, literature checking, and assistant review. Distinguish separate assistant review from independent empirical validation.
3. **Accountability:** human authors take responsibility for the final content, including AI-assisted text, claims, and artifacts. Submission metadata and submission-form disclosure remain separate actions; do not claim submission occurred.

A suitable human contribution/approval passage is: “The human authors originated the research ideas and wrote the initial manuscript drafts. They reviewed the AI-assisted revisions, made the final edits, verified the manuscript text, reported results, and references, and approved the final manuscript. The human authors take full responsibility for the final content, including material prepared with AI assistance.” Retain the detailed AI paragraph alongside this passage.

Avoid “entirely human-written,” “AI only corrected grammar,” “all analyses were performed exclusively by humans,” or “independently replicated by humans.” None follows from the attestation, and these formulations would obscure observed assistance. Human verification also does not establish new training runs, clinical validation, complete notebook execution, or an untouched confirmation cohort.

## Current propagation targets and historical records

Replace the stale non-attestation in canonical `paper-v08.tex:163`; rebuild its PDF, source ZIP, Overleaf ZIP, manifests, and dependent current QA/hash receipts. The source ZIP receives that file unchanged; the Overleaf builder renames it to `main.tex`. Update the current `README.md` and generated source README wording if adding the completion status. `OVERLEAF.md` needs no new authorship assertion, only any necessary pagination/build updates.

`reviews/editor-framing-final.md:39` describes human verification as unperformed; add a dated attestation addendum if retaining this as the current review. Preserve prior review conclusions as historical observations. The conditional cautions in `reviews/submission-rules.md:7` and `framing-literature-and-guidelines.md:55` remain correct: do not assert verification that did not occur. `qa/prior-index.md`, `qa/barrier-trial.tex`, `qa/float-trial.tex`, earlier reviews, and history snapshots are historical, not current disclosure sources; do not rewrite their history or mistake their cached wording for the current manuscript.

Pre-edit manuscript SHA-256: `2f0b53bf31344eb52f59ecb3878beecf0d6a76f52de8b11c43505238230852d2`.

## Resolution — 26 September 2026

The canonical disclosure now faithfully states the confirmed human contributions and completed verification/approval, while retaining AI-assisted framing, code and export inspection, result analysis, literature checking, drafting/revision, code generation, and assistant review. It claims neither exclusive human production of intermediate work nor new human tests or independent empirical validation. `reviews/human-authorship-attestation.json` accurately records the user confirmation as its basis and separates facts not inferred. The stale non-attestation has been removed; the current README and generated source README also record the confirmed status. Earlier descriptions above remain the pre-edit audit record.

Inspected the revised PDF's pages 10–11 at normal reading size. The two disclosure paragraphs and references are readable, with no overlap or clipping; disclosure/references remain on pages 10–11 of a 23-page PDF with nine main pages. This is a review of faithful reporting of the author's attestation, not independent certification of the human work.

Resolved source SHA-256: `575bca7406ba7224450bd54a29a7033773b1977266cd0db1f0d685b1f1fb1403`.
Resolved PDF SHA-256: `d6d7ffd9c729ad194bae3ba8824881758105fa5641f33b5c5c38843396605e0a`.
