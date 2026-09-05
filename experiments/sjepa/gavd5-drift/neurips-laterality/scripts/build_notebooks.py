"""Materialize the thin, ordered notebooks for the laterality experiment suite.

The notebooks deliberately contain orchestration and audit displays only.  All
scientific logic lives in the importable ``laterality`` package, so a notebook
cannot silently become a second implementation of the protocol.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


SUITE_ROOT = Path(__file__).resolve().parents[1]

KERNEL_METADATA = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "codemirror_mode": {"name": "ipython", "version": 3},
        "file_extension": ".py",
        "mimetype": "text/x-python",
        "name": "python",
        "nbconvert_exporter": "python",
        "pygments_lexer": "ipython3",
        "version": "3",
    },
}


def markdown(source: str):
    return new_markdown_cell(dedent(source).strip())


def code(source: str):
    return new_code_cell(dedent(source).strip(), execution_count=None, outputs=[])


def bootstrap_cell():
    """Fresh-kernel bootstrap that works from the repo, suite, or a descendant."""
    return code(
        """
        from pathlib import Path
        import sys

        import matplotlib.pyplot as plt
        from IPython import get_ipython
        from IPython.display import display


        def locate_suite_root(start: Path | None = None) -> Path:
            start = (start or Path.cwd()).resolve()
            for ancestor in (start, *start.parents):
                for candidate in (ancestor, ancestor / "neurips-laterality"):
                    if (
                        (candidate / "config" / "protocol.json").is_file()
                        and (candidate / "laterality").is_dir()
                    ):
                        return candidate.resolve()
            raise FileNotFoundError(
                "Could not locate neurips-laterality from the current working directory."
            )


        SUITE_ROOT = locate_suite_root()
        if str(SUITE_ROOT) not in sys.path:
            sys.path.insert(0, str(SUITE_ROOT))

        from laterality.config import load_context

        context = load_context(SUITE_ROOT / "config" / "protocol.json")
        shell = get_ipython()
        if shell is not None:
            shell.run_line_magic("matplotlib", "inline")


        def show_inline(figure):
            display(figure)
            plt.close(figure)


        print(
            f"suite={SUITE_ROOT} profile={context.profile} "
            f"artifacts={context.artifact_root} protocol={context.protocol_digest[:12]}"
        )
        """
    )


NOTEBOOKS = {
    "00_protocol_and_governance.ipynb": [
        markdown(
            """
            # 00 — Protocol and governance gate

            This notebook freezes the computational context and reports the submission
            governance gate before any result is interpreted. The primary estimand is
            post-development, within-dataset cross-validated performance on held-out
            **source videos**. A source video is
            the independent unit; persistent person identifiers are unavailable, so the
            workflow cannot support an unseen-person claim. Folder names are dataset
            annotations, not diagnoses, and no diagnostic or clinical claim is made.

            The protocol separates three claims that must not be collapsed: an analytic
            wrapper can force odd output; an anatomy-aware encoder-plus-probe can exhibit
            useful native odd behavior; and a checkpoint can satisfy a direct strict
            token-equivariance test. Only the last two are empirical, and neither removes
            the BlazePose schema, preprocessing, architecture, or source-sampling
            assumptions supplied to the experiment.

            The checked-in governance record intentionally fails closed until an
            institutional ethics determination, a data-use review, and a derived-pose
            release review each have a dated internal reference. Public availability is
            not a substitute for those determinations. This suite never redistributes
            raw video or identity-bearing frames. Linkable identifiers, derived poses,
            embeddings, and checkpoints may be released only as completed reviews permit.
            """
        ),
        markdown(
            """
            ## What this notebook is testing

            Notebook 00 does not test a model. It tests whether the project is about to run
            under the intended computational rules and whether its submission/release gate
            is administratively complete. Think of it as the label on a sealed experiment:
            it states the question, the unit being analyzed, the selected model recipes,
            and the rules that will later decide what can be claimed.

            An **estimand** is the quantity the study intends to estimate. Here it is
            post-development, within-dataset cross-validated performance on held-out source
            videos from the Gait Abnormality in Video Dataset (GAVD). "Within-dataset"
            means the result stays inside GAVD; "held-out" means a source video is evaluated
            by a model that did not train on that source.
            It does not mean that a person is known to be absent from every other video,
            because persistent person identifiers are unavailable.

            The protocol also separates three left–right ideas. An **odd output** obeys
            $f(Mx)=-f(x)$ after input $x$ is anatomically mirrored by $M$; a mathematical
            wrapper can force this property. **Native probe behavior** asks whether an
            unconstrained read-out already behaves usefully. **Representation
            equivariance** asks the stricter question of whether internal tokens themselves
            transform according to the registered anatomical joint swap. Later notebooks
            test these separately so success on an engineered wrapper cannot be described
            as symmetry learned by the encoder.

            The governance gate uses logical **AND**, not majority vote. Every review must
            be marked `resolved` and must include both an approved internal reference and a
            date. `Unresolved` means that the repository has no recorded determination; it
            does not mean “rejected,” and it must not be guessed from public availability.

            <svg viewBox="0 0 980 230" width="100%" role="img"
                 aria-labelledby="governance-flow-title governance-flow-description"
                 xmlns="http://www.w3.org/2000/svg">
              <title id="governance-flow-title">Submission governance AND gate</title>
              <desc id="governance-flow-description">Ethics, data-use, and derived-pose
              release reviews must all be resolved with a reference and date before the
              submission and release gate is ready.</desc>
              <defs><marker id="arrow00" markerWidth="8" markerHeight="8" refX="7"
                refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#475569"/></marker></defs>
              <style>
                .review00{fill:#fff1f2;stroke:#be123c;stroke-width:1.5}
                .gate00{fill:#f8fafc;stroke:#334155;stroke-width:1.5}
                .blocked00{fill:#fee2e2;stroke:#991b1b;stroke-width:2}
                .line00{stroke:#475569;stroke-width:1.8;fill:none;marker-end:url(#arrow00)}
                .h00{font:600 15px system-ui,sans-serif;fill:#0f172a}
                .s00{font:12px system-ui,sans-serif;fill:#475569}
                .r00{font:600 12px system-ui,sans-serif;fill:#9f1239}
              </style>
              <rect class="review00" x="20" y="15" width="255" height="55" rx="9"/>
              <rect class="review00" x="20" y="87" width="255" height="55" rx="9"/>
              <rect class="review00" x="20" y="159" width="255" height="55" rx="9"/>
              <text class="h00" x="147" y="38" text-anchor="middle">Ethics determination</text>
              <text class="r00" x="147" y="57" text-anchor="middle">currently unresolved</text>
              <text class="h00" x="147" y="110" text-anchor="middle">Data-use review</text>
              <text class="r00" x="147" y="129" text-anchor="middle">currently unresolved</text>
              <text class="h00" x="147" y="182" text-anchor="middle">Derived-pose release review</text>
              <text class="r00" x="147" y="201" text-anchor="middle">currently unresolved</text>
              <rect class="gate00" x="370" y="75" width="180" height="80" rx="10"/>
              <text class="h00" x="460" y="105" text-anchor="middle">All three complete?</text>
              <text class="s00" x="460" y="128" text-anchor="middle">status + reference + date</text>
              <text class="h00" x="460" y="148" text-anchor="middle">AND</text>
              <path class="line00" d="M275 42 C330 42 330 95 370 95"/>
              <path class="line00" d="M275 114 L370 114"/>
              <path class="line00" d="M275 186 C330 186 330 135 370 135"/>
              <rect class="blocked00" x="650" y="75" width="290" height="80" rx="10"/>
              <text class="h00" x="795" y="105" text-anchor="middle">Submission/release readiness</text>
              <text class="r00" x="795" y="132" text-anchor="middle">BLOCKED until every input is complete</text>
              <path class="line00" d="M550 115 L650 115"/>
            </svg>
            """
        ),
        markdown(
            """
            ## How to read the protocol and governance progress display

            This audit has three short stages: validate the governance record, assemble the
            protocol snapshot, and render the status-only figure. It normally finishes in
            seconds. The bar makes the active check explicit and turns an exception into a
            visible failed stage. Because the stages are brief and differently sized, its ETA
            is only a rough indication; `estimating…` is normal at the beginning.

            A completed progress bar means the files were read and the checks ran. It does
            not mean that the governance gate passed. The separate red or green governance
            figure remains authoritative about readiness.
            """
        ),
        bootstrap_cell(),
        code(
            """
            from laterality.config import model_config
            from laterality.governance import load_governance, submission_readiness
            from laterality.visualization import governance_figure
            from notebook_progress import NotebookTaskProgress

            governance_progress = NotebookTaskProgress(
                "Protocol and governance audit progress",
                "stage",
                refresh_seconds=0.25,
            )
            governance_progress.start(3, profile=context.profile)

            with governance_progress.unit(1, "Load and validate governance record"):
                governance_path = SUITE_ROOT / "governance" / "status.json"
                governance_payload = load_governance(governance_path)
                governance_readiness = submission_readiness(governance_payload)

            with governance_progress.unit(2, "Assemble the frozen protocol snapshot"):
                protocol_snapshot = {
                    "profile": context.profile,
                    "synthetic_smoke": not context.is_paper,
                    "protocol_digest": context.protocol_digest,
                    "claim": context.protocol["claim_boundary"]["primary"],
                    "unsupported_claims": context.protocol["claim_boundary"]["not_supported"],
                    "independent_unit": context.protocol["claim_boundary"]["independent_unit"],
                    "selected_folds": list(context.folds),
                    "selected_seeds": list(context.seeds),
                    "selected_variants": list(context.variants),
                    "model": model_config(context),
                    "primary_lane": context.protocol["evaluation"]["primary_lane"],
                    "constructed_repair_lane": context.protocol["evaluation"][
                        "constructed_repair_lane"
                    ],
                    "primary_seed_estimand": context.protocol["evaluation"][
                        "primary_seed_estimand"
                    ],
                    "representation_equivariance": context.protocol["evaluation"][
                        "representation_equivariance"
                    ],
                    "decision_rules": context.protocol["evaluation"]["decision_rules"],
                }

            with governance_progress.unit(3, "Render the status-only governance figure"):
                show_inline(governance_figure(context, governance_payload))
            governance_progress.complete(status="Protocol and governance audit complete")
            protocol_snapshot, governance_readiness
            """
        ),
        markdown(
            """
            ## How to interpret the current output

            **Read the red bars as gate states, not scientific failures.** The checked-in
            record currently marks the ethics determination, data-use review, and
            derived-pose release review as `unresolved`; their reference and date fields
            are empty. `governance_readiness["ready"]` is therefore `False`, and all three
            names appear in its `unresolved` list. The plot suppresses references by design
            and shows only whether each required record is complete.

            This means the repository does not currently authorize a claim that the work
            is submission-ready or that poses, embeddings, or checkpoints may be released.
            It does not itself determine whether a particular institution permits local
            analysis while reviews are pending; that question must follow the applicable
            institutional process. Never change a status to make the bar green unless the
            determination actually exists. Record only the approved internal reference and
            date—do not copy confidential review documents into the repository.

            **Then read the protocol snapshot as a contract, not a result.** For the paper
            profile it should identify all five outer folds, seeds 42–46, and the two
            variants `vanilla` and `reflection_augmented`. The model snapshot specifies 64
            frames, 33 landmarks, three coordinates, four-frame temporal patches, a
            96-dimensional embedding, four encoder layers, two predictor layers, and four
            attention heads. These values tell later artifacts what implementation they
            must match; they do not say that this model performs well.

            The primary lane `learned_single_free` means a learned encoder, one original
            input pass, and an unconstrained linear read-out. The constructed repair lane
            `learned_two_pass_odd_zero` combines original and mirrored representations to
            force odd features and uses a zero-origin read-out. Exact oddness in that lane
            proves the construction, not that the unconstrained encoder learned symmetry.
            Likewise, the registered $R^2$ and equivariance margins are future decision
            rules, not scores already achieved.

            SHA stands for Secure Hash Algorithm; SHA-256 produces the 256-bit content
            fingerprint used as the protocol digest. The displayed prefix should begin
            `6f7baefbda07` for the current
            locked protocol. A different digest is not automatically wrong, but it means
            results belong to a different protocol and must not be mixed with these
            artifacts.

            The narrow conclusion from notebook 00 is: **the paper computation is precisely
            specified, but the current submission/release governance gate is not complete.**
            This remains true even if every later model finishes successfully.

            A `smoke` run uses generated poses and a tiny model only to test plumbing,
            lineage checks, and algebraic invariants. Its metrics are **not empirical
            evidence** and must never enter a paper table. Likewise, a completed paper run
            does not override an unresolved governance gate.
            """
        ),
    ],
    "01_cohort_and_target_audit.ipynb": [
        markdown(
            """
            # 01 — Cohort and paired-valid target audit

            This notebook decides which pose sequences are suitable for the later
            analysis and checks that the left-versus-right motion measurement behaves
            as intended. A **pose sequence** is a series of video frames represented by
            estimated body-landmark coordinates instead of by the original images. A
            **cohort** is simply the collection of sequences retained for analysis.

            The data come from the Gait Abnormality in Video Dataset (GAVD). Although
            that is the dataset's published name, this project treats its folder labels
            only as dataset annotations. This notebook does not diagnose anyone and
            does not validate a clinical measurement.

            In the `smoke` profile, the poses are generated test data. Smoke mode checks
            that the software works from beginning to end; its output is not scientific
            evidence. In the `paper` profile, the same checks run on the empirical data.
            """
        ),
        markdown(
            """
            ## What this notebook is checking

            This notebook turns the available pose files into the **locked analysis
            cohort** used by later notebooks. "Locked" means that the acceptance rules
            were written down before examining the later model results. Keeping those
            rules fixed helps prevent a result from being improved by quietly changing
            which sequences are included.

            Read the notebook as a sequence of six questions:

            1. Which protocol and data profile are active?
            2. Which pose sequences pass the pre-specified quality-control (QC) rules?
            3. Can the left-versus-right motion target be computed from observed
               coordinates alone?
            4. Does anatomical mirroring reverse the target exactly and undo itself
               when applied twice?
            5. Were the audited cohort and its history saved for the later stages?
            6. What do the final chart and audit numbers mean?

            This is a **data and implementation audit**, not a model-performance result.
            No machine-learning model is trained and no diagnosis is predicted here.
            """
        ),
        markdown(
            """
            ## Plain-language glossary

            Here are the main terms used below:

            - **Quality control (QC)** means applying the pre-written checks that decide
              whether a pose sequence contains enough usable information.
            - A **body landmark** is an estimated point such as a shoulder, knee, ankle,
              heel, or foot point. Each point has three coordinates, called `x`, `y`,
              and `z`.
            - A **frame transition** is the movement from one video frame to the next
              available frame.
            - **Paired-valid** means that a left landmark and its matching right
              landmark are both visible at the start and end of the same transition.
              Comparing the two sides on exactly the same transitions avoids giving one
              side an unfair advantage because it was visible more often.
            - The **target** is the single number that later models will try to predict.
              Here it summarizes relative left-versus-right motion. It is derived from
              coordinates and is not a clinical outcome.
            - **Interpolation** means filling a short gap by estimating values between
              two observed points. Interpolation may make model input easier to use, but
              interpolated values are not allowed to define the target.
            - A **sentinel** is a placeholder stored where a coordinate is invalid. The
              validity mask tells the program to ignore it, whatever its numeric value.
            - In this notebook, **authorized landmarks** are simply the body landmarks
              selected in the frozen protocol for model input. "Authorized" here does
              not mean that an ethics or data-release review has been approved.
            - A **patch** is one four-frame block used by the model. A complete patch has
              valid information throughout that block.
            - A **finite** target is an ordinary usable number, rather than a missing or
              undefined value.
            - **Provenance** means the recorded history of the data, including which
              files, pose model, and extraction version produced them.
            - A **digest** is a long content fingerprint. SHA stands for Secure Hash
              Algorithm; **SHA-256** produces a 256-bit fingerprint. If relevant content
              changes, its digest changes, allowing later notebooks to detect a mismatch.
            """
        ),
        markdown(
            """
            ## Step 1 — Confirm the run context

            The next cell locates the experiment suite, loads the frozen protocol, and
            prints four useful identifiers:

            - `suite` is the folder containing this experiment;
            - `profile` says whether this is synthetic `smoke` data or the empirical
              `paper` run;
            - `artifacts` is the folder where this notebook writes its derived files;
            - `protocol` shows the first characters of the protocol digest. The full
              digest is a fingerprint of the rules and settings. Later files must carry
              the same fingerprint, which prevents different protocol versions from
              being mixed accidentally.

            Check this line before interpreting anything below. A smoke-profile figure
            tests the plumbing only and is not scientific evidence.
            """
        ),
        bootstrap_cell(),
        markdown(
            """
            ## Steps 2–5 — Build and verify the cohort

            The next cell performs the substantive audit in a fixed order:

            **Step 2: inventory and prepare the pose files.** The program first checks
            that the file counts and fingerprints match the frozen inventory. It then
            converts each sequence into 64 time steps with 33 body landmarks. Short gaps
            may be interpolated for model input, and a separate validity mask records
            which coordinates the model may use. The untouched observed-coordinate path
            is kept separate for calculating the target.

            **Step 3: apply locked quality control.** The `prepare_cohort` function
            checks whether a sequence contains enough usable landmark coverage, enough
            complete four-frame blocks, and enough information to calculate the target.
            A sequence either enters the cohort or receives an explicit exclusion
            reason. Importantly, acceptance does not depend on whether the target is
            positive, negative, large, or small.

            **Step 4: calculate and independently reconstruct the target.** Five matching
            left/right landmark pairs are used: shoulders, knees, ankles, heels, and
            foot-index points. For each pair, the program:

            1. keeps only transitions where both landmarks are observed at both ends;
            2. calculates how fast each side moved;
            3. takes the median, or middle, speed for each side; and
            4. computes `(left speed - right speed) / (left speed + right speed)`.

            The five pair values are averaged to make one target. A simple example is a
            left speed of 3 and right speed of 2, which gives `(3 - 2) / (3 + 2) = 0.2`.
            Positive values indicate more left-side motion under this definition;
            negative values indicate more right-side motion. The code recalculates this
            value from the saved pair contrasts and checks that the answer matches to
            within a tiny floating-point tolerance. "Floating point" is the computer's
            approximate way of storing decimal numbers.

            **Step 5: test the mirror rules.** An anatomical mirror flips the horizontal
            coordinate and swaps each named left landmark with its right partner. That
            operation must reverse the sign of the target. Applying it twice must return
            the original coordinates and validity mask. This mirror-twice property is
            sometimes called an **involution**. The program also replaces invalid
            coordinates with enormous sentinel numbers and confirms that the target does
            not change. If any assertion fails, execution stops because later results
            would not have a trustworthy left-versus-right interpretation.

            Finally, the accepted arrays, manifest, and metadata are saved with a cohort
            digest. A **manifest** is a table listing the retained sequences. **Metadata**
            is information describing the data rather than the pose values themselves.
            Later notebooks use the digest to prove that they loaded this exact handoff.
            """
        ),
        markdown(
            """
            ## The paired-valid target in one picture

            The same visibility rule is applied to both sides before speeds are compared.
            For example, if the left knee is visible across 20 transitions but the right
            knee is visible across only 12 of those, the calculation does not compare 20
            left transitions with 12 right transitions. It keeps only transitions jointly
            visible for both knees at both endpoints. This is what **paired-valid** means.

            <svg viewBox="0 0 1050 285" width="100%" role="img"
                 aria-labelledby="target-flow-title target-flow-description"
                 xmlns="http://www.w3.org/2000/svg">
              <title id="target-flow-title">Construction and mirror audit of the paired-valid target</title>
              <desc id="target-flow-description">Observed coordinates and validity masks
              select jointly visible transitions. Median left and right speeds form a
              normalized contrast for each of five landmark pairs, which are averaged into
              the target. Anatomical mirroring must negate that target.</desc>
              <defs><marker id="arrow01" markerWidth="8" markerHeight="8" refX="7"
                refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#475569"/></marker></defs>
              <style>
                .box01{fill:#f8fafc;stroke:#334155;stroke-width:1.5}
                .valid01{fill:#ecfdf5;stroke:#047857;stroke-width:1.5}
                .target01{fill:#eff6ff;stroke:#2563eb;stroke-width:1.8}
                .mirror01{fill:#fff7ed;stroke:#c2410c;stroke-width:1.5}
                .line01{stroke:#475569;stroke-width:1.8;fill:none;marker-end:url(#arrow01)}
                .h01{font:600 14px system-ui,sans-serif;fill:#0f172a}
                .s01{font:12px system-ui,sans-serif;fill:#475569}
              </style>
              <rect class="box01" x="15" y="35" width="170" height="78" rx="9"/>
              <text class="h01" x="100" y="64" text-anchor="middle">Observed coordinates</text>
              <text class="s01" x="100" y="85" text-anchor="middle">plus validity masks</text>
              <text class="s01" x="100" y="102" text-anchor="middle">no interpolated target values</text>
              <rect class="valid01" x="235" y="35" width="190" height="78" rx="9"/>
              <text class="h01" x="330" y="64" text-anchor="middle">Paired-valid transitions</text>
              <text class="s01" x="330" y="85" text-anchor="middle">both sides visible</text>
              <text class="s01" x="330" y="102" text-anchor="middle">at both endpoints</text>
              <rect class="box01" x="475" y="20" width="220" height="108" rx="9"/>
              <text class="h01" x="585" y="49" text-anchor="middle">One landmark pair</text>
              <text class="s01" x="585" y="70" text-anchor="middle">median left speed L</text>
              <text class="s01" x="585" y="88" text-anchor="middle">median right speed R</text>
              <text class="h01" x="585" y="112" text-anchor="middle">contrast = (L − R) / (L + R)</text>
              <rect class="target01" x="745" y="35" width="285" height="78" rx="9"/>
              <text class="h01" x="887" y="64" text-anchor="middle">Coordinate-derived target</text>
              <text class="s01" x="887" y="85" text-anchor="middle">mean across five registered pairs</text>
              <text class="s01" x="887" y="102" text-anchor="middle">shoulders, knees, ankles, heels, foot-index</text>
              <path class="line01" d="M185 74 L235 74"/><path class="line01" d="M425 74 L475 74"/><path class="line01" d="M695 74 L745 74"/>
              <rect class="mirror01" x="235" y="185" width="460" height="72" rx="9"/>
              <text class="h01" x="465" y="213" text-anchor="middle">Anatomical mirror audit</text>
              <text class="s01" x="465" y="235" text-anchor="middle">flip horizontal coordinate + swap left/right landmarks</text>
              <rect class="target01" x="745" y="185" width="285" height="72" rx="9"/>
              <text class="h01" x="887" y="213" text-anchor="middle">Required result</text>
              <text class="h01" x="887" y="238" text-anchor="middle">target(mirror(x)) = −target(x)</text>
              <path class="line01" d="M330 113 L330 185"/><path class="line01" d="M695 221 L745 221"/>
            </svg>

            A contrast is bounded between -1 and +1 when speeds are nonnegative and their
            sum is positive. A value of
            +0.2 means the median left speed exceeded the right speed under this formula; it
            does not mean “20% impairment.” Mirroring swaps $L$ and $R$, so the numerator
            changes sign while the denominator stays the same. That algebra explains why
            exact sign reversal is the required implementation check.
            """
        ),
        markdown(
            """
            ## How to read the cohort-audit progress display

            The five stages cover pose inventory and quality control, independent target
            reconstruction, mirror-twice checks, artifact saving, and figure rendering. The
            mirror stage also reports how many accepted sequences have been checked, so a
            large empirical cohort does not look frozen.

            ETA is based on completed stage durations, but cohort preparation is usually much
            more expensive than plotting or saving. The estimate can therefore move sharply
            after the first stage. Progress describes completed computation only; the final
            attrition counts and assertion results determine whether the cohort is valid.
            """
        ),
        code(
            """
            from collections import Counter

            import numpy as np

            from laterality.data import prepare_cohort, save_cohort
            from laterality.geometry import anatomical_mirror
            from laterality.visualization import cohort_figure
            from notebook_progress import NotebookTaskProgress

            cohort_progress = NotebookTaskProgress(
                "Cohort and target audit progress",
                "stage",
                refresh_seconds=0.25,
            )
            cohort_progress.start(5, profile=context.profile)

            with cohort_progress.unit(1, "Inventory, prepare, and quality-control pose sequences"):
                cohort = prepare_cohort(context)

            with cohort_progress.unit(2, "Reconstruct targets and verify target contracts"):
                reconstructed_targets = np.asarray(
                    [
                        np.mean(row[np.isfinite(row)])
                        for row in cohort.pair_contrasts
                    ],
                    dtype=np.float64,
                )
                assert np.allclose(
                    reconstructed_targets,
                    cohort.table["target"].to_numpy(dtype=np.float64),
                    rtol=0.0,
                    atol=1e-12,
                )

                target_contract = cohort.attrition["target_contract"]
                assert target_contract["checked_finite_targets"] >= len(cohort.table)
                assert target_contract["maximum_mirror_antisymmetry_error"] <= 1e-10
                assert target_contract["maximum_invalid_sentinel_error"] <= 1e-12

            checked_involutions = 0
            with cohort_progress.unit(
                3,
                "Verify mirror-twice restoration for every accepted sequence",
                total_steps=len(cohort.table),
            ):
                for checked_involutions, (xyz, valid) in enumerate(
                    zip(cohort.model_xyz, cohort.model_valid),
                    start=1,
                ):
                    mirrored_xyz, mirrored_valid = anatomical_mirror(xyz, valid)
                    restored_xyz, restored_valid = anatomical_mirror(
                        mirrored_xyz, mirrored_valid
                    )
                    assert np.array_equal(restored_xyz, xyz)
                    assert np.array_equal(restored_valid, valid)
                    cohort_progress.update_unit(completed_steps=checked_involutions)

            with cohort_progress.unit(4, "Save the verified cohort handoff and summarize exclusions"):
                artifact_paths = save_cohort(context, cohort)
                exclusion_reason_counts = Counter(
                    item["reason"] for item in cohort.attrition["exclusions"]
                )
                attrition_summary = {
                    key: value
                    for key, value in cohort.attrition.items()
                    if key != "exclusions"
                }
                attrition_summary["exclusion_reason_counts"] = dict(
                    sorted(exclusion_reason_counts.items())
                )

            with cohort_progress.unit(5, "Render the attrition and target-distribution figure"):
                show_inline(cohort_figure(context, cohort))
            cohort_progress.complete(status="Cohort and target audit complete")
            {
                "attrition": attrition_summary,
                "cohort_digest": cohort.cohort_digest,
                "target_contract": target_contract,
                "checked_model_lane_mirror_involutions": checked_involutions,
                "artifacts": {key: str(value) for key, value in artifact_paths.items()},
            }
            """
        ),
        markdown(
            """
            ## Step 6 — Read the figure and audit record

            **Start with the left panel.** "Attrition" means the reduction from the
            starting set to the usable set. `Input poses` is the number of available pose
            sequences presented to quality control. `QC eligible` is the subset retained
            for analysis, and `Excluded` is the remainder. In the displayed paper-profile
            run, 625 of 642 sequences are retained and 17 are excluded. Thus about 97.4%
            are retained and 2.6% are excluded. The excluded bar is not a model error
            rate and does not label people as good or bad data. It records sequences that
            could not satisfy the fixed coverage or target-computability rules.

            **Then read the right panel.** This is a histogram, which groups numeric
            values into ranges and uses bar height to show how many sequences fall in
            each range. The horizontal axis is the coordinate-derived motion contrast,
            and the vertical axis is the number of retained sequences. The black line
            marks zero. Values to its right indicate relatively more left-side motion;
            values to its left indicate relatively more right-side motion. Values near
            zero indicate similar motion on the two sides under this specific formula.
            The presence of values on both sides confirms that the target keeps a sign.
            It does not establish that the sample is clinically symmetric, unbiased, or
            representative of a population. This chart is not a diagnosis, clinical
            scale, class label, or model-performance score.

            **Finally, read the printed audit record below the plot.** The main fields
            mean the following:

            - `input_sequences`, `accepted_sequences`, and `excluded_sequences` repeat
              the counts shown in the left panel;
            - `accepted_sources` counts source videos rather than pose sequences. One
              source video can produce more than one sequence, so this number is smaller;
            - `inventory` contains the frozen file counts and provenance fingerprints;
            - `exclusion_reason_counts` explains why sequences failed quality control.
              A semicolon means that a sequence failed more than one check;
            - `target_contract` reports the mirror and invalid-sentinel test errors.
              Zero is ideal, and extremely small values can arise from decimal rounding;
            - `checked_model_lane_mirror_involutions` counts how many processed model
              inputs passed the mirror-twice check;
            - `cohort_digest` is the content fingerprint used for later history checks;
            - `artifacts` lists the saved handoff files. The compressed NumPy file
              (`.npz`) stores arrays, the comma-separated values file (`.csv`) stores the
              manifest table, and the JavaScript Object Notation file (`.json`) stores
              metadata.

            The inventory gives useful context for the first bar. The run began with
            666 annotation files. Of these, 642 had a matching pose archive, meaning a
            saved bundle of extracted body coordinates; 24 did not. That is why the
            chart begins at 642 rather than 666. The annotations refer to 103 source
            videos, and the final cohort contains usable sequences from 93 of them.
            `extraction_version_counts` shows which software version produced each pose
            archive, while `pose_model` names the body-landmark detector. These values
            are provenance checks, not measures of scientific performance.

            The exclusion labels can contain `target_not_computable`, meaning the five
            left/right pairs did not all have enough shared visible transitions;
            `insufficient_authorized_coverage`, meaning too little of the selected model
            input was valid; or `insufficient_authorized_patches`, meaning too few
            complete four-frame blocks remained.

            In this run, `checked_finite_targets` can be larger than the 625 accepted
            sequences. That is expected: the target checks run before all coverage rules,
            so a sequence can have a valid target and still be excluded for insufficient
            model-input coverage.

            The key takeaway is modest: the cohort is internally consistent with the
            frozen quality-control and mirror rules. This figure alone says nothing about whether a
            model learns laterality or generalizes to new videos or people.
            """
        ),
        markdown(
            """
            ## Specific interpretation of the current paper-profile findings

            The complete inventory forms a useful funnel. There are 666 annotation files,
            642 matching pose archives, and therefore 24 annotations without a usable pose
            archive at this stage. Locked quality control then retains 625 of the 642 pose
            sequences and excludes 17. The retention rate is 97.35%. The accepted sequences
            come from 93 source videos, while the official inventory contains 103 source
            videos. Sequence and source counts are different because one source video can
            yield multiple extracted sequences.

            The 17 exclusion records break down into four exact combinations:

            | Recorded reason combination | Sequences |
            |---|---:|
            | target not computable + insufficient authorized coverage | 8 |
            | target not computable + insufficient coverage + insufficient complete patches | 4 |
            | insufficient authorized coverage only | 3 |
            | target not computable only | 2 |

            These rows describe failed technical requirements, not participants or clinical
            categories. Because one row can fail several checks, the semicolon-separated
            combination should be read as one sequence's complete reason record. The target
            was finite for 628 sequences even though only 625 entered the cohort. This is
            expected: three sequences could form a target but failed a later model-input
            coverage rule.

            The accepted target distribution spans approximately -0.195 to +0.215. Its mean
            is -0.0061, median is -0.0050, and middle 50% runs from about -0.0434 to +0.0286.
            There are 336 negative and 289 positive sequence values, with no exact zeros.
            This tells us that the computed quantity covers both signs and is centered near
            zero at the sequence level. It does not show clinical balance, population
            representativeness, or absence of confounding. The histogram counts sequences,
            so a source producing many sequences contributes more bars; later performance
            metrics deliberately give each source equal total weight.

            Both recorded algebraic error maxima are exactly 0.0 in this run:
            `maximum_mirror_antisymmetry_error` and
            `maximum_invalid_sentinel_error`. All 625 accepted model inputs also passed the
            mirror-twice restoration check. These are strong implementation checks: the
            mirror and validity-mask code obey the frozen mathematical contract on the
            audited inputs. They do not validate the target as a medical measurement and do
            not show that a learned model will recover it.

            The narrow conclusion is: **the empirical cohort handoff is complete and
            internally consistent with the locked inventory, quality-control, validity,
            target-reconstruction, and mirror rules.** The next notebook may split these 93
            sources. No model-performance claim begins until the held-out evaluation.
            """
        ),
    ],
    "02_source_level_splits.ipynb": [
        markdown(
            """
            # 02 — Source-level outer and inner splits

            Splits are assigned on a table with exactly one row per source video and
            expanded to sequences afterward. Every clip, mirror, and augmentation from a
            video inherits that video's fold. Outer-test sources are never used for
            encoder training, checkpoint selection, read-out tuning, or preprocessing
            statistics. Inner folds tune read-outs using outer-training sources only.

            Dataset annotations keep source counts reasonably balanced across folds and
            later support one explicitly named confounding control; they are not diagnoses
            or primary prediction targets. Multiple videos
            may still depict the same unidentified person, so these folds establish
            held-out-video—not held-out-person—evaluation.
            """
        ),
        markdown(
            """
            ## What experiment-design problem does this notebook solve?

            A **split** assigns observations to roles before model fitting. The dangerous
            shortcut would be to split 625 sequences independently. Sequences cut from the
            same source video share scene, camera, clothing, pose-extraction behavior, and
            often adjacent motion. Putting sibling sequences in both training and testing
            would let the model encounter source-specific information before evaluation.
            This is called **data leakage**.

            Notebook 02 prevents that problem by collapsing the cohort to 93 source videos,
            assigning folds on that source table, and only then expanding assignments back
            to sequences. It creates two nested levels:

            - The five **outer folds** estimate held-out-source performance. One fold is
              sealed as the final test set while the other four supply training data.
            - Four **inner folds** divide only the current outer-training sources. Notebook
              04 uses them to select the ridge regularization strength of the linear
              read-out. They never open the outer-test sources and do not select or stop the
              encoder trained in notebook 03.

            **Stratification** uses dataset annotations to keep source counts as even as the
            available integer counts allow. It does not train a diagnosis and does not
            guarantee equal sequence counts, because sources yield different numbers of
            sequences. The split seed `20260904` makes the assignment reproducible; it is
            separate from optimization seeds 42–46 used during model training.

            <svg viewBox="0 0 1040 330" width="100%" role="img"
                 aria-labelledby="split-flow-title split-flow-description"
                 xmlns="http://www.w3.org/2000/svg">
              <title id="split-flow-title">Nested source-video splitting workflow</title>
              <desc id="split-flow-description">Ninety-three source videos are assigned to
              five outer folds. For one outer fold, 18 or 19 test sources remain sealed and
              74 or 75 training sources are subdivided into four inner read-out folds. All
              sequences inherit the role of their source video.</desc>
              <defs><marker id="arrow02" markerWidth="8" markerHeight="8" refX="7"
                refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#475569"/></marker></defs>
              <style>
                .box02{fill:#f8fafc;stroke:#334155;stroke-width:1.5}
                .train02{fill:#ecfdf5;stroke:#047857;stroke-width:1.5}
                .test02{fill:#fff7ed;stroke:#c2410c;stroke-width:1.7}
                .inner02{fill:#eff6ff;stroke:#2563eb;stroke-width:1.5}
                .line02{stroke:#475569;stroke-width:1.8;fill:none;marker-end:url(#arrow02)}
                .h02{font:600 14px system-ui,sans-serif;fill:#0f172a}
                .s02{font:12px system-ui,sans-serif;fill:#475569}
              </style>
              <rect class="box02" x="15" y="115" width="175" height="88" rx="9"/>
              <text class="h02" x="102" y="145" text-anchor="middle">Locked cohort</text>
              <text class="s02" x="102" y="168" text-anchor="middle">93 source videos</text>
              <text class="s02" x="102" y="188" text-anchor="middle">625 sequences inherit source</text>
              <rect class="box02" x="240" y="115" width="180" height="88" rx="9"/>
              <text class="h02" x="330" y="145" text-anchor="middle">Five outer folds</text>
              <text class="s02" x="330" y="168" text-anchor="middle">stratified by annotation</text>
              <text class="s02" x="330" y="188" text-anchor="middle">each source tests once</text>
              <path class="line02" d="M190 159 L240 159"/>
              <rect class="train02" x="480" y="45" width="205" height="88" rx="9"/>
              <text class="h02" x="582" y="75" text-anchor="middle">Outer-training sources</text>
              <text class="s02" x="582" y="98" text-anchor="middle">74 or 75 sources</text>
              <text class="s02" x="582" y="118" text-anchor="middle">encoder may use these</text>
              <rect class="test02" x="480" y="220" width="205" height="88" rx="9"/>
              <text class="h02" x="582" y="250" text-anchor="middle">Outer-test sources</text>
              <text class="s02" x="582" y="273" text-anchor="middle">18 or 19 sources</text>
              <text class="s02" x="582" y="293" text-anchor="middle">sealed until final evaluation</text>
              <path class="line02" d="M420 145 C450 145 450 89 480 89"/>
              <path class="line02" d="M420 175 C450 175 450 264 480 264"/>
              <rect class="inner02" x="755" y="25" width="260" height="145" rx="9"/>
              <text class="h02" x="885" y="55" text-anchor="middle">Four inner read-out folds</text>
              <text class="s02" x="885" y="79" text-anchor="middle">inner train: 55–57 sources</text>
              <text class="s02" x="885" y="99" text-anchor="middle">inner validation: 18–19 sources</text>
              <text class="s02" x="885" y="124" text-anchor="middle">select ridge penalty only</text>
              <text class="s02" x="885" y="146" text-anchor="middle">outer test remains untouched</text>
              <path class="line02" d="M685 89 L755 89"/>
              <rect class="test02" x="755" y="220" width="260" height="88" rx="9"/>
              <text class="h02" x="885" y="250" text-anchor="middle">Final held-out evaluation</text>
              <text class="s02" x="885" y="273" text-anchor="middle">one outer fold at a time</text>
              <text class="s02" x="885" y="293" text-anchor="middle">performed in notebook 04</text>
              <path class="line02" d="M685 264 L755 264"/>
            </svg>
            """
        ),
        markdown(
            """
            ## How to read the split-construction progress display

            Four short stages load the locked cohort, assign nested source-video folds, save
            the lineage-bound split manifest, and render the source-count summary. The work
            is deterministic and normally completes in seconds. A full bar confirms that the
            stages ran; the printed fold census and non-overlap assertions remain the evidence
            that the split itself is complete and leakage-controlled.
            """
        ),
        bootstrap_cell(),
        code(
            """
            from laterality.data import load_cohort
            from laterality.splitting import build_source_splits, save_splits
            from laterality.visualization import split_figure
            from notebook_progress import NotebookTaskProgress

            split_progress = NotebookTaskProgress(
                "Source-level split construction progress",
                "stage",
                refresh_seconds=0.25,
            )
            split_progress.start(4, profile=context.profile)

            with split_progress.unit(1, "Load and validate the locked cohort"):
                cohort = load_cohort(context)

            with split_progress.unit(2, "Assign nested folds on the source-video table"):
                split_config = context.protocol["splits"]
                splits = build_source_splits(
                    cohort.table,
                    context.protocol["data"]["conditions"],
                    outer_folds=split_config["outer_folds"],
                    inner_folds=split_config["inner_folds"],
                    seed=split_config["seed"],
                )

            with split_progress.unit(3, "Validate lineage and save the split manifest"):
                split_artifact = save_splits(context, cohort, splits)

            with split_progress.unit(4, "Summarize and render source-count balance"):
                fold_summary = [
                    {
                        "fold": fold["fold"],
                        "train_sources": len(fold["train_sources"]),
                        "test_sources": len(fold["test_sources"]),
                        "train_source_counts": fold["train_source_counts"],
                        "test_source_counts": fold["test_source_counts"],
                        "inner_folds": len(fold["inner_readout_folds"]),
                    }
                    for fold in splits["folds"]
                ]
                show_inline(split_figure(context, splits))
            split_progress.complete(status="Source-level split construction complete")
            {
                "split_artifact": str(split_artifact),
                "source_census": splits["source_census"],
                "folds": fold_summary,
            }
            """
        ),
        markdown(
            """
            ## Step-by-step interpretation of the split results

            ### 1. Check the source census before the bars

            The split contains all 93 accepted source videos and all 625 accepted
            sequences from notebook 01. The source census is 9 `cerebralpalsy`, 28
            `myopathic`, 29 `normal`, 9 `parkinsons`, and 18 `stroke` source videos. These
            strings are dataset annotations used for stratification; they are not diagnoses
            made by this project and are not the primary target.

            With five folds, an annotation represented by 9 sources can contribute only one
            or two sources to each test fold. Counts of 28, 29, and 18 similarly imply test
            allocations of roughly 5–6, 5–6, and 3–4. The plotted bars show exactly these
            integer patterns. Small differences of one are expected and are the closest
            possible balance; identical fold bars are not mathematically available for all
            five annotations.

            ### 2. Read each bar as held-out source videos, not sequences

            The current paper split is:

            | Outer fold | Train sources | Test sources | Train sequences | Test sequences | Test annotation counts (CP / M / N / P / S) |
            |---:|---:|---:|---:|---:|---|
            | 0 | 74 | 19 | 436 | 189 | 2 / 5 / 6 / 2 / 4 |
            | 1 | 74 | 19 | 443 | 182 | 2 / 5 / 6 / 2 / 4 |
            | 2 | 74 | 19 | 553 | 72 | 2 / 6 / 6 / 1 / 4 |
            | 3 | 75 | 18 | 548 | 77 | 1 / 6 / 6 / 2 / 3 |
            | 4 | 75 | 18 | 520 | 105 | 2 / 6 / 5 / 2 / 3 |

            Here CP abbreviates `cerebralpalsy`, M `myopathic`, N `normal`, P
            `parkinsons`, and S `stroke`. The abbreviations are only table space-savers.

            Notice that folds 0 and 2 both test 19 sources but contain 189 versus 72 test
            sequences. That is not a splitting failure. It shows why sequence-level
            splitting would be misleading: some source videos yield many more sequences
            than others. Later metrics give each source equal total weight so fold 0 does
            not receive extra influence merely for containing more clips.

            The five test-source sets are disjoint and collectively contain every one of
            the 93 sources exactly once. Their test-sequence counts also sum to 625. For any
            chosen fold, its training and test source sets do not overlap. Mirrors and
            augmentations inherit the source assignment and therefore cannot cross the
            boundary later.

            ### 3. Interpret the inner folds as tuning partitions only

            Every outer-training set is split four ways for read-out tuning. When the outer
            training set has 74 sources, an inner fit uses 55 or 56 sources and validates on
            18 or 19. When it has 75 sources, an inner fit uses 56 or 57 and validates on 18
            or 19. Each outer-training source serves as inner validation exactly once.

            Because sources have unequal numbers of sequences, the corresponding inner
            sequence counts range from 266 to 450 for fitting and from 80 to 197 for
            validation. Again, this variability is expected. The inner validation scores
            are source-weighted in notebook 04. Most importantly, none of these inner folds
            contains the current outer-test sources.

            A concrete example helps. In outer fold 0, 19 sources and their 189 sequences
            are placed behind the final-test boundary. The encoder in notebook 03 uses only
            the remaining 74 sources. Later, ridge candidates are compared using four
            rotations inside those 74 sources. Only after a penalty is selected is the
            read-out refit on all 74 and evaluated on the untouched 19.

            ### 4. Understand what the artifact proves—and what it cannot prove

            The saved split manifest carries the cohort, context, protocol, and split
            digests. Later stages reject it if those fingerprints do not match. This proves
            deterministic bookkeeping and source-level non-overlap under the available
            `video_id` field.

            It cannot prove held-out-person generalization. Two source videos could depict
            the same unidentified person, and this project deliberately does not infer
            identity. It also cannot show that a model performs well; no encoder or read-out
            score appears in this notebook.

            The narrow conclusion is: **the accepted cohort has been partitioned into
            complete, non-overlapping, annotation-stratified outer source folds, with inner
            tuning folds confined to each outer-training set.** This is the leakage-control
            foundation required by notebooks 03–05.
            """
        ),
    ],
    "03_fold_local_training.ipynb": [
        markdown(
            """
            # 03 — Fold-local, label-blind training

            Each encoder starts from a recorded seed-matched initialization and is trained
            only on its outer-training source videos. Sampling is source-uniform before a
            sequence is selected, preventing long videos from dominating. Dataset folder
            annotations do not enter the representation objective. Reflection augmentation
            is a registered training variant, not a post-result repair.

            Vanilla and reflection-augmented checkpoints use the same initialization,
            source draws, target masks, geometric views, optimizer schedule, and update
            count. Reflection has its own random stream, so consuming augmentation draws
            cannot perturb sampling or masking. This makes their difference a paired
            recipe ablation under the registered seeds—not a universal causal claim.

            Checkpoint lineage binds protocol, cohort, split, allowed sources, fold, seed,
            variant, and implementation. A mismatch fails instead of silently reusing a
            stale model. `smoke` runs validate execution only; paper evidence requires the
            locked paper profile, every registered fold and seed, and subsequent held-out
            evaluation. Training cannot resolve the ethics or data-use gate.
            """
        ),
        markdown(
            """
            ## What experiment does this notebook perform?

            This notebook performs a **paired training-recipe ablation**. An ablation is a
            controlled comparison in which one ingredient is changed while the remaining
            ingredients are kept matched. Here the changed ingredient is anatomical
            reflection augmentation:

            - `vanilla` uses the original training poses and never applies reflection;
            - `reflection_augmented` reflects a sampled pose with probability 0.5 by
              reversing the horizontal coordinate and exchanging corresponding left and
              right landmarks.

            The scientific motivation is to learn whether exposure to reflected examples
            later improves left–right behavior. Notebook 03 cannot answer that question by
            itself. Its job is to produce the correctly isolated encoders. Notebooks 04
            and 05 subsequently test their held-out predictive utility and their direct
            response to reflection.

            The model is an **S-JEPA**, or Skeleton Joint-Embedding Predictive
            Architecture. It hides some valid space-time pose patches and learns to predict
            their latent representations. It is trained jointly with **VICReg**, short for
            Variance–Invariance–Covariance Regularization. In simple terms, VICReg asks two
            views of the same pose to have similar summaries while discouraging every
            example from collapsing to the same representation and discouraging redundant
            representation dimensions. The laterality target and dataset condition label
            never enter either loss. “Label-blind” does not mean anatomy-blind: landmark
            identities and the registered left/right mapping are still part of the pose
            representation and augmentation.

            Each outer fold seals away 18 or 19 source videos. A fresh encoder sees only
            that fold's other 74 or 75 sources. For each fold, seeds 42 through 46 create
            five reproducible optimization repeats, and each seed is matched across the two
            variants. Consequently, the paper design trains $5\\times5\\times2=50$
            encoders. These are 50 fitted models, not 50 independent datasets: folds reuse
            some training sources and seeds reuse the same fold, so the job count must not
            be treated as an inferential sample size.

            <svg viewBox="0 0 1060 360" width="100%" role="img"
                 aria-labelledby="training-flow-title training-flow-description"
                 xmlns="http://www.w3.org/2000/svg">
              <title id="training-flow-title">Fold-local paired training workflow</title>
              <desc id="training-flow-description">The locked cohort is divided into five
              outer folds. For each fold, training sources feed matched vanilla and
              reflection-augmented jobs across five seeds while test sources remain sealed.
              Fifty checkpoints then move to held-out evaluation in later notebooks.</desc>
              <defs>
                <marker id="arrow03" markerWidth="8" markerHeight="8" refX="7" refY="4"
                        orient="auto" markerUnits="strokeWidth">
                  <path d="M0,0 L8,4 L0,8 z" fill="#475569"/>
                </marker>
              </defs>
              <style>
                .box03 { fill:#f8fafc; stroke:#334155; stroke-width:1.5; rx:10; }
                .train03 { fill:#ecfdf5; stroke:#047857; stroke-width:1.5; rx:10; }
                .hold03 { fill:#fff7ed; stroke:#c2410c; stroke-width:1.5; rx:10; }
                .variant03a { fill:#eff6ff; stroke:#2563eb; stroke-width:1.5; rx:10; }
                .variant03b { fill:#fff1f2; stroke:#e11d48; stroke-width:1.5; rx:10; }
                .line03 { stroke:#475569; stroke-width:1.8; fill:none;
                          marker-end:url(#arrow03); }
                .t03 { font: 15px system-ui, sans-serif; fill:#0f172a; }
                .s03 { font: 12px system-ui, sans-serif; fill:#475569; }
                .h03 { font: 600 15px system-ui, sans-serif; fill:#0f172a; }
              </style>
              <rect class="box03" x="15" y="125" width="155" height="82"/>
              <text class="h03" x="92" y="153" text-anchor="middle">Locked cohort</text>
              <text class="s03" x="92" y="175" text-anchor="middle">93 source videos</text>
              <text class="s03" x="92" y="193" text-anchor="middle">625 sequences</text>

              <rect class="box03" x="215" y="125" width="155" height="82"/>
              <text class="h03" x="292" y="153" text-anchor="middle">Outer fold</text>
              <text class="s03" x="292" y="175" text-anchor="middle">repeat for folds 0–4</text>
              <text class="s03" x="292" y="193" text-anchor="middle">split by source video</text>
              <path class="line03" d="M170 166 L215 166"/>

              <rect class="train03" x="420" y="55" width="175" height="82"/>
              <text class="h03" x="507" y="83" text-anchor="middle">Training sources</text>
              <text class="s03" x="507" y="105" text-anchor="middle">74 or 75 sources</text>
              <text class="s03" x="507" y="123" text-anchor="middle">labels excluded from loss</text>
              <rect class="hold03" x="420" y="230" width="175" height="82"/>
              <text class="h03" x="507" y="258" text-anchor="middle">Sealed test sources</text>
              <text class="s03" x="507" y="280" text-anchor="middle">18 or 19 sources</text>
              <text class="s03" x="507" y="298" text-anchor="middle">never used in notebook 03</text>
              <path class="line03" d="M370 154 C392 154 394 96 420 96"/>
              <path class="line03" d="M370 178 C392 178 394 271 420 271"/>

              <rect class="variant03a" x="645" y="20" width="180" height="82"/>
              <text class="h03" x="735" y="48" text-anchor="middle">Vanilla</text>
              <text class="s03" x="735" y="70" text-anchor="middle">reflection probability 0</text>
              <text class="s03" x="735" y="88" text-anchor="middle">five matched seeds</text>
              <rect class="variant03b" x="645" y="125" width="180" height="82"/>
              <text class="h03" x="735" y="153" text-anchor="middle">Reflection augmented</text>
              <text class="s03" x="735" y="175" text-anchor="middle">reflection probability 0.5</text>
              <text class="s03" x="735" y="193" text-anchor="middle">five matched seeds</text>
              <path class="line03" d="M595 82 L645 61"/>
              <path class="line03" d="M595 111 L645 152"/>

              <rect class="box03" x="870" y="73" width="170" height="90"/>
              <text class="h03" x="955" y="101" text-anchor="middle">50 checkpoints</text>
              <text class="s03" x="955" y="123" text-anchor="middle">300 epochs each</text>
              <text class="s03" x="955" y="141" text-anchor="middle">1,200 updates each</text>
              <path class="line03" d="M825 61 C847 61 847 102 870 102"/>
              <path class="line03" d="M825 166 C847 166 847 133 870 133"/>

              <rect class="hold03" x="815" y="245" width="225" height="82"/>
              <text class="h03" x="927" y="273" text-anchor="middle">Held-out evaluation</text>
              <text class="s03" x="927" y="295" text-anchor="middle">not performed here</text>
              <text class="s03" x="927" y="313" text-anchor="middle">continue to notebooks 04–05</text>
              <path class="line03" d="M595 271 L815 278"/>
              <path class="line03" d="M955 163 L955 245"/>
            </svg>
            """
        ),
        markdown(
            """
            ## How to read the live progress display

            The training cell below can train many independent encoders, so it reports
            progress at two levels. A **job** is one variant, outer fold, and random seed.
            A variant is one registered training recipe. An outer fold identifies which
            source-video group is held out, and a seed identifies one reproducible random
            starting point. An **epoch** is one scheduled round of source-balanced batches,
            while an optimizer update is one step that changes the learned weights.

            Within a newly trained job, the display shows completed epochs and optimizer
            updates, current training loss, elapsed time, and estimated time remaining. The
            overall bar counts all registered jobs. A previously saved checkpoint counts as
            complete only after its lineage and model-state fingerprints have been
            validated.

            The estimated time of arrival (ETA) is intentionally adaptive. At first it
            uses the median historical duration of compatible checkpoints produced on the
            same device, if any exist. Otherwise, an estimate appears after the first new
            epoch. Once five new epochs are available, their recent median speed takes over.
            This is more stable than extrapolating from one unusually fast or slow epoch,
            but it remains an estimate: thermal throttling, other work on the computer, or
            suspending the laptop can change it.

            The display updates in place about once per second, so it does not create
            thousands of notebook output lines. Re-running the cell safely reuses every
            valid completed checkpoint. Training is checkpointed after a whole job, not
            after every epoch. If execution is interrupted in the middle of a job, earlier
            completed jobs remain reusable, but the interrupted job starts again.

            Training loss describes the self-supervised optimization objective. It is not
            validation accuracy or held-out performance. Those quantities are calculated
            later, in notebooks 04 and 05, after the outer-test sources are evaluated.
            """
        ),
        bootstrap_cell(),
        code(
            """
            import pandas as pd

            from laterality.data import load_cohort
            from laterality.splitting import load_splits
            from laterality.training import train_selected
            from laterality.visualization import training_figure
            from notebook_progress import NotebookTrainingProgress

            cohort = load_cohort(context)
            splits = load_splits(context, cohort)
            progress = NotebookTrainingProgress(refresh_seconds=1.0)
            training_summaries = train_selected(
                context,
                cohort,
                splits,
                progress_callback=progress,
            )
            show_inline(training_figure(context, training_summaries))
            pd.DataFrame(training_summaries).drop(columns="history").sort_values(
                ["variant", "fold", "seed"]
            ).reset_index(drop=True)
            """
        ),
        markdown(
            """
            ## Step-by-step interpretation of the completed paper run

            ### 1. First establish what completed successfully

            The saved output is from the **paper profile**, not the synthetic smoke test.
            It contains all 50 registered jobs: 25 vanilla checkpoints and 25
            reflection-augmented checkpoints. Every row reports 1,200 optimizer updates,
            which equals 300 epochs times four updates per epoch. No run stopped early and
            no non-finite loss was reported. This is evidence that the fixed training plan
            completed; it is not yet evidence that the representations are useful.

            The repeated `enable_nested_tensor ... norm_first` messages are PyTorch
            performance warnings. They say that one optional nested-tensor optimization was
            not used by the Transformer configuration. They do not indicate missing data,
            leakage, an invalid checkpoint, or failed optimization. A traceback, a
            non-finite-loss exception, a lineage mismatch, or a missing row would be a
            substantively different warning sign.

            ### 2. Read the left plot as an optimization diagnostic

            The solid line is the mean training loss at each epoch across the 25 fold/seed
            jobs for that variant. The translucent band runs from the smallest to the
            largest observed job loss at that epoch. It is a **range**, not a confidence
            interval, standard error, or test of a difference between variants.

            In the saved run, mean loss changed as follows:

            | Epoch | Vanilla mean loss | Reflection-augmented mean loss |
            |---:|---:|---:|
            | 1 | 11.923 | 11.870 |
            | 10 | 3.015 | 3.060 |
            | 50 | 1.829 | 1.902 |
            | 100 | 1.535 | 1.605 |
            | 200 | 1.309 | 1.350 |
            | 300 | 1.241 | 1.299 |

            Both variants therefore reduced their mean objective by about 89% from epoch 1
            to epoch 300, and all 50 individual jobs ended below their own epoch-1 loss.
            Most of the reduction happened early. Improvement continued after epoch 200,
            but it was smaller: approximately 0.069 for vanilla and 0.052 for the augmented
            recipe. The average over the last 20 epochs was 1.246 for vanilla and 1.299 for
            reflection augmentation, close to the final-epoch means. This pattern is
            consistent with stable optimization approaching a plateau rather than obvious
            divergence.

            Individual epoch losses still fluctuate because every epoch samples sequences,
            masks, and geometric views. The checkpoint is deliberately the fixed epoch-300
            checkpoint, not whichever epoch happened to have the smallest training loss.
            The median minimum-loss epoch was 280 for vanilla and 276 for reflection
            augmentation, which also shows why a single lowest point should not be treated
            as a selected model.

            ### 3. Do not use the loss gap to choose a winning variant

            Final loss averaged 1.241 for vanilla and 1.299 for reflection augmentation.
            In matched fold/seed pairs, augmented minus vanilla final loss averaged +0.058,
            had a median of +0.034, and ranged from -0.076 to +0.228. The augmented job had
            the smaller final loss in 6 of 25 pairs.

            These are useful optimization descriptions, but they do **not** demonstrate
            that reflection augmentation harms or helps the scientific outcome. The
            augmented model is trained on a different input distribution because each
            sampled pose has a 0.5 chance of being reflected. Moreover, this self-supervised
            objective is not held-out laterality error. For example, fold 2/seed 44 ended at 1.175 for
            vanilla and 1.171 for augmentation—almost identical—whereas fold 4/seed 45
            ended at 1.168 and 1.396. Neither pair tells us which encoder predicts the
            coordinate-derived target better or transforms more correctly under reflection.
            Those comparisons require the paired held-out measurements in notebooks 04
            and 05.

            Loss is also not a percentage. A loss of 1.2 does not mean 120% error or 80%
            accuracy. It is the sum of the latent prediction objective and a weighted
            VICReg regularizer. Its scale is meaningful mainly for detecting learning,
            instability, and unusual differences under the fixed implementation.

            ### 4. Read the right plot as a fairness-of-exposure check

            For each job, the right panel divides the most frequently drawn training source
            by the least frequently drawn training source. Exactly 1.0 would mean identical
            counts. The observed ratios ranged from 1.054 to 1.087, so the most frequent
            source received at most about 8.7% more draws than the least frequent source.

            Every job made 24,000 source draws: 300 epochs times four batches times 20
            sequences. Each epoch first visits every one of its 74 or 75 training sources,
            guaranteeing at least 300 draws per source, and then uses source-uniform padding
            to reach the fixed four-batch budget. Across jobs, the recorded extremes were
            309 to 340 draws. This small spread is compatible with the intended balanced
            random padding and gives no sign that a long source video dominated merely
            because it contained more extracted sequences.

            The vanilla and augmented job in every matched fold/seed pair have identical
            minimum and maximum draw counts. This is expected because their source-sampling
            stream is shared, while reflection decisions use a separate stream. The two
            variant columns therefore have the same vertical pattern. Within either column,
            multiple jobs with the same ratio can overlap, so fewer than 25 dots may be
            visually distinguishable.

            ### 5. Interpret each table column literally

            - `variant` identifies the training recipe, not a result category.
            - `fold` identifies which source-video group was sealed away. Fold 0 is not
              earlier, easier, or more important than fold 4.
            - `seed` identifies a reproducible random initialization and random stream. It
              is not a subject identifier and larger seed numbers are not better.
            - `optimizer_updates` verifies equal compute. All rows should show 1,200.
            - `final_loss` is the mean total training loss in epoch 300. It is neither a
              validation loss nor test performance.
            - `minimum_source_draws` and `maximum_source_draws` summarize source exposure.
              Their ratio is what appears in the right plot.
            - `checkpoint` is the local artifact used downstream. Its lineage records the
              allowed training sources and forbidden test sources so an incompatible file
              fails validation instead of being silently reused.

            ### 6. State the narrow conclusion

            Notebook 03 supports the following conclusion: **all registered fold-local,
            label-blind training jobs completed under equal update budgets; both recipes
            learned a substantially lower and stable training objective; and source
            exposure remained closely balanced and paired across recipes.**

            It does not yet support “reflection augmentation is better,” “the encoder is
            reflection-equivariant,” “the target is predictable on held-out sources,” or
            any diagnostic or clinical claim. Notebook 04 next applies each fold-local
            checkpoint to the sources that checkpoint never saw and fits read-outs without
            opening the outer-test targets. Notebook 05 aggregates those paired held-out
            results and applies the registered decision rules. Only those stages can answer
            the scientific comparison posed here.
            """
        ),
    ],
    "04_held_out_evaluation.ipynb": [
        markdown(
            """
            # 04 — Held-out source-video evaluation

            For each fold and seed, the encoder and every fitted read-out see only
            outer-training sources. Ridge selection uses source-disjoint inner folds;
            scaling, fitting, neutral-band calibration, and target scale are train-only.
            Predictions for original and mirrored poses are then produced once on the
            held-out sources with the same fitted read-out.

            The lane factorial separates single-pass free prediction, two-pass odd/even
            parity, and the zero-origin read-out constraint for both learned and paired
            initial encoders. Measured visibility, acquisition, annotation, and combined
            nuisance lanes are accompanied by a learned-plus-nuisance incremental lane.
            The target-component oracle is self-consistency only, never a baseline.

            Before any read-out, the notebook also compares `E(Mx)` with `S E(x)` for all
            33 joints on exactly common-valid tokens. Its strict error is residual energy
            divided by total representation energy: zero is exact, while unrelated
            equal-energy representations are near one. `S` swaps joints but does not fit,
            rotate, sign-flip, or align latent channels. This conservative diagnostic is
            evaluated per checkpoint against its paired initialization. Mirrors remain
            paired transformations, not new test cases.
            Results remain about source videos, while folder labels remain annotations and
            no clinical or unseen-person interpretation is permitted. Synthetic smoke
            scores are pipeline diagnostics, not evidence.
            """
        ),
        markdown(
            """
            ## What this notebook evaluates

            Notebook 03 produced encoders; notebook 04 asks how each frozen encoder behaves
            on source videos that were sealed away from its training fold. Nothing here
            updates encoder weights. A paper-profile **evaluation job** is one training
            variant, one outer fold, and one optimization seed. Five folds, five seeds, and
            two variants therefore give $5\times5\times2=50$ evaluation jobs.

            Within a job, the outer-test targets remain untouched while the read-out recipe
            is chosen. Seven **ridge regression** penalties are compared using the four inner
            folds made only from outer-training sources. Ridge regression is a linear
            prediction rule whose penalty discourages unstable, excessively large
            coefficients. After the penalty is selected, the read-out is refitted on all
            outer-training sources and used once on the held-out sources. Scaling,
            missing-value imputation, the neutral band, and target scale are also calculated
            from training sources only.

            The 16 registered feature **lanes** answer deliberately different questions:

            - the primary `learned_single_free` lane asks what a single encoder pass plus an
              unconstrained linear read-out can predict;
            - learned and paired-initialization two-pass lanes separate odd and even content;
            - zero-origin odd lanes impose exact sign reversal and are construction controls;
            - visibility, acquisition, annotation, and combined nuisance lanes test measured
              non-representation explanations;
            - the learned-plus-nuisance lane asks whether learned features add information
              beyond those measured controls; and
            - the target-component oracle reconstructs a target from its own ingredients. It
              checks self-consistency and is never a fair predictive baseline.

            Before fitting any read-out, the strict token audit compares $E(Mx)$ with
            $S E(x)$. Here $E$ is the encoder, $x$ is a pose sequence, $M$ anatomically
            mirrors the coordinates, and $S$ swaps the corresponding left/right token
            joints without fitting an alignment. The error is squared residual energy
            divided by total representation energy. Zero means exact agreement; lower is
            better. The calculation uses only tokens valid in both paths.

            <svg viewBox="0 0 1080 340" width="100%" role="img"
                 aria-labelledby="evaluation-flow-title evaluation-flow-description"
                 xmlns="http://www.w3.org/2000/svg">
              <title id="evaluation-flow-title">Held-out evaluation workflow for one checkpoint</title>
              <desc id="evaluation-flow-description">One fold-local checkpoint encodes
              outer-training and sealed outer-test poses. Inner training folds select ridge
              penalties for sixteen read-out lanes. The final read-outs predict original and
              mirrored held-out poses, while a direct token audit compares the learned and
              paired-initial encoders.</desc>
              <defs><marker id="arrow04" markerWidth="8" markerHeight="8" refX="7"
                refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#475569"/></marker></defs>
              <style>
                .box04{fill:#f8fafc;stroke:#334155;stroke-width:1.5}
                .train04{fill:#ecfdf5;stroke:#047857;stroke-width:1.5}
                .test04{fill:#fff7ed;stroke:#c2410c;stroke-width:1.6}
                .audit04{fill:#eff6ff;stroke:#2563eb;stroke-width:1.5}
                .line04{stroke:#475569;stroke-width:1.8;fill:none;marker-end:url(#arrow04)}
                .h04{font:600 14px system-ui,sans-serif;fill:#0f172a}
                .s04{font:12px system-ui,sans-serif;fill:#475569}
              </style>
              <rect class="box04" x="15" y="125" width="180" height="84" rx="9"/>
              <text class="h04" x="105" y="154" text-anchor="middle">Frozen checkpoint</text>
              <text class="s04" x="105" y="177" text-anchor="middle">variant + fold + seed</text>
              <text class="s04" x="105" y="197" text-anchor="middle">encoder weights do not change</text>
              <rect class="train04" x="250" y="35" width="210" height="105" rx="9"/>
              <text class="h04" x="355" y="65" text-anchor="middle">Outer-training sources</text>
              <text class="s04" x="355" y="88" text-anchor="middle">four inner source folds</text>
              <text class="s04" x="355" y="108" text-anchor="middle">select ridge penalty</text>
              <text class="s04" x="355" y="128" text-anchor="middle">fit scaler + read-out</text>
              <rect class="test04" x="250" y="215" width="210" height="105" rx="9"/>
              <text class="h04" x="355" y="245" text-anchor="middle">Sealed outer-test sources</text>
              <text class="s04" x="355" y="268" text-anchor="middle">original pose x</text>
              <text class="s04" x="355" y="288" text-anchor="middle">anatomical mirror Mx</text>
              <text class="s04" x="355" y="308" text-anchor="middle">targets never tune the read-out</text>
              <path class="line04" d="M195 151 C220 151 220 87 250 87"/>
              <path class="line04" d="M195 183 C220 183 220 267 250 267"/>
              <rect class="box04" x="525" y="35" width="225" height="105" rx="9"/>
              <text class="h04" x="637" y="65" text-anchor="middle">Sixteen registered lanes</text>
              <text class="s04" x="637" y="88" text-anchor="middle">learned + initialization floors</text>
              <text class="s04" x="637" y="108" text-anchor="middle">odd/even + nuisance controls</text>
              <text class="s04" x="637" y="128" text-anchor="middle">same train-only tuning rule</text>
              <path class="line04" d="M460 87 L525 87"/>
              <rect class="audit04" x="525" y="215" width="225" height="105" rx="9"/>
              <text class="h04" x="637" y="245" text-anchor="middle">Direct token audit</text>
              <text class="s04" x="637" y="268" text-anchor="middle">compare E(Mx) with S E(x)</text>
              <text class="s04" x="637" y="288" text-anchor="middle">learned versus paired initial</text>
              <text class="s04" x="637" y="308" text-anchor="middle">common-valid tokens only</text>
              <path class="line04" d="M460 267 L525 267"/>
              <rect class="audit04" x="815" y="125" width="245" height="84" rx="9"/>
              <text class="h04" x="937" y="154" text-anchor="middle">Saved held-out rows</text>
              <text class="s04" x="937" y="177" text-anchor="middle">prediction + mirror prediction</text>
              <text class="s04" x="937" y="197" text-anchor="middle">strict token errors + lineage</text>
              <path class="line04" d="M750 87 C785 87 785 151 815 151"/>
              <path class="line04" d="M750 267 C785 267 785 183 815 183"/>
            </svg>
            """
        ),
        markdown(
            """
            ## How to read the live evaluation progress display

            The overall bar counts evaluation jobs, not sequences or read-out lanes. The
            active label names the variant, outer fold, and seed. A fresh job performs four
            expensive encoder passes—learned and paired-initial encoders on original and
            mirrored poses—then tunes all 16 lanes. The display updates after each job, so
            a single job can remain active for a while without implying that execution has
            stalled.

            A saved comma-separated values (CSV) file is only a **cached candidate**. It
            counts as reused after the checkpoint lineage, file fingerprint, row coverage,
            source weights, and result digest all validate. A mismatch stops the run rather
            than silently recomputing into a questionable directory. A fresh result is
            written atomically, meaning an incomplete file does not replace a valid result.

            Estimated time of arrival (ETA) appears after the first newly computed job and
            uses the median duration of completed new jobs. Cache validation time is not used
            to predict fresh computation. Source folds contain different sequence counts,
            and CPU, GPU, or Apple Metal Performance Shaders (MPS) load can vary, so the
            estimate is a guide rather than a deadline. Re-running the cell safely validates
            and reuses completed jobs; an interrupted active job is the only work that may
            need to be repeated.
            """
        ),
        bootstrap_cell(),
        code(
            """
            from laterality.data import load_cohort
            from laterality.splitting import load_splits
            from laterality.visualization import evaluation_figure
            from notebook_progress import (
                NotebookTaskProgress,
                evaluate_selected_with_progress,
            )

            cohort = load_cohort(context)
            splits = load_splits(context, cohort)
            evaluation_progress = NotebookTaskProgress(
                "Held-out evaluation progress",
                "job",
                refresh_seconds=1.0,
            )
            evaluations = evaluate_selected_with_progress(
                context,
                cohort,
                splits,
                progress=evaluation_progress,
            )

            evaluation_summary = (
                evaluations.groupby(["variant", "fold", "seed", "lane"], as_index=False)
                .agg(test_sequences=("sequence_id", "size"), test_sources=("video_id", "nunique"))
                .sort_values(["variant", "fold", "seed", "lane"])
                .reset_index(drop=True)
            )
            representation_summary = (
                evaluations[
                    evaluations["lane"]
                    == context.protocol["evaluation"]["primary_lane"]
                ]
                .groupby(["variant", "seed"], as_index=False)
                .agg(
                    learned_strict_error=(
                        "learned_strict_equivariance_error",
                        "mean",
                    ),
                    initial_strict_error=(
                        "initial_strict_equivariance_error",
                        "mean",
                    ),
                    minimum_common_tokens=(
                        "learned_strict_equivariance_common_tokens",
                        "min",
                    ),
                )
                .sort_values(["variant", "seed"])
                .reset_index(drop=True)
            )
            display(evaluation_summary)
            display(representation_summary)
            show_inline(evaluation_figure(context, evaluations))
            """
        ),
        markdown(
            """
            ## Step-by-step interpretation of the evaluation output

            ### 1. Verify completeness before interpreting magnitudes

            In a complete paper run, `evaluation_summary` has 800 rows: 50 checkpoint jobs
            times 16 lanes. Within a particular fold, every lane must report the same held-out
            source and sequence counts because lanes are alternative feature/read-out recipes
            applied to the same test observations. Expected source counts are 19, 19, 19, 18,
            and 18 for folds 0 through 4. Expected sequence counts are 189, 182, 72, 77, and
            105. A missing job, lane, source, or sequence is an integrity failure—not a zero
            score and not something aggregation should ignore.

            The table does not yet report prediction quality. Its job is to prove coverage:
            `variant` names the training recipe, `fold` names the held-out source partition,
            `seed` names the optimization repeat, and `lane` names the feature/read-out
            recipe. Repeated test counts across lanes are expected rather than duplication.

            ### 2. Read the representation summary as a diagnostic

            `representation_summary` has one row for each variant/seed combination after
            pooling the five outer folds, normally ten rows. `learned_strict_error` measures
            the trained encoder, while `initial_strict_error` measures that checkpoint's
            paired pre-training initialization. Lower values mean closer agreement with the
            fixed anatomical token swap. `minimum_common_tokens` is the smallest number of
            jointly valid tokens supporting any included comparison; it must remain at or
            above the registered minimum of eight.

            This compact printed summary uses an ordinary mean over sequence rows and is
            descriptive. The plotted comparisons and notebook-05 inference balance sources
            so a video that yielded many clips does not dominate the scientific conclusion.

            ### 3. Interpret both panels of the figure

            In the left panel, the initial and learned bars summarize strict token error for
            each variant. The dashed line is the registered absolute error margin of 0.1.
            A learned bar below its initial bar suggests improvement, and a learned bar below
            the dashed line suggests small absolute error. Neither visual suggestion alone is
            a passed claim: notebook 05 requires source-bootstrap confidence bounds for both
            the absolute margin and learned-minus-initial improvement.

            In the right panel, each dot is a paired source/checkpoint comparison. The
            horizontal coordinate is initial-encoder error and the vertical coordinate is
            learned-encoder error. A dot below the diagonal favors the learned encoder; a dot
            above it favors the initialization. For example, `(0.30, 0.12)` indicates a large
            reduction but still misses the 0.1 absolute margin, whereas `(0.08, 0.07)` meets
            the point margin but shows little training improvement. The registered conclusion
            needs both kinds of evidence with uncertainty included.

            ### 4. Keep prediction and symmetry questions separate

            A representation can transform cleanly yet carry too little information to
            predict the coordinate-derived target. Conversely, a read-out can predict the
            target while internal tokens fail the strict swap test. That is why this notebook
            stores predictions, mirrored predictions, and direct token errors separately.
            Exact oddness in a zero-origin two-pass lane is imposed by construction; it does
            not prove that the single-pass encoder learned equivariance.

            ### 5. Interpret the current paper-profile findings

            The completed paper evaluation contains all 50 jobs, all 16 lanes, and 100,000
            saved rows. Across the five outer folds, each of the 625 accepted sequences and
            each of the 93 source videos is tested exactly once for every variant/seed
            combination. The minimum common-token count is 196, comfortably above the
            registered minimum of eight, so these strict-error values are not being driven by
            the minimum-support cutoff.

            After giving each source equal total influence, the learned strict-error means are
            0.1053 for reflection augmentation and 0.1138 for vanilla, compared with 0.0832
            for their matched initial encoders. Lower is better, so training did not improve
            the descriptive mean: learned error is higher than initial error for every seed in
            both variants. The five learned seed means range from 0.0831 to 0.1233 for
            reflection augmentation and from 0.0845 to 0.1324 for vanilla. Only two augmented
            seed means and one vanilla seed mean fall below 0.1.

            The source-level picture is mixed rather than uniform. Learned error is lower than
            paired-initial error in 220 of 465 source/seed comparisons (47.3%) for reflection
            augmentation and 192 of 465 (41.3%) for vanilla. A few sources have much larger
            errors, which is why the means exceed the medians and why source-level uncertainty
            matters. Reflection augmentation has the lower descriptive learned mean by about
            0.0084, but that point difference is not yet a supported augmentation effect.

            The narrow descriptive reading is therefore cautious: **the direct token audit
            currently gives no visible evidence that training improved strict anatomical-swap
            equivariance over initialization, and several learned seed means miss the 0.1
            absolute margin.** Notebook 05 must still compute paired source-bootstrap
            confidence bounds before the registered gate receives its final `SUPPORTED` or
            `NOT SUPPORTED` status. These token results also say nothing yet about held-out
            target predictability.

            ### 6. State only the handoff conclusion

            A complete notebook-04 run establishes that all registered checkpoints were
            evaluated on their untouched source folds with train-only read-out selection and
            that the paired mirror/token diagnostics were saved. It still does not establish
            positive predictive utility, a statistically supported symmetry effect, external
            generalization, unseen-person performance, or a clinical result. Notebook 05
            performs the source-balanced aggregation and applies the locked decision rules.
            """
        ),
    ],
    "05_aggregate_statistics.ipynb": [
        markdown(
            """
            # 05 — Source-balanced aggregation and uncertainty

            Scientific gates are computed for each checkpoint before aggregating across
            the registered seeds. Squared mirror residuals therefore cannot cancel across
            seeds, and predictive performance is the mean single-checkpoint metric—not an
            accidentally stronger ensemble. The mean-prediction ensemble remains clearly
            labeled as a secondary operational estimand.

            A training-induced symmetry statement is conjunctive: the native single-pass
            probe needs an absolute positive held-out R² bound, low native output error,
            and an advantage over its paired initialization; the direct learned-token
            error must also beat initialization and meet its separate absolute margin.
            Analytic oddness is checked for every row and seed, but is only an implementation
            property. Constructed learned content additionally needs positive absolute R²
            and an advantage over the identically constructed random floor.

            Source-cluster bootstraps carry every sequence and seed from each sampled
            source together. They remain conditional on the fixed cross-fitted checkpoints.
            Nuisance contrasts diagnose only the measured controls; they cannot exclude
            unmeasured camera, uploader, pose-estimation, or repeated-person effects. A
            failed gate is a result, never permission to retune the locked protocol.
            """
        ),
        markdown(
            """
            ## How held-out rows become scientific evidence

            Notebook 05 does not train or evaluate another model. It combines the
            cross-fitted rows from notebook 04 while preserving the independent unit: the
            source video. **Cross-fitted** means every prediction came from a checkpoint
            whose outer-training set excluded that prediction's source video.

            Several layers of aggregation must stay distinct:

            1. Sequence predictions are weighted so every source video contributes total
               weight one, even when sources yield different numbers of sequences.
            2. Each checkpoint receives its own metric before seeds are averaged. This is
               the primary estimand and prevents a mean-prediction ensemble from appearing
               to be a typical single model.
            3. A source-cluster **bootstrap** repeatedly samples source videos with
               replacement. Every sequence, seed, and paired measurement belonging to a
               sampled source travels together. The resulting 95% confidence interval (CI)
               describes uncertainty across the observed source-video sample, conditional
               on the fixed fitted checkpoints.
            4. Registered gate decisions compare CI boundaries with thresholds fixed in
               the protocol. A threshold is not moved after seeing the result.

            The primary predictive score is source-balanced $R^2$, called the coefficient
            of determination. A value of 1 is perfect prediction, 0 matches the weighted
            mean-only reference, and a negative value is worse than that reference. Mean
            absolute error (MAE) is the average absolute prediction distance in target
            units; lower is better. The normalized antisymmetry error measures how far
            predictions on original and mirrored inputs depart from exact sign reversal;
            zero is exact and lower is better.

            <svg viewBox="0 0 1080 345" width="100%" role="img"
                 aria-labelledby="report-flow-title report-flow-description"
                 xmlns="http://www.w3.org/2000/svg">
              <title id="report-flow-title">Source-balanced aggregation and decision workflow</title>
              <desc id="report-flow-description">Held-out prediction rows are grouped by
              source and checkpoint. Single-checkpoint metrics and paired controls enter
              source-cluster bootstraps. Confidence bounds are compared with locked
              predictive, output-symmetry, and representation-symmetry gates. Governance
              remains a separate requirement for submission readiness.</desc>
              <defs><marker id="arrow05" markerWidth="8" markerHeight="8" refX="7"
                refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#475569"/></marker></defs>
              <style>
                .box05{fill:#f8fafc;stroke:#334155;stroke-width:1.5}
                .metric05{fill:#ecfdf5;stroke:#047857;stroke-width:1.5}
                .boot05{fill:#eff6ff;stroke:#2563eb;stroke-width:1.5}
                .gate05{fill:#fff7ed;stroke:#c2410c;stroke-width:1.5}
                .gov05{fill:#fff1f2;stroke:#be123c;stroke-width:1.5}
                .line05{stroke:#475569;stroke-width:1.8;fill:none;marker-end:url(#arrow05)}
                .h05{font:600 14px system-ui,sans-serif;fill:#0f172a}
                .s05{font:12px system-ui,sans-serif;fill:#475569}
              </style>
              <rect class="box05" x="15" y="125" width="190" height="90" rx="9"/>
              <text class="h05" x="110" y="155" text-anchor="middle">Held-out rows</text>
              <text class="s05" x="110" y="178" text-anchor="middle">all folds, seeds, variants</text>
              <text class="s05" x="110" y="198" text-anchor="middle">original + mirror predictions</text>
              <rect class="metric05" x="255" y="125" width="205" height="90" rx="9"/>
              <text class="h05" x="357" y="155" text-anchor="middle">Source-balanced metrics</text>
              <text class="s05" x="357" y="178" text-anchor="middle">one source = total weight one</text>
              <text class="s05" x="357" y="198" text-anchor="middle">each checkpoint scored first</text>
              <path class="line05" d="M205 170 L255 170"/>
              <rect class="boot05" x="510" y="125" width="205" height="90" rx="9"/>
              <text class="h05" x="612" y="155" text-anchor="middle">2,000 source bootstraps</text>
              <text class="s05" x="612" y="178" text-anchor="middle">paired rows move together</text>
              <text class="s05" x="612" y="198" text-anchor="middle">estimate + 95% CI</text>
              <path class="line05" d="M460 170 L510 170"/>
              <rect class="gate05" x="765" y="55" width="285" height="120" rx="9"/>
              <text class="h05" x="907" y="85" text-anchor="middle">Conjunctive empirical gates</text>
              <text class="s05" x="907" y="108" text-anchor="middle">absolute predictive utility</text>
              <text class="s05" x="907" y="128" text-anchor="middle">native output behavior</text>
              <text class="s05" x="907" y="148" text-anchor="middle">strict token equivariance</text>
              <path class="line05" d="M715 154 L765 125"/>
              <rect class="gov05" x="765" y="220" width="285" height="90" rx="9"/>
              <text class="h05" x="907" y="250" text-anchor="middle">Separate governance gate</text>
              <text class="s05" x="907" y="273" text-anchor="middle">ethics + data use + release</text>
              <text class="s05" x="907" y="293" text-anchor="middle">statistics cannot override it</text>
              <path class="line05" d="M715 186 C740 186 740 265 765 265"/>
            </svg>
            """
        ),
        markdown(
            """
            ## How to read the live aggregation progress display

            The reporting cell exposes 13 named stages. Early stages load and validate all
            evaluation files, build the explicitly secondary seed-mean ensemble, and compute
            ordinary and high-coverage metrics. Later stages create separate source
            bootstraps for checkpoint performance, native output symmetry, and strict token
            equivariance before writing the overview and final summary.

            The registered paper profile uses 2,000 bootstrap repetitions. These are
            vectorized calculations, so the display reports scientifically meaningful stages
            rather than printing 2,000 nearly identical lines. Stages have unequal costs:
            loading may be quick while a bootstrap can dominate. The ETA uses the median
            completed stage time and can move substantially early in the run. Treat it as a
            rough planning aid, not a promised completion time.

            The progress wrapper temporarily adds timers around the existing reporting
            functions and restores them afterward. It passes through the same arguments and
            return values, so progress reporting does not alter resampling, metrics, decision
            rules, or saved scientific tables. If any expected evaluation artifact is missing
            or incompatible, the active stage turns into an explicit error rather than a
            partially completed report.
            """
        ),
        bootstrap_cell(),
        code(
            """
            from laterality.data import load_cohort
            from laterality.splitting import load_splits
            from laterality.visualization import report_dashboard
            from notebook_progress import (
                NotebookTaskProgress,
                aggregate_and_save_with_progress,
            )

            cohort = load_cohort(context)
            splits = load_splits(context, cohort)
            reporting_progress = NotebookTaskProgress(
                "Statistical aggregation progress",
                "stage",
                refresh_seconds=0.5,
            )
            report = aggregate_and_save_with_progress(
                context,
                cohort,
                splits,
                progress=reporting_progress,
            )
            dashboard_path = (
                context.artifact_root / "report" / "enhanced_evidence_dashboard.svg"
            )
            show_inline(
                report_dashboard(context, report, output_path=dashboard_path)
            )

            print("Report summary:")
            print(report["summary"])
            print("\\nPer-checkpoint source-bootstrap estimands and controls:")
            display(report["checkpoint_bootstrap"])
            print("\\nNative output symmetry by seed (no cancellation):")
            display(report["native_symmetry_by_seed"])
            print("\\nNative output symmetry source bootstrap:")
            display(report["native_symmetry_bootstrap"])
            print("\\nDirect representation equivariance by seed:")
            display(report["representation_by_seed"])
            print("\\nDirect representation equivariance source bootstrap:")
            display(report["representation_bootstrap"])
            print("\\nSecondary mean-prediction ensemble metrics:")
            display(report["metrics"])
            print("\\nHigh-pose-coverage sensitivity metrics:")
            display(report["high_coverage_metrics"])
            print("\\nOptimization-seed diagnostics:")
            display(report["seed_metrics"])
            print("\\nSecondary ensemble-prediction bootstrap contrasts:")
            display(report["ensemble_bootstrap"])
            """
        ),
        markdown(
            """
            ## Step-by-step interpretation of the report

            ### 1. Read the dashboard as four related questions

            **Top left: absolute predictive utility.** Bars show the mean single-checkpoint,
            source-balanced $R^2$ for the native lane and the constructed odd lane. Error bars
            are 95% source-bootstrap CIs. A positive point estimate is not enough: the locked
            absolute-utility gate requires the entire lower confidence bound to exceed zero.
            For example, $R^2=0.20$ with CI `[-0.05, 0.38]` remains inconclusive because the
            interval includes performance no better than the mean-only reference.

            **Top right: direct representation audit.** Lower strict token error is better.
            Learned encoders are compared with paired initial encoders, and the horizontal
            line marks the absolute 0.1 margin. Support requires both an upper confidence
            bound below 0.1 and a learned-minus-initial upper bound below zero. An estimate of
            0.08 with upper bound 0.12 does not pass the absolute gate.

            **Bottom left: native output symmetry.** This panel measures the unconstrained
            single-pass probe's normalized mirror residual checkpoint by checkpoint. Lower is
            better. The error is squared before aggregation, so positive and negative mirror
            deviations cannot cancel across seeds. Constructed zero-origin odd lanes are
            checked elsewhere for exactness but cannot establish native behavior.

            **Bottom right: registered diagnostics.** Green means that the exact Boolean rule
            evaluated to `True`; red `NOT SUPPORTED` means at least one required bound failed.
            It does not mean the software failed or that the opposite scientific claim has
            been proved. The top-level training-induced symmetry statement is deliberately
            conjunctive: native predictive utility, native output behavior, strict token
            error, and training improvement must all pass together.

            ### 2. Use the summary as the authoritative claim ledger

            `paper_run_complete` verifies the paper profile, all five folds, all five seeds,
            and both variants. `synthetic_evidence=True` identifies smoke output that must not
            enter a paper claim. `integrity_gates` records exact constructed oddness and run
            completeness. `empirical_diagnostics` records each predeclared scientific gate
            separately, including useful learned content beyond paired initialization and
            measured nuisance controls.

            `submission_ready` is stricter than good statistics. It also requires the
            governance record from notebook 00 to be complete. Therefore a scientifically
            favorable paper run can still be submission-blocked, and resolving governance
            cannot turn an unfavorable empirical diagnostic into a favorable one.

            ### 3. Distinguish the detailed tables

            - `checkpoint_bootstrap` contains the primary mean-single-checkpoint estimands and
              paired performance contrasts. Look at `estimate`, `ci95_low`, and `ci95_high`
              together rather than selecting only a favorable seed.
            - `native_symmetry_by_seed` and `representation_by_seed` preserve each seed before
              aggregation. They make instability or one-seed dependence visible.
            - `native_symmetry_bootstrap` and `representation_bootstrap` carry those paired
              quantities through source-level resampling and feed the registered gates.
            - `metrics` reports the secondary mean-prediction ensemble. It can describe how an
              operational ensemble behaves but must not replace the primary checkpoint
              estimand.
            - `high_coverage_metrics` repeats the secondary metrics only for sequences with at
              least 90% authorized pose coverage. Agreement is reassuring sensitivity
              evidence; disagreement is something to report, not permission to discard the
              main analysis.
            - `seed_metrics` describes checkpoint-to-checkpoint variation, while
              `ensemble_bootstrap` contains secondary paired contrasts for the ensemble.

            ### 4. Interpret nuisance controls narrowly

            Visibility, acquisition, and dataset-annotation lanes ask whether recorded
            technical variables alone predict the target. The learned-plus-nuisance contrast
            asks whether learned features add information beyond those measured variables.
            Passing that contrast does not exclude unmeasured camera angle, uploader, repeated
            individuals, pose-estimation behavior, or other confounding.

            ### 5. Report null and mixed results without redesigning the test

            A failed CI gate is a valid result. For example, useful $R^2$ with poor strict
            token equivariance supports predictability but not the registered symmetry claim;
            low token error with nonpositive $R^2$ supports transformation consistency but not
            useful decoding. The protocol requires these outcomes to remain separate and
            forbids post-result threshold changes. All conclusions remain within the Gait
            Abnormality in Video Dataset (GAVD), at the held-out-source-video level, and are
            neither unseen-person nor clinical claims.

            ### 6. Interpret the current paper-profile gate pattern

            The paper run is computationally complete: all folds, seeds, and variants are
            present, and every registered constructed odd-output check is exact. Those are
            integrity successes. The empirical claims, however, are not supported by the
            registered confidence-bound rules:

            | Question | Estimate and 95% source-bootstrap CI | Registered reading |
            |---|---|---|
            | Native vanilla predictive utility | $R^2=0.0598$ `[-0.0253, 0.1257]` | Not supported; lower bound is not above 0 |
            | Learned native minus paired initialization | `-0.0180` `[-0.0385, 0.0025]` | Not supported; learned features do not beat the initialization floor |
            | Constructed odd predictive utility | $R^2=0.0430$ `[-0.0436, 0.1128]` | Not supported; lower bound is not above 0 |
            | Constructed learned minus random floor | `-0.0587` `[-0.0955, -0.0174]` | Learned construction is reliably worse than its paired random floor |
            | Native output antisymmetry error | `0.2155` `[0.1935, 0.2363]` | Not supported; upper bound exceeds the 0.1 margin |
            | Strict learned token error | `0.1138` `[0.0951, 0.1384]` | Not supported; upper bound exceeds 0.1 |
            | Learned minus initial token error | `+0.0306` `[0.0158, 0.0476]` | Training significantly worsened this error; improvement required a value below 0 |
            | Reflection minus vanilla primary $R^2$ | `+0.0041` `[-0.0056, 0.0128]` | No supported predictive advantage for reflection augmentation |
            | Learned increment beyond measured nuisance | `+0.0542` `[-0.0162, 0.1198]` | Not supported; interval crosses 0 |

            Reflection augmentation does reduce strict learned token error relative to vanilla
            by `-0.0084` with CI `[-0.0102, -0.0069]`. This is a precise relative improvement,
            but it does not rescue the main claim: the augmented learned error is still 0.1053
            with an upper bound of 0.1278, and it remains significantly worse than its own
            paired initialization by +0.0221. Relative improvement and satisfaction of an
            absolute scientific criterion are different questions.

            The secondary seed-mean ensemble has modest positive native $R^2$ values—0.0662
            for vanilla and 0.0705 for reflection augmentation—but these cannot replace the
            primary mean-single-checkpoint estimates. The paired random single-pass ensemble
            is higher at 0.0836. Restricting to the 600 high-coverage sequences from 91 sources
            gives similar native ensemble values, 0.0604 and 0.0647, so the main pattern is not
            reversed by that sensitivity subset.

            The registered final reading is therefore: **this complete run does not support
            native predictive utility, native output symmetry, strict learned representation
            equivariance, useful learned content in the constructed repair, or a
            reflection-augmentation predictive advantage.** It does support that the
            engineered odd lanes are algebraically exact and that reflection augmentation
            modestly improves strict error relative to vanilla, while both learned variants
            remain worse than initialization under the strict audit. Governance is also
            unresolved, so `submission_ready` is `False` independently of these empirical
            results.
            """
        ),
    ],
    "06_external_subject_gate.ipynb": [
        markdown(
            """
            # 06 — Optional external subject-indexed manifest gate

            The internal experiment cannot establish unseen-person generalization because
            GAVD provides no persistent subject identifier. This notebook only validates a
            separately supplied, custodian-indexed external pose manifest. It does not
            download media, infer identity, train a model, or manufacture an external
            result. Subject identifiers must be provided under an approved data contract,
            and subjects must be disjoint across train, validation, and test partitions.

            Validation is fail-closed behind ethics, data-use, and derived-pose reviews
            scoped specifically to the external dataset; GAVD reviews cannot authorize it.
            When neither required environment setting is supplied, the optional study is
            simply **not configured / not run**; this is a completed preflight decision,
            not an error and not a failure of notebooks 00–05. Supplying only one required
            setting is an incomplete configuration and remains blocked. Setting an
            environment variable cannot itself supply authorization. Passing this gate
            validates structure and authorization prerequisites; it does not run an
            external evaluation or constitute confirmation. Dataset labels, if
            present outside this contract, remain annotations rather than diagnoses, and
            no clinical claim follows from structural validation.
            """
        ),
        markdown(
            """
            ## What is—and is not—being checked

            The Gait Abnormality in Video Dataset (GAVD) has source-video identifiers but no
            persistent person identifier. Source-disjoint folds therefore cannot prove that
            the same person is absent from every fold. This optional notebook defines a
            narrow route for a different dataset whose custodian has supplied subject IDs.
            It validates prerequisites only; there is no external model evaluation in this
            suite.

            An **environment variable** is a setting supplied to the notebook process rather
            than written into the code. `LATERALITY_EXTERNAL_MANIFEST` must point to a
            comma-separated values (CSV) manifest, and `LATERALITY_EXTERNAL_GOVERNANCE` must
            point to a governance record explicitly scoped to that external dataset.
            `LATERALITY_EXTERNAL_POSE_ROOT` may optionally name the allowed pose-file root.
            Paths and authorization records can contain sensitive operational information,
            so they are not hard-coded into the notebook. The configuration cell first uses
            values already exported to the Jupyter process. For any missing value, it checks
            `neurips-laterality/.env` and then the repository-root `.env`, reading only these
            three names. It displays their presence and source but redacts their values.

            Every manifest row must contain six fields:

            | Field | Plain-language meaning |
            |---|---|
            | `dataset_reference` | The same external dataset named by governance |
            | `sequence_id` | A unique identifier for one pose sequence |
            | `subject_id` | A custodian-supplied person identifier; never inferred here |
            | `pose_path` | An existing pose file inside the declared root |
            | `split` | `train`, `validation`, or `test` |
            | `joint_schema` | Exactly `BlazePose33`, the supported 33-landmark layout |

            Validation checks governance before opening the manifest. It then rejects empty
            fields, duplicated sequence IDs or pose files, missing files, paths escaping the
            allowed root, unsupported splits or joint layouts, dataset-reference mismatch,
            and any subject appearing in more than one split. Both train and test must contain
            at least one subject. These rules trust the custodian-supplied `subject_id`; they
            do not perform face recognition or guess identity.

            <svg viewBox="0 0 1080 325" width="100%" role="img"
                 aria-labelledby="external-flow-title external-flow-description"
                 xmlns="http://www.w3.org/2000/svg">
              <title id="external-flow-title">Fail-closed external manifest validation workflow</title>
              <desc id="external-flow-description">Required environment settings lead first
              to external-dataset governance validation, then manifest and pose-file checks,
              followed by subject-disjoint partition checks. Supplying neither required
              setting produces a neutral optional-not-configured result. A partial or invalid
              supplied contract is blocked. Passing produces a validated contract but no
              evaluation result.</desc>
              <defs><marker id="arrow06" markerWidth="8" markerHeight="8" refX="7"
                refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#475569"/></marker></defs>
              <style>
                .box06{fill:#f8fafc;stroke:#334155;stroke-width:1.5}
                .check06{fill:#eff6ff;stroke:#2563eb;stroke-width:1.5}
                .pass06{fill:#ecfdf5;stroke:#047857;stroke-width:1.6}
                .block06{fill:#fff1f2;stroke:#be123c;stroke-width:1.6}
                .skip06{fill:#fffbeb;stroke:#b7791f;stroke-width:1.6}
                .line06{stroke:#475569;stroke-width:1.8;fill:none;marker-end:url(#arrow06)}
                .h06{font:600 14px system-ui,sans-serif;fill:#0f172a}
                .s06{font:12px system-ui,sans-serif;fill:#475569}
              </style>
              <rect class="box06" x="15" y="105" width="180" height="100" rx="9"/>
              <text class="h06" x="105" y="135" text-anchor="middle">Environment settings</text>
              <text class="s06" x="105" y="158" text-anchor="middle">manifest + governance</text>
              <text class="s06" x="105" y="178" text-anchor="middle">optional pose root</text>
              <text class="s06" x="105" y="195" text-anchor="middle">settings are not approval</text>
              <rect class="check06" x="245" y="105" width="185" height="100" rx="9"/>
              <text class="h06" x="337" y="135" text-anchor="middle">Governance first</text>
              <text class="s06" x="337" y="158" text-anchor="middle">external dataset named</text>
              <text class="s06" x="337" y="178" text-anchor="middle">three dated reviews</text>
              <text class="s06" x="337" y="195" text-anchor="middle">evaluation scope explicit</text>
              <path class="line06" d="M195 155 L245 155"/>
              <rect class="check06" x="480" y="105" width="185" height="100" rx="9"/>
              <text class="h06" x="572" y="135" text-anchor="middle">Manifest + files</text>
              <text class="s06" x="572" y="158" text-anchor="middle">required fields unique</text>
              <text class="s06" x="572" y="178" text-anchor="middle">paths exist inside root</text>
              <text class="s06" x="572" y="195" text-anchor="middle">BlazePose33 only</text>
              <path class="line06" d="M430 155 L480 155"/>
              <rect class="check06" x="715" y="105" width="185" height="100" rx="9"/>
              <text class="h06" x="807" y="135" text-anchor="middle">Subject partitions</text>
              <text class="s06" x="807" y="158" text-anchor="middle">one subject → one split</text>
              <text class="s06" x="807" y="178" text-anchor="middle">train and test nonempty</text>
              <text class="s06" x="807" y="195" text-anchor="middle">identity never inferred</text>
              <path class="line06" d="M665 155 L715 155"/>
              <rect class="pass06" x="940" y="45" width="125" height="95" rx="9"/>
              <text class="h06" x="1002" y="75" text-anchor="middle">Validated</text>
              <text class="s06" x="1002" y="98" text-anchor="middle">contract only</text>
              <text class="s06" x="1002" y="118" text-anchor="middle">no evidence yet</text>
              <path class="line06" d="M900 135 L940 105"/>
              <rect class="skip06" x="15" y="245" width="275" height="60" rx="9"/>
              <text class="h06" x="152" y="271" text-anchor="middle">Neither required setting supplied</text>
              <text class="s06" x="152" y="291" text-anchor="middle">OPTIONAL STUDY NOT CONFIGURED / NOT RUN</text>
              <path class="line06" d="M105 205 L105 245"/>
              <rect class="block06" x="420" y="250" width="300" height="60" rx="9"/>
              <text class="h06" x="570" y="276" text-anchor="middle">Partial configuration or invalid contract</text>
              <text class="s06" x="570" y="296" text-anchor="middle">BLOCKED / NOT RUN with a reason</text>
              <path class="line06" d="M337 205 C337 235 480 235 480 250"/>
              <path class="line06" d="M572 205 L572 250"/>
              <path class="line06" d="M807 205 C807 235 660 235 660 250"/>
            </svg>
            """
        ),
        markdown(
            """
            ## How to configure the optional external study legitimately

            A fresh checkout is supposed to report **not configured / not run**: the
            repository cannot ship another dataset's subject identifiers, authorization
            record, or local file paths. The internal GAVD workflow remains usable. To opt
            into the external contract check, use the following sequence rather than
            replacing the guard with a permissive flag.

            1. **Confirm authorization for this exact dataset and use.** Begin with
               `governance/external-status.template.json`, but keep the completed copy
               outside version control. It must name one stable `dataset_reference`, retain
               the exact scope `subject_disjoint_pose_evaluation`, and record a genuine
               status, internal reference, and date for the ethics, data-use, and
               derived-pose-release reviews. All three reviews must be resolved.
            2. **Obtain subject IDs from the custodian.** Use persistent pseudonymous IDs,
               not names and not identities inferred from video, URLs, or gait. Assign every
               sequence belonging to one person to only one of `train`, `validation`, or
               `test`. Make the split before looking at test outcomes.
            3. **Create the CSV manifest.** Copy the header from
               `governance/external-manifest.template.csv`. Every pose path must name a
               unique existing file beneath the optional pose root, every row must say
               `BlazePose33`, and train and test must both contain subjects. The
               `dataset_reference` must match governance exactly.
            4. **Configure local paths.** Copy the names from
               `governance/external.env.template` into the repository-root `.env` and
               replace its examples with real paths. Relative paths in that file are
               interpreted relative to the `.env` file, so Jupyter's launch directory does
               not change their meaning.
            5. **Run from the first cell.** The configuration summary identifies a missing
               key without exposing its value. Once both required settings are present, the
               validator reports the first governance, manifest, path, or partition rule
               that fails. Correct the source record; never invent a review or weaken a
               check merely to obtain a green display.

            The detailed operator checklist is in
            `docs/EXTERNAL_EVALUATION_GATE.md`. Reaching green in this notebook means only
            `manifest contract validated; evaluation not run`. Producing predictions and an
            external metric requires a separate, pre-specified implementation that does not
            yet exist in this suite.
            """
        ),
        markdown(
            """
            ## How to read the validation progress display

            This gate normally finishes in seconds because it reads metadata and checks file
            existence; it does not load every pose array or run a model. It accounts for two
            checks: configuration discovery and contract validation. “100%” means that the
            notebook reached a terminal preflight decision; it does not mean an external
            evaluation ran.

            If neither required setting is present, configuration discovery completes and
            contract validation is explicitly skipped. The indicator therefore ends at 100%
            in amber as `Not configured / not run`. If only one setting is present, the
            remaining check is accounted for as blocked and the display ends at 100% in red.
            When both are present, the second check validates governance, CSV structure, pose
            paths, and subject partitions. A red result means the completed check rejected
            the supplied contract; green means only that its prerequisites passed.
            """
        ),
        bootstrap_cell(),
        code(
            """
            from laterality.external import (
                ExternalEvaluationBlocked,
                ExternalManifestError,
                validate_external_manifest,
            )
            from notebook_external_config import (
                external_gate_figure,
                load_external_gate_settings,
            )
            from notebook_progress import NotebookTaskProgress

            gate_progress = NotebookTaskProgress(
                "External manifest gate progress",
                "check",
                refresh_seconds=0.5,
            )
            gate_progress.start(
                2,
                profile=context.profile,
                note=" Validation is normally brief; no external evaluation is run here.",
            )

            with gate_progress.unit(1, "Discover external-gate configuration"):
                external_settings = load_external_gate_settings(
                    dotenv_paths=(SUITE_ROOT / ".env", SUITE_ROOT.parent / ".env"),
                )
                configuration_status = external_settings.redacted_summary()

            if external_settings.configuration_state == "not_configured":
                external_status = {
                    "status": "not configured / not run",
                    "gate_state": "not_configured",
                    "validator_called": False,
                    "reason": (
                        "No external manifest or external governance path was supplied. "
                        "The optional external study was not configured and was not run; "
                        "this does not block the internal notebooks 00–05."
                    ),
                    "configuration": configuration_status,
                    "evidence_created": False,
                }
                gate_progress.finish_skipped(
                    external_status["reason"],
                    status="Preflight complete — optional study not configured",
                )
            elif external_settings.configuration_state == "incomplete":
                missing_names = ", ".join(external_settings.missing_required)
                configured_names = ", ".join(external_settings.configured_required)
                external_status = {
                    "status": "blocked / not run",
                    "gate_state": "configuration_blocked",
                    "validator_called": False,
                    "reason": (
                        f"External-gate configuration is incomplete: configured "
                        f"{configured_names}; missing {missing_names}. Supply the matching "
                        "real custodian record; templates are not approvals."
                    ),
                    "configuration": configuration_status,
                    "evidence_created": False,
                }
                gate_progress.block(
                    external_status["reason"],
                    account_for_remaining=True,
                )
            else:
                gate_progress.start_unit(
                    2,
                    "governance, manifest, files, and subject partitions",
                    detail="Fail-closed validation of the supplied external contract",
                )
                try:
                    external_cohort = validate_external_manifest(
                        external_settings.manifest,
                        external_settings.governance,
                        pose_root=external_settings.pose_root,
                    )
                except (ExternalEvaluationBlocked, ExternalManifestError) as error:
                    gate_progress.complete_unit()
                    external_status = {
                        "status": "blocked / not run",
                        "gate_state": "contract_blocked",
                        "validator_called": True,
                        "reason": str(error),
                        "configuration": configuration_status,
                        "evidence_created": False,
                    }
                    gate_progress.block(external_status["reason"])
                else:
                    external_status = {
                        "status": "manifest contract validated; evaluation not run",
                        "gate_state": "contract_validated",
                        "validator_called": True,
                        "sequences": external_cohort.n_sequences,
                        "subjects": external_cohort.n_subjects,
                        "train_subjects": len(external_cohort.train_subject_ids),
                        "validation_subjects": len(external_cohort.validation_subject_ids),
                        "test_subjects": len(external_cohort.test_subject_ids),
                        "configuration": configuration_status,
                        "evidence_created": False,
                    }
                    gate_progress.complete_unit()
                    gate_progress.complete(status="Manifest prerequisites validated")

            show_inline(external_gate_figure(context, external_status))
            external_status
            """
        ),
        markdown(
            """
            ## Step-by-step interpretation of the gate result

            ### 1. Start with `gate_state`

            `not_configured` is the ordinary fresh-checkout state. Neither required record
            was supplied, so the optional study was not attempted. The amber bar and 100%
            progress mean configuration preflight finished and the inapplicable contract
            check was explicitly accounted for as skipped. This does not invalidate or
            block the internal held-out-video experiment in notebooks 00–05.

            `configuration_blocked` means exactly one of the two required paths was supplied.
            Read `configuration["configured_required"]` and `missing_required` to distinguish
            what was found from what is absent. `contract_blocked` means both paths existed
            but a governance or manifest rule rejected them. Examples include an unresolved
            external review, a dataset-reference mismatch, a missing pose file, or one
            subject appearing in both train and test. Red 100% means the checks finished
            with a fail-closed decision; it is not a frozen kernel.

            `contract_validated` is green. It means the supplied rows passed the structural
            and authorization prerequisites. Correct an underlying record only when
            authoritative information supports the correction; do not weaken the validator
            or invent a review to change the color.

            ### 2. Then inspect `validator_called`

            It is `False` for `not_configured` and `configuration_blocked`, because there was
            no complete pair of records to validate. It is `True` for both contract states.
            The configuration report identifies sources while `paths_redacted` remains
            `True`, so configured path values are not printed.

            ### 3. Interpret a green bar narrowly

            `manifest contract validated; evaluation not run` means the supplied rows passed
            the structural and authorization prerequisites. `sequences` counts manifest rows,
            `subjects` counts unique custodian-provided subject IDs, and the three partition
            fields show how many unique subjects belong to train, validation, and test. Those
            subject sets are disjoint by construction.

            Even in this green state, `evidence_created` remains `False`. No encoder was fitted
            or transferred, no hyperparameter was selected, no test prediction was produced,
            and no external metric or confidence interval exists. The chart explicitly says
            “evaluation not run” to prevent prerequisite validation from being misreported as
            external confirmation.

            ### 4. Keep authorization and scientific evidence separate

            Environment variables point to records; they do not grant permission. Governance
            can authorize a future subject-disjoint pose evaluation, but it cannot guarantee
            good performance. Conversely, a technically plausible model cannot bypass missing
            authorization. A separate, pre-specified external evaluation implementation and
            result would be required before making any unseen-person statement.

            The narrow conclusion has three legitimate forms: **optional study not configured
            and not run**, **supplied external contract blocked and evaluation not run**, or
            **manifest prerequisites validated but evaluation still not run**. None is a
            clinical claim, and none by itself is external empirical evidence.
            """
        ),
    ],
}


def build_notebook(filename: str, cells: list) -> Path:
    notebook = new_notebook(cells=cells, metadata=KERNEL_METADATA.copy())
    stem = filename.removesuffix(".ipynb").replace("_", "-")
    for index, cell in enumerate(notebook.cells):
        cell["id"] = f"{stem[:48]}-{index:02d}"
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    output = SUITE_ROOT / filename
    nbformat.write(notebook, output, version=4)
    return output


def main() -> None:
    written = [build_notebook(filename, cells) for filename, cells in NOTEBOOKS.items()]
    for path in written:
        print(path.relative_to(SUITE_ROOT.parent))


if __name__ == "__main__":
    main()
