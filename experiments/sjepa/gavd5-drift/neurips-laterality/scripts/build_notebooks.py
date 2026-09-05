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
            post-development, within-GAVD cross-validated performance on held-out
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
        bootstrap_cell(),
        code(
            """
            from laterality.config import model_config
            from laterality.governance import load_governance, submission_readiness
            from laterality.visualization import governance_figure

            governance_path = SUITE_ROOT / "governance" / "status.json"
            governance_payload = load_governance(governance_path)
            governance_readiness = submission_readiness(governance_payload)

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
            show_inline(governance_figure(context, governance_payload))
            protocol_snapshot, governance_readiness
            """
        ),
        markdown(
            """
            A `smoke` run uses generated poses and a tiny model only to test plumbing,
            lineage checks, and algebraic invariants. Its metrics are **not empirical
            evidence** and must never enter a paper table. Likewise, a completed paper
            run does not override an unresolved governance gate.
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
            - A **digest** is a long content fingerprint. This project uses the Secure
              Hash Algorithm 256-bit form, abbreviated **SHA-256**. If relevant content
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
        code(
            """
            from collections import Counter

            import numpy as np

            from laterality.data import prepare_cohort, save_cohort
            from laterality.geometry import anatomical_mirror
            from laterality.visualization import cohort_figure

            cohort = prepare_cohort(context)
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
            for xyz, valid in zip(cohort.model_xyz, cohort.model_valid):
                mirrored_xyz, mirrored_valid = anatomical_mirror(xyz, valid)
                restored_xyz, restored_valid = anatomical_mirror(
                    mirrored_xyz, mirrored_valid
                )
                assert np.array_equal(restored_xyz, xyz)
                assert np.array_equal(restored_valid, valid)
                checked_involutions += 1

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

            show_inline(cohort_figure(context, cohort))
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
        bootstrap_cell(),
        code(
            """
            from laterality.data import load_cohort
            from laterality.splitting import build_source_splits, save_splits
            from laterality.visualization import split_figure

            cohort = load_cohort(context)
            split_config = context.protocol["splits"]
            splits = build_source_splits(
                cohort.table,
                context.protocol["data"]["conditions"],
                outer_folds=split_config["outer_folds"],
                inner_folds=split_config["inner_folds"],
                seed=split_config["seed"],
            )
            split_artifact = save_splits(context, cohort, splits)

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
            {
                "split_artifact": str(split_artifact),
                "source_census": splits["source_census"],
                "folds": fold_summary,
            }
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
        bootstrap_cell(),
        code(
            """
            import pandas as pd

            from laterality.data import load_cohort
            from laterality.splitting import load_splits
            from laterality.training import train_selected
            from laterality.visualization import training_figure

            cohort = load_cohort(context)
            splits = load_splits(context, cohort)
            training_summaries = train_selected(context, cohort, splits)
            show_inline(training_figure(context, training_summaries))
            pd.DataFrame(training_summaries).drop(columns="history").sort_values(
                ["variant", "fold", "seed"]
            ).reset_index(drop=True)
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
        bootstrap_cell(),
        code(
            """
            from laterality.data import load_cohort
            from laterality.evaluation import evaluate_selected
            from laterality.splitting import load_splits
            from laterality.visualization import evaluation_figure

            cohort = load_cohort(context)
            splits = load_splits(context, cohort)
            evaluations = evaluate_selected(context, cohort, splits)

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
        bootstrap_cell(),
        code(
            """
            from laterality.data import load_cohort
            from laterality.reporting import aggregate_and_save
            from laterality.splitting import load_splits
            from laterality.visualization import report_dashboard

            cohort = load_cohort(context)
            splits = load_splits(context, cohort)
            report = aggregate_and_save(context, cohort, splits)
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
            If either required environment setting is absent, the validator is deliberately
            not called and the status is **blocked / not run**. Setting an environment
            variable cannot itself supply authorization. Passing this gate
            validates structure and authorization prerequisites; it does not run an
            external evaluation or constitute confirmation. Dataset labels, if
            present outside this contract, remain annotations rather than diagnoses, and
            no clinical claim follows from structural validation.
            """
        ),
        bootstrap_cell(),
        code(
            """
            import os

            from laterality.external import (
                ExternalEvaluationBlocked,
                ExternalManifestError,
                validate_external_manifest,
            )
            from laterality.visualization import external_gate_figure

            manifest_setting = os.getenv("LATERALITY_EXTERNAL_MANIFEST")
            external_governance_setting = os.getenv(
                "LATERALITY_EXTERNAL_GOVERNANCE"
            )

            if not manifest_setting or not external_governance_setting:
                external_status = {
                    "status": "blocked / not run",
                    "validator_called": False,
                    "reason": (
                        "Both LATERALITY_EXTERNAL_MANIFEST and an external-dataset-"
                        "scoped LATERALITY_EXTERNAL_GOVERNANCE record are required."
                    ),
                    "evidence_created": False,
                }
            else:
                pose_root_setting = os.getenv("LATERALITY_EXTERNAL_POSE_ROOT")
                try:
                    external_cohort = validate_external_manifest(
                        manifest_setting,
                        external_governance_setting,
                        pose_root=pose_root_setting,
                    )
                except (ExternalEvaluationBlocked, ExternalManifestError) as error:
                    external_status = {
                        "status": "blocked / not run",
                        "validator_called": True,
                        "reason": str(error),
                        "evidence_created": False,
                    }
                else:
                    external_status = {
                        "status": "manifest contract validated; evaluation not run",
                        "validator_called": True,
                        "sequences": external_cohort.n_sequences,
                        "subjects": external_cohort.n_subjects,
                        "train_subjects": len(external_cohort.train_subject_ids),
                        "validation_subjects": len(external_cohort.validation_subject_ids),
                        "test_subjects": len(external_cohort.test_subject_ids),
                        "evidence_created": False,
                    }

            show_inline(external_gate_figure(context, external_status))
            external_status
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
