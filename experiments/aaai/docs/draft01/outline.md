# Manuscript outline and conditional narratives

This outline is written before the complete new-study result is available. It supplies three evidence-dependent narratives, not expected findings. Select a narrative only after reconciling the frozen real-run artifacts, diagnostics and review record. Keep all primary results, including contrary controls. Do not change the score direction, eligible cohort or source folds to obtain a preferred story.

The manuscript concerns a controlled experiment: whether the help supplied by other joints is a useful normal-reference signal for unusual gait. It is not a new clinical world model or the first predictive skeleton anomaly detector. The [implemented protocol](../context_study_protocol.md), [paper plan](../paper_plan.md), [results report](../context_study_results.md), [review record](../context_study_review.md) and [primary-source claim map](primary_source_claim_map.md) govern the final text.

## A title that survives an uncertain outcome

A suitable neutral title is **How Much Does the Rest of the Body Help? A Source-Held-Out Study of Predictive Gait Context**. It makes the question concrete without announcing a gain before it exists.

Possible outcome-specific titles should remain conditional:

| Evidence observed | Possible title direction |
| --- | --- |
| A supported primary improvement, with adequate controls | Predictive Context Profiles for Gait Anomaly Ranking. |
| An imprecise primary comparison | Evaluating the Uncertain Value of Predictive Context for Gait Anomaly Detection. |
| A supported primary degradation under adequate learning | When Predictive Context Profiles Underperform Raw Skeleton Surprise. |

Avoid “clinical diagnosis,” “causal neuromechanics,” “robust across populations,” “state of the art,” and “world model of neurological disease” in the title. These claims exceed the current observations and comparisons.

## Abstract structure

Use five connected sentences, populated from the final artifacts:

1. **Problem:** Abnormal walking can remain predictable, so prediction discrepancy alone may miss changes in relationships between body regions.
2. **Question and design:** Compare the same hidden joint patches under own-joint, patch-local and full-context access using a normal-trained skeleton JEPA with frozen representations, coordinate and contextual-latent targets, and independent matched heads.
3. **Evaluation:** State the retained source population, normal-source support, four source-held-out folds, normal-reference construction and matched controls. Explicitly call the task GAVD folder-label ranking.
4. **Measured finding:** Report the actual primary difference and its paired conditional interval, then the most informative contrary or supporting control. If incomplete, do not replace the missing comparison with a favorable secondary result.
5. **Implication and limit:** State what the comparison establishes under this recipe, followed by the principal generalization and interpretation limit.

There are no numerical performance placeholders in this outline. Counts and effect estimates must be inserted from the frozen run. The update budget and mask count are protocol details; they are not evidence of convergence.

## Seven-page body

### 1. Introduction and question, approximately one page

Open with a hidden left-ankle patch: can its own visible history predict it, and how much does the rest of the body help? Explain that both reduced and increased help may be unusual. A tightly repeated compensation need not create a large residual.

Connect the question to varied neurological and muscle constraints without mapping a folder to one mechanism. Stroke, Parkinsons, cerebral palsy and myopathic gait motivate movement-level questions; myopathic is not a single neurological etiology. A camera sees projected motion after control, mechanics, compensation and pose-estimation errors have acted.

Conclude the introduction with bounded contributions:

- A matched comparison of information access before contextual encoding, with local-coordinate and contextual-teacher targets.
- A source-level three-region normal-reference profile and explicit support, reference and observation-budget controls.
- An empirical finding, positive or adverse, on a traceable GAVD derivative, with the limitations that determine its interpretation.

Only include the third contribution once it is known. Engineering completeness and a teaching package are reproducibility strengths, not sufficient algorithmic novelty by themselves.

### 2. Related work and scope, approximately half a page

Use the claim map rather than an unstructured list of JEPA papers. Explain the distinction from Skeleton S-JEPA's representation learning, normal-only gait analysis in FSGait and GenGait, SeeKer's conditional keypoint densities and confidence-aware scores, and STG-NF's pose likelihoods.

Our fixed uniform masks replace Skeleton S-JEPA's original motion-aware selection. VICReg, a small sixteen-joint model, frozen encoders and matched context heads are disclosed adaptations. The primary study does not use a learned stochastic latent variable or the optional positional-noise ablation.

A central novelty question is whether the *difference between matched information conditions*, rather than another predictor or anatomical heatmap, yields a useful and interpretable signal. If ridge explains the finding equally well, the contribution is the comparison and evidence, not a necessary JEPA architecture.

### 3. Data, observations and evaluation roles, approximately one page

Describe the all-twelve-folder binary endpoint precisely: `normal` versus every other folder. Explain why exercise, pregnancy and style are included without calling them medical pathology. Keep the five-class condition task separate.

Use an attrition diagram from the fresh 348-source annotation inventory, including 32 normal sources, through media availability, extraction, window QC, fixed-mask profile support and final source folds. Populate every retained count from saved artifacts. These source IDs are not verified patient identities.

Describe the sixteen joints, four-frame tokens, 32-frame windows, 15 Hz sampling, confidence handling, NaNs, pelvis centering and prefix torso scaling. Explain how observation eligibility is frozen before folds without reading losses. Report what visibility-dependent selection removes.

Show four outer source folds, with five non-test normal sources reserved in each fold for diagnostics and the reference. Other normals fit the anomaly models; abnormal sources do not. Record prior exposure from the earlier five-class study. New folds on previously examined recordings remain exploratory.

### 4. Matched predictive experiment, approximately one and a half pages

First describe normal-only pretraining, fixed update counts and representation-health diagnostics. State that VICReg discourages collapse and redundancy rather than separating disease classes. Freeze the encoder, full-input EMA teacher and center after pretraining.

Then describe the six heads: own, patch-local and full access crossed with coordinate and latent targets. Each target type uses identical initial head parameters, separate optimizers, common draws and equal update budgets. Show that restrictions apply before the encoder's first attention layer. Hidden values never enter the branch; the coordinate target and teacher distribution remain detached targets.

Name two interpretive boundaries. Shared pelvis/torso normalization already supplies a body reference to the own-joint condition. A four-frame patch-local control includes short motion as well as geometry. The task is completion, because context may lie after the hidden patch.

Define per-joint benefit as own minus full error. Show aggregation within window, sequence and source, then equal joint weights in left-leg, right-leg and arm groups. Derive the shrunk diagonal normal reference and its two-sided distance in a few equations. Do not call that distance a probability or an exact likelihood ratio.

### 5. Experiments and measured findings, approximately one and a half pages

Lead with the primary matched comparison: latent profile distance versus full-context raw KL on identical twelve-joint support and aggregation. Show fold support and the normal-count-weighted within-fold AUROC. Report the paired source-bootstrap difference and its conditioning. Do not pool raw distances from differently fitted folds.

Use a compact control table for constituent own/full distances, patch-local access, local-coordinate profiles, source-balanced ridge and the random-backbone control with fitted heads. Include the fixed four-patch observation-budget and leave-one-reference-out sensitivities. The random-backbone latent teacher has different targets, so coordinate results help interpret that control.

Report five-class RF and frozen-encoder probe results separately because they receive abnormal training labels. Their source-label rows divide one source's evaluation weight, and inner normal references stay excluded from fitting. Do not place their macro-F1 beside a normal-only AUROC as if they measured the same task.

### 6. Diagnostic interpretation and limitations, approximately one page

Show a small, transparently selected failure or sensitivity gallery with video, pose support and signed profiles. Report paired timing and observation edits without interpreting an edited leg as a simulated disease. Geometry damage and support loss are part of the result.

When measured, include the optional four-factor teacher diagnostic in a compact panel. It compares original/edited predictor context and original/edited frozen teacher targets on **eight unchanged arm and right-leg joints**. The signed components are predictor change, teacher change, interaction and their total, in raw per-target KL units. They do not decompose the nonlinear three-region anomaly score, and the absent left-leg region is not reconstructed from the eight available joints.

End with the limits most likely to affect the result: few normal sources, five-reference estimation, visibility selection, shared normalization, contextual targets, previously examined sources, missing published baselines, and lack of verified independent participants or external sites. Completed blinded phenotype annotation should only be claimed if independent reviews and agreement measurements exist.

## Conditional narrative A: the primary result is positive

Use this narrative only if the primary latent-profile difference is positive and its uncertainty and support justify saying more than “a higher point estimate.” Report the full interval even if it is favorable. Check that the result is not carried by one fold, one reference source, a small visibility subset or a changed target weighting.

The strongest supported story would be: the source-level pattern of predictive benefit provides information beyond matched raw discrepancy under this normal-only recipe, and the coordinate, patch-local, ridge and random-backbone controls identify what part of that information depends on learned representations and broader temporal context.

There are narrower positive stories:

- If only latent profiles improve, emphasize a useful contextual-feature score while retaining teacher construction as an explanation. Do not claim a validated movement-coordination marker.
- If ridge matches the coordinate profile, emphasize the value of comparing information conditions. Do not claim that JEPA is required.
- If the random-backbone control matches the trained model, the study has not demonstrated a benefit from self-supervised learning.
- If patch-local context explains the gain, describe local geometry and short motion rather than long-range coordination.

Even a favorable conditional interval does not establish superiority over SeeKer or STG-NF, clinical specificity, population robustness or causal explanation. These limits should appear near the result and in the abstract's final sentence.

## Conditional narrative B: the primary result is inconclusive or near zero

Distinguish uncertainty from equivalence. An interval spanning useful improvement and useful degradation supports an inconclusive finding. It does not establish that context is useless. A precise estimate near zero can support a limited absence-of-benefit statement only relative to a justified, declared practical effect scale; this study should not invent an equivalence margin after seeing results.

A useful narrative is that additional joint context may improve normal reconstruction without improving abnormal-source ranking. If this pattern is observed, it separates two often conflated claims: learning a prediction task and obtaining an anomaly detector. The fixed support, teacher-target and reference controls help identify why those claims can diverge.

If normal learning itself is inadequate, the manuscript must be narrower: the bounded laptop-scale recipe did not establish a useful effect. Do not infer a general failure of JEPA, predictive learning or the biological hypothesis. Report training curves and the missing adequacy evidence prominently.

The value of this paper would come from a transparent empirical boundary, carefully controlled comparisons and reproducible failure analysis. A long list of unfinished extensions would not compensate for an unclear primary result.

## Conditional narrative C: the primary result is negative

If the paired estimate supports degradation relative to raw full-context KL, report it directly. Do not reverse the benefit sign, swap target type or choose a different split as a replacement headline. Secondary findings can explain the primary failure, but must remain secondary.

Potential explanations are hypotheses to test, not automatic conclusions. Subtracting two errors may amplify estimation noise. A five-source reference may be unstable. More predictable abnormal movement may not align with the chosen regional distance. Shared normalization, patch-local geometry or contextual teacher construction may dominate the contrast. Use the available controls to determine which explanation has supporting evidence.

The compelling story would be a demonstrated boundary on a plausible predictive-benefit score, with diagnostics that show where the reasoning fails. It becomes stronger if learning adequacy is established and the failure persists under the fixed observation budget and reference sensitivity. Without adequacy, limit the conclusion to this implementation and training budget.

## Tables and figures to populate only from artifacts

| Item | Required content | Evidence source |
| --- | --- | --- |
| Figure 1 | Hidden ankle under three information conditions; coordinate versus teacher target. | Implemented access rules, with illustrative labels. |
| Figure 2 | Cohort attrition and source roles. | Fresh manifest, ingestion and profile-eligibility ledgers, fold registry. |
| Table 1 | Retained sources, normal support and category composition. | Final frozen metadata and source registry. |
| Table 2 | Primary ranking comparison and matched controls. | Source scores, fold metrics and paired bootstrap. |
| Table 3 or small panel | Training adequacy and reference/budget sensitivities. | Histories, support tables and sensitivity files. |
| Secondary panel | Supervised RF/probe condition recognition. | Per-fold condition metrics and paired source-label predictions. |
| Diagnostic panel | Timing/observation behavior and optional teacher factors. | Development-control protocol, geometry and raw component files. |

Use readable vectors and concise captions. An illustrative diagram should say that it is illustrative. The teacher panel must say “raw KL, eight unchanged joints,” and cannot reuse the three-region profile axis as if it were that score.

## Readiness gates before drafting the final abstract

Confirm that all four real folds completed with the declared controls and fixed corpus. Review normal training adequacy. Check that every metric uses the declared source support and weighting, that the interval matches the point estimate, and that missing outputs are visible rather than silently filled.

Confirm that all twelve notebooks execute in fresh kernels in the intended modes and that synthetic outputs are labeled separately. Read the systematic implementation review and resolve or disclose its findings. Confirm that any claimed teacher diagnostic or intervention result has an actual saved real-data artifact. Blinded annotation and published-baseline reproduction remain separate gates.

The paired bootstrap conditions on fixed models, folds, normal references and observed source IDs. Five inner normals and at most 32 normal source IDs before filtering constrain inference; more abnormal videos do not repair that. Unknown repeated people and prior data exposure remain unresolved. The lack of a validated published baseline prohibits a state-of-the-art conclusion.

The final narrative should survive those limits without needing them hidden in a footnote. If it does not, narrow the paper's claim rather than broadening its language.
