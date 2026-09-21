# Notebook 06: results, contribution value, and paper readiness

The completed experiment supports a useful pilot-study conclusion: **this S-JEPA
procedure has not demonstrated a consistent advantage over an average-pose
classifier on this small collection**. That finding is worth explaining and
testing further. It does not establish that temporal learning cannot help, that
the methods are equivalent, or that the model can diagnose a condition.

The [notebook tutorial](../06_capstone_rf_vs_sjepa.ipynb) explains all five systems,
their configurations, the training steps, the metrics, and the MS errors. This
document focuses on what another researcher could learn from the result and how
to develop a credible submission. Findings refer to the saved full-budget laptop
run reviewed on September 20, 2026, not smoke tests or earlier datasets.

## What the completed comparison establishes

Each of 88 clips from 41 source videos has one held-out prediction from each of
five systems. S-JEPA learns 96 features and uses a frozen-encoder linear probe.
The RF uses designed joint-angle summaries. Visibility and mean pose each use
66 fixed features with the same linear-classifier settings as S-JEPA. Majority
predicts the most frequent training-clip label without reading the test skeleton.

| Comparison | Measured result | Conclusion we can defend |
|---|---|---|
| S-JEPA versus mean pose | Source macro-F1 0.457 versus 0.452; mean pose leads in four folds | No consistent advantage over the order-free control has been demonstrated |
| Source versus clip weighting | Mean pose leads under clip weighting: 0.410 versus S-JEPA's 0.397 | The apparent model ordering depends on how repeated-source clips count |
| Extra training | Continued loss is lower in all folds; validation F1 improves in two of five | Better masked-feature loss does not consistently translate into better label prediction in this recipe |
| Variation across test folds | S-JEPA source F1 ranges from 0.217 to 0.780 | Performance is sensitive to the held-out groups; the cause remains unresolved |
| MS errors | S-JEPA source-weighted MS F1 0.443; 8 of 29 MS clips correctly labeled | Correct MS predictions occur, but many clips are still missed |
| Current RF implementation | Source F1 0.411; a duplicated ankle-range feature was found | This is a result for the current extractor, not yet a validated comparison with its intended feature set |

F1 is not accuracy. Source weighting divides each source's total weight of one
among its clips; it does not combine them into a patient-level diagnosis. The
five folds share training data, and no confidence intervals or repeated-seed
results are present. Neither a small aggregate gap nor a fold-win count is a
statistical significance or equivalence test.

## How valuable are these findings?

The **strongest candidate contribution is the controlled empirical comparison**:
an average-pose control challenges the interpretation that a learned model's
score demonstrates useful temporal gait learning. That is a meaningful question
for small skeleton datasets. The present evidence is preliminary, because it
comes from one collection and one training seed, without an untrained-encoder
or matched temporal-order ablation. We should describe the observation as
dataset- and recipe-specific until those checks are available.

The source/clip comparison and continuation results are useful supporting
lessons. They explain why evaluation units and downstream validation matter
here. Those principles are already familiar, so we should not present them as
new metrics or new learning theory. Their value comes from a clear, reproducible
case study that helps readers avoid overinterpreting similar experiments.

The reusable evaluation pipeline is a practical contribution if others can run
it with permitted inputs. Source-grouped splits, linear probes, Random Forests,
caching, and process parallelism are established techniques. Combining them
carefully supports the study but does not automatically establish algorithmic
novelty. The observed 216-second full run is not a quantified acceleration claim
without a matched timing baseline. The [performance guide](15-capstone-performance.md)
documents the implementation choices separately from model-quality evidence.

The limitations also identify what **is not a contribution yet**: a superior
gait classifier, an explanation of the source of confounding, a clinical
biomarker, a new foundation model, or an agent/LLM method. We have not tested
those claims. Nor is fixing the RF coding error itself a substantive negative
result about Random Forests.

## What remains before submission?

There is enough here to begin a focused draft and obtain feedback. It would be
premature to describe the current comparison as fully validated. That is a
manageable research status, not a dismissal of the work: the saved experiment
has replaced a vague hope of improvement with a specific question to investigate.

1. **Repair the comparison before polishing the claim.** The legacy
   [angle extractor](../../../ambient/classification/features.py) populates
   `left_ankle_range` using right-ankle statistics. Both ankle-range columns
   are identical in the saved features. Correct it under a separate code-change
   task, add a regression test with unequal left/right ranges, and rerun the
   affected comparison with fresh compatible cache entries. Retain and label
   the original result. We do not know whether the correction will improve RF.
2. **Test the central interpretation directly.** Include an untrained encoder
   with the same readout/probe, and a predefined temporal-order control. State
   whether order is changed during training, evaluation, or both. Merely
   shuffling test frames also introduces an input change, so it is not by
   itself a clean estimate of the benefit of temporal learning.
3. **Quantify how stable the comparison is.** Run a declared set of seeds,
   retain every result, and use paired source-level comparisons. If bootstrapping
   fixed OOF predictions, resample whole sources and explain that the interval
   is conditional on those trained models. This does not capture all uncertainty
   from training, split choice, or having inspected this collection before.
4. **Document the data boundaries and reproduction path.** Review label
   provenance, duplicates, possible repeat participants, exclusions, and
   permissions. Release code, configuration, split manifests, and predictions
   where permitted. Do not equate a source ID with a verified participant or
   assume the code license covers all source videos.

These additions support a bounded pilot or lessons-learned paper; an external
cohort would be especially important for stronger generalization or clinical
claims. Reusing the current folds after looking at their results is further
development, not a newly untouched test. Declare that history and freeze the
next comparisons in advance.

A suitable draft could ask: **“Do learned skeleton features outperform
average pose on a small source-grouped gait collection?”** Organize it around
the question, the five-system comparison, the fold and weighting results, and
the limits of the inference. Avoid a title announcing superior MS diagnosis.
We have not established priority over related literature, so do not use “first.”

## Workshop options

The dates below were checked against organizer or parent-conference pages on
**September 20, 2026**. They are planning information, not a guarantee that the
calls will remain unchanged. Venue affiliation and scientific fit are separate
questions. An accepted workshop at a reputable conference can still be the wrong
audience for this particular study.

| Option | Verified submission information | Fit for this project |
|---|---|---|
| ICLR 2027 workshops | Workshop-selection notifications: November 29, 2026; suggested paper deadline: February 1, 2027. A specific workshop's deadline is not yet established here. [Official call](https://iclr.cc/Conferences/2027/CallForWorkshops) | Best direction to watch for representation learning, temporal data, small-data evaluation, or health ML. Choose from the confirmed program and its individual calls. |
| PerFail 2027, with IEEE PerCom | Listed paper deadline: November 17, 2026. The call seeks lessons from negative results and explicitly excludes results based merely on coding bugs. [Workshop call](https://perfail-workshop.github.io/), [parent workshop list](https://percom.org/list-of-accepted-workshops/) | Plausible after correction and controlled follow-up, if the sensing/pervasive-computing connection is substantive. A low score alone is insufficient. |
| PerAgents 2027, with IEEE PerCom | Listed paper deadline: November 3, 2026. Focus: pervasive agentic systems and multimodal foundation-model systems. [Workshop call](https://peragents.github.io/2027/), [parent workshop list](https://percom.org/list-of-accepted-workshops/) | Low fit now: there is no evaluated agent, LLM, or foundation-model deployment. A new systems experiment would be needed, not just revised terminology. |
| AgentArch 2027, with IEEE ICSA | Parent lists it as accepted; abstract deadline December 16 and paper deadline December 20, 2026. [Parent call and dates](https://conf.researchr.org/track/icsa-2027/icsa-2027-workshops), [AgentArch scope](https://agentarch.org/) | Low fit now: it studies agent architectures, coordination, and human-agent collaboration. Our current classifier experiment does not address those questions. |
| ACM The Web Conference 2027 workshops | Parent timetable lists workshop papers due January 4, 2027; workshop proposals are still being selected. [Official workshop call](https://www2027.thewebconf.org/workshops/) | Watchlist only. No specific suitable agent/LLM workshop is verified here, and the current study lacks a clear Web or agent contribution. |

The ICLR October 9, 2026 date concerns **organizer proposals**, not submission of
our research paper. Its current call also contains an inconsistent year in the
event-date sentence; use the eventual confirmed schedule for travel decisions.
PerAgents' page contains conflicting anonymity guidance, so verify the current
submission instructions before preparing a paper for it. Avoid committing to
fees or travel based only on a workshop's name or an old call.

My recommendation is to develop the honest representation-learning study,
monitor the ICLR program, and consider PerFail if the corrected experiments
produce a substantive lessons-learned account. Do not add an LLM solely to match
a fashionable venue. A separate agent study could be worthwhile, but would need
its own research question, baselines, and evaluation. No acceptance probability
can be justified from the present scores.

## Evidence and reproducibility record

The retained run is
[`capstone-20260921T011305448893Z`](../artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/capstone-20260921T011305448893Z/).
Its UTC timestamp falls on September 20 in the project's Pacific timezone.

- [`results.json`](../artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/capstone-20260921T011305448893Z/results.json)
  records the configuration, selection decisions, metrics, and execution time.
- [`oof.json`](../artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/capstone-20260921T011305448893Z/oof.json)
  contains the 88 paired clip predictions. Recomputing with
  `sjepa.full_experiment.summarize_oof` reproduces every saved aggregate metric.
- [`provenance.json`](../artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/capstone-20260921T011305448893Z/provenance.json)
  identifies code, inputs, environment, device, registry, and training budgets.
- The retained RF feature cache has 82 columns; all five training partitions
  retain 15 varying columns, including the duplicated ankle-range columns.
  This is a code-and-feature inspection, not a corrected RF evaluation.

The notebook's executable cells, execution counts, and saved outputs are retained
unchanged by this writing update. Its narrative lives in
[`scripts/notebook_06_tutorial.py`](../scripts/notebook_06_tutorial.py), called by
the shared notebook generator. The empirical values are an explicitly dated
snapshot; regeneration does not execute a new experiment or update those values.
