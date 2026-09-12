# Residual-head failure and the path to distillation

**Historical first pass.** The real models and their cache/predictions were later
supplied and evaluated. Read the [saved-model evaluation](saved-model-evaluation.md)
for the confirmed mechanisms and revised next steps. In particular, it identifies
an additional inner-fold scaling failure and shows that zero initialization alone
is insufficient. Artifact-access limitations below describe the initial pass.

Investigation date: 11 September 2026. Evidence: real `gate-v2`, notebook batch
`haic-xaGGjyT7`; separate local synthetic mechanism probes.

**Keep the completed STOP. Do not begin student or adapter training. The next
experiment should diagnose and calibrate the residual predictor using the cached
development data.** There is a reproducible failure in the present head: it can
learn an essentially perfect training fit while making large, unnecessary
corrections on new examples. Its high-dimensional baseline-input branch explains
almost all the error in a controlled reproduction. This is a strong candidate
explanation for the real run, not yet a measured decomposition of its checkpoints.

## What the real result establishes

All nine executed notebooks completed, including five outer folds. The recorded
cohort contains 50 clips from 43 source videos. All direct-v2 readiness checks
passed. Notebook 04 reports complete, nonsynthetic measurement and STOP, with all
advancement fields false. It is different from the legacy audit-blocked run.

| Predictor | Mean held-out R², approximately | Gain over ridge |
| --- | ---: | ---: |
| RGB + nuisance ridge | 0.2717 | Reference |
| Real skeleton + residual head | 0.0284 | −0.243242465 |
| Time-shuffled skeleton + residual head | 0.0285 | −0.243225799 |
| Mismatched skeleton + residual head | 0.0286 | −0.243054259 |
| No skeleton + residual head | 0.0285 | −0.243175954 |

The primary skeleton-specific increment is **−0.00006651 R²**, with paired
source-bootstrap 95% interval **[−0.00027454, +0.00008082]**. Only one of three
seeds favors real skeletons; 28.6% of paired bootstrap draws favor them. Every
seed loses substantially to ridge. The small paired interval describes these
fitted predictors; the bootstrap does not refit the models or repeat selection.

The common degradation is much larger than the differences between skeleton
conditions. A pipeline that fails even with no skeleton needs diagnosis before
this becomes a claim that skeleton motion is intrinsically uninformative.

Source: [retained notebook 04](../../../notebook_runs/04_results_and_next_decision.ipynb).
The [previous notebook evaluation](../../../work/artifacts/notebook-results-evaluation-2026-09-11/report.md)
contains execution identities, gate arithmetic and limitations. Full R² values
above are averaged from rounded report entries; gains use its embedded exact JSON.

**Evidence boundary:** the original cache, model files and prediction Parquet
files are absent locally. Non-interactive SSH to HAIC failed authentication.
The saved notebook outputs and implementation were inspected; the original
predictions and report seal have not been independently recomputed here.

## 1. Reproduced mechanism: unconstrained baseline-feature directions

The head is additive:

\[
\hat y = g(x) + c(s,x), \qquad
c(s,x) = W_x x + W_s h(s) + b.
\]

Here `g` is the fitted ridge baseline, `x` contains **2,382** standardized
baseline inputs, and `h(s)` is a width-64 temporal skeleton encoder. The baseline
inputs consist of three 768-dimensional video summaries and 78 nuisances.
The correction head contains **664,192 trainable parameters**; **609,792** belong
to the direct `x → correction` weights. Each outer fit sees only 39–41 clips;
inner fits see fewer. See
[the head implementation](../../../src/gavd6_sjepa/research_directions/future_innovation/fi_residual_models.py).

The output layer starts with random weights. This is the default in the pinned
[PyTorch 2.6 Linear implementation](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/linear.py#L105-L113).
It therefore starts by perturbing a useful baseline, including through feature
directions for which the small training sample supplies no constraint.

For training design matrix `X`, decompose the baseline weights into a component
in the row space of `X` and an orthogonal component:

\[
W_x = W_{\parallel}+W_{\perp}, \qquad XW_{\perp}^{\mathsf T}=0.
\]

Removing `W_perp` cannot change predictions on those training rows. It can change
predictions on new rows. A 40-row centered design has rank at most 39, leaving
at least 2,343 of 2,382 directions outside its row space. This does **not** imply
that real held-out data vary strongly in all those directions; that is exactly
what the saved-model diagnostic must measure.

The loss cannot identify this component from the training examples. AdamW's
coordinatewise updates can move weights into these directions, so they should
not be described as literally unchanged initialization. Its explicit decay is
also weak over this run: at learning rate 0.001 and decay 0.1, 200 decay steps
alone retain about 98% of a weight's magnitude.

### Controlled intervention

The probe used the unchanged production `train_head`, 40 training examples,
256 independent test examples, 2,382 standardized Gaussian inputs, width 64,
256 residual outputs, and the frozen 25/50/100/200 budgets. It set the true
residual to zero: the ideal correction is exactly zero. A zero residual is
compatible with nonconstant teacher targets when the baseline is perfect; this
probe is not an R² calculation on constant teacher targets.

| Seed | Final recorded training MSE | Held-out correction MSE | Held-out MSE after removing only `W_perp` |
| --- | ---: | ---: | ---: |
| 7 | 1.18 × 10⁻¹⁰ | 0.341883 | 1.89 × 10⁻⁷ |
| 19 | 1.03 × 10⁻¹⁰ | 0.342214 | 2.05 × 10⁻⁷ |
| 31 | 1.15 × 10⁻¹⁰ | 0.345291 | 3.32 × 10⁻⁷ |

Training predictions changed by less than 4.6 × 10⁻¹⁵ under the float64
decomposition. More than 99.9999% of held-out squared error disappeared.
Separately, initializing the output weights and bias to zero, with all other
training choices unchanged, produced exactly zero error in all three seeds.

An easy positive control confirmed that zero initialization does not freeze the
network: with a planted skeleton signal and `x=0`, its held-out MSE was
0.00000649, versus 0.340544 for zero correction and 0.339546 for no skeleton.
This only checks sample-pairing sensitivity in an easy problem. It does not
prove temporal-order sensitivity or recovery amid competing high-dimensional RGB
inputs, and it is not evidence about GAVD.

The local runtime was Torch 2.13.0 / NumPy 2.5.2, not the pinned HAIC Torch 2.6
environment. Production source fingerprints match the recorded run; exact
numerical reproduction in the HAIC environment remains a separate check.

Reproduce with:

```bash
.venv/bin/python scripts/research_directions/future_innovation/probe_future_innovation_residual_failure.py \
  --output work/artifacts/fi-mechanism-probe-new.json
```

The [measured JSON](../../../work/artifacts/future-innovation-root-cause-2026-09-11/mechanism-probes.json)
records the source/script hashes, dimensions, runtime, histories and interventions.
The model source and original run are unchanged.

![Synthetic training, held-out error and null-component intervention](../../../work/artifacts/future-innovation-root-cause-2026-09-11/mechanism-probe.png)

## 2. Two amplifiers visible in the training code

**The head is trained on in-sample ridge residuals.** The ridge and residual head
see the same fitting examples. In a wide design, ridge can nearly interpolate
those examples while retaining substantial error on unseen sources. The head
then learns a much smaller error distribution than the one it must correct at
test time. This is not source leakage, and in-sample residual fitting is not
universally invalid; the concern here is its interaction with a nearly
interpolating baseline and an underconstrained correction head.

A separate synthetic noisy-teacher probe using the production ridge illustrates
the issue. At alpha 0.1, training residual MSE was 1.71 × 10⁻⁹ while held-out
residual MSE was 1.16448. Even at alpha 1000, the respective values were 0.08491
and 1.16377. These are generated-data examples, not the selected real penalties.
The real training/held-out residual ratio still needs measurement.

**Selection cannot retain ridge alone.** The frozen grid contains only positive
training budgets. It selects the best residual head among those candidates and
always adds its correction. The no-skeleton arm is another fitted head; it is
not a zero-correction fallback. All candidates can lose to ridge and one will
still be selected. This explains how a correctly executed nested procedure can
deliver consistently harmful corrections without a runtime error.

See [residual construction and selection](../../../src/gavd6_sjepa/research_directions/future_innovation/fi_nested_training.py).
The saved fit summaries already contain the information needed to compare the
chosen inner head loss with ridge's inner loss; no new fit is needed for that
comparison. Allowing zero correction in a future protocol is a useful safeguard,
but does not guarantee outer-test improvement.

## 3. Confirm or reject the explanation on gate-v2

Run the following CPU-only inspection from a checkout containing the new script
on HAIC, or against a copied run. It never refits a model, runs the teacher,
updates provenance, changes a report, or produces a new gate decision.

```bash
.venv/bin/python scripts/research_directions/future_innovation/diagnose_future_innovation_fits.py \
  --run-root /hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/outputs/future-innovation/gate-v2 \
  --output work/artifacts/gate-v2-fit-diagnosis.json
```

Use `--fold 0` for a partial diagnostic. Output must be a new file outside the
run. To inspect locally, copy `config/`, `manifests/`, `teacher-cache/`, `models/`,
`predictions/`, `qc/` and `reports/`. The script resolves cached arrays within the
copied root and verifies their hashes. Only use trusted experiment artifacts:
loading the saved ridge model uses joblib/pickle.

It checks the relevant configuration/cache/fold receipts and any report seal,
reconstructs saved predictions, and records these diagnostics for each fit:

| Measurement | What it resolves |
| --- | --- |
| Correction energy and twice its alignment with baseline error | Whether the head adds noise or predicts useful residual structure |
| Baseline-input null component, its removal, and training invariance | Whether the reproduced mechanism explains real held-out degradation |
| Skeleton-plus-bias contribution | Whether most correction magnitude comes from the shared RGB/nuisance branch |
| Training versus held-out ridge residual MSE | Whether the residual training target is severely suppressed |
| Best selected inner loss versus zero-correction inner loss | Whether selection knowingly chose among uniformly harmful candidates |
| Feature rank, scaled magnitudes, constant training columns | Whether conditioning or previously absent categories amplify the error |

For baseline residual `e` and correction `c`, the exact diagnostic identity is
`MSE(e-c) - MSE(e) = E[c²] - 2 E[e*c]`, with source weights and valid features.
Large correction magnitude without enough alignment makes the model worse.
Null removal is a post hoc intervention, not a newly validated model or a
replacement for the sealed result. Reported diagnostic MSE is within-fold;
it must not be presented as the experiment's pooled featurewise R².

If removal does not recover most of the lost performance, lower this mechanism's
priority and investigate the observed alternative: feature scaling spikes,
training/test residual mismatch, skeleton contribution, or failed prediction
reconstruction. Do not force the real data to match the synthetic explanation.

The tests reconstruct all 12 heads in a synthetic direct-v2 fold, repeat the
inspection after relocating the run, verify every input file's hash and mtime
is unchanged, and reject a modified checkpoint. They also independently check
the null-space intervention. This validates the tool, not the scientific result.

## 4. Proceed in four bounded steps

1. **Finish the saved-model diagnosis.** Preserve `gate-v2` and its STOP. The
   confirmed software mechanism warrants investigation; the absent real artifacts
   currently prevent a definitive attribution of the −0.243 R² loss.

2. **Calibrate a separately versioned development experiment.** First isolate
   zero output initialization using the same cached features and source splits.
   Then compare residual training strategies if measured residual suppression
   warrants it. Include a literal zero-correction candidate, selected using inner
   sources only. Report all prespecified interventions; avoid changing several
   components and attributing the outcome to one. The 50 inspected clips remain
   development data, regardless of their original outer-fold labels.

   Add a simple reference predictor that jointly ridge-regresses the target on
   baseline inputs and fixed skeleton summaries. A skeleton-block weight of zero
   can recover the baseline candidate. This tests whether useful low-dimensional
   skeleton structure exists without the randomly initialized wide correction
   branch. Match representations, preprocessing and selection across controls.

   If using cross-fitted residual targets, generate them only within the current
   training-source partition. Convert each subfit prediction back to original
   teacher units before subtracting it, then apply a common training-only target
   scale. Otherwise different subfold standardizations create incompatible
   target vectors. Cross-fitting is a candidate remedy for suppressed residuals,
   not a promise of improvement.

   Before interpreting real data, require a zero-residual control, a noisy null
   control, a known skeleton-signal control with realistic competing inputs, and
   a known temporal-order signal whose benefit disappears under the designated
   shuffle. The existing two convolution layers have a five-frame receptive field
   before global averaging, so block-shuffle sensitivity should be demonstrated.

3. **Re-establish predictive evidence before expanding the study.** Freeze the
   revised implementation, primary target/horizon, controls and decision rules
   before new evaluation. Keep source-balanced splits, matched no-skeleton and
   mismatch controls, seed results and uncertainty. Reserve previously unexamined
   source videos for confirmation; do not repeatedly tune on them. Choose sample
   size from development source-level variability and the effect of interest,
   rather than treating more windows as more independent sources. A calibrated
   predictor that still shows no skeleton benefit is a reason to stop or revise
   the target hypothesis, not to increase student capacity.

4. **Only then test representation benefit and distillation.** Compare raw
   skeletons, frozen pretrained S-JEPA and a frozen randomly initialized S-JEPA
   with matched heads. Test whether the proposed innovation target can actually
   be predicted by a model receiving skeletons alone. A teacher correction that
   uses RGB/nuisance inputs is not automatically an appropriate skeleton-only
   target. Measure the exported skeleton-predictable component explicitly, with
   source-held-out prediction and retrieval controls, before training a student.
   Evaluate distillation against matched students without innovation distillation
   and the raw-skeleton reference. Adapter training remains conditional on the
   frozen-representation result, as in the proposal.

The current head is additive in skeleton and baseline inputs. Future conditional
heads could also exploit interactions that are unavailable to a skeleton-only
student. More generally, conditional predictive improvement and transferable
skeleton-only information are distinct claims; both need evidence.

## Interpretation limits that persist after a head repair

The target is a person-region token at frames 38–39 contextualized by all 64
frames. It is not an isolated observation of gait eight frames into the future.
The direct protocol checks causal prefix inputs but does not establish target
person/motion selectivity. Source videos are not verified participant identities.
The no-skeleton control retains validity, and the baseline contains aggregate
confidence and missingness, so the primary contrast concerns additional
coordinate/confidence history under that conditioning.

These limits do not explain the numerical head failure by themselves. They
matter when interpreting any repaired positive result. A useful predictive
comparison would still need to show that the learned innovation supports the
intended skeleton representation and retrieval task.

The [original distillation proposal](../../../notes/world-model-extensions/proposals-03/02-future-innovation-distillation.md)
remains design context. The [frozen direct-v2 protocol](direct-gate-protocol.md)
continues to define the completed run. This investigation does not amend its
thresholds or grant advancement.
