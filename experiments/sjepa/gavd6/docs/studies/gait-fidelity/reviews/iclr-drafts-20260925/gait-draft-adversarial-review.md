# Independent adversarial evidence review

Reviewed 2026-09-25: new scientific and overview Markdown drafts; outputs/iclr/configs, ledgers, CSVs, comparisons, diagnostics; current matched objective/model/measurement implementations; rendered PNG previews for primary contrasts, response benchmarks, readout repair and actual response example.

## Required corrections

1. **Development raw-motion count 68 is unsupported and apparently wrong.** Core prepare receipt (`walking-core/ledger.json:790`) records 751 unique raw-motion hashes over both train/development. Train receipts (`ledger.json:1234`) record 692 available hashes. Data validation counts hashes and forbids their occurrence across splits (`synthetic_training_v2/contracts.py:89–93`, reused by `gait_fidelity/data.py:60–76`). Therefore development has 59 by subtraction, if the prepare receipt is used; alternatively omit this cell because individual development raw-motion IDs are absent from the transfer. Do not claim this count comes from coverage exports.
2. **Replace 'Ordinary JEPA' with 'core reference-paired JEPA'.** `ordinary_jepa` is a distinct implemented model arm not used in this experiment. Likewise change 'Ordinary paired JEPA' in any new caption to avoid accidental implementation misidentification.
3. **Clarify five families versus three follow-up variants.** The overview's eight is numerically true: all five core matched objective comparisons and all three follow-up representation variants worsen waveform error with original scalar change supervision. Calling them eight distinct model families overstates architecture diversity; say 'all five core families and all three follow-up variants.'
4. **Inference readout must cover missing joints explicitly.** Residual addition applies at observed joints; missing inputs get an absolute coordinate prediction. The full methods paragraph currently only describes adding corrections; a short second clause completes it.
5. **Primary repair contrast is delta JEPA only.** Ensure figure C/title/caption identify delta; the endpoint dense-minus-low-scalar secondary is numerically different. The actual values and ViTPose scope are correct.

## Useful precision improvements

6. Give the response coupling-by-readout interaction if discussing ordering reversal: +1.08517082° [−0.65174394, 2.84754284], descriptive crossed person/seed 95% interval. This is in `jepa-response/evaluation/readout-control-comparisons.json`, `comparisons.coupling_by_readout_interaction.response_error`. Positive means the delta advantage is larger with change supervision. The interval does not establish a dependable interaction.
7. Say **follow-up JEPA gradients** were clipped on nearly every pretraining update: five runs 2000/2000, one1999/2000. Core receipts do not retain this statistic. Clipping is an optimization sensitivity concern, not proof that the run failed to learn.
8. Full JEPA error equation should define the centering and temperature scaling sufficiently to avoid turning e into unspecified representation distance: e=H(p/τ_s−stopgrad((t−c)/τ_t)), with H subtracting the per-token feature mean. The teacher receives projected clean reference poses. This is not plain coordinate-space differencing or a future-prediction world model.
9. The zero-response comparison is correctly bounded to pooled failure-inclusive error. Keep the clear-observation result as a descriptive qualification: direct3.90° vs zero5.81°, occluded11.19°. Do not elevate raw direction accuracy66.1% alone to evidence against chance without defining the class balance.
10. If adding human-facing reproducibility detail, the transfer has69 verified selected files; three optional `provenance/gait-fidelity-release.json` files were missing upstream. Hash verification validates this packet, not complete checkpoint or raw-data reproducibility. Existing wording is appropriately cautious.

## Numeric and scientific checks passed

- All three primary estimates, signs and displayed intervals agree with saved comparison files.
- Repair means are correctly extracted for ViTPose, rather than incorrectly using pooled means.
- Reported93% fraction is3.697397/3.976253 and is explicitly descriptive, not causal.
- Failure contribution and success contribution sum to the total;74% delta advantage from failure term is valid (0.276676/0.373054). Captions correctly avoid calling success contribution a conditional success mean.
- All16 neural response variants in the pooled response export exceed zero-response5.81083°.
- Actual response example's provenance, aggregate nature and lack of raw sample are explicit. Lexicographic selection avoids outcome-driven cherry-picking. It cannot fulfill a raw observation-to-reconstruction example; draft properly identifies this gap.
- Diagnostic interpretation correctly separates the clean-reference teacher from deployed encoder. No general impossibility claim is made from one failed ridge probe.
- Explanations distinguish nominal body-model edit from measured projected response, and distinguish development from confirmation and clinical validity.
- The negative/inconclusive central claim is proportionate: objective and failure behavior affect apparent representation utility; a successful JEPA superiority claim is not supported.
- Inspected four rendered major figure previews: values match source output; axis units, zero reference, error-bar definitions and populations are appropriate. No visual overlap, clipping or misleading scales found at supplied preview size. Publication final-print size still needs document-level inspection.

## Remaining evidence gaps rather than writing defects

No raw frames, joint arrays or per-example restored trajectories in compact export; no protected-person confirmation; no GAVD/real-video synchronized-reference result; no clinical meaningful-change margin; limited14-person reused development cohort,12fromone source collection; limited schedule/coefficients and only3fitted seeds; anatomical3Dvalidity not established; preprocessing source snapshots evolved after core freeze.
