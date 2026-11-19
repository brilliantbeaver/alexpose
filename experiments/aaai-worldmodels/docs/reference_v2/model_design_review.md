# Independent model and pre-fit design review

The evaluation specialist inspected the model and mask implementations, their tests, the model contract, the normal-development resource pilot, and the coordinator protocol before new outer-fold outcomes were fitted or inspected. The reviewer authored `v2_evaluation.py` and its evaluation contract, so this document is **not** independent acceptance of those evaluation artifacts. A separate data specialist reviews that module. The reviewer did not author the model or mask implementations or the coordinator's protocol.

## Disposition

The inspected model and mask implementations are accepted for the declared primary cohort and core coordinate-prediction experiment. No blocking causal-input, target-weighting, mask-budget, or inference-query defect was found. This is an implementation and design acceptance, not evidence of positive efficacy or neurological validity. The runner and its complete nested integration require a separate review before real evaluation.

The cohort currently has 91 verified singleton source-relation components. The runner must propagate `component_id` to the model's optional `group_id` if group-based sampling is requested. The final model contract restricts this implementation to singleton components, and the runner must reject a future multi-source component until an explicit component/source/sequence/window hierarchy is implemented and independently reviewed. This limitation does not change any fit in the current singleton cohort.

## Scientific and implementation checks

The predictor accepts exactly sixteen prefix samples and rejects a 24-sample input. Future coordinates and future support enter fitting through separate arguments, and the prediction API always produces the same eight-time/eight-distal-joint queries. Artificially hidden and naturally missing coordinates are erased before filling, feature scaling, and inference. Reference joints and latest anchors are protected by canonical masks. The motion-balanced policy has a uniform sampling floor, so an observation without an adjacent valid motion pair can still be masked.

All seven policy families use a common eligible-token definition. M1 and later policies match their exact rounded hidden count. Structured priorities can truncate the final block; their descriptions and audit do not falsely claim every realized mask is a complete tube. Random groups preserve the anatomical group's sizes and time extent, with realized partial geometry disclosed. Adaptive pattern information is explicitly acknowledged, with the separate residual-pattern diagnostic retained.

Current-posture and posture-plus-velocity modes use the declared distal-joint anchors. The latter computes robust slopes and median absolute deviations from actual recent decoder intervals, with separate support counts. These are projected articulation features rather than measured physical speed. The 24-feature compact posture reference is clearly identified as a new fit, rather than a relabeling of the older repeated-posture ridge.

The masked ridge's coordinate weights normalize within each leg, and its unpenalized intercept/scaling behavior agrees with an independent solver. Neural loss likewise gives equal weight to both legs regardless of target density. Source-balanced sampling separates the source, sequence, window, and mask-bank random-number streams. Parameter initialization, backbone, source exposure, steps, and future support are matched across the main masking comparisons. All model families reject abnormal training labels.

The protocol freezes posture-plus-velocity ridge as the primary anomaly candidate before outcomes. It separates that reduced-training calibration ensemble from the all-permitted-normal-source forecasting models. It retains persistence, historical temporal controls, the current-posture reference, low-motion anomaly comparison, pattern diagnostics, and complete mask reporting. No abnormal label chooses the model family, direction, duration, or calibration rotation.

## Independently executed evidence

The reviewer ran the model, masking, and evaluation test files together. The result was **36 passed in 3.48 seconds**. Twelve evaluation tests were written by this reviewer; the 24 model/mask tests were written by the model author. Test roles are disclosed rather than presenting all 36 as independently authored checks.

A separate reviewer-written numerical calculation generated 31 synthetic windows, seven random input features, asymmetric left/right target densities, and nonuniform training weights. It compared **all 128** masked ridge output-coordinate solutions with independently weighted scikit-learn Ridge fits. The maximum prediction discrepancy was **4.88e-15**. A separate NumPy calculation of per-leg observed error followed by equal-leg averaging matched the Torch masked loss within **4.44e-16**. These synthetic checks validate arithmetic; they are not empirical model-performance results.

The reviewer additionally loaded the new verified real cohort and selected one deterministic representative window per source. All **637** combinations of 91 source windows and seven mask policies passed support, protection, and matched-hidden-count checks using actual decoder timestamps. No new model was fitted for that structural check, and its results do not measure a masking benefit.

## Reviewed identities

| Artifact | SHA256 |
| --- | --- |
| `worldmodels/v2_models.py` | `07a3326f06705002429ddd66cbac444583917354603729cf5cdeffc49c711657` |
| `worldmodels/v2_masks.py` | `050b08ca75849eb48796f2e79afaf89c8040138fabf61c2b8ea711b3e32f8cd2` |
| `tests/test_v2_models.py` | `a3f4720941623afc08177ce302c2de3755d0875001a51ed47c6f6d5111fa653e` |
| `tests/test_v2_masks.py` | `4148a3d846bd42ecad002d2c8dc6217dabc45b3edc355a08c9ee2760d7bdec8a` |
| `docs/reference_v2/model_contract.md` | `3489df39850ecc6d99bf94fc9c49c70c1189553da3211f31b7652fbc8634b20e` |
| `docs/reference_v2/protocol.md` | `7b7560939585a6849ccf85c9e6b096d5327bbfe6e5c50ef5b314e70ddfe1f71f` |

A material edit to these artifacts reopens the relevant checks. The coordinator's final release record must identify the exact accepted implementation, complete runner, and protocol. No acceptance here authorizes changing the primary outcome after evaluation.

## Complete staged-runner integration review

The evaluation reviewer subsequently inspected the full `v2_runner.py`, including normal-only alpha/duration selection, all-T reference fits, masking fits, disjoint calibration rotations for both posture-plus-velocity and the matched posture reference, nuisance-distance fitting, source aggregation, alternate-partition refitting, and final endpoint construction. The runner rejects multi-source components until their training hierarchy is extended and reviewed. Its real entry point checks the accepted protocol, implementation hashes, and frozen configuration before fitting.

**M1: unequal inner-fold sizes initially changed source weighting.** Compact alpha, historical-leg alpha, and M0 duration selection originally averaged three validation-fold means. With different fold sizes this does not give equal weight to each normal source. Before any real outer fit, the model owner corrected all three paths to pool individual held-out source errors and added a regression test whose selected penalty differs under the incorrect weighting. Selection records now retain those source errors for independent replay.

**M2: accepted configuration needed an explicit guard.** The reviewer and coordinator required the real entry point to compare the normalized requested configuration with the configuration stored in the accepted freeze. This prevents a different grid, seed family, or mask policy from accidentally using the same accepted study identity. The owner added a failure test. These were pre-fit software corrections, not responses to efficacy outcomes.

A reviewer-owned complete synthetic execution used **73 windows from 25 generated sources**, with unequal windows/sequences per source and asymmetric target visibility. It completed in **27.27 seconds**. The reviewer verified **680** actual checkpoint training memberships and hashes, rebuilt **30** nuisance references and empirical calibration distributions from retained source rows, and replayed **60** calibration/outer prediction trajectories bit for bit using the corresponding saved checkpoints. Every calibration source remained outside the forecasting and nuisance construction that produced its reference. All **1,630** saved artifact hashes matched. Reconstructing saved source MSE from window trajectories differed by at most **9.89e-17**, and a CSV round trip changed a rank score by at most **2.78e-17**. The initial audit used an unnecessarily strict bitwise assertion for CSV floating-point values; it was corrected to a declared 1e-12 numerical tolerance while trajectory replay remained bitwise.

The synthetic source and check record are retained under `artifacts/executed-reference-v2/review/runner_synthetic.py` and `runner_synthetic.json`. Temporary generated model arrays were removed after the audit. These are software and resource checks, not real GAVD forecasting or anomaly evidence. A separate final reviewer independently evaluates the whole integrated study and the exact final hashes.

## Runner identities after the final guard review

- `worldmodels/v2_runner.py`: `cb8e57a3e32311b163d9938e941a31eac299c58db7b23eec4d59ce37d4a029fb`.
- `tests/test_v2_runner.py`: `3247214cb2f07eaf921c7018cb97c612156d0ceabceed60a830f23c4dd60a721`.

**M3: pin the accepted input cohort as well as its code.** The final real entry point compares the cohort identity and membership file hashes with the accepted freeze before loading observations or fitting. A new regression test changes the membership bytes and verifies rejection before the loader is called. This guard prevents an accidental different cohort from inheriting the same approved run configuration. It does not change forecast or score calculations. The final whole-study reviewer separately records complete acceptance.

The final runner implementation is accepted for the declared pre-fit release, subject to the coordinator and whole-study reviewer matching these hashes with the accepted freeze. Source-weight selection and freeze guards are resolved before any real outer efficacy fit.
