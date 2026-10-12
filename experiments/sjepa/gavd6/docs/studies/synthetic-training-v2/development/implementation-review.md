# Independent implementation review — 18 September 2026

Reviewer context: a separate Codex worker inspected actual new config, contracts, data, preparation, extraction, runtime, models, training, protocol and an initial workflow integration snapshot. This worker also performed the independent retained-pilot audit and authored audit.py/evaluation.py; those two modules are **not independently reviewed by their own author**. The coordinator must separately inspect them. No real GPU fit or new scientific outcome exists in this review.

The strongest competing explanation for any later benefit is ordinary paired coordinate denoising plus a learned readout. Latent loss reduction, noncollapsed embeddings, smoother trajectories, or passing fixtures cannot distinguish that explanation. The proposed study includes the right comparators, but their empirical contrasts remain pending.

## Findings and checks

| ID | Priority | Direct evidence and consequence | Smallest adequate correction / current disposition |
|---|---|---|---|
| R1 | High | The first contracts/data version retained NaN confidence at unobserved joints, while models.validate_inputs required finite confidence everywhere. Real missing-detection tracks and the provided obstruction fixture therefore failed before fitting. | Coordinator added a documented input-only numeric fill preserving observation flags and original artifact scores. Independent mutation test now passes; target changes cannot change normalized inputs. |
| R2 | High | Initial preparation admitted reserved validation audit rows and omitted their reservation flag from produced records. An existing protected recording could become development evidence despite the final bundle validator's reservation guard. | Coordinator excludes reserved rows before loading motion arrays and retains protection metadata. An independently mocked authoritative registry/audit test now verifies exclusion. The broader join to actual reservation/exposure manifests still needs integration inspection. |
| R3 | High | validate_records indexed split consistency by canonical_person_id but did not bind original person_id to that canonical identity. A copied original person with a new canonical label, window and motion passed in a second split. Reproduced by a two-row adversarial fixture. | Coordinator added consistent canonical mapping and original split per original person ID, alongside canonical component/motion checks. Independent regression now passes. |
| R4 | High | RunConfig.require_gpu_scope accepted projected_gpu_hours=NaN and measured_gpu_hours either NaN or −1. Comparisons against a finite authorized cap then failed open or reduced recorded spending. Reproduced without any GPU or scheduler action. | Coordinator now validates finite/nonnegative cost fields, checks authorized ledger totals, and guards preparation. Independent NaN/negative regression now passes. Runtime aggregate cap enforcement remains an integration check. |
| R5 | Medium | Initial prepare_source extracted every configured estimator on every admitted split, with no route to reserve an extractor for development only. TrackBundle.save itself did not receive held_extractor. | Enforce excluded estimator family before training extraction; validate configured held identity on every fitting bundle. Pending integration inspection. |
| R6 | Medium | Contract grouping uses canonical_person_id but initial evaluation data records retained original person_id. Using originals for paired uncertainty could count two known aliases as independent people. | Coordinator's workflow now maps evaluation person_id to canonical_person_id and preserves source_person_id. This is a required integration contract; the generic evaluation module expects already audited group identities. |
| R7 | Medium | Early masking sampled all fixed-grid slots while protocol stated 50% of observed eligible input tokens. Coordinate and latent losses also needed matched query/support weighting for partial/missing patches. | Model worker revised masking to observed tokens, adds separate missing-input queries, and matches coordinate patch/window weighting. Reviewer independently ran all 11 current model/training tests successfully, including matched missing queries, exact resume and a measured timer stop. |
| R8 | High | Initial workflow _fit wrote the same environment-gpu.json during direct and JEPA stages. The first receipt owned that file; another Slurm job ID in the second stage would alter its bytes and invalidate earlier receipts. | Use immutable stage/fit-specific runtime records. Sent to coordinator; final source-path inspection pending. |
| R9 | High | Initial workflow checked one projected budget before all seed/arm fits and did not reconcile elapsed GPU time or cap the cumulative stage/run during execution. A successful projection check is not runtime enforcement. | Account for prior preparation, attempts/retries, fit and inference costs, then stop before the remaining authorized scope is exhausted. Source preparation and final workflow need the same ledger contract. Pending integration recheck. |
| R10 | Medium | Initial _evaluate hardcoded complete=False, so every future source Gate B was insufficient regardless of supplied measurements. RunConfig already referenced a decision specification, but this snapshot did not consume it. | Implement supported explicit development adjudication or clearly mark that empirical adjudication is unimplemented; never imply executable gates solely from a placeholder. Pending integration recheck. |
| R11 | Medium | Low-level fit_arm supported exact checkpoint resume, but the first workflow never passed resume_from. Retrying an interrupted stage encountered its retained checkpoint and raised FileExistsError. | Provide a concrete explicit resume selection bound to identity, while retaining completed-stage reuse. Pending integration recheck. |
| R12 | High | code_identity omitted shared renderer/estimator/body-model-producing modules, and preparation's nominal cache identity omitted actual checkpoint/audit/render/body asset hashes. Saved adjacent metadata and NPZ hashes did not bind those inputs into the requested identity. The shape hash used metadata without actual shape coefficients. | Include producing code and actual supplied asset content in preparation identity, with schema/preprocessing, and hash actual shape/geometry. Pending integration recheck. |

The five independent tests in test_adversarial_contracts.py initially produced three passing tests and two failing tests (four failing subcases): R1 and R2 corrections passed; R3/R4 reproduced; confirmation access was denied before any NumPy array load. These tests define required behavior, not expectations that defects remain present. Final dispositions must record rerun status.

After coordinator corrections, all **five independent adversarial tests pass**. No source data or GPU access was used. This verifies the identified contracts, not the prospective research effect.

## Observed strengths and remaining limits

The target and model-input APIs are separate and allowlisted. Missing joints retain output queries; residuals are added only to observed finite inputs, with absolute predictions for missing inputs. The teacher runs without gradients and ordinary/paired JEPA share centered loss and stability code. Initialized states are retained. Coordinate and latent arms use separate frozen-readout fitting. Actual observation/confidence channels are common to the static comparison.

The static arm uses each frame's coordinates **after common whole-window input normalization**. The origin/scale therefore carry low-dimensional full-window coordinate information; this is a preprocessing-matched control, not an absolute no-history information boundary. Preserve that distinction in captions.

Resume signatures include array content, configuration, caller identity, model/training/objective code and runtime; checkpoints retain optimizer and RNG state. Existing worker tests cover deterministic resumed fits and changed-data rejection. Independent review has inspected these paths, but does not relabel worker-authored tests as its own experiments.

The HAIC runtime guard checks exactly Torch 2.6.0+cu124, Torchvision 0.21.0+cu124, MMCV 2.1.0, a Slurm allocation, CUDA availability and actual CUDA operators. This code is appropriate preparation, not evidence that this session ran on HAIC. This reviewer did not connect to HAIC or execute CUDA operators.

Synthetic proxy landmarks, supplied foreground boxes, and confidence-source limitations are accurately disclosed in the protocol. Real temporal preservation still requires independent temporally dense annotations. Existing reserved-source files are not recovered by a new manifest path or random seed. Source preparation must bind actual reservation/exposure content and checkpoint/render/preprocessing identities before a source result can be certified reproducible.

## Claim verdicts before empirical execution

| Claim | Verdict |
|---|---|
| Retained pilot means and oracle arithmetic reproduce | Supported at the saved aggregate level; raw inference/person-bootstrap replay unavailable. |
| The new implementation restores missing inputs without target-mask inference access | Supported by inspected design and fixture checks after integration corrections; source behavior pending. |
| Paired latent prediction beats coordinate training | Insufficient evidence: no new source fit. |
| Synthetic image adaptation exceeds augmented-real replay | Insufficient evidence: independent Gate A not run. |
| Real gait timing or clinical measurements are preserved | Insufficient evidence: independent real temporal references absent. |
| Personalized teaching has useful opportunity on the old panel | Not supported: extra retrospective headroom is about 0.0406%; no achieved personalization gain. |
| Fixture execution authorizes confirmation or GPU expansion | Rejected; fixtures are software evidence only. |

Readability review: retain “synthetic proxy,” “recording” where identity is unknown, “privileged clean-target supervision,” and “offline restoration.” Do not shorten these into “ground truth,” “independent people,” “self-supervised gait learning,” or “forecasting” in result captions. None of the new source effects is significant or measured yet.
