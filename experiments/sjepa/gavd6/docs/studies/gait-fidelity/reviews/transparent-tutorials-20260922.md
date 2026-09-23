# Review of the transparent Gait Fidelity tutorials

The revision covers all twelve notebooks in `notebooks/gait_fidelity`. It follows the worked-equation and executable-tensor style of the multiple-sclerosis tutorials while retaining the Gait Fidelity body-12 model, objectives and evaluation protocol. The numerical examples appear directly in the generated notebooks; `lesson_*.py` files organize their build-time text.

Three authors investigated and implemented separate areas: data and masking; model architecture and optimization; evaluation and uncertainty. Each area was then reviewed by an author of another area. The primary agent independently traced the implementation, added the five experimental-control tutorials and integrated the sequence. A fourth reviewer assessed the final flow without authoring any of its lessons.

## Findings and corrections

| Finding | Resolution |
| --- | --- |
| Earlier prose called a token one joint at one time. | Explain and construct four-frame joint tokens, including channel and flattening order. |
| Graph masking could be described as an arbitrary graph-growth algorithm. | Show the implemented anatomical region sampler, cyclic intervals, exact budget and possible truncation. Compare all five policies bit for bit with production. |
| A sparse demonstration row could leave fewer than two eligible input tokens. | Select a teaching row by declared metadata and input support, disclose skipped demonstration candidates, retain every study record and test empty/single-token inputs separately. |
| The small mask-control audit assumed nonempty hiding probabilities and interval distributions. | Preserve unsupported values as missing and verify parity with the production audit. |
| A target-only interpretation overstated the JEPA comparison. | Explain that the feature target, loss, predictor/projector, regularizer and moving-average teacher belong to the declared pretraining recipe. |
| Calling the static baseline temporally independent hid its auxiliary inputs. | State that it retains full-window confidence, availability, time and input normalization while excluding cross-frame coordinate trajectories. |
| Patch supervision could be mistaken for an all-frames-valid condition. | Derive the any-valid-reference-frame rule, then show its patch/window reductions. |
| A zero first-step encoder gradient could be mistaken for unchanged parameters. | Explain the zero-initialized output layer and distinguish loss gradients from AdamW weight decay. |
| `torch.manual_seed` inside a CPU-only RNG fork could change a notebook kernel's accelerator RNG. | Seed `torch.random.default_generator` directly for all six CPU scratch constructions, preserving their CPU values without seeding CUDA/MPS/XPU generators. |
| Pooled representation regularization could be interpreted as context-only pooling. | State that the mean includes all patch–joint output slots, including missing/query slots. |
| A supported-condition mean could be presented as an unconditional population result. | Show missing-reference exclusions, prediction-failure penalties, coverage and family/person weighting together. |
| Standalone demonstration plots could be mistaken for source evidence. | Label software fixtures and generated CPU model examples on their plots. |

The source-parity checks cover camera projection, input-only normalization, exact masks, channel packing, attention, coordinate decoding, coordinate/feature/regularization losses, teacher updates, readout freezing, movement losses, calibration, temporal filtering, donor assignments, minibatch endpoint exposure, evaluation contrasts and crossed bootstrap intervals. The explanations identify input-only normalization, loss-only reference validity and train-only control fitting as distinct leakage boundaries.

## Validation and limits

All twelve final notebooks passed their 94 code cells in real Jupyter kernels. The 50 study tests and six layout tests passed, and reconstruction verified 915 retained artifacts and 39,168 neural per-window metric rows. Model states for all 135 optimization phases, all 102 final prediction exports, seven evaluation CSV tables, calibration parameters and comparison intervals exactly matched the earlier fixture. The 161 scientific source and launcher files in the frozen identity were unchanged. The final independent reviewer confirmed resolution of the accelerator RNG finding and reported no outstanding findings.

The [validation receipt](../records/transparent-tutorials-20260922.json) records the final notebook hashes, execution counts, tests and comparison with the earlier CPU fixture. Scratch examples are isolated from the production random-number streams and retained checkpoints. The earlier fixture provides a numerical preservation check for model states, final prediction arrays and evaluation tables; run timestamps, paths and elapsed times are intentionally excluded from that equality comparison.

These checks do not certify source rendering quality, anatomical landmark agreement, HAIC dependencies or H100 runtime. The unchanged source study uses the inherited development population and still requires independent confirmation for stronger scientific claims. The tutorials explain these limits alongside the computed outputs.
