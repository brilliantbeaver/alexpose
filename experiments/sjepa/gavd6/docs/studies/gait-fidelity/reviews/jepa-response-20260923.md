# JEPA response follow-up: independent review

**23 September 2026 · implementation and CPU validation; HAIC execution pending**

[Protocol](../methods/jepa-response.md) · [HAIC guide](../../../../slurm/gait-fidelity/JEPA_RESPONSE.md)

Three agents implemented separate areas: objectives/calibration/training;
parent reuse and Slurm execution; and diagnostics/evaluation. The training
author subsequently reviewed the execution and evaluation code, while the
diagnostics author independently reviewed the training mathematics and tutorial.
The primary agent reviewed the integrated interfaces, ran the full study suite,
executed notebook F in a Jupyter kernel and inspected rendered figures.

## Scientific findings and corrections

- The paired and endpoint losses share tensors, temperatures, reference support
  and a coefficient. Their difference is the cross-endpoint residual inner
  product. This tests residual coupling; it does not assign physical units to
  latent magnitude or establish a unique benefit of correct pairing.
- All four reference frames must be valid at both queried endpoints. An
  unsupported auxiliary pair retains its original base loss. Reductions average
  tokens within pairs before averaging pairs.
- Coordinate residuals must use one common observation-derived pair scale.
  Direct subtraction of separately normalized residuals would mix units.
- Encoder-only coefficient calibration fails at the zero-initialized coordinate
  head. Calibration now uses all trainable parameters, the declared precision,
  32 fixed training batches and restored random-generator states.
- Historical cross-entropy and teacher-entropy logs had different reductions and
  center timing. New CE/entropy/KL diagnostics use identical probabilities and
  reductions before updates; historical differences are not treated as KL.
- Probe outcomes cannot admit or stop model fits. The implementation uses only
  the registered E/P/T ridge diagnostics, without adding a coefficient search,
  MLP probe or new representation arm.
- The tutorial now distinguishes the student from the privileged teacher:
  reference validity is withheld from the student but legitimately supplied to
  the teacher. Identical-reference teacher differences isolate normalization;
  student differences can also reflect the independently hidden context.

## Runtime and reproducibility findings

- The original phase key would merge the new JEPA variants. Explicit variant,
  formula, support and calibration identities now separate checkpoints and
  upstream dependencies. Existing core/full plan identities remain unchanged.
- A prepared parent dataset is a distinct dependency type. No rendering jobs
  are invented in the child ledger, and the legacy `source_bundle` interface is
  not reused. Hash checks cover prepared arrays, admission records, completed
  predictions and original code. Receipt contents must agree with ledger result
  pointers and registered phases.
- Child scientific settings inherit the verified parent, including its actual
  selected update schedule. Initialization refuses edited model, masking,
  sampling, optimizer, measurement or evaluation settings.
- Deadline checks at submission alone could allow late queued jobs to overrun.
  A worker supervisor now enforces the absolute cutoff independently of the
  coordinator. Process-group cleanup also stops descendants if the leader
  exits first. Ordinary core workers preserve their previous execution path.
  The final Slurm review changed signal-only termination to ordinary `scancel`
  for retained child jobs, which also cancels pending allocations.
- Failed allocations remain charged. Diagnostics retry with their original
  four-hour allocation from the six-hour recovery allowance. Runtime projection
  includes repeated verification of the complete profiling artifact tree.
- Cached evaluation now verifies upstream child predictions as well as its own
  receipt and the parent. Evaluation and numerical reconstruction share a lock;
  reconstruction writes to temporary storage. Parent feature exports use the
  recorded interpreter with `-I -B`, preventing bytecode writes to that checkout.

## Verification and remaining limits

The focused training checks include exact first-update parity with the original
CE/VICReg path, endpoint/mask exposure, teacher detachment, frozen readouts,
interrupted resume and invalid calibration rejection. The full study suite and
fresh core-to-child fixture exercise the broader workflow; the accompanying
[validation record](../records/jepa-response-validation-20260923.json) contains
counts and source hashes.

The vector workflow passed independent visual review. Plot review added
in-image software-fixture labels and explicit empty-panel coverage; conditional
response plots retain the actual reference range even when predictions fail.
Notebook F subsequently executed all eight code cells in a Jupyter kernel,
including the saved child tables and figures. A final independent read-through
confirmed that its equations and the HAIC commands match the implemented paths.

No blocking issue remained in these reviews. Local tests use CPU Torch 2.13;
the established HAIC environment uses Torch 2.6 with CUDA 12.4. Actual H100
operators, throughput, queue behavior and source-data results remain subject to
HAIC preflight and profiling. Neither the software fixture nor review approval
is evidence of a scientific JEPA advantage.
