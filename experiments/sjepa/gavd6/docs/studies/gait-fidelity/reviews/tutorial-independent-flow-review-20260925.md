# Independent tutorial flow and algorithm review, 25 September 2026

Reviewer scope: the data/start/experiment revisions, architecture and response
revisions, and the new readout-repair tutorial. This reviewer did not author
those changes. Evaluation/probe additions authored by this reviewer are assessed
separately by the parent task's other reviewer.

## Findings and resolution

1. **The response result reader silently omitted the real comparison.**
   The canonical response `comparisons.json` is keyed directly by metric, rather
   than by `primary`. A `get('primary', {})` returned an empty table for the
   downloaded result. The architecture author corrected the reader to accept the
   direct mapping and require `response_error`. The author reran all response
   cells and reported the actual 0.3730535-degree estimate, interval
   [−1.110272, 1.760337], 14 people and three seeds. The corrected reader was
   independently inspected.

2. **The new held-estimator assertion initially skipped the latest source run.**
   The source configuration omits a top-level `held_extractor`; preparation
   defaults it to `vitpose`. The first tutorial version used an empty fallback,
   so the check did nothing. The data author corrected this to consult the
   bundle's `held_extractor_family`, then the configuration and production's
   `vitpose` default. Both extractor identity and family columns are checked.

3. **Fixture scope needed to distinguish the default from a selected matrix.**
   The overview's statement that fixtures retain the full matrix was too broad
   once the execution runner exposed `--experiment-set core`. The revised
   overview now explains that full is the fixture default, while selecting core
   exercises the latest source matrix and hierarchical sampler on generated
   data. Optional topology and re-pairing examples remain labeled as unexecuted
   scientific controls in the completed core study.

4. **The CPU repair example must not seed CUDA generators.**
   The initial G tutorial used `torch.manual_seed` within
   `fork_rng(devices=[])`. On a GPU host that call also seeds CUDA generators,
   although only CPU state is restored. The parent changed the two CPU model
   construction seeds to `torch.random.default_generator.manual_seed` and
   labeled the final assertion a CPU Torch RNG check. Both changes were
   independently inspected after implementation.

5. **Core and follow-up sessions use different artifact layouts.**
   Tutorials 00–06 and A–E use `Study.bundle_path`, which requires the core
   preparation receipt. Selecting the current response or repair session can
   therefore produce an irrelevant missing-preparation error. The parent added
   an early `configure` guard for recognized response and repair study kinds,
   explaining the parent-session requirement and pointing to F, G and 07. F and
   G are standalone mathematical lessons; 07 uses the compact packet. The guard
   was independently inspected. The evaluation tutorial now explicitly places
   its transition to 06 after the probe examples below it.

No scientific result, model source, checkpoint, data split or study configuration
was altered during this review. All findings above are resolved in the inspected
source. The parent records integrated execution separately.

## Algorithm trace

The data revision correctly distinguishes a condition record from a source
window, motion and person. Its condition-grid count includes baseline and exact
no-change duplicates. Metadata identity checks prohibit cross-split people,
motion hashes and source families. The explicit endpoint table reproduces
`training.paired_indices`, including baseline reuse, metadata ordering,
training-only selection and matched conditions. The naming permutation changes
coordinates, confidence and availability together, while references and physical
timestamps remain fixed. Its double-swap and production equality checks cover
correct, whole-window and temporary swaps.

The current hierarchical sampler selects a person, raw motion, source window and
pair uniformly at successive levels, with replacement. Its code, ordering and
probability factorization agree with `hierarchical_pair_groups` and
`draw_hierarchical_pairs`. The extended re-pairing control is correctly separated
from this sampler and from the later readout repair. It is not presented as a
completed core experiment.

The architecture exposes channel packing, time/joint identities, noncausal
attention and residual coordinate prediction. The new feature-predictor
calculation reproduces its Transformer → LayerNorm → linear path and input
gradient. References remain training-loss or privileged-teacher data. The
teacher has an EMA copy of the encoder architecture; it is not a deployment
input branch.

The response derivation preserves the distinction between any-valid reference
support for the base feature objective and all-valid, both-endpoint support for
the auxiliary. Unsupported auxiliaries have a differentiable zero without
zeroing the base objective. Its centered, temperature-scaled residual formula,
equal-pair reduction and coordinate residual conversion to common pixel units
match the source. The full forward example replays the same translated-view
random draws when checking cross-entropy plus VICReg and the auxiliary. It
compares every trainable parameter gradient, not just a scalar total.

The response calibration uses one shared JEPA coefficient determined by the
larger auxiliary gradient energy and 10% of the base initial RMS; it does not
claim that both auxiliaries individually reach 10%. The optimizer uses the
combined gradient before clipping and updating, followed by teacher EMA and the
center computed from pre-update teacher tokens. Reference access and the
training-person interpretation of probe folds remain explicit.

The readout repair preserves `[pair, endpoint, time, joint, xy]` ordering and
equal-pair angular reduction with unequal reference-frame counts. Its manual
percentiles, angle geometry, fixed reference support, scalar response, dense
paired angle response and short-segment terms follow the production definitions.
Dense differences run across movement states at matching times; they are not
temporal derivatives. The coordinate objective remains squared error in
input-derived normalized units, unlike the reference-box-normalized distance
reported by evaluation.

The repair coefficient matches the weighted low-scalar angular gradient RMS,
using squared energies over all readout parameters and fixed training batches.
It does not target coordinate RMS or update parameters during calibration. The
two scratch arms share initial weights and batches, then evolve separately.
Their optimizer states are separate; encoder and teacher weights remain frozen,
and there is no teacher EMA during readout fitting. Deployment receives only
estimated coordinates, scores, availability and time. The real primary scope is
correctly stated as delta-JEPA dense versus low-scalar ViTPose waveform error,
with reused development people and no protected confirmation result.

## Remaining validation boundary

The lesson authors ran the small numerical parity examples; this reviewer
inspected the calculations and their source counterparts. The parent performs
the ordered real-kernel notebook execution and records the generated notebook
hashes. Those CPU checks establish implementation fidelity on teaching arrays,
not independent reruns of HAIC training or new scientific confirmation.

## Final documentation and execution-claim review

The final pass inspected `ALGORITHM_GUIDE.md`, `README.md`, `ENVIRONMENT.md`,
`VISUAL_WALKTHROUGH.md`, the refresh record, notebook 00's matched-initialization
explanation and G's new AdamW-state checks. Three additional presentation/API
issues were reported and corrected by the parent:

- F's examples perform a scratch optimizer update, so README's statement that
  they "do not train models" was replaced by the accurate restriction on saved
  study checkpoints and source fits.
- The algorithm guide's opening reading order now agrees with README and G:
  00–06 followed by F/G, with 07 alongside for actual results.
- F's compact-packet reader now honors `GF_EVIDENCE_ROOT`, matching the documented
  optional evidence-root override used by G, 06 and 07.

The corrected files were read after these changes. Notebook 00 now accurately
states that same-seed output networks begin with matching weights and are fitted
separately. G compares each readout parameter's AdamW state, including its first
and second moments, against the source-loss update path after each scratch step.
The scientific scope, calibration targets, training-versus-probe identity split
and core-versus-repair estimator distinction remain consistent across the guides.

The refresh record explicitly reports the integrated fixture execution as
incomplete: two attempts stopped with disk-space errors. It separately identifies
the 64 targeted passing tests, successful standalone G calculation and successful
07 execution. It does not present any of those as a completed ordered execution
of 00–06 plus A–G. Historical validation receipts are explicitly historical.
This review ran no storage-heavy validation and found no remaining overstatement
of integrated completion in the inspected documentation.
