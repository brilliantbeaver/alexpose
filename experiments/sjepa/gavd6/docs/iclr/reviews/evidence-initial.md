# Independent evidence and statistical audit

Audit date: 25 September 2026. Reviewer role: evidence auditor and statistical reviewer, independent of manuscript drafting. This review reads the retained results, source measurement definitions, experiment receipts, existing critical analyses, and notebook documentation. It creates new analysis artifacts without changing the experiment evidence.

## Defensible laterality claim

The evidence supports a controlled evaluation of **side-sensitive projected movement restoration**. The main measurement preserves the right-minus-left ordering of knee excursions, while a separate geometric diagnostic checks whether predicted lower-limb joint pairs are closer to the named or swapped projected reference. These measurements expose different failure modes. Direct end-to-end coordinate training substantially improves the geometric diagnostic in this synthetic development panel. The tested frozen JEPA representations have much greater geometric assignment failure under global input renaming, and paired feature-difference supervision does not establish a response-error advantage over matched endpoint supervision.

This is a useful empirical evaluation claim. It does not establish anatomical side recovery without an external anchor, disease laterality, affected-limb classification, identifiability, or world-model forecasting. Laterality means retaining side-sensitive information in the declared image-plane measurements and stress tests.

The primary research question can be stated as: Does prediction of paired clean-reference feature differences improve the preservation of a signed movement response when noisy pose trajectories must also tolerate left-right input renaming? The current evidence gives an unresolved primary comparison and a strong descriptive counterexample to interpreting a small response gain as general laterality fidelity.

## Independent numerical verification

The new [audit script](../evidence/independent_evidence_audit.py) read all person-level CSVs and the response condition-person CSV, checked all 69 transferred-file hashes, and independently reconstructed 16 saved metric comparisons across the three stages to absolute tolerance 1e-10. These 16 comparisons include secondary endpoints; there are three declared primary comparisons. The [verification JSON](../evidence/independent-evidence-verification.json) records the checks and new exploratory contrasts.

Every method has the complete 14-person by three-seed panel. The core has 16 methods, of which ten are learned; the response follow-up has 16 learned methods, including ten imported from the core; the repair evaluates nine methods, including retained fits. Shared stages reuse the same people and predictions. They are not independent replications.

The independently generated [condition-person table](../evidence/laterality-condition-person.csv) has 9,408 rows. Its [mean table](../evidence/laterality-condition-means.csv) can supply a scientific figure. For endpoint metrics, including geometric assignment, the nonheld stratum contains four states (baseline, exact no-change copy, 5-degree edit, 10-degree edit) and the held stratum contains one (15-degree edit), requiring weights 4:1. For response metrics the corresponding weights are 2:1 because baseline and the no-change diagnostic are excluded. The recovered means match the published person table to 1e-10. Response direction rates are deliberately excluded from this reaggregation because their eligibility denominator depends on reference-change magnitude.

Naively averaging the historical core `by-condition-person.csv` is unsafe: it also separates motion-label groups and can upweight people represented in more categories. The new condition tables use the more explicit response export and preserve its person/seed units.

## Laterality measurements and results

For side s, let q_s be the 95th minus 5th percentile of the image-plane hip-knee-ankle angle. The signed scalar is A = q_R - q_L. The response is ΔA = A_b - A_a, and its error is the absolute difference between predicted and reference responses. The sign of ΔA is the direction of change in the right-minus-left measurement. It is not itself the identity of the manipulated, impaired, or clinically affected side. Equal bilateral changes can cancel in A, and common errors in endpoint A values can cancel in ΔA. The waveform and per-leg excursion outcomes therefore remain necessary.

The geometric assignment diagnostic is independently defined in `src/gavd6_sjepa/research_directions/gait_fidelity/evaluation.py:68`. At each timestamp and each of the hip, knee, and ankle pairs, it compares the sum of Euclidean errors under named and swapped assignment. Reference pairs enter the denominator only if both joints are valid and separated by at least 4 pixels. A swapped assignment that is better by more than 2 pixels is wrong; named and swapped errors within 2 pixels are ambiguous; nonfinite outputs are missing. Wrong, ambiguous, and missing cases all count as failures. The diagnostic does not use the angular reference-support denominator, and it is not a calibrated classifier with a demonstrated 50% chance level.

All values below are person-balanced means averaged over three fitted seeds; assignment is a percentage and angular errors are degrees.

| Procedure | Geometric assignment failure | Response error | Waveform error | Response direction accuracy |
|---|---:|---:|---:|---:|
| Unchanged estimated coordinates (core) | 49.91% | 12.688 | 18.571 | Not exported |
| Direct coordinate-only restoration | 20.95% | 7.544 | 12.073 | 66.09% |
| Plain JEPA, coordinate-only readout | 48.64% | 10.691 | 17.453 | 49.32% |
| Delta JEPA, scalar-change readout | 50.46% | 9.988 | 21.765 | 48.44% |
| Endpoint JEPA, scalar-change readout | 49.69% | 10.361 | 22.031 | 47.84% |

Direction is scored among changes with reference |ΔA| > 1 degree, with failed predictions counted as incorrect. These observed percentages must not be described as chance-adjusted accuracy.

| Naming condition | Direct coordinate-only | Delta JEPA, scalar | Endpoint JEPA, scalar |
|---|---:|---:|---:|
| Correct | 18.61% | 24.15% | 24.70% |
| Global swap | 23.29% | 83.40% | 81.12% |
| Temporary swap | 20.97% | 43.82% | 43.24% |

The global corruption swaps all left-right body-12 coordinate, confidence, and validity channels throughout the sequence; the temporary corruption swaps the middle third. Reference coordinates are unchanged by these naming corruptions. Physical mirroring is a separate three-dimensional transformation with corresponding anatomical index exchange and fresh projection. The camera remains fixed across each comparison. Similar mean errors on original and mirrored conditions do not establish per-example equivariance or exact sign reversal of the projected measurement.

New, post hoc descriptive comparisons use the same paired crossed person/seed bootstrap as the response study, 2,000 draws with random seed 731. Positive values below indicate lower failure for the candidate. These intervals are unadjusted for selection and multiplicity.

| Candidate versus comparator | Assignment improvement, percentage points | Crossed 95% interval | People favoring candidate after seed averaging |
|---|---:|---:|---:|
| Direct versus unchanged, all conditions | 28.95 | [28.15, 29.89] | 14/14 |
| Direct versus plain JEPA/base, all conditions | 27.69 | [26.27, 29.11] | 14/14 |
| Direct versus delta JEPA/scalar, all conditions | 29.50 | [27.98, 31.55] | 14/14 |
| Direct versus delta JEPA/scalar, global swap | 60.11 | [57.65, 63.39] | 14/14 |
| Delta versus endpoint JEPA/scalar, all conditions | -0.77 | [-1.53, -0.17] | 0/14 |

The large direct-versus-frozen difference is a practical comparison, because direct training updates the entire encoder for 4,000 coordinate-supervised steps, while JEPA receives 2,000 feature-pretraining steps and 2,000 frozen-readout steps. It does not isolate whether freezing, objective choice, representational capacity, optimization, or the residual decoder causes the difference. The last comparison is especially relevant to a laterality-centered narrative: delta has a slightly lower scalar response point estimate while its geometric assignment score is worse. It must remain an exploratory observation, with its small magnitude and reused development set disclosed.

The per-leg excursion errors also prevent a single-metric ranking. Direct/base has mean left/right excursion errors 17.944/17.187 degrees, compared with delta/scalar 15.697/15.293. Thus direct is not best on every exported follow-up endpoint even though it is best on every metric in the older nine-metric core person table. A manuscript saying simply “direct is best on all metrics” would be false once per-leg outcomes are included.

## Primary comparisons and statistical interpretation

| Stage and declared endpoint | Improvement estimate | Declared 95% interval |
|---|---:|---:|
| Core: paired JEPA versus direct, both scalar readouts; response | 0.6881 degrees | [-0.6402, 1.9767], crossed bootstrap |
| Response: delta versus endpoint JEPA, both scalar readouts; response | 0.3731 degrees | [-1.1103, 1.7603], crossed bootstrap |
| Repair: delta dense versus delta low scalar, ViTPose only; waveform | 0.2789 degrees | [-0.1949, 0.7526], person t interval |

All three primary intervals include zero. Neither superiority nor equivalence is established. Repair's primary inference averages the three fixed fitted seeds within each person and uses a paired Student t interval over 14 people; it is conditional on those fits. A crossed bootstrap is secondary. Putting all three comparisons in one figure requires clear interval labeling and a warning that endpoints and populations of observation conditions differ; it is not a meta-analysis.

The response primary's 0.3731-degree gain decomposes into 0.2767 degrees from the failure contribution and 0.0964 degrees from the successful-output contribution. The failure component explains 74.16% of the point estimate. Delta's total 9.9880 degrees is 6.2096 successful-output contribution plus 3.7784 failure contribution; endpoint's 10.3611 is 6.3060 plus 4.0551. These additive terms retain the original denominator. A success-conditional mean uses a different denominator and cannot replace them.

The zero-response benchmark is 5.8108 degrees, lower than the pooled failure-inclusive response error of all 16 learned variants. It produces no trajectory, so waveform and assignment values are unavailable rather than zero. Direct/base does better than zero on RTMPose alone and clear observations, but those are descriptive overlapping strata. Neither a scalar point-estimate improvement nor approximately 50% direction accuracy establishes reliable pooled movement sensitivity.

The original paired scalar-plus-geometry readout raises waveform error in all eight available matched model/representation families (five core, three response). In repair, lowering the scalar coefficient from 1.0 to 0.1 already improves delta-JEPA ViTPose waveform error by 3.6974 degrees; dense supervision adds an unresolved 0.2789-degree primary gain. The favorable endpoint-JEPA dense result, 0.7182 [0.3875, 1.0489] degrees, is secondary and cannot replace the declared delta comparison. No noninferiority margin was set for response, and a response interval crossing zero does not establish preserved response accuracy.

## Population, training, and artifact limits

The executed training population contains 112 people, 692 available raw-motion hashes, 1,645 source windows, and 315,840 expanded endpoint rows. These counts come from ledger `sampling_coverage`, not the larger planned cohort. The development population has 14 people and 155 windows, with 12 BioMotionLab_NTroje people and two KIT people; each fitted model has 55,800 endpoint records and 33,480 nonzero response pairs. Repeated conditions, mirrors, seeds, or model variants do not add independent people. All three ledgers retain complete scheduler status for their recorded attempts (48 core, 29 response, 18 repair), with no failed attempts recorded there. This does not mean all angular measurements succeeded.

The planned counts, 113 training people and 15 development people, precede preparation exclusions and must not be reported as actual samples. The fourteen planned confirmation people have no completed independent confirmation result. The confirmation exposure audit is not ready. The analysis therefore concerns sequentially adapted development experiments, with no independent participant confirmation, clinical validation, or GAVD evaluation.

Each observation window has 128 frames at 25 Hz (about 5.1 seconds). The source uses a kinematic right-knee edit with nominal levels 0, 5, 10, and 15 degrees; 15 degrees and ViTPose extraction are held from model training. The nominal three-dimensional edit is not the actual projected response. The model consumes observed two-dimensional coordinates, confidence and observation flags; paired states and clean references supply privileged training supervision. This is offline restoration of an available window, not future prediction, action-conditioned dynamics, or planning.

The downloaded packet is sufficient to reproduce these summary analyses but omits raw pose arrays, checkpoint inference, full feature arrays, and complete learning curves. The three absent source release receipts are declared in the transfer inventory. Hash agreement verifies the copied evidence, not upstream rendering or training from scratch. CPU notebook fixtures are software teaching examples, while notebook 07 reads the same downloaded empirical packet; neither adds an independent experiment.

## Substantive objections to enforce in all seven reviews

| Objection | Severity | Evidence | Required correction | Residual limitation |
|---|---|---|---|---|
| Calling geometric failure anatomical or affected-side accuracy | Critical | Assignment implementation includes wrong, ambiguous, and missing projected pairs; no anatomical anchor | Define its denominator and three failure categories, use geometric wording | Real anatomical validation requires new references |
| Claiming delta JEPA improves the primary response or demonstrates laterality preservation | Critical | Response interval includes zero; pooled zero benchmark wins; assignment is worse than endpoint | Report unresolved contrast and complementary outcomes | More independent people and seeds are needed |
| Calling near-50% scores chance performance | Major | Assignment is not binary; response sign balance is not established | Report observed scores without chance language | Calibrated chance comparison would need a defined null and denominator audit |
| Promoting post hoc laterality strata into primary evidence | Major | These strata were selected after completed development results | Label exploratory, show large effects as descriptive, preserve registered primary | Confirmation must freeze the new hypothesis |
| Averaging held/nonheld strata equally | Major | Unequal counts of endpoint and response levels | Use 4:1 endpoint and 2:1 response weights | None after correct aggregation |
| Claiming direct wins every metric | Major | Per-leg excursion outcomes favor several scalar methods | Name the specific metrics and practical scope | Multi-objective ranking remains unresolved |
| Calling dense supervision a demonstrated repair mechanism | Major | Low scalar achieves most waveform recovery; dense primary unresolved | Separate coefficient effect from dense-vs-low comparison | Sparsity, temporal targets, and optimization are not independently isolated |
| Inferring deployment information from favorable teacher probes | Major | Teacher receives references; all 21 deployment-encoder probes exceed zero MSE | Name branch, input, training population, decoder limitations | A fixed poor probe cannot prove information absence |
| Treating the three stages as replication | Critical | Shared 14 people and imported predictions | Describe sequential development and copied controls | Independent confirmation requires new experiments |
| Treating scalar direction as changed-side identity | Major | ΔA is a signed change in right-minus-left excursions | Explain meaning and bilateral cancellation | Independent side-specific clinical targets are absent |

Revision can make the current scientific argument clear and accurate. It cannot establish the desired method advantage, population generalization, anatomical side recovery, clinical relevance, or a causal account of the optimization failure. Those limits should remain visible in the final version rather than be written away.
