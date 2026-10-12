# Independent review of the paper-style proposal figures

Reviewed 21 September 2026 by the data-plan agent, which did not generate these figures. The reviewer authored the separate data specification, so this review is independent of figure construction rather than independent of the entire proposal. Figures 13–16 were each opened as a native PNG and as a 900-pixel-wide PNG: eight rendered assets inspected in total. Their SVG sources and both preview versions are identified by hash below. Figures 01–12 are outside this review's scope.

## Decision

The four figures pass the visual checks for clipping, text overlap, crossed connectors and legibility at both inspected sizes. They present one main idea each, retain useful whitespace and avoid relying on color alone for the masking states. The proposal status and invented arithmetic are generally explicit. Three scientific-communication revisions should be made before final acceptance; none requires redesigning the layout.

| Figure | Visual assessment at native and 900 px | Scientific and communication assessment | Required revision |
| --- | --- | --- | --- |
| 13, data to claims | Three rows and three columns are easy to follow. Solid/dashed connectors have clear spacing and a readable legend. No text escapes its box. | Counts and GAVD-overlap status agree with the data page. Local annotation and clinical access are correctly pending. However, the heading “Reference supervision” and footer “Reference labels supervise training and evaluation” blur the separation between fitting references and held-out clinical scoring. The solid-arrow legend can also suggest the newly proposed intervention dataset already exists. | Use “Reference evidence”; explicitly separate training-only supervision from held-out scoring. Describe the solid arrows as the existing pairing pipeline and state that new intervention data remain to be prepared. |
| 14, paired JEPA | The two phases are distinct; the teacher weight-update connector avoids the data paths. Native and reduced previews remain readable. | Reference features supervise training without entering the student observation path; the teacher has no loss gradient; frozen-encoder readout and deployment are distinguished. “EMA,” queries and tokens are explained within the figure. | No required change. Keep the proposed status and the training-only reference statement when embedding it. |
| 15, graph-time masking | Cells, joint rows and time blocks align cleanly. Dots and crosses make hidden versus missing states distinguishable beyond color. The legend and coverage note remain readable at 900 px. | Both draws preserve the same naturally missing token and vary side and interval. The labels correctly distinguish illustrative masks from results. The stick graph places anatomical left on the left of the page without stating a viewing convention; a reader could interpret it as a frontal pose and reverse the anatomical side. | Label it as a schematic joint graph rather than a camera view, or add explicit anatomical L/R labels to the highlighted graph. Preserve the joint labels in the two token grids. |
| 16, response and observation error | Both panels have clear roles and ample space around the arithmetic. No connector-label overlap occurs. All units and caveats are readable at 900 px. | The arithmetic is correct: +2° − (+4°) = −2° for response distortion, and +1° − 0° = +1° for the observation contrast. The example is explicitly illustrative, camera-dependent and nonclinical. The sentence “The starting position error alone…” is inaccurate in context because the values shown are errors in a bilateral angle-excursion summary, not joint position errors. | Replace with “Error in motion A alone cannot describe this loss of movement response” or an equally clear measurement-level statement. |

## Source and interpretation checks

Figure 13 agrees with the committed data roles: existing synthetic-development infrastructure, the 91-clip/41-source local development collection, and conditional independently referenced clinical data. The data specification also records that raw AMASS source files are not present in this checkout. Availability of a pairing pipeline must therefore not be read as a guarantee that every input for the proposed new experiment has already been imported locally.

Figure 16 uses the proposal's first engineering endpoint: right-minus-left projected knee excursion, where excursion is the 95th minus 5th percentile of the two-dimensional hip–knee–ankle angle over a fixed reviewed interval. Its lower panel holds motion and camera fixed while adding occlusion, so the equal 6° reference values are appropriate. It also warns that a camera change can legitimately change a projected reference. The one-degree observation contrast moves the estimate closer to the reference in this invented example; the text correctly says that observation changes error rather than claiming it necessarily worsens error.

The figures do not establish efficacy, statistical significance, clinical validity or completion of the proposed experiments. The review checks explanatory accuracy and rendered layout, not model implementation or physiological validity. Full method details, including reference-valid eligibility, sparse-input behavior, coverage and matched controls, remain in the linked protocol text rather than being squeezed into the diagrams.

## Reviewed asset hashes

The following hashes identify the first reviewed render. Any revised assets require a second visual check and a new hash record; they should not be described as passing this review unchanged.

| Asset | SHA-256 |
| --- | --- |
| `images/13-data-to-claims.svg` | `5063be430b93f77399e4547a5535507608c0dfba1563b1a9bc8f6b4015d36bf3` |
| `images/previews/13-data-to-claims.png` | `f1af4f4d7d76ba6b927ac69a30050d6c2b503d9a03dc3c013ef18b37f17bb640` |
| `images/previews/13-data-to-claims-900.png` | `b329e37ebc0fad95252c72a29eee6ec326367ec772d1e0d732c8e4d30a087eb7` |
| `images/14-paired-jepa-method.svg` | `e6bf4f020533460e7cbbc6dc3c1689109128c22b426c620179866f386abc277f` |
| `images/previews/14-paired-jepa-method.png` | `3f8583fd4147e585cda5c9a3a65a9812cd914e5f80199e277cd8969d265a2b71` |
| `images/previews/14-paired-jepa-method-900.png` | `d4f2c30a284c9493ab9e797a1be5367ac97e7089f0fdfceeb68ea43fc7bc534c` |
| `images/15-graph-time-mask.svg` | `cde494d3af06688ef5bf0ac330f146840d35f56d4182cfeaca35bfddf0708832` |
| `images/previews/15-graph-time-mask.png` | `d3fb98115f139def4ae99b7f3382f7900f39926b5d7c82338b92b3db7dab56b5` |
| `images/previews/15-graph-time-mask-900.png` | `e54c314730c7eb31b19e361c3cdbd8c89d0e4106b7e20d5f85383555f99f7506` |
| `images/16-response-estimand.svg` | `d526a6985c02470f8f00d858ceb29abf843a76f72621795337eaf304232c1105` |
| `images/previews/16-response-estimand.png` | `b648395652cd47aa155d3655829d206e4713a330ec7e92ebe7eb83a41dbda1fe` |
| `images/previews/16-response-estimand-900.png` | `df5ac96245298d20dec964aa79f5b8a7d68f5266dd97e1343511b3ff15ff0786` |


## Revision check and final acceptance

The reviewer reopened the revised native and 900-pixel previews of Figures 13, 15 and 16: six additional rendered inspections. Figure 14's SVG and both PNG hashes remain unchanged from the first inspection. All requested corrections are resolved. Figure 13 now distinguishes reference evidence, training supervision and held-out scoring, and names new intervention pairs as pending. Figure 15 identifies the drawing as a schematic body graph with anatomical sides and no implied camera view. Figure 16 uses “Baseline measurement error,” matching the quantity in the example.

**Accepted for inclusion in the proposal.** All four final figures are readable at the inspected sizes, with no observed clipping, excessive text overlap or ambiguous connector crossings. The longer Figure 13 footer and arrow legend fit within their allocated space at 900 px. The figures continue to state proposed or illustrative status and do not turn conditional reference data into completed clinical evidence. This acceptance applies to the final hashes below.

| Final asset | SHA-256 |
| --- | --- |
| `images/13-data-to-claims.svg` | `2e6966cd5ed8a476e1f1bbf3a3890760208172c1259d05ee1126cf98fe6ef9d2` |
| `images/previews/13-data-to-claims.png` | `bb79e2058701f029b0532e39908069c119e1217004ab3740a07466df9eec0111` |
| `images/previews/13-data-to-claims-900.png` | `19ac70dea204ade898934afd6254158bb237efb759f7e1e17e0be4379879f232` |
| `images/14-paired-jepa-method.svg` | `e6bf4f020533460e7cbbc6dc3c1689109128c22b426c620179866f386abc277f` |
| `images/previews/14-paired-jepa-method.png` | `3f8583fd4147e585cda5c9a3a65a9812cd914e5f80199e277cd8969d265a2b71` |
| `images/previews/14-paired-jepa-method-900.png` | `d4f2c30a284c9493ab9e797a1be5367ac97e7089f0fdfceeb68ea43fc7bc534c` |
| `images/15-graph-time-mask.svg` | `cec0c10caa9318e3ae17a19ad62fc89526b689aa72fc75e3d40c65c968d78035` |
| `images/previews/15-graph-time-mask.png` | `ed786cb0d869a87183609aa4bf0243c1f51e33e1fb6482b364b51faaa6c98216` |
| `images/previews/15-graph-time-mask-900.png` | `8d7221fc4b1987e2b74d8c33eb24329c5fbf464b956c4bc461dd6bd340b34338` |
| `images/16-response-estimand.svg` | `cdb651e2b396c31241af6cfdfa277644f846b1741d71eb09b65870bf3919ffe1` |
| `images/previews/16-response-estimand.png` | `226ffa34250540879ea2bb9b25a8cf499d0ad70f4989394e56922571a9f0e86f` |
| `images/previews/16-response-estimand-900.png` | `6affa7b66ce45751b136f410aeb3fa74e488171ccdeefef831ebb8483bce8b17` |
