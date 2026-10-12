# Review of the paper structure and interactive proposal

21 September 2026. Scope: reorganized study files, the paper-style README and HTML, the explicit data plan, four new explanatory figures and the interactive response calculation. New restoration experiments remain pending.

## Organization and editorial decisions

The root now contains the current `README.md` and its generated `proposal.html`. Data, method protocols, literature, source evidence, images, scripts, reviews and historical records have distinct homes and indexes. The complete previous synthesis is retained in [history](../records/history/proposal-before-organization-20260921.md), while the active proposal follows motivation, question, data, experimental design, method, evaluation and reproducibility. It includes no results or discussion section.

The data plan distinguishes committed development assets from conditional source-motion and clinical extensions. It records exactly what supplies observations, references and identities, how the data connect to the methods, and the admission gates. It also identifies the first engineering endpoint: signed projected knee excursion, with its image-plane interpretation and reference-support requirements. Clinical contact timing remains a separately gated measurement.

Four new diagrams explain data-to-claim relationships, paired JEPA, graph-time masking and response distortion. Existing diagrams remain available in the full visual index. The offline HTML is generated from the Markdown scientific source, with section navigation, figure enlargement and an illustrative response explorer added by the builder. It loads no remote runtime assets.

## Independent adversarial review and corrections

| Review | Finding and correction |
| --- | --- |
| [Scientific review](paper-science-20260921.md) | Added coordinate-pretrained/frozen-readout controls for the new loss. End-to-end direct training remains a separate practical comparison; it cannot alone isolate a representation-objective interaction. |
| Scientific review | Required common reference-supported physical timestamps across all four experimental cells, both limbs and methods. Unsupported full contrasts remain reported. |
| Scientific review | Made prediction failure affect the primary decision through a declared all-attempted score or joint coverage/error confidence rule. Reporting failures beside an incomplete mean is insufficient. |
| [Data and figure review](paper-figures-20260921.md) | Distinguished training supervision from held-out reference scoring and clarified that the existing synthetic pipeline still needs new intervention pairs. |
| Figure review | Labelled the anatomy drawing as a joint graph with no implied camera view; corrected “position error” to measurement error in the excursion example. |
| [Interaction review](paper-interaction-20260921.md) | Closed the mobile contents panel after selection, reserved space above anchor targets and allowed scrolling in a short desktop sidebar. |
| Interaction review | Disabled controls until JavaScript initializes, preserved the static explanatory case without JavaScript, displayed quarter-degree settings consistently and gave each figure link a distinct accessible name. |

The data author did not generate the four new figures and independently opened all eight native/900px previews. The scientific reviewer did not author the rewritten proposal or data plan. The figure author independently reviewed the coordinating author's interactive implementation. Reviewers returned for the revised artifacts and recorded final dispositions and hashes.

## Verification and remaining limits

The two figure builders parse SVGs, check label/connector geometry and generate native and 900px PNG previews. Original twelve diagrams remain unchanged; all four new diagrams received independent rendered-image review. Rebuilding the builders verifies that their new locations resolve correctly.

The interaction review exercised 158,565 combinations of true change, clean/degraded retained change and observation offsets, including zero and negative changes. It checked the identities `I = R_degraded − R_clean = N_B − N_A`, presets, output updates and revised mobile/no-JavaScript behavior through source and simulated-DOM checks. These checks validate the illustrative calculation, not a research model.

The [current validation receipt](../records/paper-validation-20260921.json) records local links and fragments, generated assets, relocated evidence preservation, reviewed hashes and code syntax. Historical JSON receipts retain their original bytes and original scope; the [organization map](../records/organization-20260921.json) explains their subsequent locations.

**Live browser rendering remains unverified.** The browser-control service reported no available browser. Consequently, actual viewport rendering, focus behavior, assistive technology and native video playback were not exercised here. The rendered scientific diagrams, local asset links and interactive arithmetic were checked independently; source/DOM review is not represented as a browser test.

The proposal still requires data/reference admission, implementation of the new measurement and paired-change loss, a new physical-time pose cache, independent real annotations and fresh confirmation groups. Documentation and audit completion do not imply completed model training or clinical validation.
