# Citation check for instruction sections 2 and 3

Checked on November 15, 2026 against the primary records and the existing source notes. The canonical instruction file was not edited. No wrong article identifier, author attribution, or method title was found among the requested references. Their short method descriptions are supported.

## One substantive citation improvement

The cerebral-palsy row mentions contracture, growth, and musculoskeletal constraints. Its nearby [Steele, Rozumalski and Schwartz citation](https://pubmed.ncbi.nlm.nih.gov/26084733/) supports altered EMG synergy complexity and associations with impairment, but does not directly establish the growth/contracture part. Retain that distinction, but add a directly relevant primary source rather than making the EMG study carry the whole sentence.

A bounded addition is [Herskind and colleagues, *Muscle growth is reduced in 15-month-old children with cerebral palsy*](https://pubmed.ncbi.nlm.nih.gov/26510820/), DOI 10.1111/dmcn.12950. Its observational comparison supports altered muscle growth; the proposed connection to contracture is an interpretation, not a proven causal pathway. An appropriately cautious sentence is: “Motor-control limitations coexist with altered muscle growth and musculoskeletal constraints, which may contribute to contracture.” For the selective-control portion, the already cited clinical advice links [Chruscikowski and colleagues](https://pubmed.ncbi.nlm.nih.gov/27889619/), DOI 10.1016/j.gaitpost.2016.11.031.

## Requested method references verified

| Reference | Verified identity and scope |
|---|---|
| [FSGait](https://openaccess.thecvf.com/content/ACCV2024/html/Duan_FSGait_Fine_Grained_Self-Supervised_Gait_Abnormality_Detection_ACCV_2024_paper.html) | Bingzhi Duan, Xiaoyue Wan and Xu Zhao; ACCV 2024. Normal-gait training, separate reconstruction/prediction modules for posture/temporal anomalies, and sequence/frame/joint outputs are supported. The URL is correct. Direct opening returned HTTP 403, but the indexed primary CVF record and its PDF verified the attribution. |
| [SeeKer](https://arxiv.org/abs/2506.18368) | Anja Delić, Matej Grcić and Siniša Šegvić; *Sequential keypoint density estimator: an overlooked baseline of skeleton-based video anomaly detection*, ICCV 2025. Conditional keypoint distributions and detector-confidence score weights are correctly described. |
| [Beyond Marginals](https://arxiv.org/abs/2509.15033) | Padmaksha Roy, Almuatazbellah Boker and Lamine Mili. The title and separation of marginal, temporal, and inter-variable structure are correct. The primary record lists TMLR 2025. |
| [GenGait v2](https://arxiv.org/abs/2604.01997v2) | Elisa Motta and colleagues; *GenGait: A Transformer-Based Model for Human Gait Anomaly Detection and Normative Twin Generation*. Normal-only masked learning, joint inconsistency, and normative reconstruction are supported. Keep the existing source-note boundary: validation uses simulated abnormal patterns performed by held-out normative participants, not a clinical disease-validation cohort. |
| [PCI](https://pubmed.ncbi.nlm.nih.gov/17503027/) | Plotnik, Giladi and Hausdorff, 2007; DOI 10.1007/s00221-007-0955-7. The original measure uses bilateral event phase and assesses aging and Parkinson's disease. |
| [Q-PCI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11435969/) | Galor, Zeilig and Plotnik, 2024; DOI 10.3390/s24186105; PMID 39338850. The cited four-limb study uses healthy adults, as the canonical text states. |

The S-JEPA and MAMP links also resolve to their intended papers. The LTI asymmetry, lightweight ST-GCN, and confidence-filtering entries match their primary records. None requires a citation correction.

## Clinical identifiers verified

The NLM PubMed XML records confirm the following matches where browser pages returned a challenge: 20007501 is Clark and colleagues' post-stroke motor-module study; 38709654 is Cleland and Madhavan's interlimb coordination/temporal-symmetry study; 7953597 is Morris and colleagues' Parkinsonian hypokinesia study; 25502948 is Sterling and colleagues' dopaminergic arm-swing study; and 26084733 is Steele and colleagues' CP synergy study. The publisher page for DOI 10.1186/s12984-025-01718-5 confirms Vandekerckhove and colleagues' longitudinal Duchenne study.

The canonical text appropriately treats these as motivation for observable measurements. It does not claim that GAVD keypoints recover muscle modules, dopamine state, lesions, or weakness. Retain the boundary that Duchenne is one muscle-disease example rather than a definition of every myopathic clip.
