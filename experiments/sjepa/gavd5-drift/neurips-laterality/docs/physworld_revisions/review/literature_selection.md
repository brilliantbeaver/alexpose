# Primary-source selection for the revision

Sources were checked on 9 September 2026. The manuscript uses the literature to motivate questions and bound novelty, not to import performance gains into GAVD. Source titles and links below identify the selected primary research. Preprints are described as such unless a proceedings record was directly checked.

| Source | Use in the paper | Boundary |
|:--|:--|:--|
| [S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf), Abdelfattah and Alahi, ECCV 2024 | Establishes prior skeleton feature prediction | Our implementation and endpoint are an adaptation; action-recognition findings are not gait laterality results. |
| [Masked Motion Predictors are Strong 3D Action Representation Learners](https://arxiv.org/abs/2308.07092), Mao et al., 2023 | Motivates motion-informed sampling | The local arm adapts the official-code sampler; it does not reproduce the full motion-target method. |
| [Less is More: Compact-Token Masked Feature Prediction for Skeleton Representation Learning](https://arxiv.org/abs/2603.10648), Do et al., 2026 preprint, v3 dated 3 August | Recent connected-region masking precedent | Anatomical tubes alone are insufficient novelty. |
| [seq-JEPA: Autoregressive Predictive Learning of Invariant-Equivariant World Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/2f63d2963526bdd9ff1b8bcc2dc9905a-Abstract-Conference.html), Ghaemi et al., NeurIPS 2025 | Transformation-sensitive representation context | This study does not reproduce its architecture or world-model tasks. |
| [Soft Equivariance Regularization for Invariant Self-Supervised Learning](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3be6511c8f56d0dca4b5ed59fdf9b2f4-Abstract-Conference.html), Lee et al., ICLR 2026 | Supports evaluating geometry together with utility | Does not demonstrate that the local reflection loss will improve gait. |
| [V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning](https://arxiv.org/abs/2603.14482), 2026 preprint | Motivates testing dense or intermediate supervision | Larger image/video evidence; no assumed expected local gain. |
| [Human-JEPA: A Human-Centric Vision Model that Perceives and Anticipates](https://arxiv.org/abs/2608.21160), August 2026 preprint | Motivates separating encoder utility from predictor utility | RGB human-video evidence differs from pose laterality. |
| [GaitJEPA author repository](https://github.com/AVAuco/GaitJEPA) | Establishes existing JEPA-for-gait work | Silhouette recognition is a different task; no broad first-JEPA-for-gait claim. |
| [VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning](https://arxiv.org/abs/2105.04906), Bardes et al. | Explains the regularizer | No claim that regularization excludes every form of information loss. |
| [Computer Vision for Clinical Gait Analysis: A Gait Abnormality Video Dataset](https://arxiv.org/abs/2407.04190), Ranjan et al.; associated IEEE Access DOI, 2025 | Dataset attribution | The local 625-clip subset is not the full published GAVD inventory. |
| [Gait asymmetry in community-ambulating stroke survivors](https://pubmed.ncbi.nlm.nih.gov/18226655/) | Spatial and temporal asymmetry motivation | No local stroke-effect estimate or affected-side prediction. |
| [Arm swing magnitude and asymmetry during gait in the early stages of Parkinson's disease](https://pubmed.ncbi.nlm.nih.gov/19945285/) | Motivation to preserve anatomical side identity | Shoulder landmarks do not directly measure arm swing. |
| [The Effect of Increased Gait Speed on Asymmetry and Variability in Children With Cerebral Palsy](https://pubmed.ncbi.nlm.nih.gov/32082235/) | Heterogeneity of gait symmetry | Local annotation does not resolve unilateral/bilateral subtype. |
| [Longitudinal Alterations in Gait Features in Growing Children With Duchenne Muscular Dystrophy](https://pmc.ncbi.nlm.nih.gov/articles/PMC9201072/) | Pelvic and multijoint movement beyond one signed contrast | The myopathic GAVD annotation does not identify Duchenne muscular dystrophy. |
| [Gait asymmetry in children with Duchenne muscular dystrophy: evaluated through kinematic synergies and muscle synergies of lower limbs](https://pmc.ncbi.nlm.nih.gov/articles/PMC10388506/) | Cautions against equating a simple symmetric measure with all normal movement | Motivation only; no clinical transfer claim. |

Other primary work inspected includes I-JEPA, V-JEPA 2, Group Equivariant CNNs and ASMa. They were not all included in the final manuscript because the closest skeleton and recent geometric precedents more directly explain its contribution. Omitting a tangential citation does not imply the method is novel.

The official workshop call is used only to establish topic fit, double-blind review, format and the posted submission dates. The latest manuscript primarily addresses articulated geometry and evaluation; no claim is made to contribute new multimodal sensors or physical-property estimation. [Physical World AI call](https://physworld-org.github.io/physworld.github.io/cfp/)

