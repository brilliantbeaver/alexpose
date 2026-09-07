# Before Health Agents Interpret Movement: Lessons from a Gait Representation Study

*Companion extended abstract. The workshop does not list a separate extended-abstract track; the LaTeX source is canonical.*

## Abstract

We argue that movement summaries supplied to future health assistants should preserve what was measured, how recordings were weighted, and which uses were tested. A JEPA-style gait case illustrates why. For the same 64 normal-annotated validation clips, feature similarity across training stages averages to 0.89 with equal clip weights and 0.70 with equal video weights; one video supplies 60 clips. Neither value establishes retained predictive ability or a change in patient health. An exploratory comparison also favors simple pose summaries over learned features on 20 held-out videos. These observations motivate an evidence record accompanying movement summaries and a proposed test of its value for interpretation. No generative assistant or clinical intervention was evaluated.

## Position and clinical relevance

A health assistant could eventually combine video-derived movement summaries with other patient information to support mobility review. A similarity score can compare a fixed recording across two model versions, rather than a person’s movement across two visits. Repeated clips can also make one recording dominate an apparently broad evaluation.

Our position is that the measurement and evaluation conditions should remain attached to movement results when they are passed downstream. Model Cards already address intended uses and evaluation limits (Mitchell et al. 2019), and the V3 framework distinguishes stages of validation for digital measurements (Goldsack et al. 2020). We apply those principles to a concrete gait representation case, concentrating on recording weights and the distinction between model coordinates and useful function. The encoder uses a joint-embedding predictive architecture inspired by S-JEPA (Abdelfattah and Alahi 2024); it does not implement a clinical agent.

## Case study and observations

The study uses 639 quality-screened clips from 97 YouTube videos drawn from GAVD, a manually curated resource for clinical gait analysis (Ranjan et al. 2025). Its selected annotations cover normal gait and Parkinson’s, stroke, myopathic, and cerebral-palsy categories, without independent diagnostic verification here. All clips from a video share its role: 59 videos for training, 18 for validation, and 20 for testing. The evidence comes from one split and one training initialization. Video separation does not guarantee person separation across uploads.

A compact encoder receives 33-landmark poses resized to 64 frames and predicts hidden features at selected body landmarks. Training begins with normal-annotated clips, then cumulatively adds the other categories. Category labels guide that order but are not targets in the encoder loss. We reanalyze retained predictions and similarity summaries, without rerunning encoder training; some runtime settings were not fully recorded.

### Holding similarities fixed, changing the weights.

For each normal-annotated validation clip, cosine compares its features after normal-only training with its features after final training. There are 64 clips from five videos, with one video contributing 60 clips. Equal clip weighting assigns that video about 94% of the total weight and gives mean cosine 0.89. Averaging first within videos and then equally across videos assigns it 20% and gives 0.70. The same recorded similarities enter both calculations.

These are different summaries of the recordings, not competing estimates of patient stability. Equal video weights are not universally preferable and do not imply equal patient weights. Moreover, either average can change as model coordinates change without establishing whether useful movement information was lost. Functional retention needs an actual prediction task assessed across model versions.

### Testing a simpler input baseline.

Three separately fitted logistic-regression classifiers predict the dataset annotations using pose summaries, frozen learned features, or landmark availability without coordinates. Validation selects regularization, after which classifiers are refitted on the 77 training and validation videos. The same 20 test videos give:

| Input                       | Correct videos | Balanced accuracy |
|:----------------------------|:--------------:|:-----------------:|
| Pose and movement summaries |     10/20      |       0.44        |
| Learned pose features       |      6/20      |       0.26        |
| Landmark availability only  |      6/20      |       0.25        |

Descriptive results from one split and initialization. Balanced accuracy averages within-category recall over the five annotations.

All three classifiers misclassify the three stroke-annotated test videos. Two categories have only two test videos each. Selection also uses averaged features per video, whereas testing averages the category probabilities predicted for its clips; these procedures need not agree. Without repeated runs and a matched untrained encoder, the result supports retaining a pose baseline but cannot isolate the effect of pretraining or establish a general ranking.

## Proposal, counterarguments, and next evaluation

We propose that each movement summary include its measured quantity, reference model, recording unit and weights, observed result, and interpretation limits. For the weighting example, that record would state: fixed clips compared across training stages; five equally weighted videos; mean cosine 0.70; predictive retention and patient mobility change untested. This specializes established reporting guidance to the result being interpreted, rather than proposing a new general validation framework.

A follow-up could compare the same assistant’s interpretations with a bare score against interpretations with this record, using blinded review of unsupported mobility claims and recognition of insufficient evidence. Tests should distinguish changes in recordings from model updates and independently measured movement changes. The record’s benefit remains a hypothesis; documentation may be ignored or add burden. Clinical usefulness requires a defined workflow and evaluation of safety and human factors (Vasey et al. 2022).

A larger model or better evaluation could favor learned representations without eliminating the need to state what a summary measures. Forecasting remains untested and would require inputs and preprocessing restricted to the observed past.

The public-video sample, uncertain cross-upload identities, estimated pose geometry, and incomplete training records limit generalization. GAVD’s annotation repository uses the MIT License; separately hosted videos have additional access and use conditions (GAVD project 2026, 2024). Project-specific ethics and data-use reviews remain unresolved, and no approval or exemption is claimed. The evidence supports a focused position about interpreting movement inputs, with agent behavior and patient benefit left for subsequent evaluation.

## References

Abdelfattah, Mohamed, and Alexandre Alahi. 2024. “S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition.” *Computer Vision – ECCV 2024*, 367–84. <https://doi.org/10.1007/978-3-031-73411-3_21>.

GAVD project. 2024. *GAVD: MIT License*. GitHub repository. <https://github.com/Rahmyyy/GAVD/blob/main/LICENSE>.

GAVD project. 2026. *Gait Abnormality Video Dataset: Repository and Data-Use Statement*. GitHub repository. <https://github.com/Rahmyyy/GAVD>.

Goldsack, Jennifer C., Andrea Coravos, Jessie P. Bakker, et al. 2020. “Verification, Analytical Validation, and Clinical Validation (V3): The Foundation of Determining Fit-for-Purpose for Biometric Monitoring Technologies (BioMeTs).” *Npj Digital Medicine* 3: 55. <https://doi.org/10.1038/s41746-020-0260-4>.

Mitchell, Margaret, Simone Wu, Andrew Zaldivar, et al. 2019. “Model Cards for Model Reporting.” *Proceedings of the Conference on Fairness, Accountability, and Transparency*. <https://doi.org/10.1145/3287560.3287596>.

Ranjan, Rahm, David Ahmedt-Aristizabal, Mohammad Ali Armin, and Juno Kim. 2025. “Computer Vision for Clinical Gait Analysis: A Gait Abnormality Video Dataset.” *IEEE Access* 13: 45321–39. <https://doi.org/10.1109/ACCESS.2025.3545787>.

Vasey, Baptiste, Myura Nagendran, Bruce Campbell, et al. 2022. “Reporting Guideline for the Early-Stage Clinical Evaluation of Decision Support Systems Driven by Artificial Intelligence: DECIDE-AI.” *Nature Medicine* 28: 924–33. <https://doi.org/10.1038/s41591-022-01772-9>.
