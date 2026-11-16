# Reading the methods before building the experiment

This review distinguishes published methods, inspected implementations, and proposed gait adaptations. The source check was performed on November 15, 2026, using the date supplied by the workspace. No cited action-recognition result is a result on this GAVD cohort.

## 1. Resolve the two meanings of S-JEPA

The requested references use **S-JEPA for two different methods**. Abdelfattah and Alahi's ECCV 2024 paper is **Skeleton JEPA**. Giakoumoglou, Giakoumoglou, and Stathaki's November 2026 paper is **Stochastic JEPA** for images. The latter does not establish that adding stochastic latent variables improves neurological gait classification. Throughout this project, call the base model a *Skeleton-JEPA adaptation* and name the optional latent-variable extension separately. Cite both papers when combining their ideas.

## 2. Skeleton JEPA: what each component learns

[Abdelfattah and Alahi, ECCV 2024, sections 3 and 4.2](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) group consecutive observations of each joint into tokens. The view encoder sees visible tokens from an augmented skeleton. A predictor receives those representations and positioned mask tokens. The target encoder sees the full sequence; its output is selected at masked positions. The teacher receives no loss gradients and follows the view encoder by exponential moving average (EMA).

The published objective compares distributions across latent dimensions:

\[
q=\operatorname{softmax}((h_{teacher}-c)/\tau_t),\qquad
p=\operatorname{softmax}(h_{prediction}/\tau_p),\qquad
L=-\operatorname{mean}_{masked}\sum_d q_d\log p_d.
\]

The running center balances target dimensions. The reported temperatures are 0.06 for targets and 0.1 for predictions; center momentum is 0.9. Published masking favors motion. Downstream experiments use the target encoder. The reference configuration has eight encoder layers, five predictor layers, width 256, four-frame tokens, 90% masking, and 1,200 epochs. It is not a laptop training prescription. [Paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf)

The [project page](https://sjepa.github.io/) provides method diagrams and a paper link. Its currently exposed links and the [author's project list](https://moo-osama.github.io/projects/) do not identify a downloadable official skeleton implementation. We therefore do not claim code-level reproduction of that paper.

## 3. MAMP and the required uniform masking

[Mao et al., ICCV 2023](https://arxiv.org/pdf/2308.07092) train a transformer encoder and a smaller decoder to predict masked temporal motion, rather than teacher embeddings. Motion is a coordinate difference across a chosen temporal stride. MAMP's main masking strategy assigns greater probability to joints with stronger motion, using motion-dependent probabilities with Gumbel sampling. Uniform random numbers inside that sampler do **not** make the resulting masks uniform over joints.

The [official implementation](https://github.com/maoyunyao/MAMP/blob/09e6252a5c7d061c226395d25e6aa47a69e7e58d/model_mamp/transformer.py) includes both a separate `random_masking` method and motion-aware masking. `forward` builds a motion target and `forward_loss` evaluates squared error only at hidden tokens. This is direct implementation evidence, not an inferred description from the abstract.

For this study, the user's requirement takes precedence: select a fixed-size random subset uniformly from the approved joint-time tokens, independently of position values, velocities, label, and motion magnitude. This is a **uniform-mask adaptation inspired by MAMP**, not its main published masking procedure. Use the same mask draws when comparing deterministic, positional-noise, and latent-variable variants. Test that changing coordinates while holding the random seed fixed leaves the mask unchanged.

## 4. Positional stochasticity answers “where?”

The requested OpenReview manuscript, *Predicting masked tokens in stochastic locations improves masked image modeling*, became [Bar et al., *Stochastic positional embeddings improve masked image modeling*, ICML 2024](https://proceedings.mlr.press/v235/bar24a.html). The requested OpenReview PDF encountered browser verification; the [author manuscript](https://arxiv.org/pdf/2308.00566) was accessible.

StoP perturbs masked-token positional embeddings with Gaussian noise. Its important design is weight tying:

\[
c_i=A h_i+e_i,\qquad m_j=A\epsilon_j+e_j+m,
\qquad\epsilon_j\sim\mathcal N(0,\sigma I).
\]

The same learned projection transforms context and noise, discouraging the model from suppressing noise without also losing context. Generic additive embedding noise is therefore an approximation, not the complete StoP mechanism. [Method and algorithm](https://arxiv.org/pdf/2308.00566)

The [official repository](https://github.com/amirbar/StoP/tree/9380e4c85e172b6dcfb6610049300a983f735df2) was inspected for identity and layout. Its README identifies the ICML implementation; some model machinery is in a pinned `vissl` submodule. This review did not execute that image-training pipeline.

**Our proposed gait test:** joint IDs are anatomical categories. Never jitter an index into another anatomical identity or add excluded joints. Perturb predictor position embeddings only, leaving the target correspondence unchanged. Include zero-noise evaluation as a deterministic representation check. A benefit must appear on held-out sources and survive controls for pose noise. If it disappears when missingness and camera view are matched, interpret it as an acquisition-robustness effect rather than evidence of improved physiology modeling.

## 5. Latent stochasticity answers “what could happen?”

[Giakoumoglou et al., arXiv:2611.09302v1, November 6, 2026](https://arxiv.org/abs/2611.09302) introduce a learned posterior conditioned on context and target during training, and a learned context-only prior for deployment. Their [objective](https://arxiv.org/pdf/2611.09302) is latent prediction error plus a KL penalty:

\[
\mathbb E_{z\sim q(z\mid h_c,h_t)}\|g(h_c,z)-\operatorname{sg}(h_t)\|^2
+\beta D_{KL}\{q(z\mid h_c,h_t)\Vert p(z\mid h_c)\}.
\]

The paper reports representation and uncertainty benefits on images. These remain hypotheses for gait. KL controls information through the latent variable; it does not by itself prevent encoder collapse.

The [inspected predictor](https://github.com/giakoumoglou/sjepa/blob/e851ccb15297f21f54738bad6a63ce04830488ba/src/models/vision_transformer.py) uses Gaussian prior/posterior networks and reparameterized training samples. Its default `targets=None` inference branch uses the prior mean. Repeated calls to that default do not produce a Monte Carlo uncertainty estimate. Uncertainty experiments need explicit samples from the context prior.

**Our proposed gait test:** introduce this extension only after deterministic prediction works. Compare identical encoders, seeds, windows, and update budgets. Keep the posterior training-only. For forecasting, the context encoder must receive only the past, including any normalization statistics. Sample several prior latents for one fixed past and measure both prediction accuracy and dispersion. A large dispersion is not automatically a reliable uncertainty estimate: test whether higher dispersion predicts larger held-out error, and measure calibration or risk-versus-coverage behavior.

Coordinate trajectories need a separately trained readout. A latent completion is not automatically a reconstructed skeleton, a physically feasible movement, or a treatment counterfactual.

## 6. VICReg keeps dimensions informative

[Bardes, Ponce, and LeCun, ICLR 2022](https://arxiv.org/pdf/2105.04906) combine agreement between paired representations, a hinge on each feature's standard deviation, and a penalty on off-diagonal feature covariance. This encourages nonconstant, less redundant features. It does not use condition labels and does not explicitly push clinical classes apart.

For a batch of embeddings \(Z\), the relevant regularizers are

\[
L_{var}=D^{-1}\sum_d\max(0,\gamma-\sqrt{\operatorname{Var}(Z_d)+\epsilon}),\quad
L_{cov}=D^{-1}\sum_{i\ne j}\operatorname{Cov}(Z)_{ij}^{2}.
\]

Using these terms with masked prediction is a **VICReg-inspired regularizer** unless the full paired invariance objective is also implemented. The [official implementation](https://github.com/facebookresearch/vicreg) supports the distinction between the three terms.

**Our proposed diagnostic:** measure spread across independently sampled sequences or sources, not only across joint positions. Positional embeddings can vary even if the representation ignores the input. Track effective rank, per-dimension standard deviations, and changes in embeddings when motion is shuffled. Tiny laptop batches make covariance estimates noisy; report batch composition and compare results across seeds.

## 7. The Nature article is a separate comparator

[Hui et al., Scientific Reports 16, 8100 (2026)](https://www.nature.com/articles/s41598-026-39330-9) propose topology-masked motion prediction combined with contrastive learning, hybrid augmentation, and trajectory-guided feature dropping. The accessible [article-in-press PDF](https://www.nature.com/articles/s41598-026-39330-9_reference.pdf) describes action-recognition experiments on NTU and PKU-MMD. Its masking is topology/motion dependent, so importing it would violate this study's uniform-mask requirement. It is useful related work, not the source of the skeleton S-JEPA architecture or clinical validation of this project. The publisher HTML failed access; the final HTML metadata was visible in indexed results. No official implementation was verified.

## 8. Treat the requested JEPA walkthrough as a reading aid

The [requested walkthrough](https://www.thecprogramminglanguage.com/jepa.html) is a secondary explanation. We inspected it but use primary papers for implementation decisions. In particular, successful masked video representation learning is not automatically a demonstration of future-only forecasting. EMA is an update rule, not a mathematical guarantee that a model cannot collapse. These distinctions matter for the claims in a medical-world-model paper.

## 9. A falsifiable experimental sequence

The following is our proposed study design, rather than a reported result from the sources above.

1. Establish an acquisition audit, a frozen source-grouped split, and a classical RF baseline. Report attrition from annotations to available media to usable poses.
2. Pretrain the small encoder only on normal training sources. Keep all validation and test sources out of unsupervised training, including normalization and checkpoint selection.
3. Measure normal-reference anomaly scores before adding condition data. Choose the anomaly threshold using normal validation sources and keep a copy of this normal-reference model.
4. Adapt a separate checkpoint with additional condition training sources. Use retained normal examples and inspect forgetting. Call this adapted classification, not normal-only anomaly detection.
5. Train the same linear probe on frozen representations. Compare it with the RF on exactly the same retained clips. Compare the encoder to random features and a capacity-matched supervised encoder.
6. Add the positional and latent stochasticity factors one at a time, followed by their combination. A higher test score found after repeated test inspection is not confirmatory evidence.
7. Evaluate true future-only prediction separately from uniformly masked inpainting. Compare with persistence and a simple temporal predictor at matched context length and horizon.
8. Investigate errors against camera view, pose confidence, source identity, and missingness. Joint-wise residuals localize model surprise; they do not diagnose a neurological mechanism.

The central hypothesis is that a representation trained on normal motion can reveal deviations that transfer across video sources. It is weakened if source or pose-quality controls explain the apparent effect, if RF is equally effective, or if performance is unstable across source splits. Any such outcome is scientifically useful and should remain visible in the paper.

## Access and provenance ledger

| Requested item | Evidence obtained | Limitation |
| --- | --- | --- |
| Local `.../skeleton-jepa/background` | Exact `/Users/pmui` path and plausible `/Users/alexmui/vaults` equivalents checked. | Not present in this workspace; its contents were not reviewed. |
| Skeleton S-JEPA | ECCV PDF, project page, author project page. | Official skeleton code was not linked by the inspected pages. |
| MAMP | Paper and `model_mamp/transformer.py`, commit `09e6252a5c7d061c226395d25e6aa47a69e7e58d`. | No reproduction of published benchmark training. |
| StoP | Author PDF, ICML proceedings, repository commit `9380e4c85e172b6dcfb6610049300a983f735df2`. | Requested OpenReview access challenged; repository/submodule training not replayed. |
| Stochastic latent JEPA | arXiv v1 and predictor/training code, commit `e851ccb15297f21f54738bad6a63ce04830488ba`. | alphaXiv endpoint inaccessible; equivalent primary arXiv source used. |
| Nature paper | Publisher article-in-press PDF and indexed final metadata. | Final HTML access failed; implementation unverified. |
| JEPA walkthrough | Page inspected. | Secondary source, not implementation authority. |

These remote commit identities were obtained from the GitHub API during review. They identify inspected sources only; the tutorial implementation does not depend on these repositories at runtime.
