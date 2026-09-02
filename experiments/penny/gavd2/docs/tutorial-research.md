# Research notes for the enhanced Skeleton-JEPA tutorial

Purpose: evidence and citation notes for `docs/tutorial.md`. This is not the
tutorial itself. It maps technically important claims to primary sources and
records IEEE-style reference entries that can be copied into the tutorial.

Scope: Transformer capacity, JEPA/I-JEPA/V-JEPA, VICReg, exponential-moving-
average (EMA) teachers, AdamW, gradient clipping, mixed precision, frozen
evaluation, and safe checkpoint loading. Sources are original papers,
conference repositories, accepted-paper records, or official PyTorch
documentation.

## Accuracy boundary: what the notebooks are and are not

Use the following wording in the tutorial:

> Notebook 06 is a **skeleton-specific JEPA adaptation**. Like I-JEPA and
> V-JEPA, it predicts representations for hidden content from visible context
> and obtains targets from a slowly updated encoder. Its anti-collapse loss is
> **VICReg-inspired**. It is not an exact reproduction of I-JEPA, V-JEPA, or
> the original VICReg training recipe.

Why this qualification is necessary:

- I-JEPA predicts target-block representations from a context block and updates
  the target encoder with an EMA of the context encoder [2]. Notebook 06 uses
  the same broad context/predictor/EMA-target pattern, but applies it to
  spatiotemporal skeleton tokens rather than image patches.
- V-JEPA studies feature prediction from video without pixel reconstruction,
  negative examples, text, or a pretrained image encoder [3]. Skeleton clips
  are related to video in being temporal, but notebook 06 does not implement
  the published V-JEPA architecture or data pipeline.
- Original VICReg applies variance and covariance terms separately to both
  embedding branches [4]. Notebook 06 applies its variance and covariance terms
  to the online representation only, uses a LayerNorm-normalized EMA target,
  and sets `VAR_TARGET=0.5` rather than the original paper's experimental
  `gamma=1`. The original paper's main image recipe also used weights
  `lambda=25`, `mu=25`, and `nu=1`, whereas notebook 06 uses `25.0`, `0.5`,
  and `0.04`. Therefore write “VICReg-style” or “VICReg-inspired loss,” not
  “the VICReg algorithm,” and identify the notebook values as project-specific
  experimental settings.

This distinction should also appear in a caption or note near the JEPA training
diagram.

## Claim-to-source map

### 1. Tokens, vectors, and Transformer layers

Claims safe to make:

- A Transformer encoder layer combines self-attention with a position-wise
  feed-forward network [1], [8].
- In PyTorch, `d_model` is the number of features in each token vector,
  `nhead` is the number of attention heads, and `dim_feedforward` is the hidden
  dimension of the feed-forward network [8].
- With `batch_first=True`, the expected layout is
  `(batch, sequence, feature)` [8]. This directly supports the notebook shape
  `(B, T * J, D)`.
- Multi-head attention splits the total embedding dimension across the heads;
  each head has dimension `embed_dim // num_heads` [10]. Thus the tutorial may
  state:

  ```text
  baseline: 64 / 4 = 16 values per head
  enhanced: 128 / 8 = 16 values per head
  ```

  The notebook's divisibility check is consequently necessary.
- Multi-head attention lets the model attend in multiple representation
  subspaces [1], [10]. Avoid saying that individual heads are guaranteed to
  learn specific human-interpretable roles; specialization is a useful
  intuition, not a guarantee.
- Because a Transformer has neither recurrence nor convolution to encode token
  order, the original Transformer injects positional information [1]. This
  supports the notebook's learned time and joint embeddings. Be precise: the
  notebook's *factorized time-plus-joint learned embedding* is a project design
  choice; the original paper used sinusoidal positional encodings, while I-JEPA
  uses positional tokens/embeddings for target locations [2].

Suggested high-school explanation:

> A token is one small record presented to the model. Here, one token describes
> one body joint at one frame. The three input numbers `(x, y, confidence)` are
> projected into a longer learned vector. Attention lets each token gather
> useful information from other tokens, while the feed-forward network lets it
> process the gathered information.

### 2. Depth, width, heads, and feed-forward width

Claims safe to make:

- `N_LAYERS` is depth: PyTorch's `TransformerEncoder` is a stack of `N`
  encoder layers [9]. More layers create more successive rounds of attention
  and feed-forward processing. They add capacity and computation, but do not
  guarantee better learning.
- `EMBED_DIM` is width: it is the size of each token's feature vector and maps
  to PyTorch's `d_model`/`embed_dim` [8], [10].
- `N_HEADS` partitions the embedding into parallel attention heads [10].
  Increasing heads alone mainly repartitions width; it does not give every head
  the full embedding dimension.
- `FF_MULT` is a project-level convenience, not a PyTorch parameter. The
  notebook computes

  ```python
  ff_dim = embed_dim * ff_mult
  ```

  and passes that result to PyTorch as `dim_feedforward` [8]. At
  `embed_dim=128` and `ff_mult=4`, `ff_dim=512`.
- `DROPOUT` is a probability used by the encoder layer [8]. A value of `0.0`
  disables dropout. Do not tell readers that `0.1` must be better; it is an
  experiment to validate, and evaluation must use `model.eval()` [16].

Derived parameter-scaling intuition (mark as a derivation, not a quotation):

- Ignoring biases and layer-normalization parameters, a standard encoder layer
  has roughly `4D^2` attention projection parameters plus `2DF` feed-forward
  parameters. If `F = rD`, the leading term is approximately
  `(4 + 2r)D^2` per layer. Width therefore has a roughly quadratic effect on
  the large matrix weights, while layer count has a roughly linear effect.
- Report exact counts from

  ```python
  sum(p.numel() for p in context_encoder.parameters())
  ```

  rather than presenting the approximation as exact.

### 3. Token count and attention cost

Claims safe to make:

- Full self-attention has a sequence-length term of order `O(L^2 D)` in the
  complexity table of the original Transformer paper [1].
- In this project, `L = T * J = 32 * 33 = 1,056` tokens per clip. If joints stay
  fixed and `T` doubles, `L` doubles, so the pairwise attention matrix becomes
  about four times as large. Say “the quadratic attention part is about 4x,”
  not “the entire training run is exactly 4x,” because feed-forward layers,
  data movement, hardware kernels, and memory behavior also contribute.
- This supports holding `T=32` fixed for the first architecture comparison:
  changing `T` changes context length and compute at the same time, which would
  confound an architecture-only comparison.

### 4. JEPA, I-JEPA, and V-JEPA

Claims safe to make:

- A joint-embedding predictive architecture applies its prediction loss in
  representation (embedding) space rather than directly reconstructing input
  pixels [2].
- I-JEPA uses a context encoder, a predictor conditioned on positional
  information, and a target encoder. It predicts representations of target
  blocks from a visible context block [2].
- The I-JEPA target encoder's weights are updated by an EMA of the context
  encoder's weights [2].
- I-JEPA's experiments show that masking design matters: target blocks should
  be sufficiently large and the context sufficiently informative [2]. Connect
  this to the notebook's limb-over-time and time-block masks as motivation,
  while making clear that those masks are original skeleton-domain choices.
- V-JEPA investigates feature prediction as a stand-alone self-supervised video
  objective and reports evaluation with frozen model weights [3]. This supports
  the tutorial's overall “pretrain without labels, freeze, then probe” story.
- A frozen linear or shallow probe tests how accessible task information is in
  a learned representation without retraining the backbone. Both I-JEPA and
  V-JEPA use frozen-backbone evaluation protocols [2], [3]. Avoid claiming that
  a linear probe measures every kind of useful information.

Suggested analogy:

> A pixel reconstructor is asked to repaint the missing piece. A JEPA-style
> model is asked for a useful *summary* of the missing piece. For gait, that
> summary can focus on motion and body coordination instead of reproducing
> every coordinate exactly.

### 5. EMA target encoder

Claims safe to make:

- EMA teachers average model weights over successive student/context models;
  Mean Teacher established this weight-averaged-teacher pattern [5], and I-JEPA
  specifically uses EMA updates for its target encoder [2].
- Notebook update:

  ```python
  theta_target = m * theta_target + (1 - m) * theta_context
  ```

  is the standard EMA recurrence. Larger `m` changes the target more slowly.
- The heuristic memory length `1 / (1 - m)` is a mathematical approximation
  obtained from geometrically decaying weights, not a guarantee about learning.
  It gives roughly 250 updates for `m=0.996` and 1,000 updates for `m=0.999`.
  Call it an “approximate EMA time scale,” not an exact window.
- If batch size changes from 16 to 4, choosing `m=0.999` preserves the rough
  *number of examples* in that time scale (`250*16 = 1000*4 = 4000`). This is a
  project heuristic, not a recommendation made by the I-JEPA paper. It should
  be described as a motivated starting point to validate.

### 6. Representation collapse and the VICReg-style loss

Claims safe to make:

- Representation collapse means an encoder produces constant or
  non-informative outputs for different inputs [2], [4].
- Original VICReg has three named components [4]:

  1. **Invariance:** reduce mean squared distance between paired embeddings.
  2. **Variance:** keep the batch standard deviation of each feature dimension
     above a threshold using a hinge loss.
  3. **Covariance:** reduce squared off-diagonal covariance values so different
     dimensions are less redundant.

- The original paper's complete loss is a weighted sum of those components,
  with variance and covariance regularizing both branches [4].
- Notebook 06 uses these ideas differently: MSE aligns the predictor with the
  normalized EMA target, while variance and covariance regularize only the
  online representation. Use “similarity” or “prediction alignment” for its
  first logged term if that is clearer than implying a literal paired-view
  VICReg invariance loss.
- VICReg variance and covariance are batch statistics [4]. The notebook's
  `BATCH >= 2` guard is mathematically necessary for its unbiased variance and
  covariance estimates; batch size 1 is invalid for this implementation.
- With a wider embedding, the covariance matrix contains more off-diagonal
  entries. Monitoring the *weighted* loss terms is therefore important. Do not
  infer that the covariance weight must change merely from dimension; inspect
  the actual scale and downstream validation evidence.

Suggested analogy:

> Similarity says “predict the right answer.” Variance says “do not give every
> clip the same answer.” Covariance says “do not make every slot in the answer
> repeat the same fact.”

### 7. AdamW and weight decay

Claims safe to make:

- AdamW decouples weight decay from the adaptive gradient update [6].
- PyTorch describes AdamW as keeping weight decay out of the optimizer's
  momentum and variance accumulators [11].
- `LR` controls the optimizer step scale; `WEIGHT_DECAY` controls the decoupled
  shrinkage strength. Both are hyperparameters to validate.
- Changing Adam to AdamW while changing architecture, EMA, batch size, and
  training length creates an “enhanced recipe” comparison rather than a pure
  architecture ablation. This is experimental-design reasoning, not a claim
  from AdamW. For causal attribution, change one factor at a time or report the
  combined recipe honestly.

Avoid saying that AdamW universally generalizes better. The original paper
reported empirical benefits in its tested settings [6]; it is not a guarantee
for every dataset.

### 8. Gradient clipping

Claims safe to make:

- Gradient-norm clipping was proposed as a response to exploding gradients [7].
- PyTorch's `clip_grad_norm_` treats parameter gradients as one concatenated
  vector for the total norm, modifies gradients in place, and returns the total
  pre-clipping norm [12].
- In the tutorial, call clipping “a safety guard against unusually large
  updates,” not a cure for every instability. If nearly every update is clipped,
  investigate learning rate, loss scaling, and data rather than simply lowering
  the reported gradient value.
- When combining clipping with CUDA AMP/`GradScaler`, unscale gradients before
  clipping so the threshold applies to real gradients [13].

### 9. Automatic mixed precision (optional extension)

Claims safe to make:

- PyTorch AMP normally combines `torch.autocast` with
  `torch.amp.GradScaler` for float16 training [13]. Autocast chooses precision
  by operation; gradient scaling helps prevent small float16 gradients from
  underflowing [13].
- AMP can improve speed and memory use on supported hardware, but not every
  model or operation is numerically safe in lower precision [13]. Benchmark and
  verify finite losses.
- If adding AMP to this notebook, compute sensitive variance/covariance
  reductions in float32. This is a conservative project recommendation based on
  their use of variances, square roots, and covariance matrices; do not
  attribute it as a quoted VICReg requirement.

### 10. Frozen evaluation and inference mode

Claims safe to make:

- `torch.inference_mode()` is appropriate when evaluation work will not enter
  autograd; compared with `no_grad`, it removes additional view-tracking and
  version-counter overhead [14].
- `inference_mode()` does **not** set evaluation behavior. Call `encoder.eval()`
  separately so dropout and other train/eval-sensitive layers behave correctly
  [14], [16].
- `batch_first=True`, batched inputs, disabled gradients, and `.eval()` are
  among the conditions under which PyTorch attention/encoder implementations
  may use optimized inference paths [8], [10]. Say “may enable,” not “guarantees.”
- Minibatching the frozen embedding pass changes memory use, not the intended
  representation calculation. With dropout disabled and no batch-dependent
  normalization in this encoder, samples can be embedded in smaller batches.

### 11. Checkpoints and strict loading

Claims safe to make:

- PyTorch recommends saving a model's `state_dict` for flexible restoration
  [16].
- With `strict=True`, `load_state_dict` requires checkpoint keys to exactly
  match the keys expected by the reconstructed module [15]. This catches many
  architecture mismatches.
- Strict keys alone do not verify semantic configuration. Saving and checking
  `T`, `N_JOINTS`, `C`, `EMBED_DIM`, `N_LAYERS`, `N_HEADS`, `FF_MULT`, and
  `DROPOUT` is a sound project-level contract.
- `torch.load(..., weights_only=True)` restricts loading to tensors, primitive
  types, dictionaries, and explicitly allow-listed types, but checkpoints
  should still come from trusted sources [17].
- Evaluation must fail when the expected trained checkpoint is missing or
  incompatible. Silently using a fresh random encoder would answer a different
  scientific question.

## Reference-number plan for `docs/tutorial.md`

Use numbers in first-appearance order. If the tutorial introduces the project
before its software details, the following order is coherent:

1. Transformer paper.
2. I-JEPA paper.
3. V-JEPA paper.
4. VICReg paper.
5. Mean Teacher paper.
6. AdamW paper.
7. Gradient clipping paper.
8–17. Official PyTorch documentation.

When the same source supports several nearby sentences, place one citation at
the end of the paragraph. Do not cite a source for a project-specific number
that can be verified directly from notebook output; label that as “measured in
Notebook 06.”

## IEEE-style references

[1] A. Vaswani *et al*., “Attention is all you need,” in *Advances in
Neural Information Processing Systems 30*, 2017, pp. 5998–6008. [Online]. Available:
https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.
arXiv:1706.03762.

[2] M. Assran, Q. Duval, I. Misra, P. Bojanowski, P. Vincent, M. Rabbat,
Y. LeCun, and N. Ballas, “Self-supervised learning from images with a
joint-embedding predictive architecture,” in *Proc. IEEE/CVF Conf. Computer
Vision and Pattern Recognition (CVPR)*, Jun. 2023, pp. 15619–15629. [Online].
Available:
https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html.
arXiv:2301.08243.

[3] A. Bardes, Q. Garrido, J. Ponce, X. Chen, M. Rabbat, Y. LeCun,
M. Assran, and N. Ballas, “Revisiting feature prediction for learning visual
representations from video,” *Transactions on Machine Learning Research*,
2024, ISSN 2835-8856. [Online]. Available:
https://openreview.net/forum?id=QaCCuDfBk2. arXiv:2404.08471.

[4] A. Bardes, J. Ponce, and Y. LeCun, “VICReg:
Variance-invariance-covariance regularization for self-supervised learning,” in
*Proc. International Conference on Learning Representations (ICLR)*, 2022.
[Online]. Available: https://openreview.net/forum?id=xm6YD62D1Ub.
arXiv:2105.04906.

[5] A. Tarvainen and H. Valpola, “Mean teachers are better role models:
Weight-averaged consistency targets improve semi-supervised deep learning
results,” in *Advances in Neural Information Processing Systems 30*, 2017,
pp. 1195–1204.
[Online]. Available:
https://proceedings.neurips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html.
arXiv:1703.01780.

[6] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in
*Proc. International Conference on Learning Representations (ICLR)*, 2019.
[Online]. Available: https://openreview.net/forum?id=Bkg6RiCqY7.
arXiv:1711.05101.

[7] R. Pascanu, T. Mikolov, and Y. Bengio, “On the difficulty of training
recurrent neural networks,” in *Proc. 30th International Conference on Machine
Learning*, ser. Proceedings of Machine Learning Research, vol. 28, no. 3,
2013, pp. 1310–1318. [Online]. Available:
https://proceedings.mlr.press/v28/pascanu13.html. arXiv:1211.5063.

[8] PyTorch Contributors, “TransformerEncoderLayer,” *PyTorch
documentation*. Accessed: Jul. 19, 2026. [Online]. Available:
https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html.

[9] PyTorch Contributors, “TransformerEncoder,” *PyTorch documentation*.
Accessed: Jul. 19, 2026. [Online]. Available:
https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html.

[10] PyTorch Contributors, “MultiheadAttention,” *PyTorch documentation*.
Accessed: Jul. 19, 2026. [Online]. Available:
https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html.

[11] PyTorch Contributors, “AdamW,” *PyTorch documentation*. Accessed:
Jul. 19, 2026. [Online]. Available:
https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html.

[12] PyTorch Contributors, “clip_grad_norm_,” *PyTorch documentation*.
Accessed: Jul. 19, 2026. [Online]. Available:
https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html.

[13] PyTorch Contributors, “Automatic mixed precision examples,” *PyTorch
documentation*. Accessed: Jul. 19, 2026. [Online]. Available:
https://docs.pytorch.org/docs/stable/notes/amp_examples.html.

[14] PyTorch Contributors, “inference_mode,” *PyTorch documentation*.
Accessed: Jul. 19, 2026. [Online]. Available:
https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html.

[15] PyTorch Contributors, “Module.load_state_dict,” *PyTorch
documentation*. Accessed: Jul. 19, 2026. [Online]. Available:
https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict.

[16] M. Inkawhich, “Saving and loading models,” *PyTorch Tutorials*,
PyTorch Contributors. Accessed: Jul. 19, 2026. [Online]. Available:
https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html.

[17] PyTorch Contributors, “torch.load,” *PyTorch documentation*. Accessed:
Jul. 19, 2026. [Online]. Available:
https://docs.pytorch.org/docs/stable/generated/torch.load.html.

### Optional initialization reference

Use this only if the tutorial explains why it explicitly reinitializes cloned
Transformer layers.

[18] X. Glorot and Y. Bengio, “Understanding the difficulty of training deep
feedforward neural networks,” in *Proc. 13th International Conference on
Artificial Intelligence and Statistics*, ser. Proceedings of Machine Learning
Research, vol. 9, 2010, pp. 249–256. [Online]. Available:
https://proceedings.mlr.press/v9/glorot10a.html.

PyTorch's `TransformerEncoder` documentation warns that the cloned layers are
initially assigned the same parameter values and recommends manual
initialization [9]. PyTorch identifies `xavier_uniform_` as Glorot
initialization [18]. The careful claim is: “the notebook explicitly
reinitializes the cloned projection layers so they do not begin with identical
values.” Do not claim that Xavier initialization guarantees convergence.

### Optional mixed-precision paper

[19] P. Micikevicius *et al*., “Mixed precision training,” in *Proc. 6th
International Conference on Learning Representations (ICLR)*, 2018. [Online].
Available: https://openreview.net/forum?id=r1gs9JgRZ. arXiv:1710.03740.

This is the original paper-level source for mixed-precision training with loss
scaling. Prefer official PyTorch documentation [13] for exact current API usage
and [19] for the method's research background.

## Verification checklist for the final tutorial

- Use “Skeleton-JEPA adaptation” and “VICReg-style,” not “exact V-JEPA” or
  “exact VICReg.”
- Distinguish observed notebook outputs from proposed settings. An output graph
  copied from a notebook is historical evidence for the run saved in that
  notebook, not a promise about a future rerun.
- State that more depth/width increases capacity and cost, not that it
  automatically makes the encoder more accurate.
- State `128 / 8 = 16` head features and `128 * 4 = 512` feed-forward features.
- State that the attention component is quadratic in token count; avoid claiming
  exact whole-program runtime scaling.
- Explain that the EMA time-scale argument is a project heuristic.
- Explain that batch size 1 is invalid for the notebook's unbiased batch
  variance/covariance computation.
- Require both `encoder.eval()` and `torch.inference_mode()` for the frozen pass.
- Require strict state loading and explicit architecture metadata checks.
- Cite original papers for scientific ideas and official PyTorch docs for API
  behavior.
- Use IEEE bracket citations such as `[2]`, not author–date citations.
