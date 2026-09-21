# Codex Adversarial Review

Target: working tree diff
Verdict: needs-attention

The empirical calculations reproduce, but the synthesis needs a material qualification about target exposure during training.

Findings:
- [medium] Disclose target exposure caused by reflection without mask permutation (/Users/pmui/dev/alexpose/experiments/multiple-sclerosis/docs/17-research-review-and-directions.md:254)
  The limitation extends beyond unmeasured laterality effects. `random_view` swaps left/right joint data, while `train_sjepa_v2` applies the unchanged context mask and scores original target positions. A masked left knee can therefore appear, transformed, in a visible right-knee slot. A seed-42 audit of 512 masks found this exposure for 17,931/85,409 target slots conditional on reflection. This provides a potential shortcut that the discussion of successful hidden-feature learning does not acknowledge; its contribution to the observed loss decrease remains unmeasured.
  Recommendation: Explicitly distinguish this within-training target exposure from train/test leakage, qualify the masked-learning interpretation, and add a reproducible mask-permutation audit. Require a no-reflection or consistently mapped-mask comparison before attributing loss improvements to inference of hidden motion.

Next steps:
- Revise the masking interpretation while retaining the verified empirical results.
