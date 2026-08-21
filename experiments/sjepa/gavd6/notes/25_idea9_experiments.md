**Role**: You are an expert world model (JEPA) researcher well versed in human gait analysis with world models

**Task**: Ultrathink on how to best create additional notebooks in the "gavd6" project to reify idea 05-09 on gait parity. Carefully analyze the following notebooks and tutorials that document GAVD checkpoint audits and the overall vision and methodology for the full encoder re-training study:

- experiments/sjepa/gavd6/notes/ideas-claude/05-09-gait-parity/README.md
- experiments/sjepa/gavd6/notes/ideas-claude/05-09-gait-parity/README_LONG_TERM.md
- experiments/sjepa/gavd6/notes/ideas-claude/05-09-gait-parity/METHODOLOGY_LONG_TERM.md
- experiments/sjepa/gavd6/nb_05a_signed_laterality_probe.ipynb
- experiments/sjepa/gavd6/nb_05b_reflection_reach_and_futures.ipynb
- experiments/sjepa/gavd6/nb_09a_equivariant_encoder_contract.ipynb
- experiments/sjepa/gavd6/nb_09b_equivariant_futures_and_reach.ipynb

You should create notebooks to implement the full matched JEPA loop on the full GAVD dataset available:
* standard encoder;
* paired-unconstrained control;
* reflection-equivariant encoder.

For each JEPA variant, implement the training loop with the same label-free JEPA objective and anti-collapse terms, with matched data exposure and compute. Also include CUDA compatibility options, but with CPU as the default running mode. Include separate sets of hyperparameters for running the training on CPU vs GPU.

Make sure to remember that these GAVD training runs are functioning as local feasibility runs, confirming that the complete loop trains, remains non-collapsed, and preserves the geometry contract.
