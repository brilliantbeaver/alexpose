# Frozen-representation force prediction

**Historical negative feasibility result.** The 24-participant StrokePIG comparison does not support force prediction from the available frozen skeleton representations. Every evaluated arm has negative held-out R-squared.

| Chunk | Purpose |
| --- | --- |
| [Results](results/) | Participant-held comparison and the boundary on force or clinical claims. |
| [Evaluation notebook](../../../notebooks/force_prediction/01_strokepig_probe.ipynb) | Data adaptation, frozen features, controls and force readout. |
| [Model and probe implementation](../../../src/gavd6_sjepa/research_directions/reflection_equivariance/) | Shared reflection-model code used by the notebook. |
| [Representation study](../reflection-equivariance/) | Training context and separate GAVD probe limitations. |

Retain fold assignments, participant-target tables, contact tables and predictions together with their original checkpoint identity. There is no dedicated force-study Slurm workflow in this repository. The existing evaluation notebook contains its execution code and must not be treated as a current movement-preservation task.

The null result constrains claims about these representations; it does not establish a general impossibility of force inference. A new study would require an explicit new protocol and evidence, rather than reinterpretation of these saved scores.
