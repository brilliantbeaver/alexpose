# Reflection-equivariant skeleton representations

**Historical baseline study.** Compare known anatomical-reflection equivariance with ordinary S-JEPA, augmentation and matched paired controls. These controls inform the [laterality study](../latent-laterality/); they do not demonstrate movement preservation under tracking repair.

| Chunk | Contents |
| --- | --- |
| [Protocol](protocol/) | Required model comparisons, evaluation contract and interpretation limits. |
| [Results](results/) | Fixed-reflection evidence and the frozen GAVD representation probe. |
| [Development](development/) | Earlier AMASS setup, repair and pilot records. |
| [Notebooks and execution](../../../notebooks/reflection_equivariance/) | Encoder contract, GAVD checks, replication and AMASS training. |

The [reflection package](../../../src/gavd6_sjepa/research_directions/reflection_equivariance/) implements the models and probes. [Notebook builders](../../../scripts/research_directions/reflection_equivariance/) generate the retained lessons; the [AMASS training launcher](../../../slurm/reflection-equivariance/train-amass.sbatch) preserves the original command interface. Legacy identifiers in those paths are retained for reproducibility. Their local representation uses **11 lower-body landmarks**, including heel and forefoot surface landmarks; it is not a standard named skeleton or an 11-joint anatomical model.

The frozen GAVD comparison is a within-corpus diagnostic with source-video overlap. Random initialization and raw-coordinate controls do not support a representation advantage for the stored reflection variants. Read the [force-prediction result](../force-prediction/) separately before proposing biomechanical transfer. Preserve model, fold, prediction and cohort identities when interpreting any historical run.
