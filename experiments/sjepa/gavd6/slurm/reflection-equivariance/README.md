# Reflection-equivariance jobs

- [train-amass.sbatch](train-amass.sbatch) runs the historical AMASS representation-training comparison.
- [evaluate-swaps.sbatch](evaluate-swaps.sbatch) evaluates a specified frozen checkpoint under the swap-probe contract.

Read the [study overview](../../docs/studies/reflection-equivariance/README.md) and the probe's explicit input/output requirements before running either job. The [source package](../../src/gavd6_sjepa/research_directions/reflection_equivariance/) owns the numerical methods; these files preserve the existing resource and command arguments.
