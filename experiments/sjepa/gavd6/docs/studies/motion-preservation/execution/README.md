# Motion-preservation execution

The maintained notebook and Slurm guides own the runtime instructions. Use notebook 06 to diagnose the existing real-checkpoint pilot before expanding training.

- [Notebook sequence](../../../../notebooks/motion_preservation/README.md): stages 00 through 06 cover data inventory, preservation/repair evaluation, GAVD visual inspection, and repair-mechanism diagnosis.
- [Cluster guide](../../../../slurm/motion-preservation/README.md): environment, pilot configuration, dependencies and explicit final/GAVD launches.
- [Diagnose the existing pilot](../../../../slurm/motion-preservation/README.md#5-diagnose-the-existing-pilot-without-another-model-run): one CPU job reads cached predictions and saves an executed tutorial, tables, traces, and SVG figures.
- [Notebook sources](../../../../scripts/research_directions/motion_preservation): generation and execution helpers.
- [Implementation](../../../../src/gavd6_sjepa/research_directions/motion_preservation): data, geometry, rendering, model adapters, learning and reporting.

Executed notebooks belong to their identified run. Demonstration mode and software validation do not replace real data/model evaluation.

[Study overview](../README.md).
