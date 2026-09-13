# Movement representation and tracking repair

The current study asks whether video evidence can **preserve real movement while repairing tracking failures**, beyond calibrated optical-flow and simple smoothing baselines. It is an offline restoration experiment; real pretrained-model results remain to be established.

Start with the [motion-preservation study](docs/studies/motion-preservation/README.md), then follow its [six notebooks](notebooks/motion_preservation/README.md) and [HAIC execution guide](slurm/motion-preservation/README.md).

## Studies

| Study | Status and question |
| --- | --- |
| [Motion preservation](docs/studies/motion-preservation/README.md) | Current: retain true movement at comparable tracking-error removal. |
| [Future feature prediction](docs/studies/future-feature-prediction/README.md) | Historical development gates, source scaling and target-accessibility tests; no established student-distillation benefit. |
| [Latent laterality](docs/studies/latent-laterality/README.md) | Historical correspondence study; confirmation stopped after the uniform-control result. |
| [Reflection equivariance](docs/studies/reflection-equivariance/README.md) | Fixed-reflection controls and frozen-representation probes. |
| [Gait classification](docs/studies/gait-classification/README.md) | Historical label-informed representation/classification workflow. |
| [Force prediction](docs/studies/force-prediction/README.md) | Negative StrokePIG feasibility study. |
| [Perturbation response](docs/studies/perturbation-response/README.md) | Deferred protocol; availability and harmonization remain prerequisites. |

## Repository map

Each study overview groups its material into a few protocol, execution and results sections. Implementation stays in conventional [src](src/gavd6_sjepa/source_module_guide.md), [notebooks](notebooks/README.md), [scripts](scripts/navigation_guide.md), [tests](tests/) and [Slurm](slurm/README.md) directories. These are different roles, not competing copies of a study.

- [Study index](docs/studies/README.md): research questions, status and owned files.
- [Research agenda](notes/research-agenda/README.md): alternatives and shared references.
- [Repository layout and naming](docs/repository/layout.md): canonical paths, legacy compatibility and file ownership.
- [Result organization](docs/studies/output-organization.md): identified runs and preservation rules. The registered historical output bundles are not installed in this checkout.
- [Historical tutorial](docs/studies/gait-classification/tutorial.md): preserved original workflow and results.

Use the existing project environment. For the current experiment, follow the study's model/data setup and pilot instructions; a demonstration run is not a research result. No study cleanup or rename changes source reservations, final evaluation boundaries or earlier STOP decisions.
