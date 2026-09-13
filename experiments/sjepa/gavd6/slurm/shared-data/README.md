# Shared data preparation

- [download-gavd.sbatch](download-gavd.sbatch) downloads the declared GAVD source videos.
- [convert-amass.sbatch](convert-amass.sbatch) converts eligible AMASS recordings to the existing 11-landmark lower-body representation, including heel and forefoot landmarks.

These jobs retain the existing manifests, body-model requirements, runtime options and output schema. Dataset acquisition and conversion do not establish a scientific result. Follow the calling study's input and reservation contracts.
