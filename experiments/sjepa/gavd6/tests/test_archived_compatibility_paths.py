"""Keep explicitly archived imports and command launchers executable."""

from __future__ import annotations

import unittest

from gavd6_sjepa.archive.legacy_imports import (
    amass_core11_jepa,
    gait_parity_jepa,
    latent_laterality,
)
from gavd6_sjepa.data_foundations import amass_core11_conversion_pipeline
from gavd6_sjepa.research_directions.latent_laterality import laterality_corruption_inference
from gavd6_sjepa.research_directions.reflection_equivariance import (
    amass_core11_training_pipeline,
    jepa_model_architecture,
)
from scripts.archive.legacy_command_launchers import convert_amass_core11


class ArchivedCompatibilityPathTests(unittest.TestCase):
    def test_old_model_module_exports_new_implementation(self) -> None:
        self.assertIs(gait_parity_jepa.build_model, jepa_model_architecture.build_model)

    def test_old_amass_module_exports_new_implementation(self) -> None:
        self.assertIs(
            amass_core11_jepa.Core11WindowDataset,
            amass_core11_training_pipeline.Core11WindowDataset,
        )

    def test_old_laterality_module_exports_new_implementation(self) -> None:
        self.assertIs(
            latent_laterality.SequenceGaugeConfig,
            laterality_corruption_inference.SequenceGaugeConfig,
        )

    def test_old_converter_script_exports_new_implementation(self) -> None:
        self.assertIs(
            convert_amass_core11.convert_sequence_arrays,
            __import__(
                "gavd6_sjepa.data_foundations.amass_core11_conversion_pipeline",
                fromlist=["convert_sequence_arrays"],
            ).convert_sequence_arrays,
        )


if __name__ == "__main__":
    unittest.main()
