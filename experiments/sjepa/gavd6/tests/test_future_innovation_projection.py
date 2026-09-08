import tempfile
import unittest
from pathlib import Path

import numpy as np

from gavd6_sjepa.research_directions.future_innovation.fi_token_regions import (
    fixed_projection,
)
from gavd6_sjepa.shared_infrastructure.artifact_io_operations import sha256_file


class FutureInnovationProjectionTests(unittest.TestCase):
    def test_scaled_orthogonal_determinism_and_checksum(self):
        projection = fixed_projection(768)
        # Accumulate in float64 so this checks the saved matrix, not float32 BLAS rounding.
        np.testing.assert_allclose(
            projection.astype(float).T @ projection.astype(float),
            np.eye(256) * 3,
            atol=2e-7,
        )
        np.testing.assert_array_equal(projection, fixed_projection(768))
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.npy", Path(directory) / "b.npy"
            np.save(a, projection)
            np.save(b, fixed_projection(768))
            self.assertEqual(sha256_file(a), sha256_file(b))
        with self.assertRaises(ValueError):
            fixed_projection(128)
