"""Opt-in real Chumpy checks for the Python 3.11 / NumPy 1.26 study stack.

ST_TEST_CHUMPY=1 enables checks in the dedicated study/test environment.
Chumpy 0.70 fails during import before any numerical test can run.
"""
import os
import pickle
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get("ST_TEST_CHUMPY") == "1",
                     "Set ST_TEST_CHUMPY=1 in the pinned Chumpy test environment")
class ChumpyCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import chumpy
        if chumpy.__version__ != "0.71" or np.__version__ != "1.26.4":
            raise RuntimeError("This integration check requires Chumpy 0.71 and NumPy 1.26.4")
        cls.ch = chumpy

    def test_values_and_derivatives_match_analytical_polynomial(self):
        x = self.ch.array([1., 2., 3.])
        y = x * x + 2 * x
        np.testing.assert_allclose(y.r, [3., 8., 15.], atol=1e-12, rtol=0)
        np.testing.assert_allclose(y.dr_wrt(x).toarray(), np.diag([4., 6., 8.]), atol=1e-12, rtol=0)

    def test_callback_argument_inspection_works_on_python311(self):
        function = self.ch.Ch(lambda a: a * a, initial_args={"a": np.array([2., 3.])})
        np.testing.assert_allclose(function.r, [4., 9.], atol=1e-12, rtol=0)

    def test_pickle_roundtrip_preserves_arrays(self):
        # Exercises a basic array expression; no licensed model, downloaded
        # pickle or anatomical data enters this fixture.
        value = self.ch.array([[1., 2.], [3., 4.]]) * 2
        restored = pickle.loads(pickle.dumps(value))
        np.testing.assert_allclose(restored.r, [[2., 4.], [6., 8.]], atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
