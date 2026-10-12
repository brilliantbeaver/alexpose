import numpy as np
import unittest
import torch

from gavd6_sjepa.research_directions.gait_fidelity.measurements import (
    knee_angles, knee_excursion, reference_support, torch_knee_excursion, measurement_loss)


def trajectory(samples=64):
    xy = np.zeros((1, samples, 12, 2), dtype=np.float32)
    time = np.linspace(0, 2 * np.pi, samples)
    for side in (0, 1):
        xy[0, :, 6 + side] = [side * 50, 0]
        xy[0, :, 8 + side] = [side * 50, 30]
        angle = .4 + (.2 + .1 * side) * np.sin(time)
        xy[0, :, 10 + side, 0] = side * 50 + 30 * np.sin(angle)
        xy[0, :, 10 + side, 1] = 30 + 30 * np.cos(angle)
    return xy


class MeasurementTests(unittest.TestCase):

    def test_primary_metric_is_invariant_to_shared_isotropic_transform(self):
        x = trajectory()
        a = knee_excursion(x)
        b = knee_excursion(3.7 * x + [134, -12])
        np.testing.assert_allclose(a['asymmetry'], b['asymmetry'], atol=1e-4)
        swapped = x[:, :, np.arange(12).reshape(6, 2)[:, ::-1].ravel()]
        np.testing.assert_allclose(knee_excursion(swapped)['asymmetry'], -a['asymmetry'])
        assert abs(a['asymmetry'][0]) > 5


    def test_prediction_failure_cannot_change_fixed_support(self):
        x = trajectory()
        support = reference_support(x, np.ones(x.shape[:-1], bool))
        bad = x.copy(); bad[0, 8, 10] = np.nan
        m = knee_excursion(bad, support=support)
        assert m['reference_eligible'][0] and m['prediction_failure'][0]
        assert not m['supported'][0] and np.isnan(m['asymmetry'][0])
        bad = x.copy(); bad[0, 10, 10] = bad[0, 10, 8]
        assert knee_excursion(bad, support=support)['prediction_failure'][0]


    def test_common_support_and_irregular_clock_rejection(self):
        x = np.repeat(trajectory(), 4, axis=0)
        valid = np.ones(x.shape[:-1], bool); valid[2, 4, 10] = False
        support = reference_support(x, valid, groups=[[0, 1, 2, 3]])
        assert (~support[:, 4]).all()
        t = np.arange(64) / 25.; t[-1] += .01
        with self.assertRaisesRegex(ValueError, 'uniform'):
            knee_excursion(x, timestamps=t)


    def test_torch_matches_evaluator_and_has_finite_nonzero_gradients(self):
        x = trajectory()
        tensor = torch.tensor(x, requires_grad=True)
        values, supported, _ = torch_knee_excursion(tensor, np.ones(x.shape[:2], bool))
        np.testing.assert_allclose(values.detach().numpy(), knee_excursion(x)['asymmetry'], atol=1e-4)
        values.square().sum().backward()
        assert supported.all() and torch.isfinite(tensor.grad).all()
        assert tensor.grad.abs().sum() > 0


    def test_straight_tied_and_degenerate_training_angles_keep_finite_gradients(self):
        x = torch.zeros(4, 2, 32, 12, 2)
        # Two normal examples, one exactly straight, one predicted collapse.
        base = torch.tensor(trajectory(32))
        target = base[:, None].expand(4, 2, -1, -1, -1).clone()
        x[:] = target
        x[2, :, :, 10:12] = 2 * x[2, :, :, 8:10] - x[2, :, :, 6:8]
        x[3, :, :, 10:12] = x[3, :, :, 8:10]
        x.requires_grad_()
        for objective in ('paired_change', 'per_example_measurement', 'repaired_change'):
            loss, report = measurement_loss(x, target, torch.ones(target.shape[:-1], dtype=torch.bool), objective=objective)
            loss.backward()
            assert torch.isfinite(x.grad).all()
            assert report['supported_pairs'] == 4
            x.grad.zero_()
