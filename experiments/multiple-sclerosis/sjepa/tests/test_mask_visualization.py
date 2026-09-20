"""Guard the distinction between one window's mask and fresh stochastic draws."""

import numpy as np
import pytest

from sjepa.masking_v2 import mask_bank_stats, sample_mask_batch
from sjepa.viz import _animation_mask_schedule, skeleton_animation


@pytest.mark.parametrize("num_blocks", [4, 8, 16])
def test_fully_hidden_hips_in_one_draw_do_not_mean_permanent_masking(num_blocks):
    masks = sample_mask_batch(512, 33, num_blocks, np.random.default_rng(0))
    masks = masks.reshape(512, num_blocks, 33)
    # Reproduce the misleading demo: replaying draw 0 never reveals either hip.
    assert masks[0, :, [23, 24]].all()
    # Training continues advancing the RNG, so ALL joints receive both roles.
    assert (~masks).any(1).mean(0).min() >= 0.20
    assert masks.any(1).mean(0).min() >= 0.10
    assert (~masks[1, :, [23, 24]]).any(1).all()

    stats = mask_bank_stats(33, num_blocks, n_masks=512, seed=0)
    np.testing.assert_allclose(stats.joint_target_token_frac, masks.mean((0, 1)))
    np.testing.assert_allclose(stats.joint_visible_token_frac, (~masks).mean((0, 1)))
    np.testing.assert_allclose(stats.joint_always_target_frac, masks.all(1).mean(0))
    np.testing.assert_allclose(stats.token_visible_frac, (~masks).mean(0))
    # Window-level coverage alone could pass with a permanently hidden time slot.
    assert (stats.token_visible_frac > 0).all()
    assert (stats.token_visible_frac < 1).all()
    # Window coverage and token frequency must not be confused in the diagnostics.
    assert (stats.joint_visible_frac > stats.joint_visible_token_frac).all()


def test_animation_replays_motion_with_fresh_masks_and_exact_token_alignment():
    masks = np.zeros((2, 2, 33), dtype=bool)
    masks[0, :, 23:25] = True  # Hips hidden for the entire first sample.
    masks[1, 0] = True        # Legitimate all-target block; context is elsewhere.
    masks[1, 1, 0] = True
    normalized, frames = _animation_mask_schedule(masks, None, 8, 33, 4)
    np.testing.assert_array_equal(normalized, masks)
    assert frames.shape == (16, 33)
    assert frames[:8, 23:25].all()
    assert frames[8:12].all()
    assert not frames[12:, 23:25].any()
    # Divisibility alone is insufficient: 16 frames would silently stretch 2 blocks.
    with pytest.raises(ValueError, match="exactly one model window"):
        _animation_mask_schedule(masks, None, 16, 33, 4)
    with pytest.raises(ValueError, match="at least one sample"):
        _animation_mask_schedule(np.empty((0, 2, 33)), None, 8, 33, 4)


def test_rendered_markers_and_hip_status_follow_each_sample(monkeypatch, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import animation

    seq = np.random.default_rng(10).normal(size=(4, 33, 3))
    masks = np.zeros((2, 2, 33), dtype=bool)
    masks[0, :, 23:25] = True
    masks[1, 0] = True
    masks[1, 1, 0] = True

    class InspectAnimation:
        def __init__(self, fig, update, frames, **kwargs):
            self.fig, self.update, self.frames = fig, update, frames

        def save(self, path, **kwargs):
            assert self.frames == 8
            for t in range(self.frames):
                self.update(t)
                sample, frame = divmod(t, len(seq))
                hidden = masks[sample, frame // 2]
                context, target = self.fig.axes[0].collections
                np.testing.assert_array_equal(context.get_offsets(), seq[frame, ~hidden, :2])
                np.testing.assert_array_equal(target.get_offsets(), seq[frame, hidden, :2])
                texts = [text.get_text() for text in self.fig.texts]
                assert f"Mask sample {sample + 1}/2" in texts
                for j, side in [(23, "L"), (24, "R")]:
                    assert f"{side} hip ({j}): {'MASKED' if hidden[j] else 'VISIBLE'}" in texts
                if hidden.all():
                    assert "No context in this block; context is available at other times." in texts

    monkeypatch.setattr(animation, "FuncAnimation", InspectAnimation)
    skeleton_animation(seq, tmp_path / "inspected.gif", target_mask=masks, frame_group=2)


def test_multisample_gif_export_and_single_sample_compatibility(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from PIL import Image

    seq = np.random.default_rng(10).normal(size=(4, 33, 3))
    masks = np.zeros((2, 2, 33), dtype=bool)
    masks[0, :, 23:25] = True
    masks[1, :, 0:2] = True
    path = skeleton_animation(seq, tmp_path / "samples.gif", target_mask=masks, frame_group=2)
    with Image.open(path) as gif:
        assert gif.n_frames == 8
        gif.seek(0)
        first = np.asarray(gif.convert("RGB"))
        gif.seek(4)
        second = np.asarray(gif.convert("RGB"))
        assert not np.array_equal(first, second)
        # The right-hand hip labels must survive GIF export. Inspect their area,
        # away from skeleton markers and the legend; blitting used to omit them.
        h, w = first.shape[:2]
        red_status = first[int(.25*h):int(.5*h), int(.64*w):]
        blue_status = second[int(.25*h):int(.5*h), int(.64*w):]
        assert ((red_status[..., 0] > 150) & (red_status[..., 1] < 120)
                & (red_status[..., 2] < 120)).sum() > 20
        assert ((blue_status[..., 2] > 150) & (blue_status[..., 0] < 120)
                & (blue_status[..., 1] < 150)).sum() > 20
    for name, options in [
        ("unmasked", {}),
        ("static", {"masked_joints": [23, 24]}),
        ("temporal", {"target_mask": masks[0].reshape(-1), "frame_group": 2}),
    ]:
        path = skeleton_animation(seq, tmp_path / f"{name}.gif", **options)
        with Image.open(path) as gif:
            assert gif.n_frames == 4


def test_demo_refreshes_old_filenames_and_detects_stale_cell_overwrites(tmp_path):
    from PIL import Image
    from sjepa.mask_demo import verify_mask_demo, write_mask_demo

    seq = np.random.default_rng(11).normal(size=(8, 33, 3))
    result = write_mask_demo(seq, tmp_path, frame_group=2, n_samples=3)
    details = verify_mask_demo(tmp_path)
    assert details['total_frames'] == 24
    assert details['target_masks'] == result.masks.astype(int).tolist()
    assert details['visible_frames_per_joint'] == ((~result.masks).sum((0, 1)) * 2).tolist()
    for name in ['mask_demo.gif', 'mask_demo_temporal.gif', 'mask_demo_samples.gif']:
        with Image.open(tmp_path / name) as image:
            assert image.n_frames == 24
        assert (tmp_path / name).read_bytes() == result.animation.read_bytes()
    assert result.timeline.is_file()

    # Reproduce the reported failure: a stale notebook cell rewrites only the old
    # filename with its single mask. The verification must expose that mismatch.
    skeleton_animation(seq, tmp_path / 'mask_demo_temporal.gif',
                       target_mask=result.masks[0], frame_group=2)
    with pytest.raises(ValueError, match='Stale mask-demo artifact.*mask_demo_temporal'):
        verify_mask_demo(tmp_path)


def test_timeline_keeps_every_sample_time_joint_bit(monkeypatch, tmp_path):
    from matplotlib.axes import Axes
    from sjepa.viz import mask_timeline

    masks = np.zeros((3, 4, 33), dtype=bool)
    masks[0, :, 23:25] = True
    masks[1, 0, 23] = True
    masks[2, 1, 24] = True
    imshow = Axes.imshow
    seen = []

    def capture(self, values, *args, **kwargs):
        seen.append(np.asarray(values).copy())
        return imshow(self, values, *args, **kwargs)

    monkeypatch.setattr(Axes, 'imshow', capture)
    mask_timeline(masks, tmp_path / 'timeline.png')
    assert len(seen) == 1
    for sample in range(3):
        for block in range(4):
            np.testing.assert_array_equal(seen[0][:, sample * 4 + block], masks[sample, block])
