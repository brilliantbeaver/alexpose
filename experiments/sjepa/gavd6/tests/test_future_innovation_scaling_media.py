"""Expanded media regression: storage failures cannot shrink the development study."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from tests import test_future_innovation_source_discovery as discovery_tests
from gavd6_sjepa.research_directions.future_innovation.fi_contracts import read_json, write_json, sha256_file
from gavd6_sjepa.research_directions.future_innovation.fi_cohort import build_candidates
from gavd6_sjepa.research_directions.future_innovation_scaling import fi_scaling_data as data
from gavd6_sjepa.research_directions.future_innovation_scaling.fi_scaling_readiness import development_media, require_development_media


class ExpandedMediaTests(unittest.TestCase):
    def fixture(self, root):
        args, _ = discovery_tests.FutureInnovationSourceDiscoveryTests().fixture(root)
        child, sequences, videos, annotations, media = args
        frame = pd.read_csv(sequences)
        frame.loc[len(frame)] = ['missing-sequence', 'missing-video', 1, 64]
        frame.to_csv(sequences, index=False)
        sources = pd.read_csv(videos)
        sources.loc[len(sources)] = ['missing-video']
        sources.to_csv(videos, index=False)
        roster = sources.assign(role='development')
        roster.to_csv(child/'config/source-reservation.csv', index=False)
        contract = read_json(child/'config/run-contract.json')
        contract.update(cohort_size=None, minimum_sources=25, source_cap=None)
        contract['inputs_sha256'].update({str(p.resolve()): sha256_file(p) for p in (sequences, videos)})
        return args, roster, contract

    def test_shared_candidate_path_reproduces_silent_missing_source_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, roster, contract = self.fixture(Path(tmp).resolve())
            with patch('gavd6_sjepa.research_directions.future_innovation.fi_cohort.check_run', return_value=contract):
                build_candidates(*args)
            self.assertTrue((args[0]/'config/candidates-contract.json').exists())
            self.assertEqual(len(pd.read_csv(args[0]/'manifests/candidates.csv')), 50)
            self.assertIn('not found', pd.read_csv(args[0]/'manifests/exclusions.csv').reason.iloc[0])
            with self.assertRaisesRegex(FileNotFoundError, '1 of 26 development recordings'):
                require_development_media(development_media(roster, args[2], args[-1]))

    def test_prepare_guard_runs_before_candidate_freeze_and_allows_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, roster, contract = self.fixture(Path(tmp).resolve())
            root, sequences, videos, annotations, media = args
            contract['input_paths'] = dict(sequence_manifest=sequences, video_manifest=videos,
                                          annotations=annotations, youtube_dir=media)
            with patch.object(data, 'read_study', return_value=({}, root)), \
                 patch.object(data, 'initialize_data', return_value=(root, contract)), \
                 patch.object(data, 'build_candidates', side_effect=RuntimeError('guard passed')) as build:
                with self.assertRaisesRegex(FileNotFoundError, '1 of 26'):
                    data.prepare(root)
                build.assert_not_called()
                self.assertFalse((root/'config/candidates-contract.json').exists())
                inventory = pd.read_csv(root/'logs/development-media.csv')
                self.assertEqual(int(inventory.available.sum()), 25)
                (media/'all/missing-video.mp4').write_bytes(b'restored discovery fixture')
                with self.assertRaisesRegex(RuntimeError, 'guard passed'):
                    data.prepare(root)
                build.assert_called_once()

    def test_confirmation_unavailable_is_allowed_but_ambiguity_and_empty_dev_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            videos = root/'videos.csv'
            pd.DataFrame({'video_id': ['dev', 'reserved']}).to_csv(videos, index=False)
            roster = pd.DataFrame({'video_id': ['dev', 'reserved'], 'role': ['development', 'confirmation']})
            (root/'dev.mp4').write_bytes(b'fixture')
            original = roster.copy(deep=True)
            require_development_media(development_media(roster, videos, root))
            (root/'dev.avi').write_bytes(b'other export')
            with self.assertRaisesRegex(FileNotFoundError, 'Ambiguous'):
                require_development_media(development_media(roster, videos, root))
            (root/'dev.avi').unlink()
            (root/'dev.mp4').write_bytes(b'')
            with self.assertRaisesRegex(FileNotFoundError, 'Empty source video'):
                require_development_media(development_media(roster, videos, root))
            pd.testing.assert_frame_equal(roster, original)

    def test_generated_video_uses_actual_decoder_and_saves_full_window(self):
        # Synthetic pixels and stand-in poses exercise disk/decoder/alignment;
        # they supply no evidence about real GAVD pose quality or teacher signal.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('frames', 'poses', 'boxes', 'qc/alignment-overlays'):
                (root/name).mkdir(parents=True)
            video = root/'generated.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 30, (100, 100))
            self.assertTrue(writer.isOpened())
            for index in range(70):
                writer.write(np.full((100, 100, 3), 2*index, dtype=np.uint8))
            writer.release()
            boxes = np.tile(np.array([.1, .1, .6, .9]), (64, 1))
            np.savez(root/'boxes/source.npz', source_boxes=boxes)
            write_json(root/'annotations.json', [{}]*64)
            row = dict(window_id='generated-window', video_path=str(video), source_first_frame=3,
                       annotation_path=str(root/'annotations.json'), box_path=str(root/'boxes/source.npz'))
            raw = np.ones((32, 33, 4), dtype=np.float32)*.5
            history = raw.copy(); history[..., 3] = 1
            with patch.object(data, 'extract_history', return_value=(raw, history, 1.0)) as pose:
                result = data.extract_one(row, root, root/'unused-synthetic-pose.task')
            frames = np.load(result['frame_path'])['video']
            self.assertEqual(frames.shape, (64, 100, 100, 3))
            np.testing.assert_allclose(frames.mean(axis=(1,2,3)), 2*np.arange(3,67), atol=2)
            np.testing.assert_array_equal(np.load(result['pose_path'])['source_frames'], np.arange(3,35))
            self.assertEqual(pose.call_args.args[0].shape[0], 64)
            self.assertTrue(Path(result['overlay_path']).is_file())


if __name__ == '__main__':
    unittest.main()
