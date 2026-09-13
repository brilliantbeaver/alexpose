"""future feature prediction / scaling / test cohort."""


import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.future_prediction.cohort import build_candidates
from gavd6_sjepa.research_directions.future_prediction.contracts import (
    initialize_run,
    read_json,
    save_npz,
    sha256_file,
    write_json,
)
from gavd6_sjepa.research_directions.source_scaling import data
from gavd6_sjepa.research_directions.source_scaling.availability import (
    COHORT_POLICY,
    discover_available,
    freeze_available,
    require_included_windows,
    verify_available,
)
from gavd6_sjepa.research_directions.source_scaling.readiness import (
    development_media,
    require_development_media,
)
from tests.support import source_discovery_fixture

# Available-cohort amendment: freeze, isolation, processing and resumption.


def fixture(root, count=30, development=28):
    config = root/'config'; config.mkdir(parents=True)
    media = root/'videos'; media.mkdir()
    ids = [f'video-{i:03d}' for i in range(count)]
    sequences = pd.DataFrame([dict(video_id=v, sequence_id=f'{v}-seq-{j}', first_frame=1,
                                  last_frame=64, n_annotated_frames=64) for v in ids for j in range(2)])
    videos = pd.DataFrame({'video_id':ids})
    roster = pd.DataFrame([dict(video_id=v, source_group=v, role='development' if i<development else 'confirmation',
                                outer_fold=i%5 if i<development else -1, annotated_sequences=2) for i,v in enumerate(ids)])
    for name, table in [('full-sequences.csv',sequences),('full-videos.csv',videos),('source-reservation.csv',roster)]:
        table.to_csv(config/name,index=False)
    for video in ids: (media/f'{video}.mp4').write_bytes(b'synthetic discovery fixture; not real GAVD')
    return media, sequences, videos, roster

class AvailableCohortTests(unittest.TestCase):
    def test_305_development_with_12_missing_freezes_293_and_ignores_extra_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); media,sequences,videos,roster=fixture(root,348,305)
            for i in [*range(12),305,306]: (media/f'video-{i:03d}.mp4').unlink()
            (media/'unregistered-video.mp4').write_bytes(b'extra')
            before=(root/'config/source-reservation.csv').read_bytes()
            summary=freeze_available(root,media)
            self.assertEqual(summary['available_recordings'],334)
            self.assertEqual(summary['included_development_recordings'],293)
            self.assertEqual(summary['unavailable_development_recordings'],12)
            self.assertEqual(summary['reserved_confirmation_recordings'],43)
            selected=pd.read_csv(root/'config/processing-sequences.csv')
            sources=pd.read_csv(root/'config/processing-videos.csv')
            self.assertEqual(len(selected),586)
            self.assertEqual(len(sources),293)
            self.assertTrue(all(Path(p).is_file() for p in sources.video_path))
            self.assertTrue(set(sources.video_id).isdisjoint(set(roster.loc[roster.role=='confirmation','video_id'])))
            self.assertEqual((root/'config/source-reservation.csv').read_bytes(),before)
            self.assertNotIn('unregistered-video',sources.video_id.tolist())

    def test_explicit_relative_paths_and_duplicates_resolve_once_before_freeze(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); media,sequences,videos,roster=fixture(root)
            original=root/'original-manifest-location'; original.mkdir()
            explicit=original/'human readable title.mp4'; explicit.write_bytes(b'original recording')
            videos['local_path']=''
            videos.loc[0,'local_path']='human readable title.mp4'
            videos.to_csv(root/'config/full-videos.csv',index=False)
            (media/'video-000.mp4').unlink()
            (media/'video-001.mkv').symlink_to(media/'video-001.mp4')
            freeze_available(root,media,resolution_manifest=original/'videos.csv')
            sources=pd.read_csv(root/'config/processing-videos.csv').set_index('video_id')
            self.assertEqual(sources.loc['video-000','video_path'],str(explicit.resolve()))
            self.assertNotIn('local_path',sources.columns)
            (media/'video-001.avi').write_bytes(b'new conflicting export after freeze')
            verify_available(root,check_files=True)

    def test_empty_ambiguous_and_unreadable_files_are_excluded_not_processed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); media,sequences,videos,roster=fixture(root)
            (media/'video-000.mp4').write_bytes(b'')
            (media/'video-001.avi').write_bytes(b'other export')
            original_open=Path.open
            def open_path(path,*args,**kwargs):
                if path.name=='video-002.mp4': raise PermissionError('fixture unreadable file')
                return original_open(path,*args,**kwargs)
            with patch.object(Path,'open',open_path):
                inventory=discover_available(roster,root/'config/full-videos.csv',media)
            excluded=inventory.set_index('video_id').loc[['video-000','video-001','video-002']]
            self.assertFalse(excluded.included.any())
            self.assertTrue(excluded.video_path.eq('').all())
            self.assertIn('Unreadable',excluded.loc['video-002','reason'])

    def test_new_downloads_do_not_change_selection_and_selected_loss_blocks_only_decoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); media,*_=fixture(root)
            missing=media/'video-000.mp4'; missing.unlink()
            freeze_available(root,media)
            before=(root/'config/processing-videos.csv').read_bytes()
            missing.write_bytes(b'newly downloaded')
            verify_available(root,check_files=True)
            self.assertEqual((root/'config/processing-videos.csv').read_bytes(),before)
            selected=media/'video-001.mp4'; selected.unlink()
            with self.assertRaisesRegex(FileNotFoundError,'disappeared'):
                verify_available(root,check_files=True)
            verify_available(root)  # Cached-array inspection needs no raw video.

    def test_changed_files_and_rehashed_inclusion_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); media,*_=fixture(root)
            freeze_available(root,media)
            (media/'video-000.mp4').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed after'):
                verify_available(root,check_files=True)
            path=root/'config/media-availability.csv'; table=pd.read_csv(path)
            table.loc[table.role=='confirmation','included']=True
            table.to_csv(path,index=False)
            contract=read_json(root/'config/availability-contract.json')
            contract['artifacts'][path.name]=sha256_file(path)
            write_json(root/'config/availability-contract.json',contract)
            with self.assertRaisesRegex(ValueError,'exactly the available development'):
                verify_available(root)

    def test_processing_manifest_and_cohort_cannot_reintroduce_omitted_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); media,sequences,*_=fixture(root)
            (media/'video-000.mp4').unlink(); freeze_available(root,media)
            with self.assertRaisesRegex(ValueError,'excluded by frozen availability'):
                require_included_windows(root,sequences.iloc[:1])
            path=root/'config/processing-sequences.csv'
            table=pd.read_csv(path); table=pd.concat([table,sequences.iloc[:1]],ignore_index=True)
            table.to_csv(path,index=False)
            contract=read_json(root/'config/availability-contract.json')
            contract['artifacts'][path.name]=sha256_file(path)
            write_json(root/'config/availability-contract.json',contract)
            with self.assertRaisesRegex(ValueError,'Processing manifest differs'):
                verify_available(root)

    def test_prepare_cache_and_resume_never_open_missing_or_confirmation_sources(self):
        # Production candidate/prepare/cache/reload path; pose and teacher outputs
        # are stand-ins. This is an integration fixture, not scientific evidence.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'child'; media,sequences,videos,roster=fixture(root,30,28)
            annotations=root/'annotations.csv'
            pd.DataFrame([dict(seq=row.sequence_id,id=row.video_id,frame_num=f,
                              bbox=str(dict(left=10,top=10,width=50,height=80)),
                              vid_info=str(dict(width=100,height=100)))
                          for row in sequences.itertuples() for f in range(1,65)]).to_csv(annotations,index=False)
            model=root/'model'; model.write_bytes(b'synthetic model')
            parent=Path(tmp)/'parent'
            initialize_run(parent,protocol='legacy-v1',sequence_manifest=root/'config/full-sequences.csv',
                           video_manifest=root/'config/full-videos.csv',annotations=[annotations],pose_model=model,
                           vjepa_root=root,checkpoint=model,synthetic=True,youtube_dir=media)
            build_candidates(parent,root/'config/full-sequences.csv',root/'config/full-videos.csv',[annotations],media)
            # One original candidate recording is now absent; two are confirmation.
            (media/'video-000.mp4').unlink(); freeze_available(root,media)
            for name in ('software.json','parent-snapshot.json'):
                write_json(root/'config'/name,{})
            write_json(root/'config/study.json',dict(protocol='source-learning-curve-v1',synthetic=True,cohort_policy=COHORT_POLICY))
            write_json(root/'config/teacher-contract.json',read_json(parent/'config/teacher-contract.json'))
            write_json(root/'config/nuisance-schema.json',dict(context_embedding_columns=3,columns=['n0','n1']))
            np.save(root/'config/projection-256.npy',np.zeros((4,256)))
            original=pd.read_csv(parent/'manifests/candidates.csv')
            # Keep one inherited window in an included source and another in the
            # unavailable source; only the included cache entry may be reused.
            old=pd.concat([original[original.video_id==v].head(1) for v in ['video-000','video-001']],ignore_index=True)
            old['decoded_fps']=30.0
            old['video_path']='/unmounted/old-location/source.mp4'
            arrays={k:np.zeros((2,*shape)) for k,shape in {
                'baseline':(5,),'person':(256,),'background':(256,),'skeleton':(32,33,4),'matching':(11,)}.items()}
            entries=[]
            for i,row in old.iterrows():
                path=parent/f'teacher-cache/{row.window_id}.npz'
                save_npz(path,**{k:v[i] for k,v in arrays.items()},window_id=np.array(row.window_id),binding=np.array('parent-fixture'))
                entries.append(dict(window_id=row.window_id,sha256=sha256_file(path),binding='parent-fixture'))
            pd.DataFrame(entries).to_csv(parent/'manifests/cache-index.csv',index=False)
            opened=[]
            def pose(row,directory,pose_model):
                self.assertTrue(Path(row['video_path']).is_file())
                self.assertNotIn(row['video_id'],['video-000','video-028','video-029'])
                opened.append(row['window_id'])
                result={**row,'evidence_origin':'new_processing','decoded_fps':30.0}
                for key,folder in [('pose_path','poses'),('frame_path','frames'),('model_box_path','boxes'),('overlay_path','qc/alignment-overlays')]:
                    path=directory/f"{folder}/{row['window_id']}-fixture.npz"
                    save_npz(path,fixture=np.zeros(1)); result[key]=str(path)
                return result
            def encode(adapter,row,projection):
                return dict(baseline=np.zeros(5,dtype=np.float32),person=np.zeros(256),background=np.zeros(256),
                            skeleton=np.zeros((32,33,4)),matching=np.zeros(11)),['n0','n1'],0.0
            runtime=dict(annotations=[annotations],video_root=media,pose_model=model,teacher_root=root,checkpoint=model)
            with patch.object(data,'read_study',return_value=({'synthetic':True,'cohort_policy':COHORT_POLICY},parent)), \
                 patch.object(data,'read_parent',return_value=({},old,arrays,{})), \
                 patch.object(data,'extract_one',side_effect=pose) as extracted, \
                 patch.object(data,'encode_features',side_effect=encode) as encoded:
                data.prepare(root,parent,**runtime)
                self.assertEqual(len(opened),53)
                candidates=pd.read_csv(root/'data/manifests/candidates.csv')
                self.assertEqual(len(candidates),54)
                self.assertTrue(all(Path(p).is_file() for p in candidates.video_path))
                self.assertFalse(set(candidates.video_id)&{'video-000','video-028','video-029'})
                data.cache(root,parent,adapter=object())
                cohort,actual=data.load_expanded(root,parent,require_audit=False)
                self.assertEqual(len(cohort),54)
                self.assertEqual(actual['baseline'].shape,(54,5))
                self.assertEqual(encoded.call_count,53)
                self.assertEqual(read_json(root/'data/cache-complete.json')['reused_windows'],1)
                self.assertTrue(all(Path(p).is_file() for p in cohort.video_path))
                self.assertNotIn(old.iloc[0].window_id,cohort.window_id.tolist())
                data.prepare(root,parent,**runtime)
                data.cache(root,parent,adapter=object())
                self.assertEqual(extracted.call_count,53)
                self.assertEqual(encoded.call_count,53)


# Expanded media regression: storage failures cannot shrink the development study.


class ExpandedMediaTests(unittest.TestCase):
    def fixture(self, root):
        args, _ = source_discovery_fixture(root)
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
            with patch('gavd6_sjepa.research_directions.future_prediction.cohort.check_run', return_value=contract):
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


if __name__ == "__main__":
    unittest.main()
