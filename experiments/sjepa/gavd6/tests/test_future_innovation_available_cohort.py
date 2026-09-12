"""Available-cohort amendment: freeze, isolation, processing and resumption."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from gavd6_sjepa.research_directions.future_innovation.fi_contracts import read_json, write_json, sha256_file, initialize_run, save_npz
from gavd6_sjepa.research_directions.future_innovation.fi_cohort import build_candidates
from gavd6_sjepa.research_directions.future_innovation_scaling.fi_scaling_availability import (
    COHORT_POLICY, discover_available, freeze_available, verify_available, require_included_windows)
from gavd6_sjepa.research_directions.future_innovation_scaling import fi_scaling_data as data


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


if __name__=='__main__':
    unittest.main()
