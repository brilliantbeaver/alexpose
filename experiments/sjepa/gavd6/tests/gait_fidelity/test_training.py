from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import unittest
import tempfile
import torch

from gavd6_sjepa.research_directions.gait_fidelity.data import fixture_bundle
from gavd6_sjepa.research_directions.gait_fidelity.evaluation import load_predictions
from gavd6_sjepa.research_directions.gait_fidelity.training import (
    train_phase, predict, paired_indices, repaired_pairs, shuffled_reference_indices, load_model)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.bundle = fixture_bundle(samples=32, people=3)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='gait-training-test-')
        self.tmp_path = Path(self.directory.name)
        self.config = dict(device='cpu',model=dict(width=8,encoder_layers=1,predictor_layers=1,heads=2,patch_size=4,window_size=32),
            training=dict(pretraining_updates=1,readout_updates=1,end_to_end_updates=1,batch_size=4,learning_rate=.0003,
                          repaired_trials=2,log_every=100),measurement=dict(min_frames=16,min_coverage=.8,min_segment_px=2.))

    def tearDown(self):
        self.directory.cleanup()


    def test_complete_registered_matrix_executes_and_frozen_encoders_stay_fixed(self):
        bundle, config, tmp_path = self.bundle, self.config, self.tmp_path
        recipe_path = Path(__file__).resolve().parents[2] / 'src/gavd6_sjepa/research_directions/gait_fidelity/recipes.json'
        plan = json.loads(recipe_path.read_text())
        recipes = plan['recipes'] if isinstance(plan, dict) else plan
        checkpoints = {}
        fitted = []
        for recipe in recipes:
            encoder, policy = recipe['encoder'], recipe['pretraining_mask']
            if recipe['mode'] == 'frozen_readout' and encoder != 'initialized':
                key = (encoder, policy)
                if key not in checkpoints:
                    receipt = train_phase(bundle, recipe, 'pretrain', 17, config, tmp_path / f'pre-{encoder}-{policy}')
                    checkpoints[key] = Path(receipt['checkpoint'])
                upstream = checkpoints[key]
            else:
                upstream = None
            phase = 'readout' if recipe['mode'] == 'frozen_readout' else 'end_to_end'
            receipt = train_phase(bundle, recipe, phase, 17, config, tmp_path / recipe['recipe_id'], upstream)
            assert receipt['status'] == 'complete' and receipt['frozen_encoder_unchanged']
            assert receipt['endpoint_presentations'] == 4
            predictions, indices = load_predictions(receipt)
            assert np.isfinite(predictions).all()
            assert all(bundle.records[i]['split'] == 'development' for i in indices)
            fitted.append(receipt)
        assert len(fitted) == 34 and len(checkpoints) == 11


    def test_repaired_labels_preserve_marginals_and_roles(self):
        bundle = self.bundle
        pairs = paired_indices(bundle.records)
        repaired, audit = repaired_pairs(bundle, pairs, 17, trials=2, tolerance=0., min_frames=16)
        np.testing.assert_array_equal(repaired[:, 0], pairs[:, 0])
        np.testing.assert_array_equal(np.sort(repaired[:, 1]), np.sort(pairs[:, 1]))
        assert audit['endpoint_frequency_preserved']
        assert not audit['tolerance_passed']  # residual reference mismatch is disclosed
        for original, replacement in zip(pairs[:, 1], repaired[:, 1]):
            a, b = bundle.records[original], bundle.records[replacement]
            assert a['source_family_id'] != b['source_family_id']
            for field in ('movement_state','movement_magnitude','camera_id','naming','observation','extractor'):
                assert a[field] == b[field]


    def test_shuffled_reference_is_same_person_different_family(self):
        bundle = self.bundle
        donors = shuffled_reference_indices(bundle.records, 17)
        for i, row in enumerate(bundle.records):
            if row['split'] == 'train':
                other = bundle.records[donors[i]]
                assert row['canonical_person_id'] == other['canonical_person_id']
                assert row['source_family_id'] != other['source_family_id']
                assert row['movement_state'] == other['movement_state']


    def test_inference_has_no_target_dependency_and_resume_identity_is_strict(self):
        bundle, config, tmp_path = self.bundle, self.config, self.tmp_path
        recipe = dict(encoder='initialized',pretraining_mask=None,readout_or_training_objective='base',recipe_id='initialized')
        result = train_phase(bundle, recipe, 'readout', 17, config, tmp_path/'one')
        subset = {k: v[:3] for k, v in bundle.inputs.items()}
        first = predict(SimpleNamespace(inputs=subset), result['checkpoint'], config)
        other = SimpleNamespace(inputs=subset,targets={'xy':np.full_like(subset['xy'],9999)},records=[])
        np.testing.assert_array_equal(first, predict(other, result['checkpoint'], config))
        result2 = train_phase(bundle, recipe, 'readout', 17, config, tmp_path/'one')
        np.testing.assert_array_equal(first, predict(other, result2['checkpoint'], config))
        wrong = deepcopy(config); wrong['training']['learning_rate'] *= 2
        with self.assertRaisesRegex(ValueError, 'Resume rejected'):
            train_phase(bundle, recipe, 'readout', 17, wrong, tmp_path/'one')
        with self.assertRaisesRegex(ValueError, 'cannot consume pretraining'):
            train_phase(bundle, recipe, 'readout', 17, config, tmp_path/'bad', Path(result['checkpoint']))


    def test_invalid_reference_content_cannot_change_teacher_features(self):
        bundle, config = self.bundle, self.config
        from gavd6_sjepa.research_directions.synthetic_training_v2.models import RestorationModel
        from gavd6_sjepa.research_directions.gait_fidelity.training import normalize_inputs
        inputs, normalization = normalize_inputs({k:v[:2] for k,v in bundle.inputs.items()})
        target = normalization.apply(bundle.targets['xy'][:2])
        valid = bundle.targets['valid'][:2].copy(); valid[:,4:12,8] = False
        model = RestorationModel('paired_jepa', config['model']).eval()
        teacher = dict(xy=torch.tensor(target), observed=torch.tensor(valid),confidence=torch.tensor(valid,dtype=torch.float32),
                       timestamps=torch.tensor(inputs['timestamps'],dtype=torch.float32))
        first = model.teacher(teacher)
        teacher['xy'][~teacher['observed']] = 999999
        second = model.teacher(teacher)
        torch.testing.assert_close(first,second,rtol=0,atol=0)

    def test_upstream_hyperparameters_and_pretraining_identity_are_checked(self):
        recipe = dict(encoder='coordinate',pretraining_mask='graph_time',
                      readout_or_training_objective='paired_change',recipe_id='shared')
        pre = train_phase(self.bundle, recipe, 'pretrain', 17, self.config, self.tmp_path/'pre')
        payload = torch.load(pre['checkpoint'], map_location='cpu', weights_only=True)
        self.assertEqual(payload['signature']['objective'], 'base')
        for key, replacement in [('learning_rate', .001), ('mask_fraction', .4)]:
            changed = deepcopy(self.config)
            changed['training'][key] = replacement
            with self.assertRaisesRegex(ValueError, 'Upstream pretraining mismatch: training'):
                train_phase(self.bundle, recipe, 'readout', 17, changed,
                            self.tmp_path/key, Path(pre['checkpoint']))

    def test_all_joint_embeddings_receive_supported_coordinate_gradients(self):
        from gavd6_sjepa.research_directions.synthetic_training_v2.models import RestorationModel
        from gavd6_sjepa.research_directions.synthetic_training_v2.training import _coordinate_patch_loss
        from gavd6_sjepa.research_directions.gait_fidelity.training import normalize_batch, _tensors
        from gavd6_sjepa.research_directions.gait_fidelity.masking import sample_mask, POLICIES
        raw = {k:v[:4] for k,v in self.bundle.inputs.items()}
        for policy in POLICIES:
            hidden = sample_mask(raw['observed'],policy,rng=np.random.default_rng(17))
            normalized,origin,scale,_ = normalize_batch(raw,hidden)
            inputs = _tensors(normalized,'cpu')
            target = torch.tensor((self.bundle.targets['xy'][:4]-origin[:,None,None])/scale[:,None,None,None])
            valid = torch.tensor(self.bundle.targets['valid'][:4])
            model = RestorationModel('coordinate',self.config['model'])
            # The residual layer starts at zero; emulate its first update so
            # the test assesses encoder propagation, rather than that designed
            # zero-gradient initialization step.
            torch.nn.init.normal_(model.readout.network[-1].weight,std=.01)
            mask = torch.tensor(hidden)
            predicted = model(inputs,mask)
            loss,_,supported,_ = _coordinate_patch_loss(predicted,target,valid,mask,model.cfg)
            loss.backward()
            gradient = model.encoder.joints.weight.grad
            assert supported.all() and torch.isfinite(gradient).all()
            assert (gradient.abs().sum(1)>0).all()

    def test_every_objective_receives_identical_finite_run_endpoint_counts(self):
        cfg = deepcopy(self.config)
        cfg['training']['end_to_end_updates'] = 3
        histories = {}
        for objective in ('base','paired_change','per_example_measurement','repaired_change'):
            recipe = dict(encoder='direct',pretraining_mask=None,
                          readout_or_training_objective=objective,recipe_id=objective)
            report = train_phase(self.bundle, recipe, 'end_to_end', 17, cfg, self.tmp_path/objective)
            histories[objective] = json.loads((self.tmp_path/objective/'history.json').read_text())
            assert report['endpoint_presentations'] == sum(len(row['endpoint_indices']) for row in histories[objective])
        for objective, rows in histories.items():
            for expected, actual in zip(histories['base'],rows):
                self.assertEqual(sorted(expected['endpoint_indices']),sorted(actual['endpoint_indices']),objective)
                self.assertEqual(expected['endpoint_multiset_sha256'],actual['endpoint_multiset_sha256'])
        # The control changes pair relationships while retaining actual inputs.
        self.assertTrue(any(a['endpoint_indices'] != b['endpoint_indices']
                            for a,b in zip(histories['base'],histories['repaired_change'])))

    def test_odd_stratum_cycles_keep_batches_closed_and_report_size(self):
        from gavd6_sjepa.research_directions.gait_fidelity.training import (
            _short_derangement, paired_batch_cycles, draw_pair_batch)
        families = ['a','a','b','b','c']
        rng = np.random.default_rng(17)
        perm = _short_derangement(families,rng)
        for i,j in enumerate(perm):
            self.assertNotEqual(families[i],families[j])
        pairs = np.column_stack((np.arange(5),np.arange(5)+5))
        repaired = pairs.copy(); repaired[:,1] = pairs[perm,1]
        cycles = paired_batch_cycles(pairs,repaired)
        self.assertEqual(sorted(map(len,cycles)),[2,3])
        lengths = []
        for _ in range(32):
            draw = draw_pair_batch(cycles,4,rng)
            lengths.append(len(draw))
            np.testing.assert_array_equal(np.sort(pairs[draw].ravel()),np.sort(repaired[draw].ravel()))
        self.assertTrue(max(lengths)<=6 and min(lengths)>=4)
