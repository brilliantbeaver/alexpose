import json
import tempfile
from pathlib import Path
import unittest
from gavd6_sjepa.research_directions.synthetic_training_v2.decisions import load_decision_spec
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import SCHEMA,sha256_file

class DecisionProvenanceTests(unittest.TestCase):
    def test_hash_string_without_artifact_cannot_authorize_source_gate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'spec.json';p.write_text(json.dumps({'calibration_artifact_sha256':'x'}))
            with self.assertRaises(ValueError):load_decision_spec(p)
    def test_calibration_content_candidate_independence_and_margin_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);artifact=root/'calibration.json';spec_path=root/'spec.json'
            values=dict(minimum_people=8,minimum_seeds=3,amplitude_error_max=.01,event_timing_max_s=.04,coordinate_margin=.02,displacement_margin=.05,clean_degradation_max=.01)
            # Software-only fake metadata exercises the verifier; it is never
            # published as a completed human/source calibration artifact.
            calibration=dict(schema=SCHEMA,samples=64,hz=25,evidence_status='source-run',candidate_outputs_used=False,
                reference_manifest_sha256='fixture-reference',reviewer='fixture reviewer',decision_values=values)
            artifact.write_text(json.dumps(calibration))
            spec={**values,'calibration_artifact':'calibration.json','calibration_artifact_sha256':sha256_file(artifact)}
            spec_path.write_text(json.dumps(spec));self.assertEqual(load_decision_spec(spec_path)['minimum_people'],8)
            spec['coordinate_margin']=0;spec_path.write_text(json.dumps(spec))
            with self.assertRaises(ValueError):load_decision_spec(spec_path)
            spec['coordinate_margin']=.02;calibration['candidate_outputs_used']=True;artifact.write_text(json.dumps(calibration));spec['calibration_artifact_sha256']=sha256_file(artifact);spec_path.write_text(json.dumps(spec))
            with self.assertRaises(ValueError):load_decision_spec(spec_path)
            calibration['candidate_outputs_used']=False;artifact.write_text(json.dumps(calibration))
            with self.assertRaises(ValueError):load_decision_spec(spec_path)

if __name__=='__main__':unittest.main()
