"""Build portable document evidence copies with source digests; never fit models."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / 'work/artifacts/iclr-bridge-2026-09-11'
OUT = ROOT / 'docs/studies/iclr/evidence'


def main():
    sources = {
        'cached-panel-report.json': ROOT / 'outputs/iclr-bridge-cached-20260911/reports/panel-report.json',
        'cached-panel-verification.json': WORK / 'future/verification-supplement.json',
        'laterality-aggregate-check.json': WORK / 'laterality/aggregate_recomputation.json',
        'symmetry-calibration.json': WORK / 'laterality/symmetry-calibration.json',
        'confidence-route-diagnostic.json': WORK / 'future/confidence-route-diagnostic.json',
    }
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, source in sources.items():
        content = source.read_bytes()
        (OUT / name).write_bytes(content)
        manifest[name] = {'source': str(source.relative_to(ROOT)),
                          'sha256': hashlib.sha256(content).hexdigest(),
                          'size_bytes': len(content)}
    (OUT / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (OUT / 'README.md').write_text(
        '# Evidence copies for the documents\n\n'
        'These JSON files are byte-identical copies of the measured report, independent '
        'verification, retained laterality arithmetic, synthetic calibration and post-hoc '
        'confidence-route diagnostic. Their source paths and SHA-256 digests are in '
        '`source-manifest.json`. The diagnostic explains a claim limitation and does not '
        'estimate a corrected effect. No new teacher or student-training result is present. '
        'Full models, predictions and bootstrap draws remain in the linked run root. '
        'Regenerate with `.venv/bin/python scripts/research_directions/iclr_bridge/build_evidence.py`.\n')
    print(f'Copied {len(sources)} evidence files with provenance into {OUT}')


if __name__ == '__main__': main()
