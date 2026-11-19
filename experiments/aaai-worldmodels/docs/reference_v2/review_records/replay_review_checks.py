"""Replay an exact exported reviewer script in its original scratch location."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,sys
HERE=Path(__file__).resolve().parent
PROJECT=next(p for p in HERE.parents if (p/'worldmodels/v2_data.py').is_file())
REPOSITORY=PROJECT.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    manifest=json.loads((HERE/'export_manifest.json').read_text())['files']
    names=sorted(n for n in manifest if n.endswith('.py'))
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('script',choices=names);args=parser.parse_args()
    for name in names:
        record=manifest[name];source=HERE/name
        if sha(source)!=record['sha256']:raise ValueError('Exported review source changed: '+name)
        target=PROJECT/record['original_path'];target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha(target)!=record['sha256']:
            raise ValueError('A different scratch script already exists; inspect it before replay: '+str(target))
        if not target.exists():shutil.copyfile(source,target)
    if args.script=='external_input_checks.py':
        # The replay preserves an existing human inspection. It cannot perform
        # fresh visual review of a changed picture while reporting that claim.
        manual=json.loads((HERE/'external_input_checks.json').read_text())
        for name,expected in manual['visual_pages_independently_inspected'].items():
            if sha(PROJECT/name)!=expected:raise ValueError('Human-reviewed QA image changed: '+name)
    target=PROJECT/manifest[args.script]['original_path']
    subprocess.run([sys.executable,str(target)],cwd=REPOSITORY,check=True)
if __name__=='__main__':main()
