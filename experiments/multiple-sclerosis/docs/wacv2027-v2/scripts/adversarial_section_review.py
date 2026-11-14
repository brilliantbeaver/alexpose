"""Challenge the standalone Section 4.1 note in an isolated read-only packet."""
from pathlib import Path
import json, shutil, subprocess, tempfile
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reviews/codex-section-4.1'
REPORT.mkdir(exist_ok=True)
prior=json.loads((ROOT/'reviews/codex-v7/manifest.json').read_text())
target=Path(tempfile.mkdtemp(prefix='wacv-section-review-'))
shutil.copytree(prior['target'],target,dirs_exist_ok=True)
def run(args):
 return subprocess.check_output(args,cwd=target,text=True).strip()
assert not run(['git','status','--porcelain'])
base=run(['git','rev-parse','HEAD'])
for src,name in [(ROOT/'wacv2007-v1-section-4.1-refined.md','section-4.1-refined.md'),(ROOT/'evidence/section-4.1-verification.md','section-4.1-verification.md')]:
 shutil.copyfile(src,target/name)
run(['git','add','section-4.1-refined.md','section-4.1-verification.md'])
run(['git','-c','user.name=Research review snapshot','-c','user.email=snapshot@localhost','-c','commit.gpgsign=false','commit','-qm','Standalone Section 4.1 refinement'])
head=run(['git','rev-parse','HEAD'])
lens='''Review ONLY the new section-4.1-refined.md as a standalone correction of original-draft.txt Section4.1. This follows seven full manuscript revisions, all saved; do not re-review the entire paper. Read the complete new note. Verify against notebook02 source/saved text outputs, sjepa/masking_v2.py and config, augment/train/models as needed, independent section verification and notebook audit. The notebook is source of truth; the512 bank is notebook-defined and5000 separatediagnostic. Challenge1: every mask probability, count, percentage and denominator is correct. Challenge2: target/contextguarantee vs window and timeblockcoverage, overlapbudget andperexample rules accurate. Challenge3: clinical motivation and originalSJEPA attribution are bounded, no unperformedcomparison or temporal benefit. Challenge4: reflection/readout caveats scientificallycorrect and distinct from testleakage. Challenge5: understandable by educatednonMLresearcher with no misleading detectorvisibility usage. Use read-only shell only,noMCP,externaltools,recursiveCodex or edits. Earlier primary-source verification may be assessed from retained records; don't claim new external verification. Report each HOLDS/BROKEN/NOTREACHED with exact file:line; any defect needs consequence and actionablefix. Don't report disclosed unmeasuredlimitations as newdefects. Include linecountandlastline readinganchor. End exactly Verdict: approve or Verdict: needs-attention.'''
framing=Path('/Users/pmui/.codex/skills/codex-adversarial-review/references/adversarial-framing.md').read_text()
prompt=framing.replace('{{BASE}}',base).replace('{{HEAD}}',head).replace('{{LENS}}',lens)
(REPORT/'prompt.md').write_text(prompt)
manifest={'target':str(target),'base':base,'head':head,'model':'gpt-6-astra','effort':'high','helper_status':'isolated committed packet; suite freeze helpers unavailable','same_model_family_review':True}
(REPORT/'manifest.json').write_text(json.dumps(manifest,indent=2))
with (REPORT/'prompt.md').open() as src,(REPORT/'review.log').open('w') as log:
 result=subprocess.run(['codex','exec','--sandbox','read-only','--cd',str(target),'-m','gpt-6-astra','-c','model_reasoning_effort=high','-o',str(REPORT/'review.md'),'--','-'],stdin=src,stdout=log,stderr=subprocess.STDOUT)
manifest.update(returncode=result.returncode,head_unchanged=run(['git','rev-parse','HEAD'])==head,clean_after=not run(['git','status','--porcelain']),report_written=(REPORT/'review.md').exists())
(REPORT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
assert result.returncode==0 and manifest['head_unchanged'] and manifest['clean_after'] and manifest['report_written']
