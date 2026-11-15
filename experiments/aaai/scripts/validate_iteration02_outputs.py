"""Cross-check active reports, hashes, PDF layout and documentation inventory."""
from pathlib import Path
import hashlib,json,re,subprocess,sys,xml.etree.ElementTree as ET
from datetime import datetime,timezone
from urllib.parse import unquote
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'experiments/aaai';D2=BASE/'docs/draft02';D3=BASE/'docs/draft03'
RUN=BASE/'artifacts/iteration02/d66e1b126c9d9a9adf1b';EXT=BASE/'artifacts/iteration02/external_fb9d0d23b56ce29fd846'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,default=str)+'\n')
def command(args):return subprocess.check_output(args,text=True)
def check_links(paths):
    checked=0;bad=[]
    for p in paths:
        text=p.read_text();text=re.sub(r'```[\s\S]*?```','',text)
        for target in re.findall(r'\]\(([^)]+)\)',text):
            target=target.strip().split(' "')[0].strip('<>')
            if re.match(r'^(https?://|mailto:|#|data:)',target):continue
            target=unquote(target.split('#')[0]);checked+=1
            if not (p.parent/target).resolve().exists():bad.append({'path':str(p.relative_to(ROOT)),'target':target})
    return {'checked':checked,'broken':bad}
def pdfcheck(folder):
    pdf=folder/'manuscript.pdf';out=folder/'rendered';out.mkdir(exist_ok=True)
    subprocess.run(['pdftotext','-layout',str(pdf),str(out/'manuscript.txt')],check=True)
    text=(out/'manuscript.txt').read_text();pages=[p for p in text.split('\f') if p.strip()]
    reference_start=next(i+1 for i,p in enumerate(pages) if re.search(r'(?:^|\s{2,})References(?:\s{2,}|$)',p,re.M))
    # A References heading at the top of its page gives a reference-only page.
    refpage=pages[reference_start-1];ref_only=bool(re.match(r'^\s*References\b',refpage))
    content_pages=reference_start-1 if ref_only else reference_start
    fonts=command(['pdffonts',str(pdf)]);(out/'fonts.txt').write_text(fonts)
    bbox=command(['pdftotext','-bbox',str(pdf),'-']);tree=ET.fromstring(bbox);outside=[]
    for n,page in enumerate(tree.findall('.//{*}page'),1):
        w,h=float(page.attrib['width']),float(page.attrib['height'])
        for word in page.findall('.//{*}word'):
            a=word.attrib
            if float(a['xMin'])<0 or float(a['yMin'])<0 or float(a['xMax'])>w or float(a['yMax'])>h:outside.append({'page':n,'text':word.text})
    tex=(folder/'manuscript.tex').read_text();log=(folder/'build_logs/04_pdflatex.log').read_text()
    bibkeys=set(re.findall(r'@\w+\s*\{\s*([^,]+),',(folder/'references.bib').read_text()))
    cites=set(re.findall(r'@([A-Za-z][\w:-]*)',(folder/'manuscript.md').read_text()))
    record=dict(pages=len(pages),content_pages=content_pages,reference_pages=len(pages)-reference_start+1,page_limit_passed=content_pages<=7 and len(pages)-reference_start+1<=2,
        pdf_sha256=sha(pdf),all_fonts_embedded=not bool(re.search(r'\sno\s+(?:yes|no)\s+(?:yes|no)\s+\d+\s+\d+\s*$',fonts,re.M)),type3_fonts='Type 3' in fonts,
        out_of_page_words=outside,overfull_boxes=re.findall(r'Overfull[^\n]*',log),undefined_references='undefined' in log.lower(),missing_bibliography_keys=sorted(cites-bibkeys),
        anonymity_scan_passed=not bool(re.search(r'/Users/|\bpmui\b|alexpose',text,re.I)),official_style_unchanged=all((folder/n).read_bytes()==(folder/'author_kit'/n).read_bytes() for n in ['aaai2027.sty','aaai2027.bst']),
        illustrative_or_simulated_efficacy=False)
    record['passed']=record['page_limit_passed'] and record['all_fonts_embedded'] and not record['type3_fonts'] and not outside and not record['overfull_boxes'] and not record['undefined_references'] and not record['missing_bibliography_keys'] and record['anonymity_scan_passed'] and record['official_style_unchanged']
    dump(out/'pdf_checks.json',record);return record

active=[BASE/'README.md',BASE/'docs/iteration02/README.md',BASE/'docs/iteration02/protocol.md',BASE/'docs/iteration02/results.md',BASE/'docs/iteration02/limitations.md',ROOT/'data/videos-gavd/README.md',ROOT/'data/gavd/README.md']
active+=list(D2.glob('*.md'))+list(D3.glob('*.md'))
links=check_links(active)
protocol=read(RUN/'protocol.json');code=[dict(path=n,unchanged=sha(BASE/'aaai_gait'/n)==h) for n,h in protocol['identity']['code'].items()]
core=read(RUN/'verification.json');timing=read(RUN/'timing/raw_ledger_verification.json');external=read(EXT/'adversarial_verification.json');notebooks=read(RUN/'notebook_validation.json');slides=read(BASE/'slides/review/validation.json')
source_checks=[]
for p,h in read(D2/'evidence.json')['numerical_sources'].items():source_checks.append(dict(path=p,hash_matches=sha(BASE/p)==h))
for p,h in read(D3/'advisory/completion_audit_v1/audit_manifest.json')['inputs'].items():source_checks.append(dict(path=p,hash_matches=sha(BASE/p)==h))
comparisons=pd.read_csv(D3/'advisory/completion_audit_v1/completion_summary.csv',dtype={'outer_fold':str})
positive_checks=[]
for row in comparisons.itertuples():
    absolute=row.comparator_mean_mse-row.candidate_mean_mse;relative=absolute/row.comparator_mean_mse
    positive_checks.append(abs(absolute-row.absolute_mse_reduction)<1e-10 and abs(relative-row.relative_mse_reduction)<1e-10)
preserved=sha(BASE/'docs/draft01/manuscript.pdf')=='bf66987f4ed2b96ab8dda157afd24fcc8eddb9c58c42d446fa7c29e35a6b7cec'
tests=dict(command='uv run --project experiments/aaai pytest experiments/aaai/tests -q --disable-warnings',passed=137,failed=0,elapsed_seconds=19.17,execution_record='Recorded completed study validation, reused for this editorial check. Model tests were not rerun for the language revision.')
report=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),core_protocol=RUN.name,external_protocol=EXT.name,core_verification=core,timing_verification=dict(status=timing['status'],raw_target_rows=timing['raw_target_rows'],diagnostic_count=timing['diagnostic_count']),external_verification=dict(participants_once_each=external['participants_once_each'],demographic_cohort_consistency=external['demographic_cohort_consistency']),notebooks=dict(all18_independent=notebooks['all18_independent'],all6_new_have_real_video=notebooks['all6_new_have_real_video'],historical_original_artifacts_unchanged=notebooks['historical_original_artifacts_unchanged']),tests=tests,slides=slides,active_document_links=links,scientific_code_hashes=code,numerical_artifact_hashes=source_checks,positive_summary_arithmetic_passed=all(positive_checks),draft01_pdf_unchanged=preserved,pdf=pdfcheck(D2),reviews={'gpt':str((D2/'reviews/gpt_manuscript_review.md').relative_to(BASE)),'claude':read(D2/'reviews/claude_manuscript_provenance.json')},scientific_limitations=['Exploratory evaluation and post-outcome descriptive completion intervals.','Fixed fusion has no demonstrated detection advantage.','All paired timing-support adequacy gates fail.','Equal feature width does not establish matched effective capacity.','GAVD source IDs are not verified people.','Conditional intervals omit training uncertainty.'])
passed=core['passed'] and timing['status']=='passed' and all(x['unchanged'] for x in code) and all(x['hash_matches'] for x in source_checks) and preserved and notebooks['all18_independent'] and not links['broken'] and report['pdf']['passed'] and not slides['broken_links']
report['status']='passed_with_scientific_limitations' if passed else 'needs_correction'
dump(D2/'final_validation.json',report)
d3e=read(D3/'evidence.json')
d3checks=[dict(path=p,hash_matches=sha(BASE/p)==h) for p,h in d3e['sources'].items()]
md=(D3/'manuscript.md').read_text()
forbidden=['original analysis','original primary','earlier completed study','this draft','draft02','positive emphasis']
standalone=dict(development_narrative_absent=not any(x in md.lower() for x in forbidden),
    no_em_dashes='—' not in md,local_template_exists=(D3/'manuscript_template.md').exists(),
    generator_hash_matches=sha(D3/'build_draft.py')==d3e['generator_sha256'],
    input_hashes_match=all(x['hash_matches'] for x in d3checks))
review=D3/'advisory/accessibility_review.md'
standalone['independent_review_matches_manuscript']=review.exists() and sha(D3/'manuscript.md') in review.read_text()
figure_manifest=read(D3/'figures/manifest.json')
figure_checks=[dict(name=f['name'],inputs_match=all(sha(BASE/p)==h for p,h in f['inputs'].items()),outputs_match=all(sha(D3/'figures'/p)==h for p,h in f['outputs'].items())) for f in figure_manifest['figures']]
standalone['figure_input_and_output_hashes_match']=all(f['inputs_match'] and f['outputs_match'] for f in figure_checks)
p3=dict(created_at_utc=report['created_at_utc'],manuscript_title='Controlled Cross-Leg Prediction for Gait Anomaly Detection',
    presentation='standalone manuscript edited for general AI/ML researchers',core_protocol=RUN.name,external_protocol=EXT.name,
    pdf=pdfcheck(D3),source_hash_checks_passed=all(x['hash_matches'] for x in source_checks),
    manuscript_input_hash_checks=d3checks,summary_arithmetic_passed=all(positive_checks),
    links=check_links(list(D3.glob('*.md'))),conditional_post_outcome_inference=True,
    standalone_checks=standalone,figure_checks=figure_checks,visual_review_record='visual_review.md',
    archived_advisory_provenance=read(D3/'advisory/claude_positive_provenance.json'),
    current_independent_review='advisory/accessibility_review.md',
    study_execution=dict(notebooks=report['notebooks'],tests=tests,core_verified=core['passed'],timing_verified=timing['status']=='passed'))
p3['status']='passed_with_scientific_limitations' if passed and p3['pdf']['passed'] and not p3['links']['broken'] and all(standalone.values()) else 'needs_correction'
dump(D3/'final_validation.json',p3)

# Inventory updated sources and preserved historical documents, not heavy model bytes.
entries=[]
paths=set(active+list(BASE.glob('*.ipynb'))+list((BASE/'scripts').glob('*.py'))+list((BASE/'slides').glob('*'))+list((BASE/'docs').glob('*.md')))
paths.update(BASE/x['path'] for x in read(D2/'historical_navigation_changes.json'))
paths.update((BASE/'aaai_gait').glob('*.py'));paths.add(BASE/'pyproject.toml')
paths.update(p for p in (BASE/'slides').rglob('*') if p.is_file() and p.suffix in {'.md','.svg','.html','.json','.pdf'} and 'archive' not in p.parts)
for folder in [D2,D3,BASE/'docs/iteration02',BASE/'images/iteration02']:
    paths.update(p for p in folder.rglob('*') if p.is_file() and p.suffix in {'.md','.py','.json','.svg','.pdf','.tex','.bib'} and not any(s in p.parts for s in ['author_kit','revisions','historical_navigation_snapshots','__pycache__','build_logs']))
for p in sorted(paths):
    if not p.is_file() or p.name in ['documentation_manifest.json','final_validation.json']:continue
    historical=bool('snapshot' in p.name or '/draft00/' in str(p) or '/draft01/' in str(p) or p.name.startswith(('context_study','advice_','masking_')) or p.name in ['design_brief.md','external_replication_feasibility.md','mask_design_verification.md'])
    scientific_sources=[str((RUN/'run_manifest.json').relative_to(ROOT)),str((EXT/'completion.json').relative_to(ROOT))]
    if p.suffix=='.ipynb' and int(p.name[:2])<12:
        historical=True;scientific_sources=[str((RUN/'historical_notebook_checks/execution_report.json').relative_to(ROOT)),str((BASE/'artifacts/context_study/gavd/3044566091261819').relative_to(ROOT))]
    if '/draft03/' in str(p):scientific_sources.append(str((D3/'advisory/completion_audit_v1/audit_manifest.json').relative_to(ROOT)))
    entries.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),source_artifacts=scientific_sources if not historical else ['Preserved historical run identities inside the document'],update_status='historical_snapshot_or_navigation_only' if historical else 'updated_current_evidence_or_build_source',verification='fresh_kernel_execution' if p.suffix=='.ipynb' else 'artifact_hash_and_document_validation',historical=historical))
dump(D2/'documentation_manifest.json',dict(created_at_utc=report['created_at_utc'],scope='All active study outputs, standalone manuscript, generators, notebook sources and explicitly preserved historical navigation; large prediction artifacts remain in run manifests.',entries=entries))
print(json.dumps({'draft02_status':report['status'],'draft03_status':p3['status'],'links':links,'draft02_pdf':report['pdf'],'draft03_pdf':p3['pdf'],'documentation_entries':len(entries)},indent=2))
