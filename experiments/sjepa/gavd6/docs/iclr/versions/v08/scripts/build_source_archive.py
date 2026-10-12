"""Portable typesetting and figure/table rebuild sources, without private ledgers."""
from pathlib import Path
import json,zipfile
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[3]
readme='''# Paper v08 portable source

Evaluating JEPA-Inspired Motion Representations through Geometry and Gait Asymmetry

The PDF has nine main pages, followed by disclosure/references and an appendix.
Compile from this directory with:

    tectonic paper-v08.tex --outdir . --keep-logs --keep-intermediates

To regenerate all eight editable figures and numerical tables from the included
audited data, use the package versions in requirements.txt, then:

    python scripts/build_figures.py
    python scripts/build_tables.py
    python scripts/build_case_figure.py --from-audit
    python scripts/build_summary_figures.py --from-audit

The concise appendix is the default. supplement/technical-details.tex preserves
the full technical reference and complete inventories; supplement/README.md
explains how to typeset that optional alternative.

The figure input CSV and analysis summaries are included. The generated tables
and figures are already present, so Python is optional for typesetting alone.
SVG text remains editable; PDF text and line art are vector with embedded fonts.

This archive supports paper/figure/table reconstruction, not end-to-end training
or the complete upstream evidence audit. That audit is delivered separately in
the research records and requires the original experimental results,
configurations, and training logs. Machine-specific logs and internal review
records are excluded from this portable source archive. No raw frames, per-example pose
predictions, complete checkpoints, new training or confirmation data are implied.

All new strata, cost sensitivity and expanded intervals are exploratory analyses
of the same 14 development people and three fitted seeds. All selected groups
are retained in the data tables. Human authors have confirmed their original
ideas and initial drafts, final edits, verification of the text, reported results
and references, and approval of the manuscript. The AI-use statement separately
discloses AI assistance. Actual conference submission remains a separate action.
'''
files=[]
for name in ['paper-v08.tex','appendix.tex','references.bib','requirements.txt']:
    files.append((HERE/name,name))
for dirname,patterns in [('figures',['*.pdf','*.svg','provenance.json']),('tables',['*.tex']),('evidence',['*.csv','provenance.json','case-figure-provenance.json','summary-figure-provenance.json','figure1-design-provenance.json']),('template',['*.sty','*.bst']),('supplement',['*.tex','*.md'])]:
    for pat in patterns:
        for p in sorted((HERE/dirname).glob(pat)):files.append((p,str(p.relative_to(HERE))))
for name in ['build_figures.py','paper_style.py','build_tables.py','build_case_figure.py','build_summary_figures.py']:
    files.append((HERE/'scripts'/name,'scripts/'+name))
raw='outputs/iclr/jepa-response/evaluation/per-person.csv'
files.append((ROOT/raw,'inputs/'+raw))
for p,name in files:
    if p.suffix in ['.tex','.bib','.py','.csv','.json','.svg']:
        s=p.read_text()
        for private in ['/Users/','/home/','theodoremui','tedmui','haic.stanford']:
            assert private not in s,(name,private)
out=HERE/'paper-v08-source.zip'
with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    z.writestr('README.md',readme)
    for p,name in files:z.write(p,name)
(HERE/'qa/source-archive.json').write_text(json.dumps({'archive':out.name,'entries':len(files)+1,'bytes':out.stat().st_size,'scope':'Typesetting and all figure/table reconstruction from included audited data; complete upstream audit remains repository-based.','private_path_scan':'passed','file_names':[name for _,name in files]},indent=2)+'\n')
print(out.name,len(files)+1,'entries',out.stat().st_size,'bytes')
