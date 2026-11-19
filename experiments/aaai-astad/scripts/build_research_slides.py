"""Build the measured neuromechanics study deck with local vector assets.

The old tutorial deck is archived once. Standard-library generation writes SVG,
HTML and editable Markdown. --pdf requires rsvg-convert and pdfunite; --render
also requires pdftoppm and Pillow. No remote asset or browser service is used.
Completed primary, sensitivity and control artifacts populate all measured panels.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import textwrap
import xml.etree.ElementTree as ET
from html import escape

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]
OUT = BASE / 'slides'
IMAGES = OUT / 'images'
DRAFT = BASE / 'docs/draft00'
W, H = 1280, 720
INK, TEAL, BLUE, ORANGE, MUTED = '#172e47', '#087f82', '#315cb5', '#ad521b', '#506478'
PALE, LINE, WHITE = '#f2f6fa', '#d9e3ec', '#ffffff'
SOURCES = {
 'workshop': ('ASTAD4 call, checked November 15, 2026', 'https://ismart.ece.mcgill.ca/astad4/'),
 'sjepa': ('Skeleton S-JEPA, ECCV 2024', 'https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf'),
 'mamp': ('MAMP and its official uniform-mask branch', 'https://github.com/maoyunyao/MAMP/blob/main/model_mamp/transformer.py'),
 'vicreg': ('VICReg', 'https://arxiv.org/abs/2105.04906'),
 'gait': ('GAVD dataset paper', 'https://arxiv.org/abs/2407.04190'),
 'clinical': ('Primary clinical evidence and limitations', '../docs/draft00/advisory/neuromechanics_clinical_evidence.md'),
 'roadmap': ('Implemented study and evaluation protocol', '../docs/context_study_protocol.md'),
 'biology': ('Biological motivation and world-model proposal', '../docs/draft00/neurological_research_direction.md'),
 'old': ('Completed earlier three-seed results', '../docs/results.md'),
 'real': ('Completed source-held-out context results', '../docs/context_study_results.md'),
 'run': ('Primary run and provenance', '../artifacts/context_study/gavd/3044566091261819/summary.json'),
 'long_run': ('4096-head-update sensitivity and provenance', '../artifacts/context_study/gavd/eba2bd5a6e05efa1/summary.json'),
 'source_controls': ('Supplementary motion and observation controls', '../artifacts/context_study/gavd/3044566091261819/source_controls/7407af320461a092/summary.json'),
 'development': ('Real development and teacher diagnostics', '../artifacts/context_study/gavd/3044566091261819/fold_0/development_controls/protocol.json'),
 'paper': ('Manuscript source evidence and claim limits', '../docs/draft01/primary_source_claim_map.md'),
 'fsgait': ('FSGait, ACCV 2024', 'https://link.springer.com/chapter/10.1007/978-981-96-0960-4_19'),
 'seeker': ('SeeKer, ICCV 2025', 'https://openaccess.thecvf.com/content/ICCV2025/html/Delic_Sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_video_ICCV_2025_paper.html'),
}


def txt(x, y, value, size=24, color=INK, weight='normal', anchor='start'):
    return f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>'


def paragraph(x, y, value, width=1000, size=24, color=INK, weight='normal', leading=1.38):
    # Arial's average character width is conservatively estimated for clean wraps.
    lines = textwrap.wrap(str(value), width=max(12, int(width/(size*.55))), break_long_words=False)
    return ''.join(txt(x, y+i*size*leading, line, size, color, weight) for i, line in enumerate(lines))


def rect(x, y, w, h, fill=WHITE, stroke=LINE, radius=14):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'


def line(x1, y1, x2, y2, color=TEAL, width=3, dash=None):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+'/>'


def arrow(x1, y1, x2, y2, color=TEAL):
    return f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="3" marker-end="url(#arrow)"/>'


def card(x, y, w, h, title, body, color=TEAL, size=24):
    title_lines = textwrap.wrap(title, width=max(12, int((w-48)/(24*.55))), break_long_words=False)
    title_svg = ''.join(txt(x+24,y+38+i*29,t,24,color,'bold') for i,t in enumerate(title_lines))
    return rect(x,y,w,h)+title_svg+paragraph(x+24,y+76+(len(title_lines)-1)*29,body,w-48,size)


def formula(x, y, w, value, size=28):
    return rect(x,y,w,74,fill='#e7f1f1',stroke='#b6d9da')+txt(x+w/2,y+47,value,size,TEAL,'bold','middle')


def embed(path, x, y, w, h):
    root = ET.parse(path).getroot()
    vb = root.attrib.get('viewBox','').split()
    if not vb:
        raise ValueError(f'Missing vector viewBox: {path}')
    vw,vh=float(vb[2]),float(vb[3]);scale=min(w/vw,h/vh)
    left=x+(w-vw*scale)/2;top=y+(h-vh*scale)/2
    body=''.join(ET.tostring(child,encoding='unicode') for child in root)
    return f'<g transform="translate({left},{top}) scale({scale})">{body}</g>'


def flow(stages, y=230, height=245, footer=None):
    n=len(stages);gap=32;wid=(1160-gap*(n-1))/n;body=''
    for i,(title,text) in enumerate(stages):
        x=60+i*(wid+gap);body+=card(x,y,wid,height,title,text,size=23)
        if i<n-1:body+=arrow(x+wid+4,y+height/2,x+wid+gap-5,y+height/2)
    if footer:body+=paragraph(62,y+height+58,footer,1155,24)
    return body


def table(headers, rows, widths, y=210, row_h=51, size=23):
    body=rect(60,y,1160,row_h,fill=INK,stroke=INK,radius=7)
    x=60
    for name,width in zip(headers,widths):body+=txt(x+17,y+34,name,size,WHITE,'bold');x+=width
    for r,row in enumerate(rows):
        yy=y+(r+1)*row_h
        body+=rect(60,yy,1160,row_h,fill=WHITE if r%2==0 else PALE,stroke=WHITE,radius=0)
        x=60
        for val,width in zip(row,widths):body+=txt(x+17,yy+34,val,size);x+=width
    return body


def archive():
    target=OUT/'archive/pre_neuromechanics'
    if target.exists():return
    target.mkdir(parents=True)
    for name in ('index.html','tutorial.md','tutorial.pdf','README.md'):
        src=OUT/name
        if src.exists():
            if src.suffix in ('.html','.md'):
                (target/name).write_text(src.read_text().replace('../images/','images/'))
            else:shutil.copy2(src,target/name)
    if (BASE/'images').exists():shutil.copytree(BASE/'images',target/'images')
    (target/'ARCHIVE.md').write_text('# Earlier tutorial deck\n\nSnapshot preserved before the neuromechanics study revision. This deck covers the original five-class workflow and its prior scope. The active deck is two directories above.\n')


def census():
    records=[];all_sources=set();norm_sources=set();all_rows=0
    for folder in sorted((ROOT/'data/gavd').iterdir()):
        if not folder.is_dir() or folder.name=='source':continue
        paths=sorted(folder.glob('*.csv'))
        if not paths:continue
        sources=set();nrows=0
        for path in paths:
            with path.open(newline='') as f:
                for row in csv.DictReader(f):
                    sources.add(row['id']);nrows+=1
        all_sources|=sources;all_rows+=nrows
        if folder.name=='normal':norm_sources=sources
        records.append(dict(label=folder.name,sequences=len(paths),sources=len(sources),rows=nrows))
    return dict(labels=records,sequences=sum(r['sequences'] for r in records),sources=len(all_sources),normal_sources=len(norm_sources),abnormal_sources=len(all_sources-norm_sources),rows=all_rows)


def large_figure(name):
    """Presentation adaptations of the eight cited document SVGs.

    Each original remains available as a full vector in the HTML speaker panel.
    These adaptations increase label sizes rather than shrinking a document page.
    """
    if name == 'neurological_chain':
        stages=[('Origin','A neural or muscle change alters capacity.'),
                ('Movement limits','Control, force or joint range can change.'),
                ('Walking strategy','The person adapts timing and posture.'),
                ('Observed pose','Camera projection and tracking add variation.')]
        body=flow(stages,y=225,height=240)
        return body+formula(140,520,1000,'The same visible movement can have several hidden explanations.',25)
    if name == 'condition_routes':
        rows=[('Stroke','Brain injury can alter selective control.','Leg contributions may differ.'),
              ('Parkinson’s','Movement scaling and automaticity can change.','Amplitude and coordination may separate.'),
              ('Cerebral palsy','An early brain disturbance interacts with growth.','Knee, foot and trunk patterns vary.'),
              ('Myopathic gait','Muscle disorders can reduce force capacity.','Trunk and leg strategies may change.')]
        body=rect(60,205,1160,48,fill=INK,stroke=INK)
        for x,label in [(78,'Condition family'),(365,'Origin and limitation'),(820,'Possible visible effect')]:body+=txt(x,237,label,23,WHITE,'bold')
        for i,(label,cause,effect) in enumerate(rows):
            y=262+i*85
            body+=rect(60,y,1160,78,fill=PALE if i%2==0 else WHITE,stroke=LINE,radius=8)
            body+=txt(78,y+45,label,25,TEAL,'bold')
            body+=paragraph(365,y+30,cause,400,23)
            body+=paragraph(820,y+30,effect,370,23)
        return body
    if name == 'predictable_patterns':
        body=''
        for i,(title,phase,amp) in enumerate([('Familiar alternation',math.pi,1),('Changed timing',.6,1),('Tightly linked',0,.9)]):
            x=70+i*400;y=310;w=340;h=200
            body+=txt(x+w/2,250,title,27,INK,'bold','middle')
            body+=line(x,y+h/2,x+w,y+h/2,LINE,1)+line(x,y-12,x,y+h+12,MUTED,1)
            for color,shift,a,dashed in [(BLUE,0,1,False),(ORANGE,phase,amp,True)]:
                points=[]
                for k in range(121):
                    t=k/120*math.pi*4
                    points.append((x+k/120*w,y+h/2-a*math.sin(t+shift)*h*.43))
                path=' '.join(('M' if k==0 else 'L')+f'{px:.2f},{py:.2f}' for k,(px,py) in enumerate(points))
                body+=f'<path d="{path}" fill="none" stroke="{color}" stroke-width="4"'+(' stroke-dasharray="9 7"' if dashed else '')+'/>'
            body+=txt(x+w/2,558,'Time',24,MUTED,anchor='middle')
        body+=line(420,597,470,597,BLUE,4)+txt(482,605,'Signal A',22)
        body+=line(700,597,750,597,ORANGE,4,'8 6')+txt(762,605,'Signal B',22)
        return body
    if name == 'neuromechanics_experiment':
        xs=[65,470,875];w=340;h=150;body=''
        top=[('Fit normal motion.','Use only normal training sources.'),('Sample uniform masks.','Keep the same hidden joint patches.'),('Compare context.','Own, patch-local and full branches.')]
        bottom=[('Score outer-test sources.','Normal versus all other folders.'),('Fit a normal reference.','Use only inner normal sources.'),('Compare target types.','Coordinates and frozen JEPA features.')]
        for i,(title,desc) in enumerate(top):body+=card(xs[i],215,w,h,title,desc,size=22)
        for i,(title,desc) in enumerate(bottom):body+=card(xs[i],445,w,h,title,desc,size=22)
        for i in range(2):body+=arrow(xs[i]+w+5,290,xs[i+1]-8,290)
        body+=arrow(1045,372,1045,434)
        for i in range(2):body+=arrow(xs[i+1]-7,520,xs[i]+w+8,520)
        return body
    if name == 'support_profile':
        body='';x0=315;x1=1160;center=(x0+x1)/2;scale=150
        body+=rect(center-.7*scale,230,1.4*scale,305,fill='#e4f4ef',stroke='#e4f4ef',radius=0)
        body+=line(center,230,center,535,TEAL,2)
        for i,(label,value) in enumerate([('Left-leg targets',-1.8),('Right-leg targets',.1),('Arm targets',1.6)]):
            y=290+i*95
            body+=txt(70,y+8,label,27,INK,'bold')+line(center,y,center+value*scale,y,ORANGE,4)
            xx=center+value*scale
            body+=f'<path d="M{xx},{y-10} L{xx+10},{y} L{xx},{y+10} L{xx-10},{y} Z" fill="{ORANGE}"/>'
        for v in [-2,-1,0,1,2]:body+=txt(center+v*scale,574,str(v),23,MUTED,anchor='middle')
        body+=txt(x0,214,'Less help',24,BLUE)+txt(x1,214,'More help',24,BLUE,anchor='end')
        body+=txt(center,614,'Illustrative standardized departure from normal',23,MUTED,anchor='middle')
        return body
    if name == 'intervention_controls':
        rows=[('Shift a complete limb.','Change relative timing.','Match the common interior support.'),
              ('Shift all limbs together.','Change the crop or global phase.','Keep the same window boundaries.'),
              ('Use patch-local context.','Restrict other joints to the target patch.','Separate local from longer context.'),
              ('Damage observations.','Add noise or delete observations.','Track both score change and coverage.')]
        body=rect(60,210,1160,48,fill=INK,stroke=INK)
        for x,label in [(78,'Paired intervention'),(460,'What changes?'),(855,'What is controlled?')]:body+=txt(x,242,label,23,WHITE,'bold')
        for i,(title,change,control) in enumerate(rows):
            y=266+i*82
            body+=rect(60,y,1160,74,fill=PALE if i%2==0 else WHITE,stroke=LINE,radius=8)
            body+=paragraph(78,y+31,title,335,23,TEAL,'bold')
            body+=paragraph(460,y+31,change,350,23)
            body+=paragraph(855,y+31,control,335,23)
        return body
    if name == 'physical_world_model':
        body=card(65,225,330,190,'A controller selects action.','Its choice depends on state, task and control parameters.',BLUE,23)
        body+=card(475,225,330,190,'The body produces motion.','Muscles, geometry and contact determine the next physical state.',TEAL,23)
        body+=card(885,225,330,190,'A camera measures it.','Projection, occlusion and the pose estimator produce observed keypoints.',ORANGE,23)
        body+=arrow(403,320,465,320)+arrow(813,320,875,320)
        body+=f'<path d="M640,423 L640,465 L230,465 L230,424" stroke="{BLUE}" stroke-width="3" fill="none" marker-end="url(#arrow)"/>'
        body+=txt(420,494,'State feedback',23,BLUE,anchor='middle')
        body+=formula(205,535,870,'External measurements are needed to validate hidden components.',24)
        return body
    if name == 'mechanism_ambiguity':
        body=card(65,210,315,155,'Mechanism A','For example, reduced push-off capacity.',BLUE,23)
        body+=card(65,415,315,155,'Mechanism B','For example, restricted joint range.',BLUE,23)
        body+=card(485,305,330,180,'Similar projected motion','Video alone may not distinguish the candidates.',TEAL,23)
        body+=card(920,305,300,180,'Retain uncertainty.','Report compatible explanations rather than one cause.',ORANGE,23)
        body+=arrow(385,285,475,355)+arrow(385,485,475,430)+arrow(823,395,911,395)
        body+=paragraph(485,550,'Then predict which extra measurement would distinguish the explanations.',720,26,TEAL,'bold')
        return body
    raise ValueError(name)


class Deck:
    def __init__(self):self.slides=[]
    def add(self,title,lede,body,*,tag='STUDY PROTOCOL',notes='',refs=('roadmap',),takeaway=''):
        self.slides.append(dict(title=title,lede=lede,body=body,tag=tag,notes=notes,refs=list(refs),takeaway=takeaway))
    def figure(self,title,lede,name,*,notes='',refs=('biology',),tag='RESEARCH MOTIVATION',takeaway=''):
        self.add(title,lede,large_figure(name),tag=tag,notes=notes,refs=refs,takeaway=takeaway)
        self.slides[-1]['source_figure'] = f'source_figures/{name}.svg'


def read_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def load_study_results(run_dir, sensitivity_dir):
    """Read completed, real artifacts; no metric is inferred from a checkpoint."""
    paths = []
    def read(path):
        path=Path(path);paths.append(path)
        return json.loads(path.read_text()) if path.suffix=='.json' else read_rows(path)
    def study(path):
        path=Path(path).resolve()
        summary=read(path/'summary.json');protocol=read(path/'protocol.json')
        verification=read(path/'verification.json')
        if not verification.get('passed'):raise ValueError('Artifact verification did not pass')
        if summary['status']!='completed' or summary.get('synthetic') or protocol['synthetic']:
            raise ValueError(f'Results must be completed real studies: {path}')
        if summary['completed_folds']!=[0,1,2,3]:raise ValueError('Four folds are required')
        metrics=read(path/'ranking_metrics.json')['methods']
        coverage=read(path/'evaluation_coverage.json')
        expected=summary['cohort']['sources']
        if coverage['common_sources']!=expected:raise ValueError('Unexpected incomplete common cohort')
        required=[f'{method}_{kind}_{budget}_profile_score'
                  for method,kinds in [('trained',('coordinate','latent')),('untrained',('coordinate','latent')),('ridge',('coordinate',))]
                  for kind in kinds for budget in ('all','budget4')]
        for key in required:
            if key not in metrics or len(metrics[key]['folds'])!=4:raise ValueError(f'Missing declared result cell {key}')
        histories=[];conditions=[]
        for fold in range(4):
            histories.append(read(path/f'fold_{fold}/trained_history.json'))
            conditions.append(read(path/f'fold_{fold}/condition_metrics.json')['metrics'])
        return dict(path=str(path),summary=summary,protocol=protocol,verification=verification,metrics=metrics,
                    bootstrap=read(path/'primary_paired_bootstrap.json'),histories=histories,conditions=conditions)
    primary=study(run_dir);longer=study(sensitivity_dir) if sensitivity_dir else None
    p=Path(primary['path'])
    manifest=read(p/'fresh_manifest.csv');before=read(p/'window_index_before_profile_qc.csv')
    media=read(BASE/'artifacts/context_study/ingestion/media_ledger.csv')
    source_controls=p/'source_controls/7407af320461a092'
    controls=dict(summary=read(source_controls/'summary.json'),protocol=read(source_controls/'protocol.json'),
                  metrics=read(source_controls/'ranking_metrics.json')['methods'],
                  bootstrap=read(source_controls/'paired_bootstrap.json'))
    if controls['summary']['status']!='completed' or controls['summary']['n_primary_common_sources']!=primary['summary']['cohort']['sources']:
        raise ValueError('Source controls are incomplete')
    dev=p/'fold_0/development_controls'
    components=read(dev/'teacher_decomposition/source_joint_components.csv')
    teacher=dict(protocol=read(dev/'teacher_decomposition/protocol.json'),
                 sources=len({r['source_id'] for r in components}),source_joint_rows=len(components),
                 predictor_own_abs_max=max(abs(float(r['predictor_own'])) for r in components),
                 teacher_own_min=min(float(r['teacher_own']) for r in components),
                 teacher_own_max=max(float(r['teacher_own']) for r in components))
    result=dict(primary=primary,longer=longer,controls=controls,development=read(dev/'protocol.json'),teacher=teacher,
                matched_intervals=read(p/'matched_comparator_intervals.csv'),
                reference_ranges=read(p/'reference_loo_auroc_ranges.csv'),
                prior=read(p/'prior_exploration_sources.csv'),
                census=dict(annotation_sources=len({r['source_id'] for r in manifest}),annotation_sequences=len(manifest),
                    normal_annotation_sources=len({r['source_id'] for r in manifest if r['label']=='normal'}),
                    media=sum(r['status'] in ('downloaded','verified_present') for r in media),
                    new_media=sum(r['status']=='downloaded' for r in media),pose_windows=len(before),
                    pose_sequences=len({r['sequence_id'] for r in before}),pose_sources=len({r['source_id'] for r in before}),
                    pose_normal_sources=len({r['source_id'] for r in before if r['label']=='normal'})))
    if longer and primary['summary']['cohort']!=longer['summary']['cohort']:raise ValueError('Sensitivity cohort differs')
    result['artifact_hashes']={str(path.relative_to(BASE)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return result


def metric(result,key,longer=False):
    return result['longer' if longer else 'primary']['metrics'][key]['primary_auroc']


def point_bars(items,y=230,row_h=68,x=455,width=670):
    """Zero-based AUROC bars, with the chance reference distinct from intervals."""
    body=''
    for value in (0,.25,.5,.75,1):
        xx=x+value*width
        body+=line(xx,y-25,xx,y+row_h*len(items)-18,LINE,1,'5 5' if value==.5 else None)
        body+=txt(xx,y+row_h*len(items)+12,f'{value:.2f}',19,MUTED,anchor='middle')
    for i,(label,value,color) in enumerate(items):
        yy=y+i*row_h
        body+=txt(65,yy+22,label,24,INK,'bold')
        body+=rect(x,yy, value*width,31,fill=color,stroke=color,radius=4)
        body+=txt(x+value*width+12,yy+24,f'{value:.3f}',24,color,'bold')
    return body


def add_real_result_slides(d,result):
    p=result['primary'];b=p['bootstrap'];lo,hi=b['interval_95']
    profile=metric(result,'trained_latent_all_profile_score');raw=metric(result,'trained_latent_all_raw_full')
    body=point_bars([('Latent benefit profile',profile,TEAL),('Raw full-context KL',raw,BLUE)],y=220,row_h=68)
    body+=txt(65,440,'Paired AUROC difference',25,INK,'bold')
    xmin,xmax=-.1,.3;x0=465;wide=650
    to_x=lambda v:x0+(v-xmin)/(xmax-xmin)*wide
    body+=line(to_x(0),420,to_x(0),530,MUTED,2,'5 5')
    body+=line(to_x(lo),475,to_x(hi),475,TEAL,6)
    for v in (lo,hi):body+=line(to_x(v),462,to_x(v),488,TEAL,3)
    body+=f'<circle cx="{to_x(b["point_difference"])}" cy="475" r="8" fill="{TEAL}"/>'
    for v in (-.1,0,.1,.2,.3):body+=txt(to_x(v),552,f'{v:+.1f}',21,MUTED,anchor='middle')
    body+=txt(65,598,f'Difference {b["point_difference"]:+.3f}; conditional 95% interval [{lo:.3f}, {hi:.3f}].',27,TEAL,'bold')
    d.add('The primary result does not establish an advantage.',
          'All 251 sources are paired. Each fold contributes in proportion to its normal count.',body,
          tag='MEASURED: PRIMARY 512-UPDATE STUDY',refs=('real','run'),
          notes=f"Run {p['summary']['run_name']}. Raw and profile scores use the same 12 readout joints, support and source aggregation. The point estimates are normal-count-weighted within-fold AUROCs. The interval is for their paired difference, not either bar. It conditions on fixed models, references and folds, with 1,000 source bootstrap replicates. Only 23 normal test sources occur across the four folds. The interval contains zero.",
          takeaway='The higher point estimate is inconclusive. It is not evidence of a demonstrated learned advantage.')
    rows=[]
    for label,key in [('Benefit profile','trained_latent_all_profile_score'),('Full-error distance','trained_latent_all_full_distance'),('Own-error distance','trained_latent_all_own_distance'),('Random-backbone profile','untrained_latent_all_profile_score')]:
        rows.append((label,f'{metric(result,key):.3f}'))
    body=table(['Score on the same cohort','AUROC'],rows,[830,330],y=218,row_h=61,size=26)
    c=result['matched_intervals'][0]
    body+=paragraph(65,568,f'Profile minus full-error distance: {float(c["Difference"]):+.3f}, with conditional interval [{float(c["CI_low"]):.3f}, {float(c["CI_high"]):.3f}].',1150,24)
    d.add('Constituent scores have higher point estimates.',
          'Two-sided reference calibration can explain a gain over an uncalibrated raw loss.',body,
          tag='MEASURED: MATCHED ALTERNATIVE EXPLANATIONS',refs=('real','run'),
          notes='The own/full distance controls preserve the same three regions, standardization and normal reference fitting. Their higher point estimates preclude attributing the raw-score difference specifically to benefit subtraction. Random backbone means no JEPA pretraining but identically fitted heads; its latent targets differ from trained targets. All supporting comparisons are exploratory and unadjusted for multiplicity.',
          takeaway='Neither subtraction nor self-supervised representation learning has an established advantage here.')
    body=point_bars([('Learned coordinates',metric(result,'trained_coordinate_all_profile_score'),TEAL),
                     ('Random-backbone coordinates',metric(result,'untrained_coordinate_all_profile_score'),ORANGE),
                     ('Fixed ridge coordinates',metric(result,'ridge_coordinate_all_profile_score'),BLUE)],y=240,row_h=87,x=505,width=600)
    body+=paragraph(65,585,'The random-backbone coordinate control shares the same observable target.',1150,24)
    d.add('Coordinate targets do not establish a JEPA gain either.',
          'The local-target control removes the contextual teacher from the prediction answer.',body,
          tag='MEASURED: LOCAL-TARGET COMPARISON',refs=('real','run'),
          notes='All values are profile-distance AUROCs. Coordinate MSE and latent KL have different units; their raw losses are not compared. Ridge uses fixed alpha 10 and 256 source-balanced draws, with different parameter sharing and training support. This limited ridge setting is not an optimization-equated baseline. No external published anomaly method has been validated on this protocol.',
          takeaway='A useful coordinate score alone would not prove that learned JEPA features are necessary.')
    ranges={r['target_type']:r for r in result['reference_ranges']};r=ranges['latent']
    body=card(65,225,545,245,'All eligible target patches',f'AUROC {profile:.3f}. Sources contribute all eligible targets under the fixed masks.',TEAL,26)
    body+=card(670,225,545,245,'Four target patches per joint',f'AUROC {metric(result,"trained_latent_budget4_profile_score"):.3f}. The target-count restriction changes the result.',ORANGE,26)
    body+=formula(90,515,1100,f'Leave-one-reference-out AUROC range: {float(r["minimum"]):.3f} to {float(r["maximum"]):.3f}',26)
    d.add('Observation support and the normal reference matter.',
          'Five inner normal sources define each fitted reference.',body,
          tag='MEASURED: SUPPORT AND REFERENCE SENSITIVITY',refs=('real','run'),
          notes='The reference range combines the smallest or largest fold AUROC after omitting one reference source per fold; it is a perturbation range, not a confidence interval. Models remain fixed. Four distinct annotated patches per joint do not equalize context, overlap exposures, recording duration or precision. Reference source omissions do not create independent studies.',
          takeaway='The primary profile is sensitive to support and reference choices. Report both alongside its point estimate.')
    cm=result['controls']['metrics']
    body=table(['Control','Fitting labels','AUROC'],[
        ('Simple motion distance','Normal only',f'{cm["motion_normal_distance"]["primary_auroc"]:.3f}'),
        ('Visibility and view distance','Normal only',f'{cm["nuisance_normal_distance"]["primary_auroc"]:.3f}'),
        ('Visibility and view RF','Normal + abnormal',f'{cm["nuisance_supervised_probability"]["primary_auroc"]:.3f}')],[575,370,215],y=240,row_h=72,size=24)
    mb=result['controls']['bootstrap']['motion_normal_distance']
    body+=paragraph(65,574,f'Profile minus motion distance: {mb["point_difference"]:+.3f}; conditional interval [{mb["interval_95"][0]:.3f}, {mb["interval_95"][1]:.3f}].',1150,24)
    d.add('A simple motion control is competitive.',
          'Source controls use the same 251 sources and outer source roles.',body,
          tag='MEASURED: SUPPLEMENTARY SOURCE CONTROLS',refs=('source_controls','real'),
          notes='Motion uses normal-source-fitted standardized motion features and covariance distance. Nuisance features represent availability and view. The nuisance RF receives binary abnormal training labels and is a separately supervised shortcut diagnostic. Its weak result does not prove that the neural score is free of unmeasured acquisition confounding. Four test sources have views absent from the normal fitting vocabulary. These are unadjusted supplementary comparisons.',
          takeaway='Weak nuisance prediction cannot certify absence of confounding; motion is a substantive simpler comparator.')
    cats=p['metrics']['trained_latent_all_profile_score']['categories']
    labels=[('exercise','Exercise'),('stroke','Stroke'),('myopathic','Myopathic'),('parkinsons','Parkinsons'),('cerebral palsy','Cerebral palsy')]
    rows=[(label,str(cats[key]['n_category_sources']),f'{cats[key]["normal_weighted_auroc"]:.3f}') for key,label in labels]
    body=table(['Folder comparison against normal','Abnormal sources','AUROC'],rows,[660,300,200],y=215,row_h=59,size=24)
    body+=paragraph(65,610,'These are dependent exploratory category descriptions, not diagnostic sensitivities.',1150,23)
    d.add('Exercise separates more strongly than the focal families.',
          'The primary endpoint includes every non-normal folder, including nonpathological circumstances.',body,
          tag='MEASURED: DESCRIPTIVE CATEGORY ANALYSIS',refs=('real','run'),
          notes='All five displayed categories have four supported folds. Every category comparison reuses normal sources, and mixed-category abnormal sources can also be reused. Category sample sizes are small. The observed variation does not identify an affected neural system, and a category AUROC is not a clinical diagnostic accuracy. Other folders and sparse categories remain in the full report.',
          takeaway='The all-folder result cannot be presented as neurological disease detection.')
    rows=[]
    for fold,c in enumerate(p['conditions']):
        rows.append((str(fold+1),str(c['random_forest']['n_test_sources']),f'{c["random_forest"]["macro_f1"]:.3f}',f'{c["normal_jepa_probe"]["macro_f1"]:.3f}'))
    body=table(['Outer fold','Test sources','Random Forest','Frozen JEPA probe'],rows,[220,250,310,380],y=235,row_h=65,size=25)
    body+=paragraph(65,590,'Macro-F1 is numerically higher for the Random Forest in every fold.',1150,25)
    d.add('The supervised capstone favors the Random Forest.',
          'Normal plus four focal labels define a separate five-class recognition task.',body,
          tag='MEASURED: SEPARATELY SUPERVISED CLASSIFICATION',refs=('real','run'),
          notes='Both methods receive abnormal-condition training labels and share outer folds and test support. Inner normal reference sources remain excluded from fitting. Source-label pairs share one total source weight, after averaging windows within sequences. The JEPA representation is frozen after normal-only pretraining. These are per-fold descriptive macro-F1 values, not the normal-only ranking metric or a significance test. The 4096-head sensitivity does not change these classifiers because it retains the pretrained encoder.',
          takeaway='Keep this task separate from anomaly AUROC. It does not establish a JEPA recognition advantage.')
    dev=result['development'];teacher=result['teacher'];coverage=dev['coverage'][0]
    body=card(60,225,355,270,'Paired profile coverage',f'{coverage["scorable_sources"]} of {len(dev["selected_source_ids"])} normal fitting sources retain every required region after all edits.',ORANGE,26)
    body+=card(462,225,355,270,'Own prediction is unchanged',f'Maximum absolute own-predictor component: {teacher["predictor_own_abs_max"]:.3f}. Its permitted coordinates did not change.',TEAL,24)
    body+=card(864,225,355,270,'The teacher target can change',f'Own-branch teacher component spans {teacher["teacher_own_min"]:.3f} to {teacher["teacher_own_max"]:.3f} in observed source-joint raw KL.',BLUE,24)
    body+=paragraph(65,559,f'The raw teacher diagnostic has {teacher["source_joint_rows"]} observed source-joint rows across {teacher["sources"]} fitting sources.',1150,24)
    d.add('The real diagnostic shows a contextual target effect.',
          'The left-leg timing edit preserves the eight measured arm and right-leg targets.',body,
          tag='MEASURED: DEVELOPMENT DIAGNOSTICS, FOLD 1',refs=('development','real'),
          notes='The paired full-profile control requires joint support common to all edits and leaves only five of twelve training sources scorable. This coverage is distinct from the raw eight-joint teacher diagnostic, which has eligible rows from all twelve but incomplete source-joint combinations. The own prediction component is exactly zero, while teacher-target KL can change under an edited left leg. The component range is across observed source-joint summaries, not an interval or population statistic. The identity is algebraic, not mechanism validation. These are training-normal diagnostics, not held-out tests, virtual diagnoses or a decomposition of the nonlinear three-region score.',
          takeaway='The teacher can change an unchanged joint’s target. This does not establish disease mechanism or clinical explanation.')
    if result['longer']:
        long=result['longer'];lb=long['bootstrap'];rows=[]
        for label,key in [('Latent profile','trained_latent_all_profile_score'),('Raw full-context KL','trained_latent_all_raw_full'),('Full-error distance','trained_latent_all_full_distance'),('Random-backbone latent profile','untrained_latent_all_profile_score'),('Four-patch latent profile','trained_latent_budget4_profile_score')]:
            rows.append((label,f'{metric(result,key):.3f}',f'{metric(result,key,True):.3f}'))
        body=table(['AUROC score','512 head updates','4,096 head updates'],rows,[560,280,320],y=210,row_h=58,size=23)
        body+=paragraph(65,600,f'Longer-run profile minus raw: {lb["point_difference"]:+.3f}; conditional interval [{lb["interval_95"][0]:.3f}, {lb["interval_95"][1]:.3f}].',1150,23)
        d.add('Longer head fitting does not settle the claim.',
              'The 512-head-update run stays primary. Both runs retain the same 512-update JEPA pretraining.',body,
              tag='MEASURED: NORMAL-DIAGNOSTIC-MOTIVATED SENSITIVITY',refs=('long_run','real'),
              notes=f"Run {long['summary']['run_name']} uses 4096 updates per head. It was launched because the 512-update normal coordinate full error exceeded own error in every fold, before abnormal ranking results were inspected. The primary recipe remains unchanged. The longer-run conditional interval still includes zero; full-error distance is essentially tied with the proposed profile, and random-backbone latent profile has a higher point estimate. This sensitivity is not a new co-primary endpoint and does not establish convergence.",
              takeaway='Retain the original primary. Do not select the longer schedule because a test point estimate is higher.')
        rows=[]
        for fold,(a,z) in enumerate(zip(p['histories'],long['histories'])):
            a=a['predictors'][-1];z=z['predictors'][-1]
            rows.append((str(fold+1),f'{a["validation_coordinate_own"]:.4f} / {a["validation_coordinate_full"]:.4f}',f'{z["validation_coordinate_own"]:.4f} / {z["validation_coordinate_full"]:.4f}'))
        body=table(['Fold','512 updates: own / full MSE','4,096 updates: own / full MSE'],rows,[150,505,505],y=235,row_h=65,size=24)
        better=sum(h['predictors'][-1]['validation_coordinate_full']<h['predictors'][-1]['validation_coordinate_own'] for h in long['histories'])
        body+=paragraph(65,595,f'After longer fitting, full context has lower MSE than own context in {better} of four folds.',1150,25)
        d.add('Normal predictive adequacy remains unresolved.',
              'Validation uses fixed normal development draws. These are coordinate MSE diagnostics.',body,
              tag='MEASURED: LEARNING ADEQUACY',refs=('long_run','run'),
              notes='Additional raw input cannot increase optimal Bayes squared-error risk when the ideal predictor can ignore it. Our frozen transformer re-encodes context separately, so memories are not guaranteed nested; finite heads can underfit or overfit. Normal negative benefit concerns these fitted learners, not harm from biological coordination. The longer schedule reduces error, but does not demonstrate the intended predictive ordering across folds. These are sampled batch-averaged normal-development diagnostics, not source-level anomaly metrics.',
              takeaway='The study cannot reject the broader value of gait relationships when its predictors may be inadequate.')


def build(result, inventory):
    d=Deck()
    d.add('Predictable walking can still be abnormal.',
          'We test whether the help supplied by other joints differs from a normal reference.',
          flow([('Learn from normal gait.','A normal-only model predicts hidden joint patches.'),('Compare the information.','Matched branches see a joint alone, local context, or the whole body.'),('Test real departures.','A two-sided profile is evaluated on source-held-out abnormal gait.')],footer='The completed comparison is exploratory. It does not establish a learned advantage or a clinical mechanism.'),
          tag='IMPLEMENTATION, REAL RESULTS AND CLAIM BOUNDARIES',notes='The revised study is motivated by neuromusculoskeletal constraints and altered organization. New results are distinguished from the earlier three-seed baseline throughout this deck. Every slide with toy curves is labeled as an illustration. This is an internal research deck, not an anonymized submission.',refs=('roadmap','biology'))
    d.figure('Biology produces motion before a camera measures it.',
             'Several hidden causes can produce similar visible walking patterns.','neurological_chain',
             notes='This is a conceptual forward chain, not an estimated causal graph for GAVD. Control, muscles, mechanics, compensation, projection and pose estimation each affect the observation. Naming a latent variable weakness does not validate that interpretation.',takeaway='Observed keypoints constrain an explanation; they do not uniquely identify its cause.')
    d.figure('Four condition families motivate different hypotheses.',
             'Stroke, Parkinson’s, cerebral palsy and myopathic gait have overlapping movement effects.','condition_routes',
             notes='Stroke can alter selective activation and limb contributions. Parkinson’s can affect amplitude and coordination differently. Cerebral palsy arises from a developing-brain disturbance while body mechanics can change with growth. Myopathic means muscle-related and is not one brain disorder. The generic GAVD myopathic label is not Duchenne muscular dystrophy. Clinical studies motivate questions rather than diagnoses from video.',refs=('biology','clinical'),takeaway='The inclusive term is neuromusculoskeletal. None of the four labels specifies one gait mechanism.')
    d.figure('A smooth repeated movement can have an unusual relationship.',
             'Lower prediction error is not a universal definition of normal gait.','predictable_patterns',
             tag='ILLUSTRATIVE SIGNALS',notes='These curves are invented teaching signals, not simulated diseases. An abnormal movement may become more predictable, less predictable, or equally predictable. We therefore preserve the one-sided contrast and predeclare a two-sided profile.',takeaway='The hypothesis concerns altered relationships, including unusually tight relationships.')
    body=card(70,220,520,245,'A. The ankle predicts itself.','The predictor sees visible patches of the target ankle. Hidden coordinates stay unavailable.',BLUE)
    body+=card(690,220,520,245,'B. The body helps the ankle.','The predictor also sees other visible allowed joints. It predicts exactly the same hidden patch.',TEAL)
    body+=formula(190,505,900,'Predictive benefit = own-joint error − full-context error',26)
    d.add('Hide the same left-ankle patch twice.',
          'The intended difference is available information, conditional on a shared body reference.',body,
          notes='Both branches use identical target positions, masks and valid-target support. Their normalized coordinates share pelvis centering and prefix torso scale, which already use other joints. The experiment restricts learned context rather than claiming preprocessing is ankle-independent. Uniform-mask completion may see both earlier and later observations.',takeaway='First establish useful benefit on held-out normal sources. Do not assume it exists.')
    d.figure('A matched experiment connects the hypothesis to a score.',
             'Normal fitting, information restrictions and independent source scoring have separate roles.','neuromechanics_experiment',
             tag='STUDY DESIGN',refs=('roadmap',),notes='This figure originated in the proposal. The information restrictions and normal-only roles are implemented and tested. Real scores later in this deck distinguish the completed experiment from the illustrative diagram.',takeaway='A completed software path is different from a demonstrated scientific benefit.')
    labels=inventory['labels'];rows=[(r['label'],str(r['sequences']),str(r['sources'])) for r in labels]
    body=''
    for side,part in enumerate((rows[:6],rows[6:])):
        x=60+side*600
        body+=rect(x,205,560,52,fill=INK,stroke=INK)+txt(x+18,239,'Folder',22,WHITE,'bold')+txt(x+300,239,'Sequences',22,WHITE,'bold')+txt(x+450,239,'Sources',22,WHITE,'bold')
        for i,row in enumerate(part):
            y=257+i*48;body+=rect(x,y,560,48,fill=WHITE if i%2==0 else PALE,stroke=WHITE,radius=0)
            body+=txt(x+18,y+32,row[0],22)+txt(x+315,y+32,row[1],22)+txt(x+470,y+32,row[2],22)
    d.add('Only the normal folder is normal for this endpoint.',
          f"The fresh inventory has {inventory['sequences']:,} sequences and {inventory['sources']} source IDs, including {inventory['normal_sources']} normal sources.",body,
          tag='FRESH ANNOTATION CENSUS',refs=('gait','roadmap'),notes='Counts are recomputed from current per-sequence CSVs and exclude data/gavd/source. Every other folder, including exercise, style and pregnant, is positive abnormal under the stated folder rule. Folder-source totals can overlap, so sum unique source IDs for the union. These are pre-extraction annotation counts, not independent patients or accepted poses.',takeaway='More abnormal folders broaden the test; they do not increase the normal calibration population.')
    counts=result['census'];cohort=result['primary']['summary']['cohort']
    stages=[('Annotations',counts['annotation_sources'],f"{counts['normal_annotation_sources']} normal sources"),
            ('Decodable videos',counts['media'],f"{counts['new_media']} newly acquired"),
            ('Pose-window QC',counts['pose_sources'],f"{counts['pose_normal_sources']} normal sources"),
            ('Profile support',cohort['sources'],f"{cohort['normal_sources']} normal sources")]
    body=''
    for i,(label,value,detail) in enumerate(stages):
        x=60+i*300
        body+=rect(x,230,260,235,fill=PALE,stroke=LINE)
        body+=txt(x+130,274,label,24,INK,'bold','middle')
        body+=txt(x+130,357,str(value),58,TEAL,'bold','middle')
        body+=txt(x+130,414,detail,21,MUTED,anchor='middle')
        if i<3:body+=arrow(x+268,345,x+292,345)
    body+=formula(90,510,1100,f"{cohort['windows']:,} windows | {cohort['sequences']:,} annotated sequences | {cohort['sources']} source videos",28)
    d.add('Quality filtering leaves only 23 normal source videos.',
          'The frozen profile cohort has 251 sources. All are scored by every declared main comparator.',body,
          tag='MEASURED: COHORT ATTRITION',refs=('real','run'),
          notes='The fresh census is 1874 annotation sequences and 348 sources. Acquisition recovered 332 videos. Pose QC retained 9474 windows, 1389 sequences and 311 sources. Fixed profile support retained 8950 windows, 1303 sequences and 251 sources. All 23 retained normals were in the earlier retained study index; 58 abnormal sources were also present, and 170 were absent from that index. Absence from the index does not prove no prior inspection. Videos and sequences are not verified independent people.',
          takeaway='The binary cohort contains 23 normal and 228 abnormal sources. Many windows do not replace normal diversity.')
    body=table(['Focal family','Sequences','Sources','Observable hypothesis'],[
      ('Parkinsonian', '47','11','Movement size and timing can separate.'),
      ('Stroke','76','19','The two legs may contribute differently.'),
      ('Cerebral palsy','64','11','Control and body mechanics can interact.'),
      ('Myopathic','188','30','Trunk and leg strategies may change.')],[255,130,120,655],y=218,row_h=61,size=22)
    body+=paragraph(65,575,'These four families provide interpretation examples. All eleven non-normal folders remain in the primary binary endpoint.',1150,24)
    d.add('The four focal families do not define every abnormal gait.',
          'Category analyses complement the all-folder detection question.',body,refs=('biology','clinical'),
          notes='The original five-class recognition task includes normal plus these four families and remains secondary. The small category source counts constrain interpretation. We do not claim known affected side, medication status, verified etiology or diagnostic ground truth from the folder labels.')
    body=flow([('Read current CSVs.','Hash annotations. Apply pose and profile QC before assigning source folds.'),('Recover the right person.','Use ambient extraction with reviewed timing and scaled annotation boxes.'),('Retain a traceable corpus.','Record source FPS, pose validity, exclusions and exact cache identity.')],y=220,height=245,
              footer='Videos are stored once under data/videos-gavd/<source_id>/. Failures stay in the acquisition ledger.')
    d.add('A prediction must trace back to the walking video.',
          'The current CSVs define the cohort; old feature caches are not experiment inputs.',body,
          tag='IMPLEMENTED DATA FOUNDATION',notes='Use ambient.pose.keypoint_extractor.SequenceKeypointExtractor on annotated crops. Preserve full-image XY, confidence, frame origin, source FPS and model hash. A decoded frame and a high-confidence pose do not prove that the current rendition still aligns with original annotation timing. Inspect overlays and sequence semantics. Never silently change labels or recover missing cases by favorable source selection.',takeaway='Person and time alignment require review before a predictive score becomes paper evidence.')
    body=rect(60,205,585,402,fill=PALE,stroke=PALE)
    for j,(ids,label) in enumerate([('11, 12','Shoulders'),('13, 14','Elbows'),('15, 16','Wrists'),('23, 24','Hips'),('25, 26','Knees'),('27, 28','Ankles'),('29, 30','Heels'),('31, 32','Foot indices')]):
        body+=txt(90,246+j*46,ids,26,TEAL,'bold')+txt(270,246+j*46,label,26)
    for t in range(8):
        body+=txt(737+t*57,230,t,18,MUTED,'normal','middle')
        for j in range(16):
            hidden=((t*17+j*7)%13)<6
            body+=rect(718+t*57,245+j*16,40,11,fill=ORANGE if hidden else TEAL,stroke=WHITE,radius=2)
    body+=txt(730,538,'Orange hides a joint-time patch.',22,ORANGE)
    body+=txt(730,573,'The illustrated mask is schematic.',20,MUTED)
    d.add('Mask only the 16 declared joints, uniformly.',
          'Four frames of one joint form a token. A 32-frame window has 128 tokens.',body,
          tag='FIXED JOINT AND MASK RULE',refs=('mamp','roadmap'),notes='The only indices are 11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32. The sampler ranks independent random draws over the whole token grid without motion, velocity, confidence or labels. Validity filters target accounting afterward. The three-group readout does not remove hips or shoulders from model training.',takeaway='Context restrictions change what a branch can read. They do not change the uniform target sampler.')
    body=flow([('Preserve real time.','Use 15 Hz only when source FPS is at least 15. Never bridge missing frames.'),('Keep missing values missing.','A four-frame joint patch is valid only when every XY value is observed.'),('Apply one shared gate.','Keep at least 75% valid tokens overall and 50% in each temporal half.')],y=215,height=255,
              footer='Center each frame on the pelvis. Scale with observed torso measurements from the first half only.')
    d.add('Quality control must not manufacture motion.',
          'Invalid patches are excluded from attention, target loss and pooled features.',body,
          tag='IMPLEMENTED DATA FOUNDATION',notes='The old complete-case pilot failed and led to this documented validity-aware protocol before classifier performance was inspected. Both methods use the same accepted cohort. Requiring future visibility makes suffix evaluation retrospective on evaluable futures. Camera view and missingness can still predict labels, so nuisance controls and coverage reports remain necessary.',takeaway='Quality filtering is a selection process. Report what each class and source loses.')
    body=card(60,215,360,225,'Retain the primary recipe.','Use 512 normal JEPA updates, then 512 updates for each matched head.',TEAL,25)
    body+=card(460,215,360,225,'Check normal adequacy.','Full-context coordinate error exceeded own error in all four folds.',ORANGE,25)
    body+=card(860,215,360,225,'Retain the sensitivity.','Fit heads for 4,096 updates. Keep 512 JEPA updates and the primary unchanged.',BLUE,25)
    body+=formula(205,485,870,'EMA: teacher ← 0.996 × teacher + 0.004 × online',25)
    body+=paragraph(65,597,'Longer fitting was motivated by normal diagnostics before abnormal ranking inspection.',1150,24)
    d.add('Adequate learning must be checked, not assumed.',
          'Both schedules are completed. Fixed update counts do not establish convergence.',body,
          notes='The longer run was launched from normal-development evidence before the new abnormal ranking results were inspected. The original 512-head comparison remains primary. Extra raw observations cannot worsen ideal Bayes MSE, but separately re-encoded frozen memories need not be nested, and finite heads can underfit. Negative normal coordinate benefit is a learner diagnostic, not biological harm. Actual results for both budgets appear later.',refs=('run','long_run','roadmap'))
    body=card(60,212,315,145,'Online context','Only permitted visible tokens enter.',TEAL,22)
    body+=card(460,212,315,145,'Predictor','Context and target positions predict features.',TEAL,22)
    body+=card(60,435,315,145,'EMA teacher','All observed tokens supply detached targets.',BLUE,22)
    body+=card(460,435,315,145,'Latent loss','Only valid hidden targets contribute.',ORANGE,22)
    body+=arrow(383,282,450,282)+arrow(618,365,618,425)+arrow(383,507,450,507)
    body+=arrow(215,365,215,425)+txt(236,400,'EMA after update',18,BLUE)
    body+=card(835,222,380,300,'VICReg regularizes views.','View agreement, dimension variance and covariance are monitored. No disease labels enter this loss.',TEAL,23)
    d.add('The teacher supplies features, while VICReg discourages collapse.',
          'The online branch sees only permitted observations. The target branch has no gradients.',body,
          refs=('sjepa','vicreg','roadmap'),notes='This is a small paper-based adaptation of Skeleton S-JEPA with uniform masking and VICReg. VICReg is not class repulsion and does not guarantee informative representations. Track actual encoder health and independent prediction performance. Teacher centering, temperatures and weights are fixed during matched predictor comparisons.',takeaway='Useful disease-class separation is a downstream empirical question, not a property promised by VICReg.')
    body=table(['Branch','Target-joint history','Other joints at target patch','Other-joint times'],[
      ('Own joint','Allowed','Blocked','Blocked'),
      ('Patch-local context','Allowed','Allowed','Blocked'),
      ('Full context','Allowed','Allowed','Allowed')],[270,270,385,235],y=245,row_h=66,size=22)
    body+=paragraph(70,570,'Apply the restriction before contextual encoding and at every attention layer. Test excluded-input noninterference.',1140,25)
    d.add('Information isolation must hold for each target token.',
          'A restricted prediction head is insufficient if its input already contains the whole body.',body,
          notes='For each hidden target joint and temporal patch, the own branch can read valid visible patches of that joint only. The patch-local branch also reads other visible joints in the same four-frame patch. Full sees all visible allowed tokens. Enforce restrictions throughout encoding and prediction. Hold the shared body reference fixed when testing coordinate changes. A four-frame control includes short motion, so it is not purely static.',takeaway='If local geometry and motion explain the benefit, do not call it long-range temporal information.')
    body=card(65,218,545,255,'Local coordinate target','The answer is the hidden normalized joint patch. No contextual transformer can mix other joints into it.',BLUE)
    body+=card(670,218,545,255,'Contextual teacher target','The frozen teacher can embed neighboring motion in the answer. Full context may predict that content more easily.',TEAL)
    body+=formula(185,516,910,'Compare own versus full within each target type.',27)
    d.add('The teacher can change what the answer contains.',
          'A local target control separates observable motion from contextual target construction.',body,
          notes='The core study contains coordinate-own, coordinate-full, latent-own and latent-full arms, plus the trained patch-local context control. Freeze teacher parameters, center and temperatures for fair latent comparisons. Coordinate MSE and latent KL have different units. Do not compare their raw magnitudes or select the favorable target after test inspection.',takeaway='A local target still shares pelvis centering and body scale. It is not a clinical ground truth.')
    d.figure('Use a small two-sided profile, fixed before scoring.',
             'Measure benefit for left-leg, right-leg and arm targets separately.','support_profile',
             tag='ILLUSTRATIVE PROFILE',refs=('roadmap',),notes='Left targets are 25,27,29,31. Right targets are 26,28,30,32. Arm targets are 13,14,15,16. All four joints receive equal weight within a group. Hips and shoulders remain eligible for training masks and are reported separately because centering can create geometry shortcuts. The profile names target regions, not the input joints responsible for help.',takeaway='Either direction can depart from normal. More help does not prove stronger biological coupling.')
    body=flow([('Fit normal predictors.','Use only outer-training normal sources. Exclude inner reference sources.'),('Fit the normal reference.','Estimate three means and a shrunk diagonal variance from inner normal sources.'),('Score held-out sources.','Measure the squared distance of each profile from that fixed reference.')],y=220,height=255)
    body+=formula(152,521,976,'score = mean[(benefit − normal mean)² / shrunk variance]',25)
    d.add('Only inner normal sources define the profile reference.',
          'A small diagonal reference can rank departures without pretending to be a clinical normal range.',body,
          notes='For group g, use variance .5*v_g+.5*mean(v), floored at 1e-8. Reference means and sample variances are across sources, not windows. Reserve exactly five normal reference sources per fold; leave-one-reference-out sensitivity uses the remaining four. Model schedule selection uses normal development data only. Fit coordinate and latent references separately. This distance is not a calibrated probability.',takeaway='Do not repair an unstable normal reference by using outer-test normal sources.')
    body=flow([('Targets become joint scores.','Use the same 16 fixed uniform masks and common valid target support.'),('Windows become source profiles.','Average within windows, within sequences and across sequences. Then weight joints equally.'),('Check observation support.','Require four distinct target patches per included joint. Repeat with four patches.')],y=220,height=260)
    body+=paragraph(65,550,'Apply the nonlinear profile distance once, after source aggregation. A missing group makes that source unscorable.',1150,25)
    d.add('More overlapping windows do not create independent evidence.',
          'A minimum support rule and common cohort are part of the score definition.',body,
          notes='Require at least two valid visible patches of the target joint per scored target after uniform sampling. Do not redraw masks. Distinct support identifies sequence/time/joint patches, so repeated masks and overlaps do not inflate its count. The fixed-budget sensitivity samples four eligible patches per joint with a source-keyed seed independent of loss and label. Refit its normal reference with the same budget.',takeaway='Report source coverage, target support, recording span and exclusions beside every detection result.')
    body=''
    colors=[BLUE,TEAL,ORANGE,'#7458a3']
    for r in range(4):
        body+=txt(60,248+r*77,f'Fit {r+1}',25,INK,'bold')
        for c in range(4):
            x=175+c*260;y=215+r*77;test=r==c
            body+=rect(x,y,238,59,fill=colors[c] if test else '#e8eef4',stroke=WHITE)
            body+=txt(x+119,y+38,f'Fold {c+1}: '+('outer test' if test else 'normal pool'),21,WHITE if test else INK,'bold' if test else 'normal','middle')
    body+=paragraph(65,581,'Test normals: 6, 6, 6, 5. Each fold has 57 test abnormals, 5 reference normals and 12 or 13 fitting normals.',1150,24)
    d.add('Four source folds reuse scarce normals without training on them.',
          'Each source receives one out-of-fold score. All of its sequences keep the same role.',body,
          notes='This is exploratory internal source cross-fitting. Assign folds after score-independent QC and before scoring. The original population has already been inspected, so a new fold assignment is not automatically untouched confirmatory evidence. All abnormal sources are excluded from normal-only representation and predictor fitting. Source IDs are not verified participant IDs.',takeaway='Folds improve evaluation efficiency. They do not create more independent normal people.')
    body=formula(75,218,1130,'Primary AUROC = Σ(normal test sources × fold AUROC) / Σ normal test sources',23)
    body+=card(65,340,550,235,'Keep ranks within each fitted fold.','Report every fold and its class support. Do not pool uncalibrated raw scores from different models.',BLUE)
    body+=card(670,340,545,235,'Pair uncertainty at the source level.','Resample sources within fold and binary label. Carry every method through the same bootstrap draw.',TEAL)
    d.add('The interval must estimate the same metric as the point.',
          'Weight within-fold AUROC by the number of normal test sources.',body,
          notes='This estimand gives each scarce normal test source equal weight in its within-fold abnormal-normal comparisons. The stratified paired bootstrap conditions on fitted models, fold assignments and normal references. It excludes retraining and unknown cross-source participant dependence. Report reference-removal and fixed-budget sensitivity separately. Report AP with abnormal prevalence. Tiny category supports are descriptive.',takeaway='Four folds are not four independent studies. Window-level bootstraps would overstate precision.')
    body=table(['Comparison','Supervision','What it tests'],[
      ('Matched full raw KL','Normal only','Same 12 joints, support, weights and aggregation.'),
      ('Own/full loss distances','Normal only','Whether subtraction adds information.'),
      ('Matched ridge predictors','Normal only','Whether a linear relationship is sufficient.'),
      ('SeeKer or STG-NF','Not reproduced','Published comparison remains a gap.'),
      ('RF and frozen probes','All five labels','Secondary supervised recognition only.')],[380,200,580],y=214,row_h=62,size=21)
    d.add('Compare information and supervision fairly.',
          'The benefit profile must improve on more than a weak or differently weighted score.',body,
          notes='Keep historical 16-joint raw KL secondary. The primary raw KL comparator must use exactly the profile readout support and equal joint/group weights. Include untrained frozen encoder with identically fitted heads and normal posture/excursion distance where feasible. An inspired baseline is not an exact reproduction. Published normal-only comparisons remain evidence requirements until validated.',refs=('roadmap','seeker'),takeaway='Do not attribute a gain to JEPA if matched ridge prediction explains it.')
    d.figure('Interventions test model behavior, not virtual diagnoses.',
             'Change a sequence in a controlled way, then compare matched targets and observations.','intervention_controls',
             tag='IMPLEMENTED DEVELOPMENT DIAGNOSTICS',refs=('roadmap',),notes='Relative limb shifts need longer uninterrupted segments, common interior support and articulation checks. Compare with a matched whole-body shift. Observation noise/deletion is a separate axis. Keep normalization reference fixed. Do not claim finite-crop marginals are exactly preserved without checking them. Coordinate noise is not the complete visual pose-error process.',takeaway='A shifted leg is not a simulated stroke. Quality matching alone does not prove causation.')
    body=table(['','Original target y0','Edited target y1'],[
      ('Original prediction p0','L00','L01'),('Edited prediction p1','L10','L11')],[390,385,385],y=240,row_h=73,size=26)
    body+=formula(85,497,1110,'Total change = prediction change + target change + interaction',25)
    body+=paragraph(65,607,'Eight unchanged arm and right-leg joints enter this raw-KL diagnostic.',1150,23)
    d.add('Separate prediction changes from teacher-target changes.',
          'Hold the local target patch fixed while editing eligible non-target context.',body,
          tag='IMPLEMENTED AND MEASURED RAW-LOSS DIAGNOSTIC',notes='Run run_study_controls.py with --teacher-decomposition. The relative left-leg edit leaves all local measurements and own histories of joints 13,14,15,16,26,28,30,32 unchanged. Prediction change=L10−L00; target change=L01−L00; interaction=L11−L10−L01+L00. Apply per branch, retaining identical masks and paired eligibility. The implementation checks the additive identity and frozen teacher center. Outputs are raw KL components per source and unchanged joint, not the nonlinear three-region profile. Its left-leg group is absent. This is a computational model-loss diagnostic, not disease causality. The real fold-0 run is complete. Its raw own-predictor component is zero on unchanged joints, while teacher-target changes can alter KL. The paired full-profile controls retain five of twelve fitting sources; the raw teacher diagnostic has 87 observed source-joint rows from twelve sources. These are different coverage requirements. Numerical results and limitations appear later.',takeaway='Eight unchanged joints do not define the full profile. The raw-loss identity is not a clinical explanation.')
    body=flow([('Blind the review.','Hide folder labels and model scores. Fix the source sample in advance.'),('Describe visible movement.','Record asymmetry, arm excursion, knee flexion, trunk motion and aids.'),('Record uncertainty.','Use two suitable reviewers, written definitions and an unscorable option.')],y=220,height=255,
              footer='Report agreement. Without independent review, call automatic measures proxies rather than physiological validation.')
    d.add('Visible-phenotype annotations can test interpretation.',
          'A description of movement is different from a retrospective medical diagnosis.',body,
          notes='View and pose quality must be recorded. Two seconds may not support reliable gait-cycle timing; phase analysis needs longer validated segments. Reviewers assess observable features, not paretic side, force, strength, dopamine or lesion location. If suitable reviewers are unavailable, do not manufacture annotation evidence.',refs=('biology','roadmap'),takeaway='Movement descriptions support interpretation. They do not replace the detection endpoint.')
    body=table(['Evidence gate','Required observation','If it does not hold'],[
      ('Learning','Stable development diagnostics.','Limit the claim to a bounded recipe.'),
      ('Information','Other joints help held-out normals.','Investigate whether the contrast is useful.'),
      ('Interpretation','Local target and controls survive.','Avoid a movement-relationship claim.'),
      ('Detection','Gain and interval on matched sources.','Report uncertainty or a precise null.'),
      ('Generalization','Independent people and settings.','Keep the claim internal to GAVD.')],[250,455,455],y=218,row_h=61,size=21)
    d.add('Each scientific claim needs its own evidence gate.',
          'A passing software test does not establish a useful anomaly detector.',body,
          notes='The roadmap predeclares a 0.05 AUROC improvement target, not a clinical threshold. A positive interval and point estimate of at least 0.05 do not prove that the gain is at least 0.05 unless the lower bound also exceeds it. A wide interval is inconclusive. The earlier study had only five or six normal calibration sources, too few for a finite 5% threshold. Nineteen exchangeable units are only a mathematical minimum. Source grouping does not establish clinical generalization.',takeaway='Uncertainty, missing evidence and failed hypotheses remain reportable outcomes.')
    hist=[('RF',[.267,.482,.333],TEAL),('Selected JEPA',[.244,.252,.331],BLUE),('Untrained encoder',[.293,.327,.348],ORANGE)]
    body='';left=125;top=245;chart_h=275;groupgap=330;barw=62
    for v in [0,.1,.2,.3,.4,.5]:
        yy=top+chart_h-v/.5*chart_h;body+=line(105,yy,1180,yy,LINE,1)+txt(92,yy+7,f'{v:.1f}',20,MUTED,anchor='end')
    for seed_i,seed in enumerate([42,43,44]):
        for m,(name,values,color) in enumerate(hist):
            x=left+seed_i*groupgap+m*76;val=values[seed_i];hh=val/.5*chart_h
            body+=rect(x,top+chart_h-hh,barw,hh,fill=color,stroke=color,radius=4)+txt(x+barw/2,top+chart_h-hh-10,f'{val:.3f}',20,color,'bold','middle')
        body+=txt(left+seed_i*groupgap+105,555,f'Seed {seed}',24,INK,'bold','middle')
    for i,(name,_,color) in enumerate(hist):body+=rect(185+i*330,595,20,20,fill=color,stroke=color,radius=3)+txt(216+i*330,613,name,22)
    d.add('The earlier three runs did not establish a learning advantage.',
          'Source-weighted five-class macro-F1 is shown for the completed historical study.',body,
          tag='MEASURED: EARLIER FIVE-CLASS STUDY',refs=('old',),notes='The historical cohort contains 450 sequences, 91 sources and 2,504 windows. All three classifiers use abnormal training labels. RF and the untrained encoder each numerically exceeded the selected JEPA in all three runs. Only 45 to 66 normal updates were used, so this does not establish general JEPA failure. Historical normal-only masked AUROCs were 0.200, 0.604 and 0.583; suffix AUROCs were 0.138, 0.417 and 0.567. These are separate supervision settings.',takeaway='These numbers are not results of the new own-joint versus full-context experiment.')
    add_real_result_slides(d,result)
    d.figure('A longer-term world model would represent the forward process.',
             'Control, body mechanics, state feedback and observation are separate modeling components.','physical_world_model',
             tag='LONGER-TERM PROPOSAL',refs=('biology',),notes='This is not the current JEPA architecture. A physical forward model needs validated dynamics, contact, controller and observation assumptions. GAVD supplies projected observations but does not identify forces, muscle strength or controller parameters. Additional simulation and clinical measurements are necessary.',takeaway='A physiological name for a latent variable is not physiological validation.')
    d.figure('Preserve multiple explanations when the video is ambiguous.',
             'Ask which additional measurement would distinguish compatible mechanisms.','mechanism_ambiguity',
             tag='LONGER-TERM PROPOSAL',refs=('biology',),notes='The diagram is a candidate benchmark design, not an established pair of clinically equivalent conditions. Distinct control and body parameters may produce similar projected motion. A useful posterior would retain ambiguity and predict the value of additional measurements. Simulation parameter recovery and clinical validation are different evidence levels. Generating gait from weakness already has prior art.',takeaway='The valuable goal is tested ambiguity and measurement selection, not an unsupported diagnosis from pose.')
    body=card(60,220,360,270,'A bounded empirical finding','The primary interval includes zero. Matched controls do not establish a learned JEPA advantage.',TEAL)
    body+=card(460,220,360,270,'A clear novelty boundary','The contribution is a controlled limitation study. A validated published comparator is still missing.',BLUE)
    body+=card(860,220,360,270,'A reviewable submission','Freeze claims by November 19. Submit PDF by November 20 using AAAI-27 format and double-blind review.',ORANGE)
    body+=paragraph(65,554,'ASTAD permits seven content pages plus two reference-only pages. The site does not state a cutoff timezone.',1150,24)
    d.add('Build a paper around the demonstrated limitations.',
          'The workshop deadline is November 20, 2026, verified on the official site on November 15.',body,
          tag='EVIDENCE AND SUBMISSION',refs=('workshop','fsgait','seeker','sjepa','paper','real'),notes='Normal-only gait modeling, masking, VICReg, positional noise and confidence weighting have prior art. Novelty must be a demonstrated mechanism, useful comparison or bounded empirical insight. Do not promise clinical diagnosis, future falls or population generalizability. Verify the exact OpenReview cutoff before upload. Archival proceedings are intended but not confirmed.',takeaway='Current evidence supports an exploratory methods study, not clinical detection or a demonstrated JEPA gain.')
    assert len(d.slides)>=38
    return d


def normalize_prose(value):
    # Notes are deliberately authored separately from concise on-slide labels.
    return value.replace('—', ', ')


def export(deck, inventory, result, args):
    archive();IMAGES.mkdir(parents=True,exist_ok=True)
    records=[]
    originals=OUT/'source_figures';originals.mkdir(exist_ok=True)
    for s in deck.slides:
        if s.get('source_figure'):shutil.copy2(DRAFT/'images'/Path(s['source_figure']).name,OUT/s['source_figure'])
    for i,s in enumerate(deck.slides,1):
        body=rect(0,0,W,H,fill=WHITE,stroke=WHITE,radius=0)
        body+=rect(0,0,18,H,fill=TEAL,stroke=TEAL,radius=0)
        body+=txt(60,40,s['tag'],15,TEAL,'bold')
        title_lines=textwrap.wrap(s['title'],width=65,break_long_words=False)
        if len(title_lines)>2:raise ValueError(f'Title too long at slide{i}')
        for j,title in enumerate(title_lines):body+=txt(60,88+j*41,title,35,INK,'bold')
        lede_y=147 if len(title_lines)==1 else 178
        # All current titles are intentionally short enough for a single line.
        if len(title_lines)>1:raise ValueError(f'Shorten slide{i} title for consistent layout')
        body+=paragraph(60,lede_y,s['lede'],1155,24,MUTED)
        body+=s['body']
        if s['takeaway']:
            body+=rect(45,638,1190,47,fill=PALE,stroke=PALE,radius=8)
            take_size=min(19,1155/(len(s['takeaway'])*.55))
            body+=txt(60,668,s['takeaway'],take_size,INK)
        body+=txt(60,704,'Skeleton predictive learning | Neuromechanics study | November 2026',15,MUTED)
        body+=txt(1218,704,f'{i:02d} / {len(deck.slides)}',16,MUTED,anchor='end')
        svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720" role="img"><title>{escape(s["title"])}</title><desc>{escape(s["lede"])}</desc><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto"><path d="M0,0L0,6L8,3z" fill="{TEAL}"/></marker></defs>{body}</svg>'
        name=f'{i:02d}.svg';(IMAGES/name).write_text(svg)
        records.append({k:normalize_prose(v) if isinstance(v,str) else v for k,v in s.items() if k!='body'}|{'image':'images/'+name})
    (OUT/'deck_spec.json').write_text(json.dumps(dict(date='2026-11-15',slides=records,census=inventory,result_artifact=args.run_dir),indent=2))
    if result:(OUT/'result_input.json').write_text(json.dumps(result,indent=2))
    md=['# Predictable walking can still be abnormal\n',f'A {len(records)}-slide internal research deck with completed primary, sensitivity and control results. This editable companion includes speaker notes and sources. SVG pages are fully vector.\n']
    html=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Skeleton predictive learning: neuromechanics study</title><style>html{scroll-behavior:smooth}*{box-sizing:border-box}body{margin:0;background:#112438;color:#172e47;font-family:Arial,sans-serif}.slide{width:min(1280px,100%);margin:22px auto;background:white;break-after:page}.slide>img{display:block;width:100%;height:auto}details{padding:18px 34px;font-size:19px;line-height:1.5;border-top:1px solid #dae2e9}summary{cursor:pointer;color:#087f82;font-weight:bold}details a{color:#315cb5}nav{background:#112438;color:white;padding:15px;text-align:center;font-size:18px}nav a{color:#bceeee}@media print{nav,details{display:none}.slide{width:1280px;height:720px;margin:0;overflow:hidden}body{background:white}@page{size:1280px 720px;margin:0}}</style></head><body><nav>Use arrow keys to move between slides. <a href="tutorial.pdf">Vector PDF</a> · <a href="tutorial.md">Editable Markdown</a> · <a href="README.md">Build and evidence notes</a></nav>']
    for i,s in enumerate(records,1):
        refs=[SOURCES[key] for key in s['refs']]
        notes=normalize_prose(s['notes'])
        original=(f'<p>Original detailed vector diagram:</p><img style="width:100%;height:auto" src="{s["source_figure"]}" alt="Detailed source diagram">' if s.get('source_figure') else '')
        html.append(f'<section class="slide" id="slide-{i}"><img src="{s["image"]}" alt="{escape(s["title"])}"><details><summary>Speaker notes and sources</summary><p>{escape(notes)}</p><ul>'+''.join(f'<li><a href="{escape(url)}">{escape(label)}</a></li>' for label,url in refs)+'</ul>'+original+'</details></section>')
        md.append(f'## {i:02d}. {s["title"]}\n\n**{s["tag"]}**\n\n{s["lede"]}\n\n![{s["title"]}]({s["image"]})\n\n{s["takeaway"]}\n\n{notes}\n\n'+', '.join(f'[{label}]({url})' for label,url in refs)+'\n')
    html.append('<script>let index=0;const slides=[...document.querySelectorAll(".slide")];document.addEventListener("keydown",e=>{if(["ArrowRight","ArrowDown","PageDown","ArrowLeft","ArrowUp","PageUp"].includes(e.key)){e.preventDefault();index=Math.max(0,Math.min(slides.length-1,index+(["ArrowRight","ArrowDown","PageDown"].includes(e.key)?1:-1)));slides[index].scrollIntoView();}});</script></body></html>')
    (OUT/'index.html').write_text('\n'.join(html));(OUT/'tutorial.md').write_text('\n'.join(md))
    (OUT/'README.md').write_text(f'''# Neuromechanics research slides

Open [index.html](index.html) locally, read the [editable Markdown](tutorial.md), or use the [vector PDF](tutorial.pdf). This deck contains {len(records)} vector slides, with expandable speaker notes and scientific sources. It works offline; external source links require internet access. The earlier 16-slide tutorial remains in [the archive](archive/pre_neuromechanics/index.html).

The deck teaches the matched own-joint, patch-local and full-context experiment, then presents the completed real 512-head-update primary study, the normal-diagnostic-motivated 4,096-head-update sensitivity, source controls, category descriptions, supervised classifiers and development diagnostics. Primary results come from run `3044566091261819`; the sensitivity is `eba2bd5a6e05efa1`. Motion and nuisance controls are `7407af320461a092` under the primary run. Historical three-seed results remain explicitly labeled and do not substitute for the new study.

The primary AUROC difference has a conditional interval that contains zero. Neither the primary nor longer schedule establishes a learned JEPA advantage. The population has 251 source videos but only 23 normal sources, all previously explored. These are not verified independent patients. The real teacher diagnostic concerns raw KL on eight unchanged joints and does not decompose the full nonlinear anomaly profile.

Rebuild from `experiments/aaai`:

```bash
uv run python scripts/build_research_slides.py --pdf --render
```

The defaults point to the exact completed runs above. To use reviewed replacement artifacts, pass `--run-dir` and `--sensitivity-run-dir`. The builder requires a passed artifact-verification report, a completed real summary, four folds, matching common-source coverage and all declared trained, random-backbone and ridge target/budget result cells. It reads the full-precision artifact values and only rounds for display. `result_input.json` preserves the loaded evidence and source-file hashes, while `deck_spec.json` records all slide descriptions and source links. The primary comparison is never automatically selected from the favorable schedule.

Generation uses Python's standard library. PDF export needs `rsvg-convert` and `pdfunite`; inspection also needs `pdftoppm` and Pillow. On macOS, `brew install librsvg poppler` supplies the command-line renderers. Omit `--pdf --render` to regenerate editable SVG, HTML and Markdown alone. Every page is an editable SVG in `images/`; eight detailed source vectors remain under `source_figures/` and in the expandable notes.

All changed pages must be rendered and visually inspected. The complete rendered deck and contact sheets are in `review/`; [visual_review.md](visual_review.md) records the actual inspection and evidence checks. A successful build is not a scientific validation of the findings. See the [results report](../docs/context_study_results.md), [protocol](../docs/context_study_protocol.md) and [manuscript evidence](../docs/draft01/evidence.json) for the full interpretation.
''')
    if args.pdf:
        for executable in ('rsvg-convert','pdfunite'):
            if not shutil.which(executable):raise SystemExit(f'{executable} is required for --pdf')
        pdfdir=OUT/'review/vector_pages';pdfdir.mkdir(parents=True,exist_ok=True)
        pages=[]
        for i in range(1,len(records)+1):
            page=pdfdir/f'{i:02d}.pdf'
            subprocess.run(['rsvg-convert','--format=pdf','--output',str(page),str(IMAGES/f'{i:02d}.svg')],check=True)
            pages.append(str(page))
        subprocess.run(['pdfunite',*pages,str(OUT/'tutorial.pdf')],check=True)
    if args.render:
        if not args.pdf and not (OUT/'tutorial.pdf').exists():raise SystemExit('Use --pdf before --render')
        from PIL import Image,ImageDraw
        review=OUT/'review';review.mkdir(exist_ok=True)
        for stale in [*review.glob('slide-*.png'),*review.glob('contact-*.png')]:stale.unlink()
        subprocess.run(['pdftoppm','-r','90','-png',str(OUT/'tutorial.pdf'),str(review/'slide')],check=True,stdout=subprocess.DEVNULL)
        pngs=sorted(review.glob('slide-*.png'))
        for sheet in range(math.ceil(len(pngs)/10)):
            canvas=Image.new('RGB',(1280,5*382),'#e5ebf1');draw=ImageDraw.Draw(canvas)
            for k,path in enumerate(pngs[sheet*10:(sheet+1)*10]):
                img=Image.open(path).convert('RGB');img.thumbnail((640,360))
                x=(k%2)*640;y=(k//2)*382
                draw.text((x+12,y+4),f'Slide{sheet*10+k+1:02d}',fill=INK);canvas.paste(img,(x,y+22))
            canvas.save(review/f'contact-{sheet+1:02d}.png')
    (OUT/'build_manifest.json').write_text(json.dumps({'date':'2026-11-15','slides':len(records),'census':inventory,'result_input':args.run_dir,'sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in IMAGES.glob('*.svg')}},indent=2))
    print(json.dumps({'slides':len(records),'sources':inventory['sources'],'normal_sources':inventory['normal_sources'],'pdf':bool(args.pdf),'rendered':bool(args.render),'results':bool(result)}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',default=str(BASE/'artifacts/context_study/gavd/3044566091261819'),help='Completed real primary run')
    p.add_argument('--sensitivity-run-dir',default=str(BASE/'artifacts/context_study/gavd/eba2bd5a6e05efa1'),help='Completed real longer-head sensitivity')
    p.add_argument('--pdf',action='store_true');p.add_argument('--render',action='store_true')
    args=p.parse_args();result=load_study_results(args.run_dir,args.sensitivity_run_dir)
    inventory=census();export(build(result,inventory),inventory,result,args)

if __name__=='__main__':main()
