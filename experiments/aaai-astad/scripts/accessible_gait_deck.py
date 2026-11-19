"""Plain-language slide content and visuals for the verified gait study.

This module changes presentation only. Measurements come from saved run tables.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from aaai_gait.masking import draw_masks

BASE=Path(__file__).resolve().parents[1]
ORDER=[1,3,4,5,2,6,8,7,9,10,11,13,12,14,15,16,17,18,19,20,41,43,21,22,23,24,25,26,27,28,29,30,31,32,33,34,42,35,36,37,38,39,40]
POLICY={'U':'Random','J':'Whole joint','A':'Anatomical'}
TASK={'U':'Scattered hiding','E_leg':'Grouped hiding'}
NOTE_FIELDS=[('purpose','Purpose'),('explanation','Explanation'),('visual','Reading the visual'),('evidence','Evidence and limits'),('transition','Transition')]


def revise(deck,run,external,api):
    old=deck.records; records={}; C=api.cards; T=api.table
    metrics=run.table('metrics.csv')
    score=lambda key,ep='focal',policy='U',bank='U':float(metrics.loc[metrics.policy.eq(policy)&metrics.bank.eq(bank)&metrics.endpoint.eq(ep)&metrics.method.eq(key)&metrics.metric.eq('auroc'),'value'].iloc[0])
    ext=json.loads((external/'metrics.json').read_text())
    match=json.loads((external/'age_sex_sensitivity.json').read_text())
    completion=pd.read_csv(BASE/'docs/draft03/advisory/completion_audit_v1/completion_summary.csv',dtype={'outer_fold':str})
    lab=completion.query("dataset=='external_angles' and outer_fold=='all'").iloc[0]
    paired=run.json('paired_intervals.json')
    def contrast(a,b,ep='focal',kind='method'):
        return next(v for v in paired if v['bank']=='U' and v['endpoint']==ep and v['first']==a and v['second']==b and v['contrast']==kind and (kind!='method' or v.get('policy')=='U'))
    def add(n,title,lede,draw,caption,tag='THE GAIT STUDY'):
        records[n]={**old[n-1], 'content_id':n,'title':title,'lede':lede,'draw':draw,'takeaway':caption,'tag':tag}
    def cards(items):return lambda c:C(c,items,size=25,y=232,h=320)
    def tab(headers,rows,widths,row_h=76,size=24):return lambda c:T(c,headers,rows,widths,row_h=row_h,size=size)
    def models(c):
        c.card(60,250,280,235,'Own-leg baseline','Use the visible target leg to make an initial prediction. Keep it fixed.',api.TEAL,25)
        c.card(435,220,360,145,'Own-leg correction','Use extra features from the same leg.',api.TEAL,25)
        c.card(435,410,360,145,'Both-leg correction','Use features from both legs.',api.BLUE,25)
        c.arrow(345,310,425,285);c.arrow(345,385,425,480,api.BLUE)
        c.para(870,285,'Both corrections predict the same hidden answer.',330,28,api.INK,True)
        c.para(870,440,'Compare their errors to measure the added benefit.',330,26)
    def masks(c):
        for k,policy in enumerate(['U','J','A']):
            mask=draw_masks(policy,1,rng=np.random.default_rng(17+k))[0].reshape(8,16)
            x0=100+400*k;y0=270
            c.text(x0+120,231,POLICY[policy],29,api.INK,True,align='center')
            for j in range(16):
                for t in range(8):c.box(x0+t*30,y0+j*17,27,14,api.TEAL if mask[t,j] else api.PALE,api.TEAL if mask[t,j] else '#d9e3ec',1)
            c.text(x0+120,575,'Time →',22,api.MUTED,align='center')
        c.text(60,605,'Each column is four samples. Each row is one joint. Colored cells are hidden.',23,api.MUTED)
    def timing_diagram(c):
        for r,(name,start,color) in enumerate([('Four frames earlier',1,api.BLUE),('Original timing',2,api.TEAL),('Four frames later',3,api.ORANGE)]):
            y=255+r*90;c.text(60,y+30,name,26,color,True)
            for j in range(12):c.box(440+j*60,y,52,47,color if start<=j<start+8 else api.PALE,color if start<=j<start+8 else '#d9e3ec',4)
        c.para(440,555,'Each tile is four samples. The hidden answer and target-leg observations stay fixed.',740,24)
    add(1,'Can the other leg help explain unusual walking?',
        'We test whether the opposite leg improves prediction and whether that improvement helps detect unusual gait.',
        cards([('Walking measurements','We follow joint positions over time in videos and laboratory recordings.'),('A controlled prediction','We hide part of one leg and measure what the other leg adds.'),('Anomaly detection','We then test whether the resulting score distinguishes unusual movement.')]),
        'The talk separates useful prediction from useful anomaly detection.')
    add(2,'The study uses videos and laboratory measurements.',
        'The datasets let us ask the same prediction question under different recording conditions.',
        tab(['Dataset','Evaluation units','Group composition'],[('GAVD: four condition groups','90 recordings','28 normal; 62 abnormal'),('GAVD: all abnormal folders','304 recordings','28 normal; 276 abnormal'),('Independent laboratory','178 participants','128 healthy; 50 stroke')],[460,340,360],row_h=95),
        'A video upload is not necessarily a unique person, while laboratory IDs identify participants.')
    add(3,'Start with a short hidden part of the left ankle.',
        'The visible left leg already gives clues. We ask what the right leg adds after those clues have been used.',
        cards([('The ankle over time','Visible ankle positions may help estimate the missing part.'),('The knee and foot','Nearby joints also constrain where the ankle can be.'),('The opposite leg','Right-leg observations may add information about the walking pattern.')]),
        'This illustration explains the question; it is not an observed prediction result.','AN ILLUSTRATIVE EXAMPLE')
    add(4,'Different conditions can change walking in different ways.',
        'Clinical findings guide what we measure, but the videos do not reveal the cause of an individual movement.',
        tab(['Condition group','Why its movement is relevant'],[('Stroke','The two sides may compensate differently for weakness.'),('Parkinsons','Small movements can still have regular timing.'),('Cerebral palsy','Control and body mechanics can both affect posture.'),('Myopathic gait','Muscle weakness can alter trunk and leg movement.')],[340,820],row_h=78,size=24),
        'A repeated compensation can be predictable and still differ from normal walking.')
    add(5,'Keep posture and movement size visible to the model.',
        'Changing every movement to the same size or timing can remove the differences we want to study.',
        cards([('Posture','A bent knee or tilted trunk may matter even when the motion repeats.'),('Movement size','Small steps or reduced arm movement should remain measurable.'),('Shared timing','Both legs keep the same clock so their relative timing remains available.')]),
        'The coordinates share one body scale, while the recorded movement differences remain.')
    add(6,'Prediction can use coordinates or learned features.',
        'Self-supervised learning creates training targets from the observations rather than requiring condition labels.',
        cards([('Predict observations','A model can estimate a hidden joint position directly.'),('Predict features','JEPA predicts a learned representation supplied by a teacher network.'),('Our main comparison','We use regularized linear models so every branch predicts the same coordinates.')]),
        'The separate JEPA comparison comes later; the main method does not require it.')
    add(7,'The main video comparison contains 90 recordings.',
        'We compare normal walking with four condition groups and retain every other abnormal folder in a second analysis.',
        tab(['Annotation group','Recordings'],[('Normal',28),('Parkinsons',10),('Stroke',15),('Cerebral palsy',10),('Myopathic',27)],[900,260],row_h=64,size=25),
        'The separate five-class task uses 89 recordings after excluding one ambiguous annotation.')
    add(8,'Follow each recording from annotation to joint coordinates.',
        'The current CSV files determine the cohort, and saved metadata connects every measurement to its video.',
        cards([('Read the annotations','Keep the source video, sequence and frame identifiers together.'),('Extract the joints','Crop the annotated person, estimate pose and retain confidence and missingness.'),('Build the study data','Save valid windows, exclusions, body scale and source roles.')]),
        'Videos and previews stay under data/videos-gavd; cached pose requires matching provenance.')
    add(9,'A short window contains several related observations.',
        'At 15 samples per second, 32 samples span about 2.07 seconds from the first sample to the last.',
        cards([('A time window','Windows overlap, so they are not independent people or trials.'),('A token','One token contains four consecutive samples of one joint.'),('Enough visible movement','A scored joint needs distinct valid target blocks, not repeated copies of one block.')]),
        'A window can be useful for prediction without containing enough complete walking cycles.')
    add(10,'Use a fixed set of 16 joints and score the lower legs.',
        'The joint list is fixed before training. Overall window-quality checks still include all 16 joints.',
        tab(['Joint group','BlazePose indices'],[('Shoulders, elbows and wrists','11, 12, 13, 14, 15, 16'),('Hips','23, 24'),('Knees, ankles, heels and foot indices','25, 26, 27, 28, 29, 30, 31, 32')],[670,490],row_h=100,size=23),
        'Final joint-specific prediction scores use the eight knee-to-foot measurements.')
    add(11,'Give fitting, reference estimation and testing separate roles.',
        'Each of four outer folds assigns the 28 normal recordings to three distinct groups.',
        cards([('16 fitting recordings','Fit the predictors and choose settings using three internal validation folds.'),('5 reference recordings','Estimate normal cross-leg score ranges after the model settings are fixed.'),('7 test recordings','Evaluate unseen source recordings and keep all their windows together.')]),
        'The study remains exploratory because the normal recordings informed method development.')
    add(12,'Check the predictor before interpreting its behavior.',
        'Software checks and normal-data validation test whether the model can answer the intended question.',
        cards([('Known added information','Generated examples check whether useful extra context can be recovered.'),('No added information','A second generated example checks for invented context benefit.'),('Restricted inputs','Changing a hidden input must leave the prediction unchanged, with the body reference fixed.')]),
        'Generated examples verify the estimator; they do not simulate neurological conditions.')
    add(13,'Compare two corrections to the same own-leg baseline.',
        'One correction uses extra own-leg features, while the other adds opposite-leg observations.',models,
        'Equal feature counts provide a partial control; feature rank and model flexibility still differ.')
    add(14,'Three training masks hide observations in different patterns.',
        'Every example below hides half the tokens without inspecting movement speed or the condition label.',masks,
        'These illustrative draws show the masking rules, not measured model performance.','MASKING RULES | ILLUSTRATION')
    add(15,'Keep the hiding rate equal while changing the pattern.',
        'The methods differ in which observations disappear together, not in how often each token is hidden.',
        tab(['Training mask','Hidden tokens','Chance per token'],[('Random','64 of 128','50%'),('Whole joint, then random fill','64 of 128','50%'),('Anatomical group, then random fill','64 of 128','50%')],[580,300,280],row_h=100,size=24),
        'Equal hiding rates do not make the remaining observations equally informative.')
    add(16,'Test every training mask on the same two hiding tasks.',
        'Scattered hiding leaves different local clues from grouped hiding, so comparisons stay within each task.',
        tab(['Training mask','Scattered hiding','Grouped hiding'],[(POLICY[k],'Same fixed test masks','Same fixed test masks') for k in ['U','J','A']],[340,410,410],row_h=100,size=24),
        'The grouped test scores neighboring knee-to-foot observations hidden together.')
    add(17,'Hold the rest of the prediction experiment fixed.',
        'The mask comparison reuses the own-leg baseline and changes how the correction models see missing inputs.',
        cards([('The same baseline','Every training method retains the same fitted own-leg prediction.'),('The same comparisons','Use matched sources, target definitions and fitting budgets.'),('Independent randomness','Keep mask draws separate from unrelated sampling and model choices.')]),
        'This tests correction training; it does not establish a better JEPA pretraining mask.')
    adequacy=run.table('adequacy.csv').groupby(['outer_fold','policy'])[['own','both']].mean()
    vals=np.array([[adequacy.loc[(i,k),'own']-adequacy.loc[(i,k),'both'] for k in ['U','J','A']] for i in range(4)])
    add(18,'The corrections pass the internal prediction check.',
        'These points summarize normal validation data used for model selection, before evaluating test recordings.',
        lambda c:api.points(c,vals,['Random','Whole joint','Anatomical'],'Own-leg error minus both-leg error'),
        'Positive values favor both-leg prediction; development checks do not establish test performance.')
    normal=run.json('normal_test_benefit.json')
    nr=[]
    for bank in ['U','E_leg']:
        for k in ['U','J','A']:
            z=next(z for z in normal if z['policy']==k and z['bank']==bank)
            nr.append((TASK[bank],POLICY[k],f"{z['mean_benefit']:.5f}",f"[{z['conditional_95_interval'][0]:.5f}, {z['conditional_95_interval'][1]:.5f}]"))
    add(19,'Test-set prediction benefit depends on what is hidden.',
        'Each normal recording is evaluated by its held-out fold model, using identical targets within each task.',
        tab(['Evaluation task','Training','Mean benefit','95% interval'],nr,[320,220,220,400],row_h=54,size=21),
        'Benefit is own-leg error minus both-leg error; intervals crossing zero leave its direction uncertain.')
    add(20,'Inspect one saved prediction alongside its hidden answer.',
        'All three predictions refer to the same four samples from a real normal-labeled recording.',old[19]['draw'],
        'This deterministically selected example helps inspect the method; it is not an accuracy summary.','A SAVED REAL-DATA EXAMPLE')
    add(21,'The three anomaly scores describe different quantities.',
        'Cross-leg benefit becomes an anomaly score only after comparison with separate normal reference recordings.',
        cards([('Movement score M','M compares knee posture, ankle travel and trunk tilt with normal walking.'),('Cross-leg score R','R asks whether seeing the other leg helps prediction more or less than usual in normal walking.'),('Combined score S','S adds M and R and divides by two. The weights stay fixed: S = (M + R) / 2.')]),
        'Both scores measure departure from a normal reference; neither is a probability of disease.')
    add(22,'Evaluate whole recordings rather than repeated windows.',
        'Multiple masks and overlapping windows provide repeated measurements of the same recording.',
        cards([('Combine repeated errors','Average errors and benefits for one target, then combine windows and sequences.'),('Compare within each fold','AUROC measures how often an abnormal recording ranks above a normal recording.'),('Estimate uncertainty','Resample whole recordings together across methods while keeping the fitted models fixed.')]),
        'The intervals do not include uncertainty from retraining or unknown repeated people across uploads.')
    add(23,'Movement alone ranks the video labels more accurately.',
        'Both comparisons use the same frozen scores; one includes four condition groups and the other all abnormal folders.',
        tab(['Video comparison','Recordings','Movement M','Combined S'],[('Four condition groups',90,f"{score('M'):.3f}",f"{score('S'):.3f}"),('All abnormal folders',304,f"{score('M','broad'):.3f}",f"{score('S','broad'):.3f}")],[430,210,260,260],row_h=100,size=24),
        'Values are AUROCs: higher values indicate better ranking of abnormal recordings.')
    def paired_plot(c):
        p=contrast('S','M');b=contrast('S','M','broad');a=contrast('A','U',kind='mask_policy')
        api.forest(c,[('Four groups: combined − M',p['observed'],p['conditional_95_interval']),('All folders: combined − M',b['observed'],b['conditional_95_interval']),('Four groups: anatomical − random',a['observed'],a['conditional_95_interval'])],label='AUROC difference; zero means no change',limits=(-.25,.16))
    add(24,'Use paired intervals to judge the size of a difference.',
        'Each interval compares two scores on the same recordings, with the trained models held fixed.',paired_plot,
        'The first two rows favor movement alone; the masking interval leaves the direction uncertain.')
    rows=lambda bank:[(POLICY[k],f"{score('S',policy=k,bank=bank):.3f}",f"{score('S','broad',k,bank):.3f}") for k in ['U','J','A']]
    add(25,'The scattered test does not establish a better training mask.',
        'All three rows use the same scattered-hiding task and the same combined-score definition.',
        tab(['Training mask','Four groups: AUROC','All folders: AUROC'],rows('U'),[460,350,350],row_h=95,size=23),
        'The anatomical-minus-random difference is 0.055, with a 95% interval from −0.018 to 0.127.')
    add(26,'Grouped hiding asks a separate anomaly question.',
        'Here the prediction target is a neighboring group of leg observations, with a reference fitted for this task.',
        tab(['Training mask','Four groups: AUROC','All folders: AUROC'],rows('E_leg'),[460,350,350],row_h=95,size=23),
        'Compare training masks within this table; changing the task also changes what the score measures.')
    baseline=[('Movement, 16 features','M'),('Movement, 36 features','motion36'),('Average of M and own-leg improvement','S_sham'),('Average of M and correction errors','S_constituent'),('Average of M and baseline error','S_base_error'),('Timing and waveform features','timing8'),('Isolation Forest','isolation_forest')]
    add(27,'Simple comparisons help explain the anomaly scores.',
        'These methods use the same recordings and normal-only fitting, but summarize movement in different ways.',
        tab(['Method','Four groups','All folders'],[(name,f"{score(key):.3f}",f"{score(key,'broad'):.3f}") for name,key in baseline],[680,240,240],row_h=48,size=21),
        'Prediction scores use normal-reference scaling before averaging with M. Table values are AUROCs.')
    aux=run.json('auxiliary_metrics.json')
    add(28,'A published pose model needs a missing-data adaptation.',
        'STG-NF models pose likelihoods. We adapt its graph and fill missing values using fitting-normal medians.',
        tab(['Method','Four groups','All folders'],[('Adapted STG-NF',f"{aux['focal']['stgnf_adapted']['weighted_auroc']:.3f}",f"{aux['broad']['stgnf_adapted']['weighted_auroc']:.3f}"),('Movement score',f"{score('M'):.3f}",f"{score('M','broad'):.3f}"),('Combined score',f"{score('S'):.3f}",f"{score('S','broad'):.3f}")],[680,240,240],row_h=95,size=24),
        'Only five normal recordings have complete poses, so this is an adapted comparison.')
    classifiers=pd.json_normalize(run.json('five_class_metrics.json')).groupby('method')[['macro_f1','balanced_accuracy']].mean()
    names={'random_forest':'Random Forest','learned_jepa_probe':'Learned JEPA features','random_encoder_probe':'Random encoder features'}
    add(29,'Condition classification uses labels in a separate experiment.',
        'We fit classifiers for five categories after obtaining motion features or encoder features.',
        tab(['Classifier input or method','Macro F1','Balanced accuracy'],[(names[k],f'{v.macro_f1:.3f}',f'{v.balanced_accuracy:.3f}') for k,v in classifiers.iterrows()],[620,270,270],row_h=95,size=24),
        'Macro F1 weights the five classes equally; the small learned-feature run does not improve here.')
    add(30,'Shift the other leg while keeping the hidden answer fixed.',
        'Earlier and later observations come from one valid 48-sample interval, without wrapping or padding.',timing_diagram,
        'Only visible donor joints and hip anchors are allowed, and the shared body reference stays fixed.','THE TIMING TEST | ILLUSTRATION')
    add(31,'The timing comparison leaves very little usable context.',
        'Both versions must use observations that are valid and permitted in the aligned and shifted inputs.',
        cards([('27 normal recordings','There are enough recordings to meet the planned coverage floor.'),('About 1.75 tokens','Few editable lower-leg tokens remain after matching visible observations.'),('No adequate outer comparison','All 24 model, fold and offset combinations fail the aligned-input prediction check.')]),
        'The 72 corresponding internal comparisons also fail; recording count alone is not enough.')
    timing=run.json('timing_summary.json')
    timingrows=sorted([r for r in timing['summaries'] if r['scope']=='outer_test_normal' and r['policy']=='U' and r['variant']=='timing'],key=lambda r:r['offset'])
    add(32,'A timing response needs an adequate starting prediction.',
        'The Random U model uses 27 normal test recordings; prediction is already poor with the aligned inputs.',
        tab(['Other-leg observations','Mean error change','95% interval'],[('Four frames earlier' if r['offset']<0 else 'Four frames later',f"{r['mean_effect']:.6f}",f"[{r['conditional_source_bootstrap_95_interval'][0]:.6f}, {r['conditional_source_bootstrap_95_interval'][1]:.6f}]") for r in timingrows],[370,320,470],row_h=110,size=24),
        'Positive change means higher shifted error; this response does not identify a coordination mechanism.')
    add(33,'Test the prediction method on independent participants.',
        'The laboratory dataset records both legs on one clock and provides participant identities.',
        cards([('188 selected trials','Ten healthy trials do not meet the fixed observation requirements.'),('178 eligible participants','The analysis includes 128 healthy people and 50 people after stroke.'),('Six joint-angle signals','Fit new hip, knee and ankle models, with 32 healthy test participants in each fold.')]),
        'This repeats the method with new measurements; it does not transfer video-trained weights.')
    errors=pd.read_csv(external/'heldout_prediction_error.csv').query('binary_label==0').sort_values('outer_fold')
    add(34,'The both-leg model improves laboratory prediction.',
        'Every test fold shows lower healthy-participant error than the own-leg correction.',
        lambda c:api.external_gain(c,errors.both_vs_own_relative_gain.to_numpy()*100),
        'Each fold contains 32 healthy test participants; values are relative reductions in prediction error.')
    add(35,'Laboratory anomaly ranking also favors movement alone.',
        'The cross-leg score provides a ranking, but its fixed average with movement reduces accuracy.',
        tab(['Laboratory score','AUROC'],[(name,f"{ext[k]['weighted_auroc']:.3f}") for name,k in [('Movement M','M'),('Cross-leg R','R'),('Combined S','S')]],[900,260],row_h=95,size=25),
        'Combined minus movement AUROC is −0.066, with a 95% interval from −0.106 to −0.031.')
    add(36,'Compare participants with similar age and the same sex.',
        'The fixed matching rule pairs healthy and stroke participants within five years inside each test fold.',
        tab(['Matched-participant comparison','Value'],[('Healthy/stroke pairs',sum(match['by_fold'].values())),('Mean absolute age difference',f"{match['mean_absolute_age_difference']:.2f} years"),('Movement / combined AUROC',f"{match['metrics']['M']['weighted_auroc']:.3f} / {match['metrics']['S']['weighted_auroc']:.3f}"),('95% interval for combined minus movement',f"[{match['conditional_95_interval'][0]:.3f}, {match['conditional_95_interval'][1]:.3f}]")],[790,370],row_h=77,size=23),
        'The models are unchanged, and walking speed remains unmatched.')
    add(37,'Recording conditions still limit what we can conclude.',
        'A difference between web recordings can reflect the camera, missing observations or the people being recorded.',
        cards([('Camera and viewpoint','Too few normal recordings share a view for a complete matched-view analysis.'),('Measurement quality','High-confidence joints can still lie outside the image or belong to the wrong person.'),('People and labels','Upload IDs are not verified identities, and folder labels lack standardized clinical confirmation.')]),
        'Review clips and poses alongside scores, without inventing a clinical explanation for an individual video.')
    add(38,'The method makes cross-leg evidence easier to examine.',
        'The experiments measure useful prediction and anomaly ranking separately, under explicit observation rules.',
        cards([('A controlled prediction','Keep the local estimate and compare two corrections on the same answer.'),('A task-specific benefit','Both-leg prediction helps on grouped video gaps and laboratory angles.'),('A separate scoring problem','The fixed combined score does not improve on movement alone in either dataset.')]),
        'The contribution is the tested comparison and its evidence, rather than a new masking family.')
    add(39,'Six notebooks explain how the evidence is produced.',
        'Each notebook runs independently and connects the measurements to real video or saved participant data.',
        tab(['Notebooks','What they teach'],[('12–13','Prepare observations and compare own-leg with both-leg prediction.'),('14–15','Run timing controls and construct the three anomaly scores.'),('16–17','Compare methods, inspect failures and assess the manuscript claims.')],[270,890],row_h=100,size=25),
        'The notes link the notebooks, producer commands and verified execution records.')
    add(40,'The next experiment should test when cross-leg scores help.',
        'The next study should improve model comparisons, estimate normal variation reliably and use new participants.',
        cards([('Match model flexibility','Strengthen the own-leg comparison before attributing a gain solely to added information.'),('Improve normal references','Estimate typical score variation without selecting weights on abnormal test labels.'),('Use new participants','Evaluate the frozen method on new people or repeated trials with known identities.')]),
        'These are proposed experiments; no improved detector or clinical mechanism has been demonstrated.')
    normal_errors=run.table('normal_error_table.csv').query("policy=='U' and bank=='E_leg'")
    own,both=normal_errors.loss_own.mean(),normal_errors.loss_both.mean()
    add(41,'The other leg helps complete two kinds of measurements.',
        'Both rows compare the both-leg correction with the own-leg correction on the same hidden answers.',
        tab(['Normal test population','Own-leg MSE','Both-leg MSE','Reduction'],[('GAVD: 28 recordings',f'{own:.6f}',f'{both:.6f}',f'{100*(own-both)/own:.1f}%'),('Laboratory: 128 people',f'{lab.comparator_mean_mse:.3f}',f'{lab.candidate_mean_mse:.3f}',f'{100*lab.relative_mse_reduction:.1f}%')],[430,240,260,230],row_h=110,size=23),
        'Mean squared errors use torso-scaled coordinates for video and degrees for laboratory data. Do not pool them.')
    add(42,'The laboratory improvement appears across participants.',
        'Paired summaries of the saved predictions show the size and consistency of the completion gain.',
        cards([(f'{100*lab.relative_mse_reduction:.1f}% lower mean error','The mean falls from 201.038 to 190.584 squared degrees.'),('A conditional interval',f'The 95% interval is [{100*lab.relative_ci_low:.1f}%, {100*lab.relative_ci_high:.1f}%], with fitted models held fixed.'),(f'{int(lab.units_with_lower_candidate_mse)} of {int(lab["count"])} improve','These participants have lower prediction error with the both-leg correction.')]),
        'These additional intervals were computed after inspecting results and do not include retraining uncertainty.')
    mr=[]
    for bank in ['U','E_leg']:
        z=completion.query("dataset=='GAVD' and outer_fold=='all'");z=z.loc[z.bank.eq(bank)].iloc[0]
        mr.append((TASK[bank],f'{100*z.relative_mse_reduction:.1f}%',f'[{100*z.relative_ci_low:.1f}%, {100*z.relative_ci_high:.1f}%]',f'{int(z.units_with_lower_candidate_mse)} / 28'))
    add(43,'Anatomical training helps when neighboring joints are hidden.',
        'Only the correction training mask changes; predictor design and evaluation targets remain the same.',
        tab(['Evaluation task','Error reduction','95% interval','Recordings improved'],mr,[350,260,270,280],row_h=115,size=23),
        'The percentage compares anatomical with random training; the scattered-test interval includes zero.')
    notes={}
    for filename in ['notes_01_15.json','notes_16_30.json','notes_31_43.json']:
        source=BASE/'slides/source'/filename
        if not source.exists():raise FileNotFoundError('Missing speaker-note source: '+str(source))
        part=json.loads(source.read_text())
        if notes.keys() & part.keys():raise ValueError('Duplicate note IDs')
        notes.update(part)
    for position,n in enumerate(ORDER,1):
        r=records[n];r['position']=position;r['notes_sections']=notes[str(n)]
        r['score_guide']=n in [1,21,22,23,24,25,26,27,28,35,36,38,40]
        assert set(r['notes_sections'])=={k for k,_ in NOTE_FIELDS},n
        r['notes']='\n\n'.join(r['notes_sections'][k] for k,_ in NOTE_FIELDS)
        if len(r['notes'].split())<140:raise ValueError('Notes too short: '+str(n))
        if n in [1,6,38,40,41,42,43]:r['refs']=list(dict.fromkeys(r['refs']+['draft03']))
        r['refs']=[x for x in r['refs'] if x!='draft02']
        if n in [3,5,9,12,13,14,15,16,17,21,22,30]:r['tag']='METHOD EXPLANATION | ILLUSTRATION'
        elif n>=18 and n not in [21,22,30,38,39,40]:r['tag']='RESULTS FROM SAVED REAL-DATA EXPERIMENTS'
    deck.records=[records[n] for n in ORDER]
    return deck
