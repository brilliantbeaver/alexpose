"""Build the restart paper only from completed, identity-checked real run outputs.

Run with the worldmodels project interpreter. Compilation and rendering are explicit
options because the local TeX and PDF tools are external to the project environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT.parents[1]
OUT = PROJECT/'docs/draft01'
RUN = PROJECT/'artifacts/runs/restart-dynamics-v1'
DATA = PROJECT/'artifacts/runs/restart-full10hz-v1'
LABELS = ['normal', 'parkinsons', 'stroke', 'cerebral palsy', 'myopathic']
NAMES = {'persistence': 'Persistence', 'damped_velocity': 'Damped velocity',
         'own': 'Own-leg history', 'bilateral': 'Bilateral history',
         'shifted': 'Shifted bilateral', 'static': 'Static posture',
         'own_shrunk': 'Shrunk own-leg', 'bilateral_shrunk': 'Shrunk bilateral',
         'shifted_shrunk': 'Shrunk shifted', 'static_shrunk': 'Shrunk static',
         'inference_shift': 'Inference shift', 'inference_static': 'Inference static'}
SCORES = {'bilateral': 'Bilateral MSE', 'persistence': 'Persistence MSE',
          'excess_error': 'Excess MSE', 'coordination_deficit': 'Bilateral minus own',
          'prefix_missingness': 'Prefix missingness'}
SHORT = {'persistence':'Persistence', 'damped_velocity':'Velocity', 'own':'Own',
         'bilateral':'Bilateral', 'shifted':'Shifted', 'static':'Static',
         'own_shrunk':'Own (S)', 'bilateral_shrunk':'Bilateral (S)',
         'shifted_shrunk':'Shifted (S)', 'static_shrunk':'Static (S)',
         'inference_shift':'Perturbed time', 'inference_static':'Perturbed static'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def esc(text):
    mapping = {'&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_',
               '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}',
               '^': r'\textasciicircum{}', '\\': r'\textbackslash{}'}
    return ''.join(mapping.get(c, c) for c in str(text))


def ci(item, factor=1, digits=5):
    return f"{item['estimate']*factor:.{digits}f} [{item['ci95'][0]*factor:.{digits}f}, {item['ci95'][1]*factor:.{digits}f}]"


def table(headers, rows, align=None, caption=None, label=None):
    align = align or ('l'+'r'*(len(headers)-1))
    body = ['\\begin{table}[t]', '\\centering', '\\small',
            '\\begin{tabular}{'+align+'}', '\\toprule',
            ' & '.join(esc(x) for x in headers)+r' \\', '\\midrule']
    body.extend(' & '.join(esc(x) for x in row)+r' \\' for row in rows)
    body.extend(['\\bottomrule', '\\end{tabular}'])
    if caption:
        body.append('\\caption{'+caption+'}')
    if label:
        body.append('\\label{'+label+'}')
    return '\n'.join(body+['\\end{table}'])


def md_table(headers, rows):
    return '\n'.join(['| '+' | '.join(map(str, headers))+' |',
                      '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def load_evidence():
    if not (RUN/'results.json').exists() or not (DATA/'preparation.json').exists():
        raise SystemExit('Completed results.json and preparation.json are required; no empirical draft generated.')
    result = json.loads((RUN/'results.json').read_text())
    prep = json.loads((DATA/'preparation.json').read_text())
    if result['status'] != 'complete' or prep['mode'] != 'all-eligible-sequences-full-duration':
        raise ValueError('Only completed full-cohort empirical results can enter the paper')
    manifest = json.loads((RUN/'artifact_hashes.json').read_text())
    for name, expected in manifest.items():
        if digest(RUN/name) != expected:
            raise ValueError('Run artifact hash mismatch: '+name)
    for name, key in [('windows.npz', 'data_sha256'), ('windows.csv', 'metadata_sha256')]:
        if digest(DATA/name) != prep[key]:
            raise ValueError('Input hash mismatch: '+name)
    source = pd.read_csv(RUN/'source_scores.csv')
    cohort = pd.read_csv(RUN/'cohort.csv')
    selection = pd.DataFrame(json.loads((RUN/'selection.json').read_text()))
    audit = json.loads((PROJECT/'docs/restart/data_audit.json').read_text())
    forecasts = {(row['arm'], row['horizon']): row for row in result['normal_forecast']}
    contrasts = {(row['reference'], row['method'], row['horizon']): row for row in result['paired_contrasts']}
    checks = []
    for horizon in (4, 8):
        normal = source[(source.label == 'normal') & (source.horizon == horizon)].pivot(index='source_id', columns='arm', values='mse')
        for arm in normal:
            np.testing.assert_allclose(normal[arm].mean(), forecasts[arm, horizon]['estimate'], rtol=1e-10, atol=1e-12)
            assert len(normal) == forecasts[arm, horizon]['n_sources']
        for (a, b, h), item in contrasts.items():
            if h == horizon:
                np.testing.assert_allclose((normal[a]-normal[b]).mean(), item['estimate'], rtol=1e-10, atol=1e-12)
        checks.append(f'Independent mean and paired-effect recomputation for horizon {horizon}')
    abnormal = pd.read_csv(RUN/'source_anomaly_scores.csv')
    y = (abnormal.label != 'normal').astype(int)
    for name, item in result['anomaly'].items():
        np.testing.assert_allclose(roc_auc_score(y, abnormal[name]), item['auroc'], atol=1e-12)
        np.testing.assert_allclose(average_precision_score(y, abnormal[name]), item['average_precision'], atol=1e-12)
    assert len(cohort) == result['windows']
    assert cohort.source_id.nunique() == result['sources']
    assert cohort.groupby('source_id').outer_fold.nunique().max() == 1
    checks += ['All five anomaly AUROC/AP estimates independently recomputed', 'Source-disjoint fold assignment and cohort counts checked',
               'All recorded result artifacts and input arrays match hashes']
    return result, prep, source, cohort, selection, audit, forecasts, contrasts, checks


def save_figure(fig, name):
    folder = OUT/'figures'
    folder.mkdir(exist_ok=True)
    for suffix in ('svg', 'pdf', 'png'):
        date=datetime(2026,11,16,tzinfo=timezone.utc)
        metadata={'CreationDate':date,'ModDate':date} if suffix=='pdf' else ({'Date':date.isoformat()} if suffix=='svg' else None)
        fig.savefig(folder/f'{name}.{suffix}', dpi=180, bbox_inches='tight', facecolor='white',metadata=metadata)
    plt.close(fig)


def build_figures(source, contrasts):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'svg.fonttype': 'none','svg.hashsalt':'medwm-restart-draft01-v1',
                         'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.set(xlim=(0, 10), ylim=(0, 4)); ax.axis('off')
    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08', facecolor=color, edgecolor='#314455', linewidth=.8))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=8)
    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9, color='#314455'))
    box(.1, 1.25, 1.8, 1.45, 'GAVD video\nAmbient pose\n10 Hz samples', '#e9f1f5')
    box(2.4, 2.2, 2.35, 1.3, 'Own-leg past\nShared reference', '#e4f2ec')
    box(2.4, .25, 2.35, 1.3, 'Add opposite leg\nOrdered or shifted', '#f7ecd8')
    box(5.25, 1.25, 2.2, 1.45, 'Normal-only ridge\nInner source folds\nSame future targets', '#e8e9f7')
    box(8, 1.25, 1.8, 1.45, 'Predict 0.8 s\nScore by source\nCompare controls', '#e4f2ec')
    arrow((1.98, 2.2), (2.32, 2.7)); arrow((1.98, 1.65), (2.32, .95))
    arrow((4.83, 2.7), (5.17, 2.2)); arrow((4.83, .95), (5.17, 1.65)); arrow((7.53, 1.97), (7.92, 1.97))
    ax.text(5, 3.85, 'Does correctly timed bilateral context improve prediction?', ha='center', fontsize=11, weight='bold')
    ax.text(5, -.15, 'All windows from one source stay in one outer fold. No future observation enters the predictor.', ha='center', fontsize=8)
    save_figure(fig, 'method')

    keys = [('persistence', 'bilateral'), ('own', 'bilateral'), ('shifted', 'bilateral'),
            ('static', 'bilateral'), ('persistence', 'bilateral_shrunk')]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    names = ['Persistence', 'Own history', 'Shifted history', 'Static posture', 'Persistence*']
    for i, (a, b) in enumerate(keys):
        item = contrasts[a, b, 8]
        val = item['estimate']*1000
        low, high = np.array(item['ci95'])*1000
        color = '#1f6a5b' if i < 4 else '#93631b'
        ax.errorbar(val, i, xerr=[[max(val-low, 0)], [max(high-val, 0)]], fmt='o', color=color, capsize=3, markersize=4)
    ax.axvline(0, color='#7b8490', linewidth=.8, linestyle='--')
    ax.set(yticks=range(len(keys)), yticklabels=names, xlabel='Reference MSE minus forecast MSE (×10³)')
    ax.invert_yaxis(); ax.grid(axis='x', alpha=.18)
    ax.text(0, -.36, '*Secondary shrunk bilateral forecast; others use primary bilateral.', transform=ax.transAxes, fontsize=6.8)
    fig.tight_layout()
    save_figure(fig, 'paired_effects')

    normal = source[(source.label == 'normal') & (source.horizon == 8)].pivot(index='source_id', columns='arm', values='mse')
    normal = normal.sort_values('persistence')
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    xx = np.arange(len(normal))
    ax.vlines(xx, normal.bilateral*1000, normal.persistence*1000, color='#bdc5ca', linewidth=1)
    ax.scatter(xx, normal.persistence*1000, label='Persistence', color='#647889', s=17)
    ax.scatter(xx, normal.bilateral*1000, label='Bilateral history', color='#1f6a5b', s=17)
    ax.scatter(xx, normal.bilateral_shrunk*1000, label='Shrunk bilateral', marker='x', color='#93631b', s=20)
    ax.set(xlabel='Normal source, ordered by persistence error', ylabel='MSE ×10³', xticks=xx, xticklabels=np.arange(1, len(normal)+1))
    ax.tick_params(axis='x', labelsize=6); ax.legend(frameon=False, fontsize=8, ncol=3)
    ax.grid(axis='y', alpha=.18)
    fig.tight_layout(); save_figure(fig, 'normal_sources')
    normal.reset_index().to_csv(OUT/'figures/normal_sources_data.csv', index=False)


def build_paper(evidence, author, affiliation):
    result, prep, source, cohort, selection, audit, forecasts, contrasts, checks = evidence
    final_labels = {row['label']:row for row in result['cohort_by_label']}
    candidate_labels = {row['label']:row for row in audit['candidate_after_all_existing_exclusions']['by_label']}
    gain = contrasts['persistence', 'bilateral', 8]
    own = contrasts['own', 'bilateral', 8]
    shift = contrasts['shifted', 'bilateral', 8]
    static = contrasts['static', 'bilateral', 8]
    shrink = contrasts['persistence', 'bilateral_shrunk', 8]
    selected = selection[selection.arm == 'bilateral_shrunk']
    beta_counts = selected.beta.value_counts().sort_index()
    beta_text = ', '.join(f'{float(beta):g}: {int(count)}/10' for beta, count in beta_counts.items())
    normal = source[(source.label == 'normal') & (source.horizon == 8)].pivot(index='source_id', columns='arm', values='mse')
    win_count = int((normal.persistence > normal.bilateral).sum())
    shrunk_wins = int((normal.persistence > normal.bilateral_shrunk).sum())
    relgain = 100*gain['estimate']/forecasts['persistence', 8]['estimate']
    elapsed = cohort.final_horizon_seconds
    timing = f'{elapsed.min():.3f} to {elapsed.max():.3f}'
    if gain['ci95'][0] > 0:
        forecast_claim = 'Bilateral history improved normal-source forecasting over persistence in this exploratory evaluation.'
    elif gain['estimate'] > 0:
        forecast_claim = 'The mean forecast error favored bilateral history, but its conditional interval included no improvement.'
    else:
        forecast_claim = 'The primary bilateral model did not improve mean normal-source forecasting over persistence.'
    temporal_supported = all(x['ci95'][0] > 0 for x in (gain, own, shift, static))
    if temporal_supported:
        timing_claim = 'The complete set of primary comparisons supports useful, timing-sensitive bilateral information within this observation contract.'
    else:
        timing_claim = 'The complete set of primary comparisons does not establish a timing-specific bilateral forecasting advantage.'
    if shrink['ci95'][0] > 0:
        shrink_claim = 'The secondary shrunk forecast also improves persistence, but does not improve on the unshrunk bilateral model.'
    elif shrink['estimate'] > 0:
        shrink_claim = 'Shrinkage has a favorable point estimate, with uncertainty that prevents a clear forecast-utility claim.'
    else:
        shrink_claim = 'The shrinkage amendment did not yield a positive mean gain over persistence.'
    title = 'Gait Forecasting for Medical World Models: A Source-Grouped Study'
    abstract = (f'Medical world models need observation dynamics that remain useful under simple controls. '
                f'We test whether correctly timed opposite-leg history improves short-horizon gait forecasts beyond a leg\'s own history. '
                f'Full-duration processing of selected GAVD videos yields {result["windows"]:,} eligible windows from '
                f'{result["sequences"]} sequences and {result["sources"]} source videos, including {result["normal_sources"]} normal sources. '
                f'Normal-only ridge predictors forecast eight observations at a nominal 10 Hz using nested source-grouped cross-validation. '
                f'The primary bilateral forecast reduces mean squared error by {relgain:.1f}% relative to persistence, with paired difference {ci(gain)}. '
                f'The corresponding own-history and shifted-context differences are {ci(own)} and {ci(shift)}. '
                f'A current-posture-conditioned regression is stronger still, with MSE {forecasts["static",8]["estimate"]:.5f} versus bilateral {forecasts["bilateral",8]["estimate"]:.5f}. '
                f'{forecast_claim} {timing_claim} '
                f'A predeclared movement-referenced excess-error score reaches abnormality AUROC {result["anomaly"]["excess_error"]["auroc"]:.3f}, a secondary association. '
                f'Intervals resample sources conditional on fixed cross-validation fits. The study is exploratory because earlier GAVD outcomes were known. '
                f'The contribution is a controlled evaluation of a motor-observation layer, without claims about treatment, disease progression, or neurological mechanism.')
    intro = (
        'Movement supplies an observable aspect of patient state. A gait observation model could eventually support longitudinal assessment, but low prediction error alone does not show that a system has learned clinically meaningful dynamics. Persistence may already predict a short future reasonably well, while appearance, movement amplitude, and recording conditions may distinguish dataset labels. We therefore separate useful forecasting, timing-sensitive information, and abnormality ranking.',
        'Neurological evidence motivates a relational question without fixing its answer. Laboratory gait exhibits structured intersegmental covariation [borghese1996], and bilateral phase coordination is affected in Parkinson\'s disease [plotnik2007]. Studies of stroke and cerebral palsy relate altered EMG modular organization to impaired movement [clark2010; steele2015]. These measurements differ from projected video coordinates. A predictable compensation can still be abnormal, and no video forecast score identifies a neural module or lesion.',
        'We ask whether adding the opposite leg\'s correctly timed history improves prediction beyond the target leg\'s own history. Our contribution is a fixed-time, fuller-sequence, source-grouped audit with matched timing controls and explicit negative-result conditions. Bilateral gait analysis and normal-only prediction are established ideas. We aim to determine what the available observations actually support before introducing a more complex medical model.')
    related = (
        'Skeleton S-JEPA learns action representations through latent masked-observation prediction [abdelfattah2024]. This is distinct from a strict future-only task. MoCoDAD generates future skeletons for anomaly detection [flaborea2023]; GenGait reconstructs normative gait [motta2026]; BioGait-VLM combines visual, language, and biomechanical evidence [chen2026]. These precedents preclude broad first-method claims. Strong static baselines are particularly relevant because low motion-prediction error need not require a learned motion model [martinez2017].',)
    data = (
        f'The pinned local GAVD derivative contains 1,874 annotation sequences and 348 source IDs. The requested normal, Parkinson\'s, stroke, cerebral palsy, and myopathic labels contain 666 sequences from 103 sources [ranjan2025]. Media checks retain 652 sequences from 98 sources. Two previously reviewed sources with person switches or scene cuts are excluded, leaving 641 sequences from 96 sources for extraction. Timing, person, and observation gates retain {prep["n_windows"]:,} windows from {prep["n_sequences"]} sequences and {prep["n_sources"]} sources. The common distal-target gate removes {prep["n_windows"]-result["windows"]} windows, retaining {result["windows"]:,} windows from {result["sequences"]} sequences and {result["sources"]} sources. These are video groups, not verified independent people.',
        f'We reuse cached originals under data/videos-gavd and perform {prep["actual_model_calls_this_invocation"]:,} recorded Ambient pose-model calls in this preparation invocation. We sample nearest native frames on a nominal 10 Hz grid without interpolating target observations. Each window contains 16 past and eight future samples, with stride eight. The final forecast time spans {timing} seconds relative to the last prefix frame. Actual frame indices, decoded times, confidence, and missingness are retained. Each source/native-frame tuple is decoded for pose once within a source cache, and identical window tuples are deduplicated.',
        'The observation contract uses BlazePose joints 11–16 and 23–32, confidence at least 0.5, and at least 70% observed joints in both the 16-sample prefix and eight-sample suffix. Full-image detection is followed by an annotation-box pelvis gate; unresolved geometry is rejected. This gate cannot guarantee identity when people overlap. Coordinates are centered on each frame\'s pelvis and divided by a torso scale estimated only from the prefix. Forecasts therefore describe relative 2D articulation, not physical velocity or clinical joint angles. Own-leg input is conditional on this shared body reference, which already uses both hips.',
        f'Retention differs by annotation: {final_labels["normal"]["sequences"]} of {candidate_labels["normal"]["sequences"]} normal sequences selected for extraction remain ({100*final_labels["normal"]["sequences"]/candidate_labels["normal"]["sequences"]:.1f}%), compared with {final_labels["myopathic"]["sequences"]} of {candidate_labels["myopathic"]["sequences"]} myopathic sequences ({100*final_labels["myopathic"]["sequences"]/candidate_labels["myopathic"]["sequences"]:.1f}%). Gates combine duration, geometry, identity, and observation support; this is not purely pose-quality attrition. Selection can affect pooled abnormality ranking. All 32 annotated normal sources are already in the selected scope, so other GAVD labels cannot increase its normal-source ceiling. More windows improve temporal coverage without adding independent patients.')
    method = (
        'For each target leg, we predict the next eight samples of knee, ankle, heel, and foot-index x/y coordinates. Prefix values are causally forward-filled, with zero before the first observation and explicit observation indicators. Eight fixed prefix positions, including both endpoints, limit dimensionality. The own-leg design has 104 features; the bilateral design has 208, including historical residuals, observation indicators, and anchors. Targets are residuals from each joint\'s latest observed prefix position. Output-specific ridge fits use observed targets only; imputed targets never become measurements.',
        'The own-leg model uses target-leg history. The bilateral model adds opposite-leg history. A separately trained shifted model retains the same selected values, missingness, and latest opposite-leg observation, but cyclically shifts its seven earlier observations by a window-specific offset derived from the window identity. This disrupts timing with matched capacity; a global feature permutation would instead be absorbable by ridge regression. The static-context model repeats the last observed posture as input but learns a separate future transition at each horizon. It does not repeat a static output. Persistence and velocity extrapolation with fixed damping 0.5 provide elementary references. Additional inference-time shift and static perturbations are reported separately because they can create input distribution shift.',
        'Five outer folds stratify source IDs by annotation with seed 2701. Only normal sources outside each outer fold fit motion models. Three inner source folds select ridge strength from {0.1, 1, 10, 100}; scaling is refitted inside every inner fold. A prospective secondary amendment jointly selects residual shrinkage beta from {0, 0.25, 0.5, 1}, with zero corresponding to persistence. Unshrunk models remain primary. Left and right legs are fitted separately, producing 80 saved fold/leg/arm checkpoints, some sharing the same fitted ridge solution.',
        'All methods use a common observed target support. Each leg must have at least four observed joint-time points in the first four future samples and eight over the full future; each target joint must also have a prefix observation. Coordinate errors are averaged within leg and window, then legs equally, windows within sequence, sequences within source, and sources equally. Training weights follow the same source/sequence/window hierarchy with adjustment for observed target counts. This prevents long videos or denser pose support from determining the endpoint.',
        'The primary endpoint is normal-source MSE averaged over all eight future observations, nominally 0.1 to 0.8 seconds, rather than only the final instant. Paired differences compare bilateral prediction with persistence, own history, shifted history, and static posture; positive values favor bilateral prediction. Four-step results and shrinkage are secondary. We use 5,000 source resamples for error intervals and 2,000 stratified source resamples for secondary AUROC and average precision. These descriptive intervals condition on fixed fits with overlapping training folds. They exclude retraining and patient-population uncertainty and are not multiplicity-adjusted confirmatory tests. Earlier GAVD outcomes were exposed, so the new cross-fitting study is exploratory.')
    results_text = (
        f'{forecast_claim} Persistence and bilateral normal-source MSE are {forecasts["persistence",8]["estimate"]:.5f} and {forecasts["bilateral",8]["estimate"]:.5f}. Their paired difference is {ci(gain)}, equivalent to {relgain:.1f}% of the persistence mean. Bilateral error is lower on {win_count} of {result["normal_sources"]} normal sources. The gain over own history is {ci(own)}, over the separately trained timing control {ci(shift)}, and over static posture {ci(static)}. {timing_claim}',
        f'The secondary shrunk bilateral forecast has persistence-minus-model difference {ci(shrink)} and lower error on {shrunk_wins} normal sources. Its selected beta counts across ten fold/leg fits are {beta_text}. {shrink_claim} Selection used only inner normal-source outcomes; no model or score was promoted based on outer abnormal labels. Appendix tables retain every arm, both horizons, and all paired comparisons.',
        'Abnormality ranking is a separate descriptive endpoint. Larger bilateral MSE, excess MSE above persistence, and bilateral-minus-own MSE are fixed to indicate more abnormal gait. Persistence error and prefix missingness are controls. '
        + ' '.join(f'{SCORES[key]} has AUROC {val["auroc"]:.3f} [{val["auroc_ci95"][0]:.3f}, {val["auroc_ci95"][1]:.3f}].' for key,val in result['anomaly'].items())
        + f' The abnormal-source prevalence is {result["anomaly"]["bilateral"]["abnormal_prevalence"]:.3f}. No score direction is reversed, and these comparisons do not measure clinical sensitivity or diagnosis accuracy.')
    discussion = (
        'The positive forecasting gain shows usable information in this observation contract. Correctly timed bilateral context also outperforms its matched shifted control. However, the small uncertain gain over own history and the stronger current-posture-conditioned predictor prevent a broader claim of bilateral temporal superiority. The static-context model learns horizon-specific transitions from the current state; its performance is a useful positive reference, not evidence that the output is motionless. These findings favor testing a compact current-state transition model before increasing temporal complexity.',
        'The predeclared excess-error score also has a favorable abnormality association, while raw error ranks in the opposite direction. Subtracting matched persistence error accounts for a basic component of observed motion difficulty, but does not isolate disease or neurological mechanism. Predictive dependence can arise from mechanics, speed, camera geometry, compensation, and correlated pose noise. The model has no treatment input, longitudinal disease state, clinical reward, or intervention outcome.',
        'Generalization is limited by public-video selection, uncertain participant overlap, annotation validity, acquisition differences, and quality-dependent attrition. Source grouping addresses repeated windows from a video, but cannot establish patient-disjoint or external-site evaluation. Pooled out-of-fold AUROC also assumes comparable score scales across separately fitted fold models; cross-fold offsets remain a limitation. Small normal-source support limits model selection and uncertainty estimates. GAVD alone therefore supports an exploratory observational benchmark, not deployment claims. The highest-value extension is an external cohort with verified identities, clinician assessments, repeated visits, and measured interventions.',
        f'{forecast_claim} {timing_claim} The reusable contribution is a complete evidence chain from real video inference to controlled source-level estimates. It makes forecast utility, timing-specific information, and abnormality association separately testable, which is necessary before a gait observation component can support stronger medical modeling claims.')
    sections = [('Introduction', intro), ('Related work', related), ('Data and observation contract', data),
                ('Predictors and evaluation', method), ('Results', results_text), ('Discussion and limitations', discussion)]
    main_arms = ['persistence','damped_velocity','own','bilateral','shifted','static','bilateral_shrunk']
    main_rows = [[NAMES[a], f'{forecasts[a,8]["estimate"]*1000:.3f}',
                  f'[{forecasts[a,8]["ci95"][0]*1000:.3f}, {forecasts[a,8]["ci95"][1]*1000:.3f}]'] for a in main_arms]
    cohort_map = {r['label']:r for r in result['cohort_by_label']}
    cohort_rows = [[label.title(), audit['inventory']['by_label'][label]['sources'],
                    cohort_map[label]['sources'], cohort_map[label]['windows']] for label in LABELS]
    forecast_rows = [[NAMES[row['arm']], row['horizon']/10, ci(row,1000,3)] for row in result['normal_forecast']]
    contrast_rows = [[NAMES[row['reference']], NAMES[row['method']], row['horizon']/10, ci(row,1000,3)] for row in result['paired_contrasts']]
    anomaly_rows = [[SCORES[key], f'{val["auroc"]:.3f} [{val["auroc_ci95"][0]:.3f}, {val["auroc_ci95"][1]:.3f}]',
                     f'{val["average_precision"]:.3f} [{val["ap_ci95"][0]:.3f}, {val["ap_ci95"][1]:.3f}]'] for key,val in result['anomaly'].items()]
    tables = {'cohort': {'headers': ['Annotation', 'Catalog sources', 'Final sources', 'Windows'], 'rows': cohort_rows},
              'main_forecasts': {'headers': ['Forecast', 'MSE ×1000', '95% interval'], 'rows': main_rows},
              'all_forecasts': {'headers': ['Forecast', 'Horizon (s)', 'MSE ×1000 [95% interval]'], 'rows': forecast_rows},
              'paired_contrasts': {'headers': ['Reference', 'Forecast', 'Horizon (s)', 'Gain ×1000 [95% interval]'], 'rows': contrast_rows},
              'anomaly': {'headers': ['Score', 'AUROC [95% interval]', 'AP [95% interval]'], 'rows': anomaly_rows}}
    md = [f'# {title}', f'{author}\n\n{affiliation}', '## Abstract', abstract]
    for heading, paragraphs in sections:
        md += ['## '+heading, '\n\n'.join(paragraphs)]
        if heading == 'Predictors and evaluation':
            md += [r'For prefix features $z_i$ and latest observed articulation $a_{i,j}$, the predictor is $\widehat y_{i,h,j}=a_{i,j}+\beta(z_i^\top w_{h,j}+b_{h,j})$. The primary model fixes $\beta=1$; the secondary amendment selects it inside normal-source validation.']
        if heading == 'Data and observation contract':
            md += [md_table(**tables['cohort'])]
        if heading == 'Results':
            md += [md_table(**tables['main_forecasts']), '![Paired effects](figures/paired_effects.svg)']
    md += ['## Reproducibility appendix', 'All tables use completed run artifacts. Intervals are conditional and descriptive.']
    for name in ['all_forecasts', 'paired_contrasts', 'anomaly']:
        md += [f'### {name.replace("_", " ").title()}', md_table(**tables[name])]
    md += ['### Selection', f'Bilateral shrinkage beta counts: {beta_text}. Complete per-fold and per-leg selection is in selection.csv.',
           '### References', 'Citation keys resolve to the accompanying references.bib. The compiled PDF contains the formatted bibliography.']
    (OUT/'manuscript.md').write_text('\n\n'.join(md)+'\n')
    def prose(text):
        s = esc(text.replace('–', '--').replace('×', r' x '))
        return re.sub(r'\[([a-z][a-z0-9]*(?:; [a-z][a-z0-9]*)*)\]', lambda m: r'\citep{'+m.group(1).replace('; ', ',')+'}', s)
    tex = [r'\documentclass[letterpaper]{article}', r'\usepackage[preprint]{aaai2027}',
           r'\usepackage[hyphens]{url}', r'\usepackage{graphicx}', r'\usepackage{natbib}',
           r'\usepackage{caption}', r'\usepackage{booktabs}', r'\usepackage{amsmath}',
           r'\frenchspacing', r'\urlstyle{rm}', r'\def\UrlFont{\rm}', r'\setcounter{secnumdepth}{0}',
           '\\title{'+esc(title)+'}', '\\author{'+esc(author)+'}', '\\affiliations{'+esc(affiliation)+'}',
           r'\begin{document}', r'\maketitle', r'\begin{abstract}', prose(abstract), r'\end{abstract}']
    for heading, paragraphs in sections:
        tex += ['\\section{'+heading+'}', '\n\n'.join(prose(p) for p in paragraphs)]
        if heading == 'Predictors and evaluation':
            tex += [r'For prefix features $z_i$ and latest observed articulation $a_{i,j}$, each output is',
                    r'\begin{equation}\widehat y_{i,h,j}=a_{i,j}+\beta(z_i^\top w_{h,j}+b_{h,j}).\end{equation}',
                    r'The primary model fixes $\beta=1$; secondary shrinkage is selected inside normal-source validation.']
        if heading == 'Introduction':
            tex += [r'\begin{figure*}[t]', r'\centering', r'\includegraphics[width=\textwidth]{figures/method.pdf}',
                    r'\caption{Controlled future prediction uses a common body reference, common targets, and source-separated training. The shifted control preserves observed values while changing their timing. No treatment or neural state is observed.}', r'\label{fig:method}', r'\end{figure*}']
        if heading == 'Data and observation contract':
            tex += [table(['Annotation','Catalog','Final','Windows'],cohort_rows,caption='Source-video counts before and after all observation and support gates. More windows do not imply more independent people.',label='tab:cohort')]
        if heading == 'Results':
            tex += [table(['Forecast','MSE','95\\% interval'],main_rows,
                          caption='Normal-source mean squared error, multiplied by 1,000, over the nominal 0.8-second future. The shrunk bilateral arm is secondary. Intervals condition on fixed fits.', label='tab:forecast'),
                    r'\begin{figure}[t]', r'\centering', r'\includegraphics[width=\columnwidth]{figures/paired_effects.pdf}',
                    r'\caption{Paired normal-source error differences. Positive values favor the forecast. Conditional 95\% source intervals do not include retraining or patient-population uncertainty.}',r'\label{fig:effects}', r'\end{figure}']
    tex += [r'\bibliography{references}', r'\clearpage', r'\appendix',
            r'\section{Complete exploratory results}',
            'Every declared forecast and contrast is retained below. The shorter horizon is cumulative over four future samples. These are descriptive comparisons without multiplicity adjustment.']
    for horizon in (4,8):
        rows = [[NAMES[r['arm']], ci(r,1000,3)] for r in result['normal_forecast'] if r['horizon']==horizon]
        tex += [table(['Forecast','MSE [95% interval]'],rows,caption=f'All normal-source forecasts at {horizon/10:.1f} seconds; values are multiplied by 1,000.')]
        rows = [[SHORT[r['reference']]+' / '+SHORT[r['method']],ci(r,1000,3)] for r in result['paired_contrasts'] if r['horizon']==horizon]
        tex += [table(['Reference / forecast','Gain [95% interval]'],rows,caption=f'All paired contrasts at {horizon/10:.1f} seconds; gain is reference minus forecast MSE, multiplied by 1,000. (S) denotes the secondary shrinkage variant.')]
    tex += [table(['Score','AUROC [95% interval]'],[row[:2] for row in anomaly_rows],caption='All secondary abnormality rankings. Higher scores always mean more abnormal.'),
            table(['Score','AP [95% interval]'],[[row[0],row[2]] for row in anomaly_rows],caption=f'Average precision for every declared score. Abnormal-source prevalence is {result["anomaly"]["bilateral"]["abnormal_prevalence"]:.3f}.'),
            r'\begin{figure*}[t]',r'\centering',r'\includegraphics[width=\textwidth]{figures/normal_sources.pdf}',
            r'\caption{All retained normal sources show the distribution of forecasting effects. Source ordering uses persistence error for display only. Source IDs and values are retained in the figure data file.}',r'\end{figure*}',
            r'\section{Reproducibility and scope}',
            prose(f'All {result["fit_count"]} saved model checkpoints and the input arrays are fingerprinted. The complete run contains source membership for every outer and inner fold, model selection grids, predictions, per-window and per-source errors, and source-level anomaly scores. Bilateral shrinkage beta counts across fold/leg fits are {beta_text}. The prepared cohort required {prep["actual_model_calls_this_invocation"]:,} recorded model calls and {prep["seconds"]:.1f} seconds for this invocation; fitting and scoring required {result["seconds"]:.1f} seconds. These timings are descriptive of the local environment.'),
            prose('The method predicts a measured articulation sequence under observational source shift. It is not a disease simulator, a patient-specific digital twin, a causal intervention model, or a validated clinical diagnostic system. Participant verification, external clinical evaluation, and explicit action or treatment data remain necessary extensions.'),
            r'\end{document}']
    content = '\n\n'.join(tex)+'\n'
    content = content.replace(esc('95\\% interval'), esc('95% interval'))
    assert '\u2014' not in content
    (OUT/'manuscript.tex').write_text(content)
    for name,item in tables.items():
        pd.DataFrame(item['rows'],columns=item['headers']).to_csv(OUT/f'{name}.csv',index=False)
    selection.drop(columns=['all_grid_mse']).to_csv(OUT/'selection.csv',index=False)
    shutil.copy2(RUN/'condition_results.csv', OUT/'condition_results.csv')
    shutil.copy2(RUN/'source_scores.csv', OUT/'source_scores.csv')
    write_json(OUT/'evidence.json',dict(results=result, preparation=prep,
               inputs={str(p.relative_to(PROJECT)):digest(p) for p in [RUN/'results.json',RUN/'source_scores.csv',RUN/'selection.json',RUN/'identity.json',RUN/'artifact_hashes.json',DATA/'preparation.json',PROJECT/'docs/restart/protocol.md']},
               checks=checks, primary_claim=forecast_claim, timing_claim=timing_claim, secondary_shrinkage_claim=shrink_claim,
               author_pending='pending' in author.lower(), manuscript_source_sha256=digest(__file__)))
    return forecast_claim, timing_claim, shrink_claim


def write_readiness(claims, compiled, render):
    status = 'Compiled and rendered internal draft; independent final review and author confirmation remain required.' if compiled and render else 'Generated evidence-grounded internal draft; compilation, rendering, and independent final review remain to be completed.'
    build=json.loads((OUT/'build_record.json').read_text())
    compiler=build.get('compiler','Not executed')
    compiler_version=build.get('compiler_version','Not executed')
    acceptance=OUT/'reviews/acceptance.json'
    if acceptance.exists():
        accepted=json.loads(acceptance.read_text())
        names=['manuscript.pdf','manuscript.md','manuscript.tex']
        if accepted.get('status')=='accepted' and all((OUT/name).exists() and accepted.get('artifacts',{}).get(name)==digest(OUT/name) for name in names):
            status='The exact manuscript has passed independent scientific, numerical, and visual review. Submission still requires the confirmed author block and human approval.'
    text = f'''# Submission readiness for draft01

{status}

The manuscript targets the Med-WM short-paper route of 2–4 body pages, excluding references and appendices. The build record reports the actual page boundaries. Do not submit a body outside the allowed range without revising it or deliberately switching to the 6–8 page full-paper route. The official workshop page lists November 20, 2026, Anywhere on Earth. Its OpenReview invitation deadline remains independently unverified.

## What the completed study supports

{claims[0]} {claims[1]} {claims[2]}

The results come from actual pose inference and normal-only model fitting. The generator validates result hashes and independently recomputes all displayed mean effects, AUROC, and average precision. This is exploratory source-grouped evaluation following exposure to earlier GAVD outcomes. Confidence intervals condition on fixed cross-validation fits and cannot establish patient-level clinical generalization.

## Files and reproduction

Read `manuscript.md` or `manuscript.pdf`. Editable sources are `manuscript.tex`, `references.bib`, and `../../scripts/build_restart_manuscript.py`. Figures are editable SVG and vector PDF, with PNG previews. `evidence.json` identifies the inputs. `manuscript_initial.md` preserves the pre-results narrative. `revisions/01_initial_results_draft/` preserves the first empirical source draft before independent comments were incorporated; that version has no PDF because compiler dependencies were still being resolved.

From the repository root:

```bash
experiments/aaai-worldmodels/.venv/bin/python experiments/aaai-worldmodels/scripts/build_restart_manuscript.py --compile --render
```

The actual compiler is `{compiler_version}`, invoked at `{compiler}`. BibTeX comes from the same portable TinyTeX installation, and `/tmp/medwm-docs-env/bin/python` performs PDF rendering. Exact commands are recorded in `build_record.json`; command-line options can point to equivalent tools on another machine. The official AAAI style requires pdfTeX, so Tectonic is unsuitable. The style and bibliography files are copied without modification from the existing author-kit copy, with hashes in `author_kit_provenance.json`.

## Systemic critiques and concrete improvements

| Issue | Current consequence | Concrete next step |
| --- | --- | --- |
| Unknown participant identity | Source separation does not prove patient separation. | Verify repeated participants across uploads and obtain an external participant-disjoint cohort. |
| Few normal sources | Many windows cannot establish broad normative coverage. | Add independent normal participants across sites, ages, views, and clinically measured walking conditions. |
| Observational 2D measurement | Predictions do not identify neurology, anatomy, or treatment response. | Validate pose measurements against synchronized clinical reference data and pair movement with clinical assessments. |
| Historical outcome exposure | The new study is exploratory even with nested source folds. | Freeze this protocol and evaluate a genuinely external cohort once. |
| Conditional uncertainty | Source bootstrap does not include full retraining variability. | Repeat outer partitions or bootstrap the complete training pipeline when the independent-source budget permits. |
| Identity and confidence gates | Filtering can select easier sources and retain undetected identity errors. | Review a blinded source-stratified sample of accepted and rejected sequences and document discrepancies. |
| Biological interpretation | Predictive dependence may reflect posture, view, or correlated pose error. | Add controlled view and speed strata, validated gait events, and matched sensor measurements. |
| Workshop scope | This is an observation transition model without actions or disease progression. | Frame it as a tested prerequisite for medical world models; add longitudinal and intervention data before stronger claims. |
| Authorship and submission metadata | Human author details and approval may still be pending. | Confirm the actual author block, affiliations, acknowledgments, AI-use disclosure, and portal metadata before submission. |

## Final review gates

The artifact-specific independent review is recorded under `reviews/`, with fixes in `response_to_reviews.md`. Readiness is marked accepted only when `reviews/acceptance.json` matches the exact PDF, Markdown, and LaTeX hashes. Otherwise a new or changed manuscript needs review. No public submission is performed by the build script.
'''
    (OUT/'README.md').write_text(text)


def compile_and_render(args):
    record = {'built_at_utc':datetime.now(timezone.utc).isoformat(), 'generator_sha256':digest(__file__),
              'compiled':False, 'rendered':False, 'review_status':'awaiting_independent_review'}
    if args.compile:
        pdftex = Path(args.pdftex)
        bibtex = pdftex.with_name('bibtex')
        record['compiler'] = str(pdftex)
        record['compiler_version'] = subprocess.run([str(pdftex),'--version'],text=True,capture_output=True,check=True).stdout.splitlines()[0]
        archive=Path('/tmp/medwm-tools/tinytex-darwin.tar.xz')
        if archive.exists():
            record['compiler_distribution']={'url':'https://github.com/rstudio/tinytex-releases/releases/download/v2026.11/TinyTeX-1-darwin-v2026.11.tar.xz',
                                             'archive_sha256':digest(archive),
                                             'additional_packages':['newtx','tex-gyre','psnfss','placeins','natbib','caption','booktabs','xpatch','xkeyval','etoolbox','fontaxes','mweights','xstring','carlisle','grfext']}
        record['commands'] = [[str(pdftex),'-interaction=nonstopmode','-halt-on-error','manuscript.tex'],
                              [str(bibtex),'manuscript'],
                              [str(pdftex),'-interaction=nonstopmode','-halt-on-error','manuscript.tex'],
                              [str(pdftex),'-interaction=nonstopmode','-halt-on-error','manuscript.tex']]
        record['source_date_epoch']=int(datetime(2026,11,16,tzinfo=timezone.utc).timestamp())
        compile_env=dict(os.environ,SOURCE_DATE_EPOCH=str(record['source_date_epoch']),FORCE_SOURCE_DATE='1')
        output=[]
        for command in record['commands']:
            process = subprocess.run(command,cwd=OUT,text=True,capture_output=True,env=compile_env)
            output.append('COMMAND: '+' '.join(command)+'\n'+process.stdout+'\n'+process.stderr)
            (OUT/'compilation_output.txt').write_text('\n'.join(output))
            record['compile_returncode'] = process.returncode
            if process.returncode:
                write_json(OUT/'build_record.json', record)
                raise RuntimeError('TeX compilation failed; see compilation_output.txt')
        record['compiled'] = True
    if args.render:
        if not (OUT/'manuscript.pdf').exists():
            raise ValueError('A compiled PDF is required for rendering')
        script = r'''
import json,re
from pathlib import Path
import pymupdf as fitz
from PIL import Image,ImageOps,ImageDraw
root=Path.cwd(); out=root/'rendered';out.mkdir(exist_ok=True)
for old in out.glob('page-*.*'):
 if re.fullmatch(r'page-\d+\.(png|txt)',old.name):old.unlink()
doc=fitz.open(root/'manuscript.pdf');pages=[];images=[]; refs=[]; appendix=[];overflow=[];reference_page_has_body=False
for i,page in enumerate(doc):
 text=page.get_text();(out/f'page-{i+1}.txt').write_text(text)
 if re.search(r'^References\s*$',text,re.M):
  refs.append(i+1)
  reference_page_has_body=len(text.split('References')[0].strip())>100
 if 'Complete exploratory results' in text:appendix.append(i+1)
 for block in page.get_text('blocks'):
  if block[0]<15 or block[1]<15 or block[2]>page.rect.width-15 or block[3]>page.rect.height-15:overflow.append(dict(page=i+1,box=list(block[:4]),text=block[4][:120]))
 pix=page.get_pixmap(matrix=fitz.Matrix(1.4,1.4),alpha=False);path=out/f'page-{i+1}.png';pix.save(path)
 image=Image.open(path).convert('RGB');image.thumbnail((306,396));images.append(image)
 pages.append(dict(page=i+1,characters=len(text),images=len(page.get_images()),width=page.rect.width,height=page.rect.height))
cols=3;rows=(len(images)+cols-1)//cols;sheet=Image.new('RGB',(cols*326,rows*426),'#e3e7ea');draw=ImageDraw.Draw(sheet)
for i,image in enumerate(images):x=(i%cols)*326+10;y=(i//cols)*426+20;sheet.paste(image,(x,y));draw.text((x,y-15),f'Page {i+1}',fill='black')
sheet.save(out/'contact.png')
body_pages=refs[0]-(0 if reference_page_has_body else 1) if refs else None
report=dict(pages=pages,page_count=len(doc),reference_start_pages=refs,reference_page_has_body=reference_page_has_body,appendix_start_pages=appendix,body_pages=body_pages,short_paper_body_in_range=body_pages is not None and 2<=body_pages<=4,full_paper_body_in_range=body_pages is not None and 6<=body_pages<=8,edge_overflow=overflow,pdf_bytes=(root/'manuscript.pdf').stat().st_size)
(out/'checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
'''
        process = subprocess.run([args.render_python, '-c', script], cwd=OUT, text=True,capture_output=True)
        (OUT/'render_output.txt').write_text(process.stdout+'\n'+process.stderr)
        if process.returncode:
            write_json(OUT/'build_record.json',record)
            raise RuntimeError('PDF render failed; see render_output.txt')
        record['rendered'] = True
        record['pdf_checks'] = json.loads(process.stdout)
    record['artifacts'] = {str(p.relative_to(OUT)):digest(p) for p in [OUT/'manuscript.md',OUT/'manuscript.tex',OUT/'evidence.json']}
    if (OUT/'manuscript.pdf').exists():record['artifacts']['manuscript.pdf']=digest(OUT/'manuscript.pdf')
    write_json(OUT/'build_record.json',record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compile',action='store_true')
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--pdftex',default='/tmp/medwm-tools/TinyTeX/bin/universal-darwin/pdflatex')
    parser.add_argument('--render-python',default='/tmp/medwm-docs-env/bin/python')
    parser.add_argument('--author',default='Author details pending confirmation')
    parser.add_argument('--affiliation',default='Affiliation pending confirmation')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    evidence = load_evidence()
    provenance_path=OUT/'author_kit_provenance.json'
    if provenance_path.exists():
        provenance=json.loads(provenance_path.read_text())
        for name in ['aaai2027.sty','aaai2027.bst']:
            if not (OUT/name).exists() or digest(OUT/name)!=provenance[name]['sha256']:
                raise ValueError('Bundled author-kit file is missing or modified: '+name)
    else:
        # Bootstrap only. Subsequent builds use the bundled, fingerprinted files.
        provenance = {}
        for name in ['aaai2027.sty','aaai2027.bst']:
            source=REPO/'experiments/aaai-astad/docs/draft01'/name
            if (OUT/name).exists() and digest(OUT/name)!=digest(source):
                raise ValueError('Refusing to overwrite modified author-kit file: '+name)
            shutil.copy2(source,OUT/name)
            provenance[name]={'source':str(source.relative_to(REPO)),'sha256':digest(source),'modified':False}
        origin=REPO/'experiments/aaai-astad/docs/draft01/author_kit/download_provenance.json'
        if origin.exists():
            provenance['inherited_upstream_archive']=json.loads(origin.read_text())
            provenance['inherited_provenance_sha256']=digest(origin)
        write_json(provenance_path,provenance)
    build_figures(evidence[2],evidence[7])
    claims=build_paper(evidence,args.author,args.affiliation)
    record=compile_and_render(args)
    write_readiness(claims,record['compiled'],record['rendered'])
    print(json.dumps({'output':str(OUT),'compiled':record['compiled'],'rendered':record['rendered'],'claims':claims}))


if __name__=='__main__':
    main()
