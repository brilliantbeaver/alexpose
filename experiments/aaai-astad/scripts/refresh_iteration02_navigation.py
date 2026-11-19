"""Update active navigation while archiving original historical document bytes."""
from pathlib import Path
import hashlib,json,shutil,os
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parents[1];DOC=BASE/'docs';PAPER=DOC/'draft02'
archive=PAPER/'historical_navigation_snapshots';archive.mkdir(exist_ok=True)
records=[]
paths=[DOC/p for p in ['context_study_protocol.md','context_study_results.md','context_study_sensitivity.md','context_study_review.md','current_execution_validation.md','data_audit.md','results.md','validation.md','model_validation.md','paper_plan.md','published_baseline_feasibility.md','adversarial_review.md','draft00/README.md','draft00/EXECUTIVE_SUMMARY.md','draft01/README.md']]
for p in paths:
    if not p.exists():continue
    relative=p.relative_to(BASE);copy=archive/relative;copy.parent.mkdir(parents=True,exist_ok=True)
    if not copy.exists():shutil.copy2(p,copy)
    marker='<!-- iteration02-navigation -->'
    if marker not in p.read_text():
        target=os.path.relpath(PAPER/'README.md',p.parent)
        p.write_text(f'{marker}\n> Historical study or planning record. Its scientific results and run identities remain unchanged. The executed [iteration02 study and draft02]({target}) contain the active results, six new notebooks and independent participant replication.\n\n'+p.read_text())
    records.append({'path':str(relative),'snapshot':str(copy.relative_to(BASE)),'original_sha256':hashlib.sha256(copy.read_bytes()).hexdigest(),'change':'navigation banner only'})
(PAPER/'historical_navigation_changes.json').write_text(json.dumps(records,indent=2)+'\n')
p=DOC/'iteration02/README.md'
if not p.with_name('README.planning_snapshot.md').exists():shutil.copy2(p,p.with_name('README.planning_snapshot.md'))
p.write_text('''# Executed cross-leg prediction study

The new experiments are complete. Opposite-leg information can improve normal completion, but the frozen equal-weight anomaly score performs worse than local movement alone. A separate participant-level laboratory study supports that distinction. The GAVD timing diagnostic fails adequacy on its exact paired inputs, so it does not support reliable coordination measurement.

Start with [verified results](results.md), the [implemented protocol](protocol.md), [limitations](limitations.md), [draft02](../draft02/README.md), and the [active slides](../../slides/tutorial.pdf). The six new notebooks are linked from the [experiment README](../../README.md). The earlier [planning snapshot](README.planning_snapshot.md) is preserved as a record of proposed work, not a statement of current execution status.

![Implemented retained predictor and nominal-width control.](../../images/iteration02/residual_predictor.svg)

## What ran

The GAVD cohort contains 304 source recordings, including 28 normal recordings, with 1,381 sequences and 9,442 windows. All U/J/A correction-training policies ran on both frozen U and E_leg banks. All 12 internal prediction gates passed, while outer-normal gains on U remain uncertain for U and A. The focal S-minus-M AUROC difference is −0.118, conditional interval [−0.217, −0.025].

The [independent laboratory study](external_replication_results.md) retains 178 participants after applying fixed observation support to 188 selected trials. Normal completion improves in every outer fold, while the fixed fusion again reduces anomaly ranking. These results are not transfer of video weights or clinical validation of the four GAVD condition families.

![The fixed training-policy by evaluation-bank comparison.](../../images/iteration02/mask_comparison.svg)

## Historical design and primary sources

The [canonical instructions](../../../../notes/aaai/05-iterate-02-INSTR.md), [data advice](advice_data.md), [method advice](advice_method.md), [neurological advice](advice_neuro.md), and three [skeleton](masking_method_review.md), [video](masking_transfer_review.md), and [neurological masking](masking_neuro_review.md) reviews preserve the reasoning before implementation. Their proposed experiments must be read beside the completed protocol and results. The [external feasibility review](external_replication_feasibility.md) records what was known before files were downloaded; the executed external report supersedes its availability status.

The [draft02 review record](../draft02/response_to_reviews.md) distinguishes accepted corrections, checked objections and unresolved limitations. The [readiness report](../draft02/submission_readiness.md) separates reproducibility and format from scientific strength. Historical draft01 results are unchanged.
''')
# Preserve feasibility and advice content; qualify its time-dependent status.
for name in ['external_replication_feasibility.md','design_brief.md','advice_data.md','advice_method.md','advice_neuro.md','adversarial_review.md','mask_design_verification.md']:
    p=DOC/'iteration02'/name
    if p.exists() and '<!-- iteration02-execution-status -->' not in p.read_text():
        p.write_text('<!-- iteration02-execution-status -->\n> This is the preserved design or feasibility record. Consult the [implemented protocol](protocol.md), [verified results](results.md), and [external execution report](external_replication_results.md) for current execution status.\n\n'+p.read_text())
video=ROOT/'data/videos-gavd/README.md'
video.write_text('''# GAVD local video storage

This directory contains source recordings, extraction/download ledgers and local previews. New downloads are organized by source recording ID, with previews in `<source_id>/previews/`. Legacy root-level cached videos remain at their recorded paths; the ingestion pipeline does not duplicate recordings into condition folders.

The current annotation inventory is [data/gavd/README.md](../gavd/README.md). The executed [iteration02 protocol](../../experiments/aaai/docs/iteration02/protocol.md) uses fresh CSV labels and verifies annotation, video, extraction and pose hashes. Its [case-review ledger](../../experiments/aaai/artifacts/iteration02/d66e1b126c9d9a9adf1b/video_review/review_ledger.csv) records ten selected previews, original intervals and source paths. Previews retain original playback timing. The post-outcome boundary audit is descriptive and did not change model fits or eligibility.

Source video redistribution permissions have not been cleared for the manuscript bundle. Keep previews local. The paper uses measured vector skeleton traces rather than source photographs.
''')
inventory=ROOT/'data/gavd/README.md';marker='<!-- iteration02-study-link -->'
if marker not in inventory.read_text():
    inventory.write_text(inventory.read_text()+'\n'+marker+'\nThe separately frozen [iteration02 observation study](../../experiments/aaai/docs/iteration02/protocol.md) records source attrition and hash-compatible pose eligibility. It does not replace this fresh CSV inventory. Videos and local previews remain under [data/videos-gavd](../videos-gavd/README.md).\n')
print('Updated active and historical navigation without changing historical scientific results.')
