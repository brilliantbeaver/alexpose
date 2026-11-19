"""Prespecified seed-42 positional-noise comparison on the frozen real cohort."""
from pathlib import Path
import sys,json
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import torch
from aaai_gait.settings import Settings
from aaai_gait.workflow import prepare_corpus,normal_anchor,select_probe,write_json,_evaluate
from aaai_gait.training import embed_windows
from aaai_gait.evaluation import align_probabilities,cluster_bootstrap
from aaai_gait.data import LABELS

def main():
    torch.set_num_threads(2)
    s=replace(Settings(),demo=False,seed=42,epochs=3)
    x,meta,p=prepare_corpus(s)
    encoder,history=normal_anchor(x,meta,p,s,spe=True)
    z=embed_windows(encoder,x,batch_size=s.batch_size);probe,selection=select_probe(z,meta,s.seed)
    test=meta.split.eq('test').to_numpy()
    prob=align_probabilities(probe.predict_proba(z[test]),probe.classes_,LABELS)
    seq,prob,metrics=_evaluate(meta.loc[test],prob)
    baseline=pd.read_csv(p['run_dir']/'test_predictions_normal_jepa.csv')
    if not seq.sequence_id.equals(baseline.sequence_id):raise ValueError('Unpaired sequence order')
    base=baseline[[f'p_{label}' for label in LABELS]].to_numpy()
    paired=cluster_bootstrap(seq.label,prob,seq.source_id,LABELS,other_probabilities=base)
    write_json(p['run_dir']/'spe_ablation.json',dict(synthetic=False,selection=selection,metrics=metrics,paired_spe_minus_fixed_sequence_weighted=paired,epochs=3,seed=42))
    table=seq.copy()
    for i,label in enumerate(LABELS):table[f'p_{label}']=prob[:,i]
    table.to_csv(p['run_dir']/'test_predictions_spe.csv',index=False)
    print(json.dumps({'run_dir':str(p['run_dir']),'source_macro_f1':metrics['source_weighted']['macro_f1'],'paired_ci':paired['intervals_95']['macro_f1']}))
if __name__=='__main__':main()
