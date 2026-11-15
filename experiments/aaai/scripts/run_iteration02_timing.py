"""Execute the frozen real-data timing census and normal model diagnostics."""
from pathlib import Path
import argparse
import hashlib
import json
import time

import joblib
import numpy as np
import pandas as pd

from aaai_gait.iteration_data import load_iteration_data
from aaai_gait.iteration_timing import build_timing_census,run_timing_fold
from aaai_gait.retained_prediction import assemble_selected_bundle
from aaai_gait.masking import rng_stream


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+"\n")


def summarize(table):
    summaries=[]
    for key,group in table.groupby(["scope","policy","variant","offset"]):
        usable=group[group.eligible & np.isfinite(group.effect)]
        # A source repeated across selected inner models remains one bootstrap unit.
        source=usable.groupby("source_id").effect.mean()
        values=source.to_numpy();rng=rng_stream(42,"timing_bootstrap_"+str(key))
        estimates=np.mean(values[rng.integers(0,len(values),size=(1000,len(values)))],axis=1) if len(values) else []
        summaries.append(dict(scope=key[0],policy=key[1],variant=key[2],offset=int(key[3]),
            normal_source_count=len(values),eligible_source_model_rows=len(usable),
            mean_effect=float(values.mean()) if len(values) else None,
            conditional_source_bootstrap_95_interval=np.quantile(estimates,[.025,.975]).tolist() if len(values) else None,
            paired_gain_over_base=float(usable.adequacy_gain_base.mean()) if len(usable) else None,
            paired_gain_over_own=float(usable.adequacy_gain_own.mean()) if len(usable) else None,
            mean_permitted_distal_tokens=float(usable.opposite_distal_permitted.mean()) if len(usable) else None,
            mean_fraction_with_distal_context=float(usable.has_permitted_distal_context.mean()) if len(usable) else None,
            mean_fraction_with_nonzero_edit=float(usable.has_nonzero_context_edit.mean()) if len(usable) else None,
            frozen_zero_correction_source_model_rows=int(usable.both_multiplier.eq(0).sum()),
            no_distal_context_source_model_rows=int(usable.has_permitted_distal_context.eq(0).sum()),
            max_own_prediction_change=float(usable.own_prediction_max_change.max()) if len(usable) else None))
    inner=table[table.scope.eq("inner_validation")&table.variant.eq("timing")&table.eligible]
    loss_columns=["loss_base_aligned","loss_own_aligned","loss_both_aligned"]
    grouped=inner.groupby(["outer_fold","policy","offset","diagnostic_model"])[loss_columns].mean().reset_index()
    grouped["positive_gain_over_both_controls"]=(grouped.loss_both_aligned<grouped.loss_base_aligned)&(grouped.loss_both_aligned<grouped.loss_own_aligned)
    gates=[]
    for key,group in grouped.groupby(["outer_fold","policy","offset"]):
        selected=inner[inner.outer_fold.eq(key[0])&inner.policy.eq(key[1])&inner.offset.eq(key[2])]
        risks=selected[loss_columns].mean()
        passed=(len(group)==3 and group.positive_gain_over_both_controls.sum()>=2
                and risks.loss_both_aligned<min(risks.loss_base_aligned,risks.loss_own_aligned))
        gates.append(dict(outer_fold=int(key[0]),policy=key[1],offset=int(key[2]),
            internal_folds=len(group),positive_internal_folds=int(group.positive_gain_over_both_controls.sum()),
            paired_support_adequacy_passed=bool(passed),**{k:float(v) for k,v in risks.items()}))
    return dict(status="complete",real_gavd_model_calls=True,summaries=summaries,
        paired_support_internal_fold_errors=grouped.to_dict("records"),paired_support_adequacy=gates,
        timing_gate_note="The same operational two-of-three normal-fold adequacy rule is evaluated on each exact paired-support original input. It changes interpretation, not frozen models or scores",
        estimand="Source-aggregated shifted-context MSE minus aligned-context MSE on identical targets; whole-body crop is a separate changed-target control",
        uncertainty="1000 bootstrap draws over unique source records, conditional on frozen models, selected recipe, split and budget; repeated inner-source model rows averaged first",
        internal_diagnostic="Post-selection normal diagnostic: the selected recipe used the same three internal folds. This is not independent confirmation",
        zero_correction="Frozen zero-gate rows remain included; zero timing response from such a model is mechanically guaranteed",
        support="Hip-only comparisons retain coverage but have no permitted distal edit; report them as uninformative for timing use",
        interpretation="Time-aligned context sensitivity only. Finite crop marginals and configuration change; no physiological phase, disease simulation or causal interpretation")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--cohort",type=Path,default=Path(__file__).resolve().parents[1]/"artifacts/iteration02/preparation/cohort_v2")
    parser.add_argument("--wait-seconds",type=int,default=1800);parser.add_argument("--refresh-census",action="store_true")
    args=parser.parse_args();run=args.run.resolve();out=run/"timing";out.mkdir(parents=True,exist_ok=True)
    data=load_iteration_data(args.cohort);census=out/"census"
    if args.refresh_census or not (census/"census.json").exists():build_timing_census(data,census)
    print("Timing census complete",flush=True);all_tables=[]
    code=Path(__file__).resolve().parents[1]/"aaai_gait"
    code_hashes={name:digest(code/name) for name in ("iteration_timing.py","retained_prediction.py","masking.py")}
    for fold in range(4):
        trained=run/f"fold_{fold}";deadline=time.monotonic()+args.wait_seconds
        while not (trained/"fitting_record.json").exists():
            if time.monotonic()>deadline:raise TimeoutError(f"Fold{fold} models are not ready")
            time.sleep(2)
        recipes=json.loads((trained/"selection.json").read_text())
        tasks=[]
        for internal in range(3):
            path=trained/f"inner_{internal}_models.joblib";record=joblib.load(path)
            bundles={policy:assemble_selected_bundle(grid,recipes[policy]) for policy,grid in record["grids"].items()}
            tasks.append((f"inner_{internal}","inner_validation",record["validation_sources"],bundles,[path]))
        paths=[trained/f"model_{p}.joblib" for p in ("U","J","A")]
        tasks.append(("outer","outer_test_normal",None,{p:joblib.load(path) for p,path in zip(("U","J","A"),paths)},paths))
        for name,scope,ids,bundles,paths in tasks:
            folder=out/f"fold_{fold}"/name;folder.mkdir(parents=True,exist_ok=True)
            inputs=dict(code_sha256=code_hashes,model_sha256={str(path.relative_to(run)):digest(path) for path in paths},
                selection_sha256=digest(trained/"selection.json"),cohort_artifacts_sha256=digest(args.cohort/"artifact_hashes.json"),
                selected_recipe_reuses_internal_validation=scope=="inner_validation",source_ids=ids,scope=scope,
                source_budget=8,noise_sd_original_torso=.01,dropout_fraction=.25,parent_seed=217)
            previous=folder/"driver_inputs.json"
            if (folder/"execution.json").exists():
                if not previous.exists() or json.loads(previous.read_text())!=inputs:raise RuntimeError(f"Inputs changed for {folder}; use a new timing identity")
                table=pd.read_csv(folder/"source_pairs.csv")
                print(f"Reused verified fold{fold}/{name}",flush=True)
            else:
                save(previous,inputs);started=time.monotonic()
                table,audit=run_timing_fold(data,fold,bundles,folder,source_ids=ids,scope=scope)
                print(f"Completed fold{fold}/{name}: {len(table)} source rows in {time.monotonic()-started:.1f}s",flush=True)
            if len(table):table["diagnostic_model"]=name;all_tables.append(table)
    combined=pd.concat(all_tables,ignore_index=True);combined.to_csv(out/"timing_pairs.csv",index=False)
    summary=summarize(combined);summary["census"]=json.loads((census/"census.json").read_text())["coverage"]
    summary["analysis_driver_sha256"]=digest(__file__)
    pd.DataFrame(summary["paired_support_internal_fold_errors"]).to_csv(out/"paired_support_internal_fold_errors.csv",index=False)
    pd.DataFrame(summary["paired_support_adequacy"]).to_csv(out/"paired_support_adequacy.csv",index=False)
    save(out/"timing_summary.json",summary)
    # The run-root copies are convenient artifact entry points, not new analyses.
    combined.to_csv(run/"timing_pairs.csv",index=False);save(run/"timing_summary.json",summary)
    print(f"Saved {len(combined)} timing source-model rows",flush=True)


if __name__=="__main__":main()
