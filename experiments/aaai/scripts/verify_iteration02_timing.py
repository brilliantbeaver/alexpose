"""Reconstruct every saved timing effect and source aggregate from raw ledgers."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from aaai_gait.iteration_timing import summarize_source_pairs


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",type=Path,required=True);args=parser.parse_args()
    root=args.run.resolve()/"timing";reports=[];source_tables=[]
    for folder in sorted(root.glob("fold_*/*")):
        if not (folder/"execution.json").exists():continue
        source_table=pd.read_csv(folder/"source_pairs.csv");source_tables.append(source_table)
        expected=source_table.set_index(["source_id","policy","variant","offset"])
        raw_count=0;max_effect_error=0.;max_source_error=0.;files=0
        for path in sorted((folder/"target_rows").glob("*.csv.gz")):
            rows=pd.read_csv(path);raw_count+=len(rows);files+=1
            error=float(np.max(np.abs((rows.loss_both_edited-rows.loss_both_aligned)-rows.effect)))
            max_effect_error=max(max_effect_error,error);assert error<1e-10
            fixed=rows.fixed_target.astype(bool)
            assert rows.loc[fixed,"target_max_change"].eq(0).all()
            assert rows.loc[fixed,"own_prediction_max_change"].eq(0).all()
            np.testing.assert_array_equal(rows.loc[fixed,"loss_own_aligned"],rows.loc[fixed,"loss_own_edited"])
            assert rows.loc[rows.variant.eq("zero"),"effect"].eq(0).all()
            actual,_=summarize_source_pairs(rows);actual=actual.set_index(["source_id","policy","variant","offset"])
            saved=expected.loc[actual.index];np.testing.assert_array_equal(saved.eligible,actual.eligible)
            difference=np.abs(saved.effect.to_numpy()-actual.effect.to_numpy());finite=difference[np.isfinite(difference)]
            error=float(finite.max()) if len(finite) else 0.;max_source_error=max(max_source_error,error);assert error<1e-10
        reports.append(dict(folder=str(folder.relative_to(root)),source_files=files,target_rows=raw_count,
            max_loss_difference_reconstruction_error=max_effect_error,max_source_aggregation_error=max_source_error,
            artifact_interval_seconds=(folder/"execution.json").stat().st_mtime-(folder/"protocol.json").stat().st_mtime,
            status="passed"))
        print(f"Verified {folder.relative_to(root)}: {raw_count} target rows",flush=True)
    if len(reports)!=16:raise ValueError(f"Expected all16 diagnostics, found{len(reports)}")
    combined=pd.read_csv(root/"timing_pairs.csv")
    if len(combined)!=sum(len(table) for table in source_tables):raise ValueError("Combined source-row count differs")
    result=dict(status="passed",diagnostic_count=len(reports),raw_target_rows=sum(r["target_rows"] for r in reports),
        source_model_variant_rows=len(combined),checks="All loss differences, all source reaggregations, exact fixed-target/own invariance and zero identities",
        runtime_note="Artifact interval measures protocol-file creation to execution-file creation, including scoring and I/O",
        independent_coordinate_check="independent_raw_pair_verification.json uses a separately written endpoint loop on one real observation",reports=reports)
    (root/"raw_ledger_verification.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="reports"},indent=2))


if __name__=="__main__":main()
