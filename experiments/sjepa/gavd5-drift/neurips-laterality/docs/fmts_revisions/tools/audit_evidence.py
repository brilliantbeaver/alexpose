"""Check retained summaries without pretending to recover absent predictions.

Run from any directory. The local report is not an anonymous submission file.
No training, inference, or source bootstrap is performed by this script.
"""
from pathlib import Path
import hashlib
import json
import statistics

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]


def main():
    figpath = ROOT / "docs/figures/physworld_figure_provenance.json"
    auditpath = ROOT / "docs/physworld_evidence_recomputed.json"
    figures = json.loads(figpath.read_text())
    audit = json.loads(auditpath.read_text())
    rows = figures["numbers"]["readout_rows"]
    initial = next(r for r in rows if r["representation"] == "initial_online__mean_motion")
    checks = []
    for row in rows:
        assert len(row["per_seed_r2"]) == 5
        mean = statistics.mean(row["per_seed_r2"])
        assert abs(mean - row["mean_r2"]) < 1e-12
        check = {"experiment": row["experiment"], "condition": row["condition"],
                 "representation": row["representation"], "mean_r2": mean,
                 "seed_sd_r2": statistics.stdev(row["per_seed_r2"])}
        if row["representation"] == "pretrained_teacher__mean_motion":
            difference = [x-y for x,y in zip(row["per_seed_r2"], initial["per_seed_r2"])]
            retained = audit["exploratory_new_trained_minus_initial_intervals"][row["experiment"]+"/"+row["condition"]]
            assert abs(statistics.mean(difference)-retained["difference_r2"]) < 1e-12
            assert all(x < 0 for x in difference)
            assert row["mean_mae"] > initial["mean_mae"]
            check.update(paired_seed_differences=difference,
                         mean_paired_difference=statistics.mean(difference),
                         retained_source_interval=retained["r2_ci95"])
        checks.append(check)
    for m in figures["numbers"]["mask_intervals"]:
        condition = m["subtraction"].split(" minus ")[0]
        arms = [r for r in rows if r["experiment"]==m["experiment"] and r["representation"]=="pretrained_teacher__mean_motion"]
        delta = next(r["mean_r2"] for r in arms if r["condition"]==condition)-next(r["mean_r2"] for r in arms if r["condition"]=="uniform")
        assert abs(delta-float(m["difference"])) < 1e-12
    files = [figpath,auditpath,ROOT/"docs/paper.md",ROOT/"docs/physworld_revisions/paper_v8.md",ROOT/"docs/TUTORIAL.md"]
    files += list(ROOT.glob("*.ipynb"))
    files += [ROOT/p for p in ["laterality/model.py","laterality/geometry.py","laterality/data.py","laterality_extensions/motion_structured_training.py","laterality_extensions/motion_readout.py","laterality_extensions/motion_runtime.py","laterality_extensions/comparative_evaluation.py"]]
    manifest = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report = dict(date="2026-09-10", raw_grid_available=(ROOT/"artifacts").exists(),
        current_checks="Retained seed means, paired differences, mask differences, notebook evidence types, implementation and recorded-audit consistency. No new source-bootstrap or inference.",
        retained_audit_scope=audit["uncertainty_scope"], checks=checks, sha256=manifest)
    (HERE/"review/evidence_check.json").write_text(json.dumps(report,indent=2)+"\n")
    # Anonymous numerical supplement: these are retained aggregate records.
    payload = {"evidence_status":"Real-data executed outputs and retained aggregate audit records. Raw predictions and checkpoints are not included. Source intervals are retained exploratory percentile intervals; they cannot be regenerated from these seed aggregates.",
        "cohort":{"clips":625,"source_videos":93,"outer_folds":5,"seeds":[42,43,44,45,46],"training_precision":"CUDA BF16","evaluation_precision":"FP32","updates_per_encoder":1200},
        "readout_rows":rows,"mask_intervals":figures["numbers"]["mask_intervals"],
        "correspondence":figures["numbers"]["correspondence"],
        "trained_minus_initial":audit["exploratory_new_trained_minus_initial_intervals"]}
    (HERE/"review/numerical_evidence.json").write_text(json.dumps(payload,indent=2)+"\n")
    # Preserve executed textual/HTML outputs locally, including original precision.
    outputdir = HERE/"review/notebook_outputs"
    outputdir.mkdir(exist_ok=True)
    for number in [11,12,13,14,16,18]:
        path = next(ROOT.glob(f"{number:02d}_*.ipynb"))
        notebook = json.loads(path.read_text())
        extracted=[]
        for idx, cell in enumerate(notebook["cells"]):
            for outidx,out in enumerate(cell.get("outputs",[])):
                data=out.get("data",{})
                text=out.get("text",data.get("text/plain",[]))
                html=data.get("text/html",[])
                extracted.append(dict(cell=idx,output=outidx,execution=cell.get("execution_count"),
                    text="".join(text) if isinstance(text,list) else text,
                    html="".join(html) if isinstance(html,list) else html))
        (outputdir/(path.stem+".json")).write_text(json.dumps(extracted,indent=2)+"\n")
    print(json.dumps({"seed_rows_checked":len(rows),"paired_training_conditions":5,"mask_contrasts_checked":3,"raw_grid_available":report["raw_grid_available"]}))


if __name__ == "__main__":
    main()
