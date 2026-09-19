"""Reconstruct retained pilot results and build the illustrated study writeup.

This reads saved aggregate CSVs; it does not rerun pose inference or training.
Reuses the companion proposal's figure style without modifying its artifacts.
"""
from pathlib import Path
import csv
import importlib.util
import json
from statistics import mean
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


style = module("proposal_figure_style", ROOT.parents[1] / "synthetic-training-v2/proposal/generate_figures.py")
style.ROOT, style.OUT = ROOT, ROOT / "images"
Figure = style.Figure


def audit_results():
    audit = module("pilot_aggregate_audit", REPO / "src/gavd6_sjepa/research_directions/synthetic_training_v2/audit.py")
    result = audit.reconstruct_pilot(REPO)
    # Independent check: average the selected CSV rows directly, without the
    # audit's source-outcome join, and compare within serialization tolerance.
    with (REPO / "notebook_runs/synthetic-training/run-03-v1/selectors/validation_predictions.csv").open() as f:
        decisions = list(csv.DictReader(f))
    errors = {r["method"]:r["error"] for r in result["table"]}
    for name in ("full", "source_progress_matched", "domain"):
        spec = result["selection"]["methods"][name]
        view = "source_progress" if name == "source_progress_matched" else name
        rows = [r for r in decisions if r["view"] == view and int(r["budget"]) == result["budget"]
                and r["kind"] == spec["kind"] and float(r["parameter"]) == spec["parameter"]]
        assert len(rows) == 48
        assert abs(mean(float(r["error"]) for r in rows)-errors[name]) < 1e-12
    data = ROOT / "evidence"
    data.mkdir(parents=True, exist_ok=True)
    (data / "pilot-audit.json").write_text(json.dumps(result,indent=2)+"\n")
    methods = [("replay","Post-probe replay"),("full_replay","Full-budget replay"),("pooled","Pooled synthetic"),
               ("full","Full response selector"),("source_progress_matched","Matched source-progress"),
               ("front","Fixed front lesson"),("domain","Scene/domain selector")]
    with (data / "reported-results.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["method","label","error","relative_error_reduction_percent"])
        writer.writeheader()
        for key,label in methods:
            writer.writerow(dict(method=key,label=label,error=errors[key],
                                 relative_error_reduction_percent=100*(errors["replay"]-errors[key])/errors["replay"]))
    return result, errors, methods


def diagrams():
    records=[]
    f=Figure("01-teaching-question","Does a short practice lesson reveal what to teach next?","Study hypothesis: prediction changes could reveal useful lesson preferences.",345)
    for x,title,lines in [(36,"Pretrained student",["Image pose estimator"]),(335,"Common probe",["Small, shared","training intervention"]),
                          (635,"Observed response",["Predictions after","minus predictions before"]),(935,"Select a lesson",["Predict benefit","over replay"])]:
        f.box(x,130,229,120,title,lines,style.GREEN if x==935 else style.PALE)
    for a,b in [(265,335),(564,635),(864,935)]: f.line([(a,190),(b,190)])
    f.text(600,310,"A change in prediction is observable without labels; whether it is an improvement is not.",20,anchor="middle",color=style.MUTED)
    records.append(f.save())

    f=Figure("02-data-roles","Different data serve different scientific roles","Synthetic reference errors teach the source selector; target reference labels never enter lesson choice.",490)
    f.box(36,120,540,140,"Training images",["Labeled AMASS probe and candidate lessons","Labeled COCO replay retains real-image training"],style.PALE)
    f.box(624,120,540,140,"Source diagnostic bank",["Separate labeled synthetic images","Measures known weaknesses before and after the probe"],style.SAND)
    f.box(36,305,540,140,"Unlabeled context",["Predictions before/after + frozen visual descriptors","Describes the setting without reference coordinates"],style.PALE)
    f.box(624,305,540,140,"Separate reference images",["Source: measure branch gains for fitting/validation","Real: score frozen choices only; not selector inputs"],style.GREEN)
    records.append(f.save())

    f=Figure("03-matched-branches","Every candidate restarts from the same probed checkpoint","Completed pilot: 10 common updates, then 25 or 75 more; validation selected 75.",575)
    f.box(36,120,240,108,"Original checkpoint",["Same initial student"])
    f.box(360,120,240,108,"10-update probe",["18 real + 2 synthetic / batch"])
    f.box(684,120,340,108,"Save weights and buffers",["Restore this state before each branch"],style.GREEN)
    f.line([(276,174),(360,174)]); f.line([(600,174),(684,174)])
    f.line([(854,228),(854,278),(208,278),(208,325)])
    f.line([(854,278),(604,278),(604,325)])
    f.line([(854,278),(996,278),(996,325)])
    f.box(36,325,344,125,"Eight lesson branches",["One candidate lesson per branch","75 updates: 18 real + 2 synthetic"])
    f.box(432,325,344,125,"Replay-only branch",["No additional synthetic lesson","75 updates: 20 real images"])
    f.box(828,325,336,125,"Pooled comparison",["Sample across all lessons","75 updates: 18 real + 2 synthetic"])
    f.text(600,495,"Fresh optimizer per branch. Each adapted checkpoint is scored across the 24 reference conditions.",19,anchor="middle",color=style.MUTED)
    f.box(36,520,1128,42,"",[],style.SAND)
    f.text(600,548,"Budget check: full replay starts from the original checkpoint and receives 85 real-only updates.",19,anchor="middle",color=style.INK)
    records.append(f.save())

    f=Figure("04-learn-and-select","Learn from source outcomes; select without target answers","The selector predicts a gain for each lesson. Replay is the zero-gain fallback.",620)
    f.text(36,119,"A  SOURCE FITTING",17,color=style.BLUE,weight="bold")
    f.box(36,145,335,135,"Source features",["Context + current predictions","Probe response + diagnostics","Training progress + descriptors"])
    f.box(36,322,335,120,"Source gain targets",["Replay error minus lesson error","Measured on separate references"],style.GREEN)
    f.box(490,222,260,140,"Fit gain predictor",["Standardize source features","Nearest neighbors or ridge","One gain per lesson"])
    f.box(850,222,314,140,"Validation students",["Select settings and update budget","Then freeze the selector","Reported results use this panel"],style.SAND)
    f.line([(371,209),(421,209),(421,256),(490,256)])
    f.line([(371,382),(421,382),(421,324),(490,324)])
    f.line([(750,292),(850,292)])
    f.text(36,490,"B  INTENDED REAL DEPLOYMENT — NOT ESTABLISHED BY THE RETAINED RESULTS",17,color=style.BLUE,weight="bold")
    for x,w,title,line in [(36,280,"New context + response","No real reference coordinates"),
                           (392,320,"Frozen gain predictor","Choose largest positive gain"),(792,372,"Run the chosen lesson","Use replay if no predicted gain is positive")]:
        f.box(x,513,w,87,title,[line])
    f.line([(316,555),(392,555)]); f.line([(712,555),(792,555)])
    records.append(f.save())

    f=Figure("06-evidence-boundary","What has been measured, and what remains unconfirmed","The retained results cover source adaptation and synthetic validation, not real-video transfer.",400)
    f.box(36,125,350,176,"Completed source trials",["RTMPose-M + HRNet-W32: fitting","RTMPose-S + HRNet-W48: validation","24 recurring synthetic conditions","One training seed: 17"],style.GREEN)
    f.box(426,125,350,176,"Observed pilot conclusion",["Some synthetic policies beat replay","Scene selection beats full response","No useful response increment shown","Repeated rows are not new people"],style.PALE)
    f.box(816,125,348,176,"Real evidence still needed",["Unlabeled GAVD lesson choice","Independent landmark annotations","Held ViTPose-family evaluation","Protected recording confirmation"],style.SAND)
    f.line([(386,214),(426,214)]); f.line([(776,214),(816,214)],dashed=True)
    f.text(600,365,"No trained skeleton JEPA, temporal restoration or clinical efficacy claim follows from this pilot.",19,anchor="middle",color=style.MUTED)
    records.append(f.save())
    (style.OUT/"figure-layout.json").write_text(json.dumps(records,indent=2)+"\n")
    return records


def results_chart(errors,methods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cairosvg
    plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
                         "font.size":14,"svg.fonttype":"none","svg.hashsalt":"original-synthetic-study"})
    fig,ax=plt.subplots(figsize=(12.5,5.7))
    fig.patch.set_facecolor("white")
    fig.suptitle("Scene selection outperformed the response selector",x=.03,y=.97,ha="left",fontsize=21,fontweight="bold",color=style.INK)
    fig.text(.03,.895,"Measured synthetic validation • 2 estimators × 24 conditions • selected 75-update budget",fontsize=13,color=style.MUTED)
    gains=[100*(errors["replay"]-errors[key])/errors["replay"] for key,_ in methods]
    colors=[style.TEAL if key=="domain" else style.ORANGE if key=="full" else "#8ba9cf" for key,_ in methods]
    ax.barh(range(len(methods)),gains,height=.58,color=colors)
    ax.set_yticks(range(len(methods)),[label for _,label in methods]); ax.invert_yaxis()
    for y,g in enumerate(gains): ax.text(g+.035,y,f"{g:.3f}%",va="center",fontsize=13,color=style.INK)
    ax.set_xlim(0,2.35); ax.set_xticks([0,.5,1,1.5,2]); ax.set_xlabel("Relative error reduction versus post-probe replay (%)",labelpad=12,color=style.INK)
    ax.xaxis.grid(True,color="#e3e9f0"); ax.set_axisbelow(True)
    ax.tick_params(axis="both",length=0,labelcolor=style.INK)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.subplots_adjust(left=.285,right=.955,top=.81,bottom=.20)
    fig.text(.03,.035,"Descriptive means; no confidence intervals. These selected validation outcomes are development evidence.",fontsize=12,color=style.MUTED)
    path=style.OUT/"05-pilot-results.svg"
    fig.savefig(path,metadata={"Date":None}); plt.close(fig)
    svg=path.read_text()
    start=svg.index(">",svg.index("<svg"))+1
    svg=svg[:start]+'\n<title>Measured policy improvements over replay</title><desc>Seven achieved policies, using recomputed aggregate validation errors. Scene selection has the largest improvement. No uncertainty intervals are available from these aggregates.</desc>'+svg[start:]
    path.write_text(svg)
    cairosvg.svg2png(url=str(path),write_to=str(style.OUT/"previews/05-pilot-results.png"),output_width=1200)
    assert ET.parse(path).find('.//{http://www.w3.org/2000/svg}image') is None


def html():
    # The shared builder provides the same type, spacing and print styles.
    style.build_html()
    temporary=ROOT/"proposal.html"
    text=temporary.read_text().replace("<title>Correct pose errors without erasing gait</title>","<title>Teaching pose estimators with synthetic lessons</title>")
    text=text.replace("</style>", "table { width: 100%; border-collapse: collapse; font-size: 15px; }\n"
                      "th, td { padding: 9px 12px; border-bottom: 1px solid #d5dfe7; text-align: left; }\n"
                      "th { background: #edf3fc; }\n</style>")
    (ROOT/"study.html").write_text(text)
    temporary.unlink()


if __name__ == "__main__":
    result,errors,methods=audit_results()
    style.validate_and_render(diagrams())
    results_chart(errors,methods)
    html()
    print("Six SVG figures generated, including one measured-results chart. Saved CSV arithmetic verified.")
