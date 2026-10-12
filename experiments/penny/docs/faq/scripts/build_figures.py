#!/usr/bin/env python3
"""Create original, accessible SVG diagrams for the GAVD3 concept FAQ.

All drawings are vector artwork. Quantitative values in exact-split-results
are the retained Notebook 06 results supplied by the evidence audit.
Run from any directory: python docs/faq/scripts/build_figures.py
"""
from pathlib import Path
from html import escape
import math

OUT = Path(__file__).resolve().parents[1] / "figures"
INK = "#183247"
MUTED = "#526579"
BLUE = "#285A91"
TEAL = "#157B78"
AMBER = "#9C681C"
PURPLE = "#78558F"
PALE = "#F4F7FA"
LINE = "#CCD6DE"


class Drawing:
    def __init__(self, title, desc, height=450):
        self.height = height
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="{height}" viewBox="0 0 1100 {height}" role="img" aria-labelledby="title desc">',
                      f'<title id="title">{escape(title)}</title><desc id="desc">{escape(desc)}</desc>',
                      '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#526579"/></marker></defs>',
                      f'<rect width="1100" height="{height}" fill="white"/>']
        self.text(32, 47, title, size=31, weight=700)
        self.line(32, 68, 1068, 68, color=LINE, width=1)

    def text(self, x, y, content, size=23, color=INK, weight=400, anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{escape(str(content))}</text>')

    def lines(self, x, y, lines, size=23, color=INK, weight=400, gap=30, anchor="start"):
        for i, value in enumerate(lines):
            self.text(x, y+i*gap, value, size, color, weight, anchor)

    def rect(self, x, y, w, h, fill=PALE, stroke=LINE, radius=12, dashed=False):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"'+ (' stroke-dasharray="7 5"' if dashed else '') + '/>')

    def line(self, x1, y1, x2, y2, color=MUTED, width=3, arrow=False, dashed=False):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"'+ (' marker-end="url(#arrow)"' if arrow else '') + (' stroke-dasharray="7 5"' if dashed else '') + '/>')

    def circle(self, x, y, r=7, fill=TEAL, stroke="none"):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')

    def box(self, x, y, w, h, heading, lines=(), color=BLUE, dashed=False):
        self.rect(x, y, w, h, dashed=dashed)
        self.rect(x, y, 7, h, color, color, 3)
        self.text(x+20, y+33, heading, 24, color, 700)
        self.lines(x+20, y+65, lines, 22, MUTED, gap=29)

    def note(self, value):
        self.text(32, self.height-20, value, 21, MUTED)

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{name}.svg").write_text("\n".join(self.parts+['</svg>']), encoding="utf-8")


def research_roadmap():
    d = Drawing("From walking videos to questions about health", "The research progresses from video and pose to latent representations and evaluation. Notebook 04 and 06 provide pretraining and classifier results; notebook 07 through 10 add source audits, gait probes, mask comparisons and forecast experiments with saved artifacts.", 460)
    stages = [("Video", ["Annotated walker", "and frame span"], BLUE), ("Pose", ["33 landmarks", "per video frame"], TEAL), ("S-JEPA", ["Predict hidden", "representations"], PURPLE), ("Evaluate", ["Freeze features;", "test classifiers"], AMBER)]
    for i, (h, lines, color) in enumerate(stages):
        x=32+i*264
        d.box(x, 105, 230, 132, h, lines, color)
        if i<3: d.line(x+233, 170, x+260, 170, arrow=True)
    d.box(32, 270, 510, 125, "Learn and evaluate", ["04: pretraining  •  06: classifier evaluation"], TEAL)
    d.box(565, 270, 503, 125, "Follow-up questions", ["07: source audit  •  08–10: probes and controls"], MUTED)
    d.note("Saved follow-up artifacts deepen the audit; each comparison needs its own interpretation.")
    d.save("research-roadmap")


def jepa_training():
    d=Drawing("Two encoders, one prediction task", "Visible skeleton tokens pass through the view encoder and predictor. Full tokens pass through a target encoder. The prediction is compared to the target at masked positions. Gradients update the view encoder and predictor; the target weights are updated through an exponential moving average.", 520)
    d.box(32, 112, 226, 118, "Visible tokens", ["Selected tokens", "are hidden"], BLUE)
    d.box(319, 112, 235, 118, "View encoder", ["Trainable features"], BLUE)
    d.box(614, 112, 222, 118, "Predictor", ["Fill hidden", "latent positions"], PURPLE)
    d.box(32, 302, 226, 110, "Full tokens", ["Complete sequence"], TEAL)
    d.box(319, 302, 235, 110, "Target encoder", ["No loss gradients"], TEAL)
    for x1, x2, y in [(262, 310, 167), (558, 605, 167), (262, 310, 357)]: d.line(x1,y,x2,y,arrow=True)
    d.box(862, 236, 206, 123, "Compare", ["At masked", "positions only"], AMBER)
    d.line(840, 167, 964, 167)
    d.line(964, 167, 964, 227, arrow=True)
    d.line(558, 357, 851, 357, arrow=True)
    d.line(435, 236, 435, 292, color=TEAL, arrow=True, dashed=True)
    d.text(459, 270, "EMA weights", 22, TEAL)
    d.lines(32, 458, ["EMA = exponential moving average: a slow blend of earlier and current encoder weights."], 22, MUTED)
    d.save("jepa-training")


def masks_tokens():
    d=Drawing("A token joins one landmark across four frames", "A 64-frame sequence produces 16 times 33 or 528 tokens. Masking samples approximately 60 percent of valid eligible tokens from ten of the 33 landmarks; other valid tokens stay visible. The grid is illustrative, not an actual sampled mask.", 485)
    d.box(32, 102, 365, 123, "Patch the pose sequence", ["64 frames ÷ 4 = 16 time patches", "16 × 33 landmarks = 528 tokens"], BLUE)
    d.box(32, 252, 365, 148, "Choose prediction targets", ["10 eligible landmarks out of 33", "About 60% of valid eligible tokens", "are hidden from the view encoder"], PURPLE)
    x0,y0=465,116
    for row in range(10):
        for col in range(16):
            masked = (col+2*row)%5 <3
            d.rect(x0+col*35, y0+row*22, 27, 14, PURPLE if masked else "#C1DEE0", "none", 2)
    d.text(465, 100, "Time patches →", 22, MUTED)
    d.text(460, 358, "Ten eligible landmarks; example mask only", 22, MUTED)
    d.rect(466, 381, 23, 18, PURPLE, "none", 3)
    d.text(501, 398, "Hidden target", 22)
    d.rect(724, 381, 23, 18, "#C1DEE0", "none", 3)
    d.text(759, 398, "Visible context", 22)
    d.note("60% refers to eligible valid tokens, not 60% of the whole skeleton.")
    d.save("masks-tokens")


def representation_pooling():
    d=Drawing("One sequence becomes a 384-number representation", "The frozen target encoder produces 96-dimensional token features. Means and standard deviations are pooled over all valid tokens and over the selected landmark tokens. Four 96-dimensional summaries concatenate into 384 features.", 460)
    d.box(32, 125, 232, 160, "Frozen encoder", ["96 values", "per valid token", "Weights stay fixed"], TEAL)
    for i,(h,l) in enumerate([("All landmarks", "Mean: 96"),("All landmarks", "Std. dev.: 96"),("Selected ten", "Mean: 96"),("Selected ten", "Std. dev.: 96")]):
        d.box(315, 96+i*76, 330, 65, h, [], BLUE if i<2 else PURPLE)
        d.text(625, 96+i*76+43, l, 22, INK, anchor="end")
        d.line(270, 205, 306, 128+i*76, arrow=True, width=2)
        d.line(650,128+i*76,709,241,arrow=True,width=2)
    d.box(723, 156, 345, 161, "Concatenate", ["96 + 96 + 96 + 96 = 384", "One vector per sequence", "Then fit the classifier"], AMBER)
    d.note("A latent dimension is a learned feature, without a guaranteed clinical interpretation.")
    d.save("representation-pooling")


def loss_collapse():
    d=Drawing("Prediction loss needs a second check: feature diversity", "Schematic feature points show how nearly constant embeddings can match without preserving useful variation. Centering, teacher and predictor temperatures, and collapse diagnostics accompany the loss. The dots are illustrative, not experiment results.", 450)
    d.box(32, 102, 420, 276, "Latent cross-entropy", ["Center target channels", "Apply target / predictor temperatures", "Convert channels to distributions", "Compare the two distributions", "Monitor more than the loss"], BLUE)
    d.rect(490, 102, 268, 276)
    d.rect(786, 102, 282, 276)
    d.text(512,136,"Collapsed features",24,AMBER,700)
    d.text(808,136,"Varied features",24,TEAL,700)
    for i in range(16): d.circle(626+8*math.cos(i),246+7*math.sin(i),6,AMBER)
    for i in range(16): d.circle(815+(i*53)%224,179+(i*37)%151,6,TEAL)
    d.lines(512,339,["Different inputs look alike"],21,MUTED)
    d.lines(808,339,["Variation is preserved"],21,MUTED)
    d.note("Schematic only. Diversity is necessary; it does not establish a health-specific representation.")
    d.save("loss-collapse")


def split_leakage():
    d=Drawing("A sequence split can reuse the same source video", "A sequence split places clips from a single source video in both train and test. A source-disjoint split keeps each source in one partition. Pretraining on test-source normal clips is transductive exposure even without labels.", 500)
    d.box(32, 100, 487, 324, "Sequence split", ["Different clips; sources may overlap"], AMBER)
    d.box(555, 100, 513, 324, "Source-disjoint split", ["Each video belongs to one partition"], TEAL)
    for x,kind in [(54,"sequence"),(577,"source")]:
        d.text(x+32,220,"Train",23,BLUE,700)
        d.text(x+283,220,"Test",23,PURPLE,700)
        for i in range(3):
            d.rect(x+30,242+i*42,178,30,"#E6EFF8",BLUE,4)
            d.rect(x+281,242+i*42,178,30,"#F0EAF5",PURPLE,4)
            d.text(x+44,264+i*42, f"Video {chr(65+i)} · clip",20)
            d.text(x+295,264+i*42, f"Video {chr(65+i if kind=='sequence' else 68+i)} · clip",20)
    d.note("Unlabeled pretraining exposure to a test source is transductive; audit it separately.")
    d.save("split-leakage")


def hypothesis_loop():
    d=Drawing("Each experiment should answer one narrower question", "A hypothesis-testing loop starts with a null hypothesis, defines a controlled comparison, evaluates held-out evidence with uncertainty, audits competing explanations and revises the next question.", 460)
    d.box(32, 101, 319, 126, "State a null hypothesis", ["No added condition information", "beyond the chosen baseline"], BLUE)
    d.box(396, 101, 318, 126, "Make a controlled test", ["Hold data and compute fixed", "Change one relevant factor"], TEAL)
    d.box(759, 101, 309, 126, "Evaluate evidence", ["Held-out scores", "Uncertainty and errors"], PURPLE)
    d.line(357,165,385,165,arrow=True)
    d.line(720,165,748,165,arrow=True)
    d.box(396, 284, 672, 107, "Audit competing explanations", ["Source identity • missing poses • view angle • label imbalance"], AMBER)
    d.box(32, 284, 319, 107, "Refine the next question", ["Return to a testable claim"], BLUE)
    d.line(913,233,913,274,arrow=True)
    d.line(387,337,360,337,arrow=True)
    d.line(188,277,188,237,arrow=True)
    d.note("A null hypothesis is a testable starting claim, not a conclusion selected after seeing the score.")
    d.save("hypothesis-loop")


def condition_proxies():
    d=Drawing("A condition label can travel through several pathways", "Health labels may be associated with real movement differences and with recording or extraction differences. Both can enter pose features and classifier outputs. A label prediction alone cannot distinguish these explanations.", 460)
    d.box(32, 187, 241, 108, "Condition label", ["Dataset annotation"], BLUE)
    d.box(328, 104, 394, 107, "Movement pathway", ["Timing, asymmetry, joint motion"], TEAL)
    d.box(328, 287, 394, 107, "Recording / extraction pathway", ["Camera, source, missing landmarks"], AMBER)
    d.box(795, 184, 273, 119, "Model features", ["Either pathway may", "support label prediction"], PURPLE)
    d.line(280,231,318,157,arrow=True)
    d.line(280,254,318,338,arrow=True)
    d.line(729,157,785,228,arrow=True)
    d.line(729,338,785,261,arrow=True)
    d.note("Arrows show possible associations in the dataset, not established causal effects.")
    d.save("condition-proxies")


def infill_future():
    d=Drawing("Hidden-position prediction and future prediction ask different questions", "Infill predicts masked representations using available surrounding time patches. Future prediction supplies only earlier patches and hides the entire suffix. Showing future context would invalidate a forecasting claim.", 455)
    labels=[("Infill",[0,0,1,0,1,0,0,1],"Context may occur before and after a target."),("Future prediction",[0,0,0,0,0,1,1,1],"The entire suffix is hidden; no later context is visible.")]
    for row,(title,masks,note) in enumerate(labels):
        y=118+row*156
        d.text(32,y+25,title,25,BLUE,700)
        for i,m in enumerate(masks):
            x=308+i*91
            d.rect(x,y,73,54,"#F0EAF5" if m else "#E2F1EF",PURPLE if m else TEAL,7)
            d.text(x+36,y+36,"?" if m else str(i+1),25,PURPLE if m else TEAL,700,"middle")
        d.text(308,y+90,note,22,MUTED)
    d.note("These timelines explain tasks; forecasting needs an unseen sequence and unseen-source test.")
    d.save("infill-future")


def metric_guide():
    d=Drawing("Different scores reveal different parts of performance", "Accuracy counts all correctly classified clips, balanced accuracy averages recall across classes, and macro F1 averages per-class F1. These metrics have different weighting and cannot be treated as interchangeable.", 445)
    for i,(h,lines,color) in enumerate([
        ("Accuracy",["Correct clips ÷ all test clips", "Every clip has equal weight", "Frequent classes can dominate"],BLUE),
        ("Balanced accuracy",["Average recall over classes", "Each class has equal weight", "Checks class recovery"],TEAL),
        ("Macro F1",["Average class-specific F1", "Balances precision and recall", "Each class has equal weight"],PURPLE)]):
        d.box(32+i*354,112,328,232,h,lines,color)
    d.note("Read these with the confusion matrix, class counts, and uncertainty from the actual test set.")
    d.save("metric-guide")


def motion_probes():
    d=Drawing("Probe the representation against measurable movement", "A frozen representation is tested for information related to motion descriptors such as cadence, asymmetry and joint excursion. The descriptors come from extracted pose and are proxy measurements. Compare probes with raw-coordinate and missingness baselines.", 460)
    d.box(32, 118, 285, 156, "Frozen representation", ["A fixed 384-value vector", "No encoder retraining"], BLUE)
    d.box(374, 118, 246, 156, "Simple probe", ["Fit a small mapping", "on training examples"], TEAL)
    d.box(678, 100, 390, 222, "Pose-derived targets", ["Cadence: step rhythm", "Asymmetry: left/right difference", "Excursion: joint movement range", "Sway: side-to-side movement"], PURPLE)
    d.line(323,196,364,196,arrow=True)
    d.line(626,196,668,196,arrow=True)
    d.text(32,366,"Compare with raw-coordinate and missingness baselines on held-out sources.",23,MUTED)
    d.note("Pose-derived targets are measurement proxies, not validated clinical motion-capture labels.")
    d.save("motion-probes")


def augmentation_tradeoff():
    d=Drawing("An augmentation declares which differences should matter less", "Small translation and rotation are view augmentations in training; body-width scaling occurs during input preparation. Horizontal flipping is disabled because left-right laterality may matter. Transformations must be checked for preservation of movement information.", 450)
    d.box(32,104,500,248,"View changes and preparation",["Small translation: position in the image", "Small rotation: slight orientation change", "Body-width scaling: input preparation", "Aim: reduce dependence on recording setup"],TEAL)
    d.box(567,104,501,248,"Preserve meaningful differences",["Left/right laterality may be informative", "Horizontal flipping is disabled", "Check effects on motion and visibility", "A transformation can remove useful signal"],AMBER)
    d.note("A design choice needs evaluation; its motivation does not establish that it improves the result.")
    d.save("augmentation-tradeoff")


def exact_split_results():
    d=Drawing("Saved classifier results on the same 21 test sequences", "On the exact retained split, S-JEPA classified 13 of 21 clips correctly, the saved legacy 82-feature reference 16 of 21, the majority control 7 of 21, and missingness control 7 of 21. All 21 test sequences come from sources present in training. This is a within-corpus comparison.", 500)
    rows=[("S-JEPA",13,TEAL),("Saved legacy82",16,BLUE),("Majority control",7,MUTED),("Missingness control",7,PURPLE)]
    x0=326; chartw=588
    for i,(label,n,color) in enumerate(rows):
        y=119+i*61
        d.text(32,y+28,label,24,INK,700 if i==0 else 400)
        d.rect(x0,y,chartw,35,"#EFF3F6","none",4)
        d.rect(x0,y,chartw*n/21,35,color,"none",4)
        d.text(934,y+27,f"{n}/21",23,INK,700)
        d.text(1068,y+27,f"{100*n/21:.1f}%",23,INK,anchor="end")
    d.text(x0,392,"0%",21,MUTED)
    d.text(x0+chartw/2,392,"50%",21,MUTED,anchor="middle")
    d.text(x0+chartw,392,"100% accuracy",21,MUTED,anchor="end")
    d.rect(32,418,1036,53,"#FFF6E7","#E1C895",8)
    d.text(50,453,"Caution: all 21 test clips share source videos with training; this is not a clinical test.",22,AMBER)
    d.save("exact-split-results")


def grouped_results():
    d=Drawing("Changing the split exposes a generalization gap", "Notebook 07's 70/30 sequence-split classifier accuracy was 62.1 percent on 29 clips versus a training-majority control of 48.3 percent. Six video-grouped folds across 96 clips averaged 24.0 percent accuracy with a 16.5 percentage point sample standard deviation, below the mean training-majority control of 49.0 percent. One test fold includes the normal pretraining source.", 460)
    d.box(32,103,488,249,"Sequence split: Notebook 07",["18/29 correct = 62.1% accuracy", "Training-majority control = 48.3%", "Source videos occur in train and test", "70/30 partition of all 96 sequences"],BLUE)
    d.box(555,103,513,249,"Video-grouped classifier folds",["Mean accuracy = 24.0%", "Training-majority control = 49.0%", "Sample standard deviation = 16.5 points", "Six folds × 16 held-out clips"],TEAL)
    d.note("One grouped test fold includes the pretraining source; encoder training is still transductive.")
    d.save("grouped-results")


def identity_probe():
    d=Drawing("Source video is predictable from several feature families", "Video-identity probe mean accuracy is 49.4 percent for S-JEPA, 51.7 percent for the untrained encoder, 46.0 percent for raw pose pooling, and 42.5 percent for missingness. A uniform guess among 18 videos would be 5.6 percent; this is not an empirical majority baseline.", 485)
    rows=[("Frozen S-JEPA",.4942528735632184,.12109946842359472,TEAL),("Untrained encoder",.5172413793103449,.1580198515502014,BLUE),("Raw pose pooling",.4597701149425288,.039817259944112136,AMBER),("Missingness",.42528735632183906,.019908629972056068,PURPLE)]
    x0=332; width=565
    for i,(label,v,sd,color) in enumerate(rows):
        y=112+i*66
        d.text(32,y+28,label,24)
        d.rect(x0,y,width,36,"#EFF3F6","none",4)
        d.rect(x0,y,width*v,36,color,"none",4)
        left=x0+width*(v-sd); right=x0+width*(v+sd)
        d.line(left,y+18,right,y+18,color=INK,width=2)
        d.line(left,y+10,left,y+26,color=INK,width=2)
        d.line(right,y+10,right,y+26,color=INK,width=2)
        d.text(944,y+28,f"{100*v:.1f}%",24,color,700)
    d.line(x0+width/18,94,x0+width/18,366,color=MUTED,width=2,dashed=True)
    d.text(32,408,"Bars: mean ± sample SD over 3 splits. Dashed line: 5.6% uniform guess.",22,MUTED)
    d.note("High source identification indicates confounding risk; it does not quantify how labels are predicted.")
    d.save("identity-probe")


def gait_probe_results():
    d=Drawing("Scalar gait probes recover only modest signal", "Out-of-fold R-squared for frozen S-JEPA latents ranges from 0.020 to 0.086 across six pose-derived targets. Missingness features exceed the latent scores for cadence and knee excursion. A separate phase vector probe has mean R-squared negative 0.127. All these are pose-based proxy targets.", 565)
    rows=[("Ankle cycle rate",.020,-.051,.118),("Ankle excursion asymmetry",.024,.057,-.128),("Knee excursion",.086,-.470,.174),("Pelvis sway proxy",.037,-.029,-.012),("Pelvis speed proxy",.054,-.131,.029),("Stance/swing proxy",.022,-.022,-.026)]
    d.text(32,105,"Pose-derived target",23,MUTED,700)
    d.text(586,105,"S-JEPA R²",23,TEAL,700,anchor="end")
    d.text(797,105,"Raw pose R²",23,AMBER,700,anchor="end")
    d.text(1034,105,"Missingness R²",23,PURPLE,700,anchor="end")
    for i,(h,a,b,c) in enumerate(rows):
        y=137+i*48
        d.rect(32,y-23,1036,42,PALE if i%2==0 else "white","none",4)
        d.text(47,y+7,h,23)
        d.text(586,y+7,f"{a:.3f}",24,TEAL,700,anchor="end")
        d.text(797,y+7,f"{b:.3f}",24,AMBER,anchor="end")
        d.text(1034,y+7,f"{c:.3f}",24,PURPLE,anchor="end")
    d.lines(32,463,["R² = 1 is perfect; 0 is the pooled mean-predictor reference; negative is worse.", "Separate phase-vector probe: mean R² = −0.127."],22,MUTED,gap=31)
    d.note("Targets are extracted-pose proxies; pooled scores use 96 sequences, or 95 for stance/swing.")
    d.save("gait-probe-results")


def mask_ablation():
    d=Drawing("Mask geometry: the observed difference is one test clip", "With one retained seed 42 run per mask strategy, neurologic-10 and motion-aware-10 each classify 12 of 21 test clips correctly, while random-10 and full-body-33 classify 13 of 21. This small sequence-split experiment does not establish a ranking of mask strategies.", 450)
    rows=[("Neurologic-10",12,TEAL),("Random-10",13,BLUE),("Motion-aware-10",12,AMBER),("Full-body-33",13,PURPLE)]
    for i,(h,n,color) in enumerate(rows):
        y=108+i*64
        d.text(32,y+29,h,24)
        for j in range(21): d.rect(350+j*26,y+6,18,24,color if j<n else "#E9EEF2","none",3)
        d.text(949,y+29,f"{n}/21",23,color,700)
    d.note("Single seed (42), shared sequence split. One-clip differences cannot establish mask superiority.")
    d.save("mask-ablation")


def forecast_results():
    d=Drawing("The forecasting experiments contain a useful failure", "The model's predicted future phase has mean angular error 1.531 radians versus 1.372 for persisting context, where lower is better. A latent cross-entropy comparison uses model training sequences but a leave-one-out reference, so its numerical difference cannot establish held-out forecast superiority.", 500)
    d.box(32,103,500,277,"Phase prediction: lower is better",["Predicted-model error: 1.531 radians", "Persist-context error: 1.372 radians", "The model does not beat persistence", "on this retained phase comparison"],AMBER)
    d.box(568,103,500,277,"Latent-loss comparison: audit first",["Model loss uses its training sequences", "Reference uses leave-one-out matching", "These evaluation conditions differ", "A held-out forecast test is still needed"],BLUE)
    d.note("Future latent similarity and accurate physical gait prediction require separate validation.")
    d.save("forecast-results")


def pose_pipeline():
    d=Drawing("Measurement quality enters before the encoder", "Annotated walker frame spans and bounding boxes define a cropped source video. A pose detector estimates 33 landmarks with visibility. Validity and missing frames are preserved; valid poses are normalized, resized, patched and fed to the encoder.", 455)
    stages=[("Find the walker",["Annotated frames", "and bounding box"],BLUE),("Extract pose",["33 estimated landmarks", "and visibility"],TEAL),("Retain validity",["Keep missing frames", "and detector coverage"],AMBER),("Normalize",["Center and scale", "Resize the sequence"],TEAL),("Patch tokens",["One landmark", "over four frames"],PURPLE),("Encode",["Learned token features", "then pooled summaries"],BLUE)]
    for i,(h,lines,color) in enumerate(stages):
        row=i//3; col=i%3
        x=32+col*354; y=98+row*157
        d.box(x,y,324,122,h,lines,color)
        if col<2: d.line(x+329,y+60,x+349,y+60,arrow=True)
    d.line(1068,221,1068,242,width=2)
    d.line(1068,242,46,242,width=2)
    d.line(46,242,46,251,arrow=True,width=2)
    d.note("Coordinates and monocular depth are estimates; coverage and recording effects can persist.")
    d.save("pose-pipeline")


def health_map():
    d=Drawing("Anatomy motivates targets; every link remains a hypothesis", "Ten bilateral landmarks at the shoulders, hips, knees, ankles and foot indices are eligible mask targets. Their motivation includes proximal motion, symmetry, joint excursion and gait timing. These targets do not define condition-specific biomarkers.", 485)
    cx=281
    points={"head":(cx,119),"ls":(cx-55,179),"rs":(cx+55,179),"lh":(cx-35,264),"rh":(cx+35,264),"lk":(cx-53,335),"rk":(cx+53,335),"la":(cx-65,402),"ra":(cx+65,402),"lf":(cx-94,428),"rf":(cx+94,428)}
    for a,b in [("head","ls"),("head","rs"),("ls","rs"),("ls","lh"),("rs","rh"),("lh","rh"),("lh","lk"),("rh","rk"),("lk","la"),("rk","ra"),("la","lf"),("ra","rf")]:
        d.line(*points[a],*points[b],color=LINE,width=5)
    d.circle(*points["head"],19,"#E7EDF2",LINE)
    for k,p in points.items():
        if k!="head":d.circle(*p,9,TEAL)
    for i,(h,lines,anchor) in enumerate([("Shoulders + hips",["Hip/trunk motion and symmetry"],179),("Knees",["Joint excursion and joint bending"],335),("Ankles + foot indices",["Foot movement and timing"],416)]):
        y=104+i*110
        d.box(563,y,505,89,h,lines,TEAL)
        d.line(382,anchor,552,y+44,arrow=True,width=2)
    d.text(32,462,"Ten eligible landmarks are shown in teal; associations can overlap across health categories.",21,MUTED)
    d.save("health-map")


if __name__ == "__main__":
    for fn in [research_roadmap,jepa_training,masks_tokens,representation_pooling,loss_collapse,split_leakage,hypothesis_loop,condition_proxies,infill_future,metric_guide,motion_probes,augmentation_tradeoff,exact_split_results,grouped_results,identity_probe,gait_probe_results,mask_ablation,forecast_results,pose_pipeline,health_map]:
        fn()
    print(f"Created 20 original SVG illustrations in {OUT}")
