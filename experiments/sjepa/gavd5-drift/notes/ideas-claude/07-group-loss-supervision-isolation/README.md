# Does the label-aware curriculum buy generalization or memorization? Isolating the group loss under source-video-disjoint folds

> On a small feasible set of source-video holdouts, does the explicit Stage-1..4 group-loss term improve held-out-source condition separation over a matched arm with that term zeroed, or does it only improve separation on videos the encoder already saw while leaving separation on new videos unchanged?

(If you want the step-by-step recipe for running this study rather than the science behind it, see the companion how-to guide in [METHODOLOGY.md](./METHODOLOGY.md).)

## The question in plain words

This project trains a model that watches short clips of walking. Each clip is turned into a stick-figure skeleton, and the model learns to describe the clip as a list of numbers (a vector). The hope is simple: clips from people who walk in similar ways should land close together in that number space. Training happens in five stages. Stage 0 uses only normal walking. Then Stages 1 through 4 add one condition at a time: Parkinson's, then stroke, then myopathic, then cerebral palsy.

During Stages 1 through 4 the training loss (the single score the model tries to make small) has three parts added together:

`L = L_JEPA + 0.05 * L_VICReg + 0.25 * L_group`

Reading the math:
- This says the total training loss is a weighted sum of three separate jobs the model must do at once.
- `L` is the total loss, the one number training pushes downward. Lower is better.
- `L_JEPA` is the self-supervised part: predict hidden pieces of a skeleton from the visible pieces. It never looks at the condition label.
- `L_VICReg` is an anti-collapse part: stop the model from mapping every clip to the same vector. It also never looks at the label.
- `L_group` is the label-aware part: it reads the condition label and pulls same-condition clips together while pushing different conditions apart.
- `*` means multiply, and `+` means add. So each part is scaled by its weight, then the three are summed.
- The weight `0.05` on `L_VICReg` is small: VICReg is a gentle guardrail, not the main goal. The weight `0.25` on `L_group` is larger, so the label-aware term has a real say in training, but it is still smaller than the weight of `1` sitting implicitly in front of `L_JEPA`, which stays the main job.
- All three parts are losses, so each is a non-negative number, and smaller means the model is doing that job better.
- If you set the `0.25` weight to zero, `L_group` disappears and Stages 1 through 4 stop using labels through this term. That is exactly the OFF arm this proposal builds, and it is how we test what the label-aware term is actually buying.

The first two parts are self-supervised: they do not use the condition label at all. They just ask the model to predict hidden parts of a skeleton from visible parts, and to avoid collapsing every clip to the same vector. The third part, `L_group`, is different. It reads the condition label. It pushes clips of the same condition toward a shared center (a centroid), and it pushes different-condition centers apart. Because that term uses labels, Stages 1 through 4 are not pure self-supervised learning. They are supervised representation fine-tuning. That is claim boundary 4 in the shared facts.

The plain question is: what is that label-aware term actually buying us? There are two very different stories. In the good story, `L_group` teaches the encoder a transferable rule about what makes gait look Parkinsonian or stroke-like, so a brand-new video the encoder has never seen still lands in the right region. In the memorization story, `L_group` mostly pulls together the specific videos it trained on, because in this tiny cohort the condition label is nearly the same thing as the source-video identity. All 12 normal sequences come from one YouTube video, and there are only 2 source videos for Parkinson's, 3 for stroke, 2 for cerebral palsy. So "learn the condition" and "memorize the video" are almost indistinguishable unless we test on held-out videos.

The way to tell the stories apart is to compare two numbers. The transductive number measures separation on videos the encoder trained on. The inductive number measures separation on videos held out from encoder training. If the group loss lifts the transductive number but not the inductive number, it is buying memorization, not generalization.

## Why this matters

The project's headline result is a transductive macro-F1 of 0.821 on the all-96 five-class readout, where every one of the 16 test videos and all 29 test rows also trained the encoder (notebook 06). Macro-F1 is a score between 0 and 1, higher is better, that averages how well each class is found. Because the encoder already saw those videos, this 0.821 is a seen-video score. If the label-aware group loss is the reason that number is high, and if it does not survive a source-video-disjoint split, then the honest way to report the whole project changes: the 0.821 becomes a statement about fit to seen videos, not about learned gait structure.

A positive result, meaning the group-loss-ON arm beats the group-loss-OFF arm on new videos by a pre-registered margin, confirms the belief that label-aware fine-tuning at this scale teaches transferable condition structure and not just video identity. That would justify the supervised curriculum as a design choice.

A null result, meaning ON and OFF are indistinguishable on new videos while ON still wins on seen videos, rules out that belief. It would show the group term buys transductive separation that does not transfer, which is the direct signature of memorization. A null here is informative in the sense ICLR/ICML/NeurIPS 2026 reward: it changes how the project reports every transductive score and it warns other small grouped-cohort skeleton projects that a label-aware compactness loss can inflate seen-video metrics without improving generalization.

This term is also the suspected cause of a documented drift in the normal anchor: the cosine similarity of the normal representation fell from 0.954 after Stage 1 to 0.594 after Stage 4.

Reading the math:
- Cosine similarity measures how aligned two vectors point, on a scale from -1 (opposite) through 0 (unrelated) to 1 (identical direction).
- 0.954 is very close to 1, so early on the normal representation barely moved.
- 0.594 is much lower, so by the end the normal representation had shifted a lot.
- The drop from 0.954 to 0.594 is what "substantial drift" means here: the normal clips are described quite differently after Stage 4 than after Stage 1.

Here is the mechanism, stated plainly rather than asserted. The normal centroid does not need its own group term to be active to move. The margin part of `L_group` pushes every pair of condition centroids apart, so when Stages 2, 3, and 4 add new condition centroids, the loss shoves those new centroids away from the normal centroid. Because one shared encoder produces all embeddings, the gradient that separates the new centroids also reshapes the features underlying the normal clips. The normal representation therefore drifts even though only the newly added conditions carry an actively changing group term. This is a suspected, testable cause, not a proven one: the ON-versus-OFF contrast below is what tests it, since the OFF arm removes exactly this margin pressure.

## Conference-level augmentation

This section lifts the item from a single-cohort question ("does the group loss buy generalization here") to a general methods claim, and it grounds the normal-anchor drift in a specific piece of neuroscience: the forgotten axis behind the 0.954-to-0.594 drift is the clinically load-bearing symmetric baseline.

First, one compact phrase we use below, spelled out. SSL means self-supervised learning: the model teaches itself by hiding part of the input and predicting the hidden part, using no labels at all (Jobs A and B above). "Supervised fine-tuning inside SSL" is our plain name for what the group loss does: it slips a label-reading job (Job C) into the middle of an otherwise label-free training recipe. So the recipe is mostly self-taught, with one supervised ingredient stirred in during Stages 1 through 4.

### From neuroscience source to mechanism to a skeleton-measurable feature

Start with the one condition that anchors the symmetric pole of the axis. Myopathy is a PRIMARY MUSCLE disease. The problem is in the muscle itself, not in a brain hemisphere and not in the basal ganglia (Barohn et al. 2014, PMID 25037080; Nagy and Veerapaneni, StatPearls Myopathy, NBK562290). (The basal ganglia are a set of deep brain structures that help run smooth, automatic movement, and a brain hemisphere is one of the two halves of the brain, each of which controls the opposite side of the body.) That mechanism has a direct consequence for how the body moves. The weakness follows what doctors call a limb-girdle distribution, which just means it is strongest in the muscles closest to the hips and shoulders, and it hits both sides of the body roughly equally (symmetric proximal weakness is THE characteristic pattern, Barohn 2014, PMID 25037080; "proximal" means near the center of the body, "symmetric" means equal on left and right). Because the weakness is even on both sides, myopathy sits at the SYMMETRIC pole, opposite the lateralized (one-sided) conditions. Two things follow at the skeleton level.

First, there is no significant left-right spatiotemporal asymmetry versus controls, meaning the timing and spacing of the two legs' steps look about equal on both sides (Xiong et al. 2023, PMID 37525241). Second, there is no damage to the body's rhythm generator, the internal clock that keeps a steady walking beat. Contrast Parkinson's, where damage in the basal ganglia causes a loss of automaticity, meaning movements that should run on autopilot no longer do (Redgrave et al. 2010, PMID 20944662). Because myopathy leaves the rhythm generator intact, cadence (steps per second) is preserved (Vandekerckhove et al. 2022, PMID 35721358: cadence 2.25 vs 2.21 steps per second, not significant). The abnormality that does show up is POSTURAL: anterior pelvic tilt (how far the pelvis tips forward) is 16.4 degrees versus 11.6 degrees in controls (Vandekerckhove et al. 2022, PMID 35721358).

Reading these numbers:
- 2.25 vs 2.21 steps per second is a cadence in steps taken each second. The two values are almost the same, and the study reports the difference as not significant (NS), meaning the small gap is within noise. That is the point: myopathy keeps a normal walking rhythm.
- 16.4 vs 11.6 degrees is the forward tilt of the pelvis, measured in degrees. 16.4 is larger, so the myopathy group tilts the pelvis forward more. This is the postural signature that separates myopathy from a plain normal walk even though the rhythm and the left-right symmetry look normal.

Normal gait is likewise low-asymmetry and rhythm-preserved, so normal and myopathy TOGETHER form one symmetric, rhythm-preserved baseline. The lateralized (one-sided) conditions sit off that baseline, each because of a one-sided injury:

- Stroke. The nerve wires that carry movement commands from the brain to the body cross over from one side to the other on their way down (this crossing is called the corticospinal decussation), so an injury on one side of the brain weakens the opposite side of the body. That one-sided weakness is called contralateral hemiparesis ("contralateral" means opposite side, "hemiparesis" means weakness of one half of the body) (PMID 30571044).
- Hemiplegic cerebral palsy. A one-sided patch of early white-matter injury in the brain (doctors call it periventricular leukomalacia, damage to the white matter near the fluid spaces deep in the brain) leaves one side of the body affected (Volpe 2009, PMID 19081519).
- Early Parkinson's. The disease often starts by damaging a brain pathway on one side first (the nigrostriatal pathway, a dopamine-carrying circuit deep in the brain that helps run smooth movement), so early on it too shows up more on one side (Riederer and Sian-Hulsmann 2012, PMID 22367437).

The validated number that measures the distance off that baseline is the symmetry ratio on step length, stance time, and swing time (Patterson et al. 2010, PMID 19932621).

The skeleton-measurable features that instantiate the symmetric baseline are therefore: signed and unsigned left-right symmetry ratios on step length, stance time, and swing time (Patterson 2010, PMID 19932621), a low step_length_cv, anterior pelvic tilt, and preserved cadence. All of these are recoverable at markerless-skeleton accuracy: temporal MAE 0.02 seconds per step and sagittal joint MAE 4 to 7 degrees (Stenum et al. 2021, PMID 33891585).

Here is the neuro anchor for THIS item. The documented normal-anchor cosine drift, from 0.954 after Stage 1 down to 0.594 after Stage 4, is a candidate signature of the label-aware group loss deforming exactly this symmetric-baseline geometry (normal plus myopathy). That geometry is the clinically load-bearing reference against which every lateralized deficit is defined. So the sharpened question is not just "does the group loss help" but "does supervised centroid compaction PRESERVE or DESTROY the symmetric-baseline geometry", and we answer it with per-stage V-usable information (Xu et al. 2020, arXiv:2002.10689) on three mechanism targets: the symmetry ratio, cadence, and anterior pelvic tilt. In plain words, V-usable information asks how much a simple, fixed reader (like a straight-line probe) can actually pull out of the model's fingerprints about a target. It is defined in full in the Background section below; here it is enough to think of it as "how much a plain reader can recover." If that recoverable amount drops on the symmetric-baseline targets as the curriculum runs, that is the visible sign the group loss deformed the baseline geometry.

### What generalizes beyond gavd5-drift

The transferable claim is a recipe, not a gavd5-drift number. Label-aware centroid supervision injected into a JEPA curriculum (again, this is the "supervised ingredient stirred into a self-taught recipe" defined at the top of this section) either builds transferable structure or it transductively memorizes labels, and the two are DISTINGUISHABLE by pairing the transductive-minus-inductive accuracy gap with per-stage V-usable information (Xu et al. 2020, arXiv:2002.10689) on mechanism targets. That pairing gives a general audit for when in-curriculum supervision helps, which is a live question against V-JEPA-style frozen-probe orthodoxy (Bardes et al. 2024, arXiv:2404.08471): V-JEPA argues for a frozen encoder read only by a downstream probe, whereas the deployed recipe here folds labels into the encoder during Stages 1 to 4. The gap-plus-V-information protocol is the tool that says which choice was right on a given cohort, and it transfers to any grouped small-cohort representation study that is tempted to add a label-aware compactness term.

### External-cohort note (honest scope)

No participant-disjoint skeleton cohort exists for myopathy or cerebral palsy. So skeleton-level clinical transfer of the symmetric-baseline claim is an explicit honest LIMITATION, not a headline. The closest supporting external anchors are label-level and cross-modal only. PhysioNet Gait-in-PD (gaitpdb, 93 PD plus 73 controls, force and IMU, DOI 10.13026/C24H3N) can confirm the rhythm and variability side of the axis at the label level, not at the skeleton level. Multi-view pose corpora (CASIA-B, Yu 2006; OU-MVLP-Pose, Takemura 2018; GREW, arXiv:2205.02692; Gait3D, arXiv:2204.02569, all non-clinical) can test whether the transductive-minus-inductive gap and the V-information trend REPRODUCE on a larger, participant-disjoint pose distribution, which is a methods generalization check rather than a clinical one. Pose validity for the symmetry-ratio and pelvic-tilt features is anchored to Human3.6M versus mocap (Ionescu et al. 2014, DOI 10.1109/tpami.2013.248). Skeletons still cannot recover kinetics or propulsion (Bowden et al. 2006), EMG or spasticity (Ropars et al. 2016), transverse-plane rotation, or an etiologic muscle diagnosis, so no augmentation here upgrades the n=18-source cohort into a clinical-accuracy claim.

### Feasibility delta versus the original

Overall this is medium-high feasibility, a modest addition on top of the original three-week core.

Core (about 3 weeks, unchanged in scale from the original plan). Reuse the frozen d0acc262 checkpoints per stage. Compute the transductive-minus-inductive gap on the small fixed source-video-disjoint holdout, plus the matched-step fold-local finetune arm and the sampler-leakage third arm already specified. Add per-stage V-usable information (Xu et al. 2020, arXiv:2002.10689) on the three mechanism targets (symmetry ratio, cadence, anterior pelvic tilt). Keep the collapse gate (feature std, effective rank, pairwise cosine against the 0.413745 and 0.609342 baselines) as a HARD REJECT. This retrains or fine-tunes only at the small gavd5-drift scale (159 sequences, 35 sources) with the group loss ablated at 0.25 versus 0.0, and requires NO new pretraining.

Reach (plus 2 to 3 weeks). Rerun the gap-plus-V-information protocol on CASIA-B and OU-MVLP-Pose for participant-disjoint methods generalization, and add the label-level rhythm confirmation on gaitpdb. No new clinical data collection is involved at any tier.

## Background and related work

Start from the pieces, because the treatment is a single term buried inside them.

A **token** is the smallest input unit the model reads. Here each sequence is resized to 64 frames, then 4 adjacent frames are grouped into one time patch, giving 16 time positions. With 33 BlazePose body joints, that is 33 x 16 = 528 possible joint-time tokens.

Reading the math:
- `33 x 16 = 528` says the number of tokens is joints times time positions.
- `33` is the count of body joints BlazePose tracks.
- `16` is the number of time positions, since 64 frames grouped 4 at a time gives 64 / 4 = 16.
- `x` is multiply, so every joint appears at every time position, and 33 times 16 is 528 possible joint-time tokens.

One token is a 4-frame by 3-coordinate = 12-number vector (BlazePose gives x, y, and a relative z), mapped by a linear layer into a 64-dimensional embedding (Grishchenko et al., BlazePose GHUM, 2022, arXiv:2206.11678).

Reading the math:
- `4-frame by 3-coordinate = 12-number` says one token packs 4 frames, each with 3 numbers, into a list of 12 numbers.
- `4` is the frames per time patch, `3` is the coordinates per frame (x, y, and a relative z), and `4 x 3 = 12` is the size of the raw token vector.
- A linear layer then turns that 12-number vector into a 64-dimensional embedding, which is just a longer list of 64 numbers the Transformer can work with.

An **encoder** is a small Transformer (depth 2, 4 heads) that turns tokens into feature vectors. This design follows the JEPA family, which stands for Joint-Embedding Predictive Architecture. The idea, introduced for images by Assran et al. (I-JEPA, CVPR 2023, arXiv:2301.08243) and for video by Bardes et al. (V-JEPA, 2024, arXiv:2404.08471), is to predict hidden parts of an input in feature space rather than reconstructing raw pixels or coordinates. The skeleton version is Abdelfattah and Alahi (S-JEPA, ECCV 2024, DOI 10.1007/978-3-031-73411-3_21), which supplies the architecture family here.

There are two encoders. The **view (online) encoder** sees only the visible tokens and is trained by gradient descent. The **target (EMA) encoder** sees all 528 tokens and is never updated by backprop. EMA means exponential moving average: think of it as a slow-moving average that ignores day-to-day noise. The target weights are a slow running average of the online weights, with momentum scheduled by a cosine curve from 0.999 toward 1.0.

Reading the math:
- The momentum is how much of the old target weight is kept at each step, on a scale from 0 to 1.
- "Scheduled by a cosine curve" just means the momentum value follows a smooth, gently bending path over training instead of jumping. It starts at 0.999 and eases up toward 1.0. (This is a different use of the word cosine than the cosine similarity described later, which compares two vectors; here cosine only names the shape of a smooth ramp.)
- `0.999` means each update keeps 99.9 percent of the old target and mixes in only 0.1 percent of the new online weights, so the target moves very slowly.
- Moving the schedule toward `1.0` means the target moves even more slowly as training goes on, giving steadier prediction targets late in training.

Because it is a slow copy, it gives stable prediction targets. A **predictor** (a 2-layer Transformer with a learned mask token) takes the visible features and tries to predict the target encoder's features at the hidden positions. This is the "predict hidden features" loop, and its gradient does not flow into the target (stop-gradient).

Inside `L_JEPA`, the teacher target is first centered (a running EMA center with beta 0.9), then sharpened at temperature 0.06 and detached (stop-gradient), while the prediction side uses temperature 0.10.

Reading the math:
- Temperature controls how sharp or soft a set of scores is. A small temperature makes the biggest score dominate (sharp), a larger one spreads attention out (soft).
- `0.06` on the teacher side is a very small temperature, so the teacher's target is sharp and confident.
- `0.10` on the prediction side is slightly larger, so the student's prediction is a bit softer than the teacher's target, a common asymmetry that helps training stay stable.
- The center's `beta 0.9` means the running center keeps 90 percent of its old value and mixes in 10 percent of the new batch mean each step, so the center drifts slowly and stops one dimension from dominating.

**Masking** decides which tokens are hidden. Only 12 lower-body landmarks are ever maskable targets (shoulders, hips, knees, ankles, heels, foot indices); face and arm joints are always visible context. The most you can hide globally is 12/33 = 0.364, far below V-JEPA's 75 to 90 percent, because masking a face or arm joint is forbidden.

Reading the math:
- `12/33 = 0.364` is a fraction between 0 and 1, the share of all joints you are even allowed to hide.
- `12` is the count of maskable lower-body landmarks, `33` is the total joints, and dividing gives 0.364, meaning at most about 36 percent of joints can be hidden.
- V-JEPA's 75 to 90 percent is much higher, so this study hides far less than a typical video JEPA does.

The sampler targets 0.60 of eligible tokens, and the realized eligible-token fraction drifted from 0.551 at end of Stage 0 to 0.423 at end of Stage 4.

Reading the math:
- These are fractions between 0 and 1, the share of eligible tokens actually masked.
- `0.60` is the goal the sampler aims for.
- `0.551` and `0.423` are what really happened at the end of Stage 0 and Stage 4, so the realized amount fell over the curriculum.

**VICReg** (Bardes, Ponce, LeCun, ICLR 2022, arXiv:2105.04906) stands for Variance-Invariance-Covariance Regularization. It stops collapse (every clip mapping to one vector) by forcing a variance floor per feature and penalizing correlations between features. The final feature standard deviation is 0.413745 and mean pairwise cosine is 0.609342, so the model is not fully collapsed.

Reading the math:
- Feature standard deviation measures how spread out the feature values are. A value at or near 0 would mean everything collapsed to one point.
- `0.413745` is comfortably above 0, so the features still vary and have not collapsed.
- Mean pairwise cosine is the average alignment between clip vectors, on the -1 to 1 cosine scale.
- `0.609342` is well below 1, so clips are not all pointing the same way, another sign the model did not collapse.

The **group loss** `L_group` is the label-aware term. It pulls same-condition clips toward a condition centroid (compactness) and pushes centroids apart (margin). It is active only in Stages 1 through 4. This is the single factor this proposal isolates.

Two references frame the evaluation. Kapoor and Narayanan ("Leakage and the Reproducibility Crisis in ML-based Science", 2022, arXiv:2207.07048) give the taxonomy of leakage this study fights: a held-out probe split is still contaminated if the encoder saw that video. Xu et al. ("A Theory of Usable Information Under Computational Constraints", ICLR 2020, arXiv:2002.10689) define **V-usable information**. In plain terms, it asks how much a fixed, simple reader (a "decoder family," meaning a chosen kind of guessing rule, like a linear probe) can actually pull out about the label from the features. **Mutual information** is a related idea from statistics: it measures how much knowing the features tells you about the label in principle, assuming an unlimited, perfectly clever reader. We use V-usable information for the secondary per-stage information panel because mutual information can say the answer is "in there" even when a simple reader like a linear probe cannot get to it. V-usable information matches what a realistic probe can actually recover.

## Method

The treatment is a single toggle, `L_group` on versus off, everything else matched. Making that toggle honest is the whole design.

**Step 0, regenerate the artifacts inside this clone (Week 1, gating).** This clone has no `.pt` checkpoints and no `pose_missingness_features.csv`. We first re-extract the `.npz` token caches from the source CSVs so the 528-token tensors exist, and regenerate the transductive reference checkpoint and missingness features. Nothing else starts until these are reproduced and verified.

**Step 1, bind to one fingerprint.** The transductive reference (the encoder that saw everything) is the five-stage curriculum-final checkpoint with fingerprint prefix `d0acc262` (600 curriculum epochs, 11,400 optimizer updates). A canonical lineage prefix `dba24a` has also been observed locally; every result is bound to `d0acc262` before any comparison, and the fingerprint is printed next to every number. The fold-local encoders trained below are new, and each carries its own hash, fold, and seed.

**Step 2, harmonize provenance before training.** Most normal rows use the augmented extraction path; every abnormal row uses the canonical path. A normal-vs-abnormal signal can therefore be acquisition, not gait. We adopt proposal 01's Day-3 provenance gate: harmonize to a single extraction pathway before training, OR make the primary held-out task a binary contrast with provenance regressed out.

"Regressed out" (also called provenance regression) means we first ask how much of the score can be explained by the acquisition pathway alone (which extraction path a clip came from), and then we subtract that part off, so what remains is the signal that is NOT just explained by the pathway. Think of it as removing the effect of a known nuisance variable so it cannot masquerade as gait. Five-class centroid geometry is demoted to secondary.

**Step 3, define the two matched arms precisely.** The treatment is the EXPLICIT group-loss term, not all label information. The balanced sampler and VICReg's requirement of at least 2 samples per condition per batch both leak label information even when `L_group` is zeroed. So we run three arms to bound that leakage:

- Arm ON: `L = L_JEPA + 0.05 * L_VICReg + 0.25 * L_group`, the deployed recipe.
- Arm OFF-matched: group weight set to 0, but the same balanced sampler retained, with a documented account of how VICReg's variance floor is satisfied without labels.
- Arm OFF-random: group weight 0 and a plain random sampler, to measure how much of any ON advantage the sampler alone reproduces.

Reading the math:
- The ON recipe is the same weighted sum defined earlier: the `0.25` weight in front of `L_group` is what makes the label-aware term active.
- OFF-matched sets that `0.25` weight to `0`, so the third term contributes nothing to `L` while everything else, including the sampler, stays the same.
- OFF-random also sets the weight to `0` and additionally swaps the balanced sampler for a plain random one, isolating how much benefit came from the sampler alone.

The null is scoped to the explicit term: ON vs OFF-matched is the primary contrast, OFF-random bounds sampler leakage.

**Step 4, fold-local compact fine-tune.** We do NOT retrain 11,400 updates per fold. On a small fixed feasible holdout set, each arm gets a compact matched-step fold-local fine-tune from the same Stage-0 starting point, identical steps, batch schedule, and seeds across arms. Only the group weight (and, for OFF-random, the sampler) changes.

**Step 5, measure transductive and inductive separation.** For every fold and arm we compute condition separation two ways: transductive (on videos the fold-local encoder trained on) and inductive (on the held-out source videos). Separation is measured two ways. First, a source-level held-out macro-F1 from a linear probe. A linear probe is a simple classifier (a straight-line rule) trained on top of the frozen features to guess the condition; "regularized" just means we hold its numbers back a little so it cannot overfit the tiny data. Second, a geometry statistic, meaning a number that describes the shape of where the clips land in the number space, rather than a classifier score. The one we use is minimum centroid cosine distance (the shared-facts transductive reference is 0.036718).

Reading the math:
- Minimum centroid cosine distance is the gap between the two closest condition centers, measured as 1 minus their cosine similarity, so larger means better separated.
- `0.036718` is the reference value from the seen-video setting: it is small, which matches the finding that the conditions barely separate.
- We report this alongside macro-F1 so a claim of "better separation" must show up in both a probe score and the geometry.

**Worked example (illustrative numbers).** Suppose we run the myopathic leave-one-source-out folds and, averaged across them, we read these inductive (held-out source) macro-F1 scores. All four numbers below are made up for illustration, not grounded facts.
- ON inductive macro-F1 = 0.62 (illustrative).
- OFF-matched inductive macro-F1 = 0.60 (illustrative).
- Step 1: compute the gap, 0.62 - 0.60 = 0.02.
- Step 2: compare the gap to the pre-registered margin of +0.05. Here 0.02 is smaller than 0.05.
- Step 3: even before checking sign consistency, the margin is not crossed.
- Step 4: also confirm both arms clear the missingness floor of 0.429; 0.62 and 0.60 both do, so both are reading more than visibility.
- How to read it: because 0.02 is below the +0.05 margin, this illustrative outcome is scored as a NULL, the memorization signature, especially if the same ON arm shows a large transductive-only lead. A grounded positive would instead need the inductive gap to reach at least +0.05 with the sign consistent across the myopathic folds.

Below is a short, readable sketch of the core mechanic: toggling the group-loss weight and then reading the seen-video (transductive) versus new-video (inductive) gap. It illustrates the idea and is not meant to run against real files.

```python
import numpy as np

# The one knob this proposal turns: the group-loss weight.
# 0.25 is the deployed ON recipe; 0.0 is the OFF arm.
def total_loss(l_jepa, l_vicreg, l_group, group_weight):
    # weighted sum: L = L_JEPA + 0.05 * L_VICReg + group_weight * L_group
    return l_jepa + 0.05 * l_vicreg + group_weight * l_group

# After fine-tuning each arm, we score separation two ways per arm.
# macro_f1 is between 0 and 1; higher means conditions are better separated.
def margin_verdict(on_inductive, off_inductive, missingness_floor=0.429, margin=0.05):
    gap = on_inductive - off_inductive          # ON minus OFF on NEW videos
    reads_gait = on_inductive > missingness_floor and off_inductive > missingness_floor
    if not reads_gait:
        return "below floor: arm not reading gait"
    if gap >= margin:                           # must clear the +0.05 margin
        return "positive: group loss transfers"
    return "null: transductive-only benefit (memorization)"

print(margin_verdict(0.62, 0.60))   # illustrative -> 'null' because 0.02 < 0.05
```

## The decisive experiment

The split is fixed and stated before any fitting. The independent unit is the SOURCE VIDEO, not the clip (claim boundary 1). Because per-condition source counts are tiny (normal 1, Parkinson's 2, stroke 3, myopathic 10, cerebral palsy 2), no five-class LOSO fold can hold out a normal source at all, so the PRIMARY endpoint is binary and provenance-controlled.

Pre-registered before fitting:

- **Primary held-out task:** binary held-out-abnormal separation on canonical-pathway conditions only (myopathic with 10 sources and cerebral palsy with 2 sources are the two canonical-pathway conditions that can be held out at all), with provenance regressed out of the probe. Leave-one-source-out (LOSO) for every condition with fewer than 4 sources; per-source dots plus a bootstrap CI. A CI is a confidence interval, an honest error bar: a range that says where the true score most likely sits given how little data we have. "Bootstrap" is how we build that range: we resample the data we have many times over, recompute the score each time, and see how much it wobbles. A wide bootstrap CI is the tool warning us that with this few source videos, one score alone can be misleading. Myopathic carries the sign-consistency test; cerebral palsy, with only 2 single-source folds, contributes to the pooled magnitude and per-source dots but not to the sign rule (see the minimum-source-count note below).
- **Primary endpoint:** held-out-source macro-F1 (inductive), reported alongside the transductive macro-F1 for the same arm and fold.
- **Pre-registered margin:** ON must beat OFF-matched on the inductive endpoint by at least +0.05 macro-F1, with the sign consistent across held-out sources. We use macro-F1 here because, like balanced accuracy, it weights every condition equally instead of rewarding an arm for just favoring the most common condition; "balanced-accuracy-equivalent" is only shorthand for that fair-to-every-class property. A delta smaller than +0.05, or one that flips sign across sources, is scored as a NULL (memorization: transductive-only benefit).

  Reading the math:
  - `+0.05` is the smallest inductive macro-F1 gain (ON minus OFF-matched) we agree to count as a real effect. Macro-F1 runs from 0 to 1, so +0.05 is a five-hundredths-of-the-scale improvement.
  - The gap is computed as ON inductive macro-F1 minus OFF-matched inductive macro-F1.
  - "Sign consistent" means the gap must be positive across the held-out sources, not positive on some and negative on others.
  - If the gap is under +0.05, or its sign flips across sources, the result is a NULL, meaning the group term helped only on seen videos.

- **Minimum source count for the sign rule.** The sign-consistency criterion is only meaningful when a condition contributes enough held-out folds to reveal a sign pattern. Myopathic has 10 source videos, so leave-one-source-out (LOSO) there yields 10 single-source folds and a genuine sign distribution. Cerebral palsy has only 2 source videos, so LOSO there yields just 2 folds of n=1 held-out source each, which cannot establish a sign pattern on its own. We therefore pre-register that the sign-consistency rule is evaluated only on conditions with at least 4 source videos (here, myopathic). Cerebral palsy contributes to the pooled magnitude estimate and is reported per source as dots, but it is explicitly excluded from the sign-consistency test, and we say so beside the number rather than letting 2 thin folds carry the rule.
- **Simple non-neural / nuisance baseline:** a missingness-only probe (visibility pattern, no gait coordinates), whose all-96 transductive reference is macro-F1 0.429 (notebook 06; the same control scores accuracy 0.448 and balanced accuracy 0.466). We compare against the macro-F1 floor of 0.429 throughout, matching the endpoint used in the figures. Any arm must beat this floor to count as reading gait.

  Reading the math:
  - These are all scores between 0 and 1, higher is better.
  - `0.429` is the macro-F1 of a probe that sees only which joints are visible, not their positions, so it is the "no gait information" floor.
  - `0.448` accuracy and `0.466` balanced accuracy are the same control read two other ways; they are reported for completeness.
  - Plain-language reminder of these scores: accuracy is the plain fraction of clips guessed right, which can look good just by favoring the biggest class. Balanced accuracy fixes that by averaging the fraction right within each condition separately, so a rare condition counts as much as a common one. Macro-F1 is a close cousin that also averages per-condition performance (each class weighted equally), so like balanced accuracy it does not let one large class hide poor performance on the others.
  - An arm that cannot beat 0.429 is not reading gait, only visibility patterns.

- **Hard collapse gate for BOTH arms:** feature standard deviation and effective rank must stay healthy (reference feature std 0.413745). A collapsed arm is rejected regardless of its scores.

  Reading the math:
  - Feature standard deviation near 0 signals collapse (all clips mapped to one vector); a healthy value stays well above 0.
  - `0.413745` is the reference healthy value; an arm that falls far below it is rejected no matter how good its macro-F1 looks.
  - Effective rank is a plain count of how many genuinely different directions the features spread across. Each clip vector has 64 numbers, so in principle the clips could vary along up to 64 independent directions. If the model quietly squashes everything onto just one or two directions, the effective rank drops toward 1, another warning sign of collapse even when the standard deviation still looks acceptable. A healthy effective rank stays comfortably above 1, meaning the features use many directions, not just a few.

| Contrast | Transductive (seen videos) | Inductive (held-out sources) | Reads as |
|---|---|---|---|
| ON vs OFF-matched, both high | high | high | group loss buys generalization (positive) |
| ON high, OFF-matched high; gap only transductive | ON > OFF | ON approx OFF | memorization (null, +0.05 not crossed) |
| OFF-random approx OFF-matched | - | - | sampler leakage is small |
| Any arm below missingness floor | - | below 0.429 F1 | that arm is not reading gait |
| Any arm fails collapse gate | - | - | rejected outright |

Every number is labeled with its encoder-exposure status and the checkpoint fingerprint.

## Controls and incorporated repairs

Each repair from the selection file is addressed here.

- **Single-factor, name it precisely.** The treatment is the explicit group-loss term, not all label information. Because the balanced sampler and VICReg's at-least-2-per-condition precondition leak labels into the group-weight-0 arm, we either match the sampler across arms and document how VICReg is satisfied without labels (OFF-matched), OR add a third random-sampling arm (OFF-random) to bound sampler leakage. Both are included. The null is scoped to the explicit term (Step 3).
- **Regenerate artifacts inside gavd5-drift during Week 1.** No `.pt` or `pose_missingness_features.csv` exist in this clone. We re-extract `.npz` from CSVs and regenerate reference checkpoints and missingness features first, and gate the whole study on that reproduction (Step 0, Day-5 gate).
- **Harmonize extraction to one pathway before training, or make the primary task binary with provenance regressed out.** We adopt proposal 01's Day-3 provenance gate and make the primary held-out task binary and provenance-controlled; five-class centroid geometry is demoted to secondary (Step 2, decisive-experiment table).
- **Scope compute honestly.** A small fixed feasible holdout set, a compact matched-step fold-local fine-tune (NOT 11,400 updates), per-source dots plus LOSO for every under-4-source class plus bootstrap CI, and a pre-registered +0.05 practical A-B margin so a small sign-flipping delta is scored as a null (Steps 4 and 5, decisive experiment).
- **Collapse gate as a hard reject for both arms.** Feature std and effective rank must stay healthy or the arm is rejected (decisive-experiment table).
- **Secondary panels use a FIXED Stage-0 reference target/center.** So predictive competence has one ruler. The secondary panels report per-stage V-usable condition information and residual normal predictive competence measured against a frozen Stage-0 reference, plus a held-out-abnormal contrast and a provenance regression as primary controls. Single-run cross-stage curves are demoted to descriptive, not inferential (Figures section).

Responsible-use reminder for controls: the folder labels (stroke, parkinsons) are dataset annotations, not diagnoses (claim boundary 5). Seed variation is not source variation (claim boundary 3): confirmation uses fresh seeds only after the split and margins are frozen.

## How this differs from the existing plan

The nearest existing item is `plan/04`, the motion-versus-position TARGET ablation, which retrains encoders to compare what the model is asked to predict. This proposal changes a different single factor: the label-aware GROUP LOSS, not the prediction target. As the shared facts state, ideas/07 isolates the label-aware group loss as the single factor, and no plan item does this. It is also distinct from `plan/06` (missingness confound) and `plan/05` (temporal readout on a frozen encoder): here we retrain fold-local encoders with the group term toggled, and the object is the transductive-versus-inductive gap, the direct signature of memorization versus transferable structure.

## Three-week timeline

**Week 1 (16 to 22 Aug 2026).** Regenerate `.npz` token caches from source CSVs; regenerate the transductive reference checkpoint and missingness features; bind everything to fingerprint `d0acc262`. Run proposal 01's provenance gate and choose the harmonized-pathway-or-regress-out route. Fix the small feasible source-video holdout set and pre-register the primary binary task, the +0.05 margin, and the collapse gate. Implement the three arms (ON, OFF-matched, OFF-random) and the compact fold-local fine-tune harness.

**Day-5 gate (20 Aug 2026):** continue only if the `.npz` and reference checkpoint reproduce exactly under `d0acc262`, provenance is harmonized or regressible, the holdout set holds out no encoder-seen video for the primary task, all three arms differ only in the named factor, and both arms pass the collapse gate on a smoke run.

**Week 2 (23 to 29 Aug 2026).** Run screening seeds of all three arms across the fold-local holdouts. Compute transductive and inductive macro-F1 and minimum centroid cosine distance per arm and fold. Run the missingness-only floor and the provenance regression control. Assemble per-source dots for the primary binary contrast.

**Day-14 gate (29 Aug 2026):** continue to confirmation only if ON either crosses the +0.05 inductive margin over OFF-matched with consistent sign, OR the transductive-only pattern (ON > OFF transductive, ON approx OFF inductive) is clean enough to report as an informative null. Otherwise stop and report the ambiguity.

**Week 3 (30 Aug to 5 Sep 2026).** Run fresh confirmation seeds on the frozen protocol. Produce the two-panel transductive-versus-inductive figure and the secondary per-stage V-information panel. Package split manifest, arm configs, seed-level results, and fingerprints.

## Figures

- `./images/fig3.svg`: The gentlest first figure. A five-step flow, walking clip, stick figure, list of numbers, three training jobs, then the seen-video-versus-new-video test, showing that the group loss (Job C) is the single knob this study turns on and off.
- `./images/fig4.svg`: The fair-comparison explainer. One shared Stage-0 start splits into the three matched arms (ON, OFF-matched, OFF-random), each scored twice, with the pre-registered +0.05 inductive rule that reads the result as a positive or a memorization null.
- `./images/fig1.svg`: Two-panel bar chart of held-out-source macro-F1 and minimum-centroid-cosine-distance for group-loss-ON versus OFF, split into a transductive panel and an inductive panel, each with per-source dots and bootstrap CIs.
- `./images/fig2.svg`: Secondary line panel of per-stage held-out-source V-usable condition information and residual normal predictive competence (fixed Stage-0 reference) across curriculum stages, with the held-out-abnormal control line.

## Responsible use

The condition folder labels used here (normal, parkinsons, stroke, myopathic, cerebral_palsy) are dataset annotations from GAVD (Ranjan et al., IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787), not diagnoses made by this project. Nothing in this study is a clinical screening tool. The pre-registered margins are internal project continuation rules, not clinical thresholds, and every reported number is labeled transductive or inductive so no seen-video score is mistaken for evidence of generalization.

## References

- Abdelfattah and Alahi, S-JEPA, ECCV 2024, DOI 10.1007/978-3-031-73411-3_21.
- Assran et al., I-JEPA, CVPR 2023, arXiv:2301.08243.
- Bardes et al., V-JEPA "Revisiting Feature Prediction for Learning Visual Representations from Video", 2024, arXiv:2404.08471.
- Bardes, Ponce, LeCun, VICReg, ICLR 2022, arXiv:2105.04906.
- Grishchenko et al., BlazePose GHUM, 2022, arXiv:2206.11678.
- Kapoor and Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science", 2022, arXiv:2207.07048.
- Xu et al., "A Theory of Usable Information Under Computational Constraints", ICLR 2020, arXiv:2002.10689.
- Ranjan et al., GAVD, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787.
- Barohn et al., "Approach to Peripheral Neuropathy and Myopathy", Neurol Clin 2014, PMID 25037080.
- Nagy and Veerapaneni, "Myopathy", StatPearls, NBK562290.
- Xiong et al., DMD spatiotemporal symmetry, Biomed Eng Online 2023, PMID 37525241.
- Vandekerckhove et al., DMD versus typically-developing gait, Front Hum Neurosci 2022, PMID 35721358.
- Redgrave et al., "Goal-directed and habitual control in the basal ganglia", Nat Rev Neurosci 2010, PMID 20944662.
- Riederer and Sian-Hulsmann, asymmetric nigrostriatal onset in Parkinson's, J Neural Transm 2012, PMID 22367437.
- Natali and Javed, corticospinal tract anatomy (pyramidal decussation), StatPearls, PMID 30571044.
- Volpe, "Brain injury in premature infants" (periventricular leukomalacia), Lancet Neurol 2009, PMID 19081519.
- Patterson et al., gait symmetry-index methods (symmetry ratio), Gait Posture 2010, PMID 19932621.
- Stenum et al., pose-estimation gait accuracy versus mocap, PLoS Comput Biol 2021, PMID 33891585.
- Ionescu et al., Human3.6M, IEEE TPAMI 2014, DOI 10.1109/tpami.2013.248.
- Yu et al., CASIA-B multi-view gait database, 2006.
- Takemura et al., OU-MVLP-Pose multi-view pose gait dataset, 2018.
- Zhu et al., GREW gait recognition in the wild, 2022, arXiv:2205.02692.
- Zheng et al., Gait3D, 2022, arXiv:2204.02569.
- Goldberger et al., PhysioNet Gait-in-PD (gaitpdb), DOI 10.13026/C24H3N.
