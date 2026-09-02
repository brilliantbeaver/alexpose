# Neuroscience + world-model priors for iterating the 7 proposals

This file is the added context for lifting the seven `notes/ideas-claude/` proposals from internal
dataset audits to ICLR/ICML-level, neuroscience-grounded studies. Pair every claim with the numeric
ground truth in `_shared_facts.md` (do not restate or contradict those numbers). No em-dashes.

## The single most powerful neuroscience discriminant (use this everywhere)
The clinical literature separates the conditions along a SYMMETRY axis:
- LATERALIZED / unilateral deficit: stroke (hemiparesis, one brain side), Parkinson's (typically starts
  one side), hemiplegic cerebral palsy. These raise signed left-minus-right asymmetry.
- SYMMETRIC / diffuse deficit: myopathy (systemic proximal weakness). Myopathy shows NORMAL variability
  (low step_length_cv) and a NORMAL positional_symmetry_score, which is exactly what distinguishes it
  from the lateralized conditions.
Implication: a signed, side-aware asymmetry axis is a mechanism SHARED by stroke/PD/CP-hemiplegic and
ABSENT in myopathy. This is a hypothesis-driven axis, not a dataset artifact. It also explains why the
project's asymmetry scalar is the weakest-decoded (R^2 0.154): mean/std pooling and laterality handling
may be discarding the signed side information.

## Condition-specific HIGH-priority mechanisms (from neuroscience/*.md, clinically sourced)
- STROKE (hemiparetic, lateralized): knee_asymmetry (>=17 deg flexion diff), ankle_asymmetry,
  walking_speed down (~0.5 m/s vs 1.4), cadence 50-65 steps/min, double_support up (30-40% vs 20%),
  stance_swing_ratio ~2.3 vs 1.6, stride/stance/swing symmetry indices, com_movement + postural sway,
  overall/temporal/movement symmetry scores. Keypoints: knees 25/26, ankles 27/28, hips 23/24, CoM.
- PARKINSON'S (rhythm + variability + lateralized onset): reduced stride_length (shuffling),
  HIGH stride_time_cv / step_length_cv / stride_velocity_cv (variability is the hallmark), postural
  instability (com sway), stride_length_si and per-joint symmetry indices (unilateral onset),
  phase_asymmetry (both feet on ground). Keypoints: bilateral ankles 27/28, hips/knees, CoM.
- CEREBRAL PALSY (crouch + proximal-to-distal chain): knee flexion in stance (crouch, >=30 deg),
  ankle equinus-vs-crouch, reduced hip extension, anterior pelvic_tilt, knee_asymmetry separates
  hemiplegic (unilateral) from diplegic (bilateral), slow walking_speed, short stride. Keypoints:
  knees 25/26, ankles 27/28, hips 23/24, pelvis.
- MYOPATHY (SYMMETRIC proximal weakness): reduced hip extension bilaterally, increased knee ROM
  (compensation), slow walking_speed (via short step, not cadence), LOW step_length_cv (consistent
  mechanical deficit, NOT variable), forward trunk_lean (>10 deg), HIGH positional_symmetry_score
  (i.e. symmetric), postural sway up. Keypoints: hips 23/24, knees 25/26, trunk.

## Data reality that bounds any clinical claim (from _shared_facts.md)
- Source video is the unit: 18 canonical sources (normal 1, PD 2, stroke 3, myopathic 10, CP 2); all
  12 normal clips from ONE video. Condition label is nearly collinear with source identity.
- All current results are transductive. mean/std pooling to 384-d is permutation-invariant (kills
  temporal order). Provenance confound: normal mostly augmented path, abnormal canonical path.
- Only 12 lower-body joints maskable (global cap 0.364). Laterality flip OFF by default.
- Frozen checkpoint fingerprint ea59fea0.

## The "audit -> generalizable principle" levers (from worldmodels/gait CONTEXT + lit review)
A proposal becomes top-venue when it yields a TRANSFERABLE claim, not a one-dataset number. Levers:
1. Matched-substrate target-isolation: hold everything fixed, change ONLY the prediction target
   {raw coords, MAMP one-frame motion, S-JEPA centered-sharpened latent, normalized latent regression}.
   Claim = "what to predict for gait", generalizable.
2. Mask-family comparison at EQUAL target coverage (difficulty controlled): random vs temporal-tube vs
   whole-anatomical-part vs motion-aware vs gait-specific (contralateral, half-cycle/future-phase,
   asymmetry-aware). Claim = which mask geometry preserves gait mechanisms.
3. Multi-scale temporal targets preserving TRUE elapsed time (local velocity, half-cycle coordination,
   full-cycle cadence/symmetry, multi-cycle variability). Directly addresses PD rhythm and stroke slowing.
4. Mechanism-probe readouts with a RAW-INPUT CEILING: only credit the representation if it beats a
   raw-coordinate probe on cadence/speed/stride/variability/asymmetry/sway.
5. Negative controls + representation-HEALTH on the sample axis (per-dim std, effective rank, pairwise
   cosine, input-sensitivity), plus random-encoder and initial-encoder controls.
6. Participant-disjoint confirmation cohort is the biggest audit->principle lever (mark as reach if it
   needs new data).

## World-model flavors already sketched in worldmodels/gait (fuse, do not duplicate)
- Neuro-Concept / concept-bottleneck JEPA: named latent subspaces z_asym, z_rhythm, z_post with
  per-subspace auxiliary heads + VICReg + biased masking (disentanglement).
- Cross-View / viewpoint-conditioned predictor with view_delta, leave-one-view-out, no-flip rule to
  protect lateralized stroke asymmetry (invariance as an "action").
- Hierarchical-Severity world model: fine per-frame + coarse per-cycle, horizon-conditioned FUTURE
  prediction, continuous severity target, prediction-error-tracks-severity.
- Prediction-error anomaly screening: normal-only world model, relative error as anomaly score,
  mechanism-grouped error decomposition, injection-direction test, random-encoder control.
- V-JEPA 2 action-conditioned predictor (arXiv:2506.09985) is the frontier anchor for any
  action/rollout framing.

## What makes the CURRENT 7 read as sub-venue (diagnosis to fix)
Most are internal validity audits of one small cohort (provenance, resize tax, group-loss isolation).
They are honest but do not (a) use the neuroscience mechanism priors above, (b) yield a claim that
generalizes as a method/principle beyond gavd5, or (c) engage the world-model/JEPA frontier. The
iteration must add a neuroscience-grounded hypothesis, a generalizable-principle framing (one of the
levers), and where natural a world-model capability (future prediction, action/view conditioning,
concept bottleneck, predictive-error severity).

## Extra verified citations available for the world-model framing
- Assran et al., V-JEPA 2, 2025, arXiv:2506.09985 (action-free pretrain then action-conditioned predictor).
- Bardes et al., V-JEPA, 2024, arXiv:2404.08471; Bardes et al., MC-JEPA, 2023, arXiv:2307.12698.
- Bardes, Ponce, LeCun, VICReg, ICLR 2022, arXiv:2105.04906; Zbontar et al., Barlow Twins, ICML 2021, arXiv:2103.03230.
- Xu et al., V-usable information, ICLR 2020, arXiv:2002.10689.
- Locatello et al., "Challenging Common Assumptions in the Unsupervised Learning of Disentangled
  Representations", ICML 2019 (disentanglement needs inductive bias / supervision), for concept-bottleneck framing.
- Garrido et al., intuitive physics from V-JEPA (violation-of-expectation), 2025, arXiv:2502.11831.
- Kapoor & Narayanan, leakage taxonomy, 2022, arXiv:2207.07048.
- Plus all citations already in `_shared_facts.md`.

## AUTHORITATIVE NEUROSCIENCE CITATIONS (verified against PubMed/DOI during online research 2026-08-16)
Use these to ground the neurological-source -> mechanism -> skeleton-measurable-feature chain in every proposal.
Do not invent numbers; the flagged items below need full-text confirmation before a precise value is quoted.

STROKE (lateralized upper-motor-neuron / corticospinal lesion -> contralateral hemiparesis):
- Natali/Javed StatPearls corticospinal-tract anatomy (PMID 30571044, 30521239): pyramidal decussation -> contralateral
  control, so a one-hemisphere lesion gives one-sided deficit (mechanistic basis of lateralized asymmetry).
- Li & Francisco 2015 Front Hum Neurosci 9:192 (PMID 25914638); Li, Francisco, Rymer 2021 NNR 35(7):601 (PMID 33978513):
  post-stroke spasticity from disinhibited reticulospinal drive.
- Chen, Patten, Kothari, Zajac 2005 Gait Posture 22(1):51 (PMID 15996592): reduced swing knee flexion (stiff-knee),
  circumduction, hip-hiking, shortened paretic single-support (exact degrees are full-text; cite pattern not number).
- Campanini, Merlo, Damiano 2013 Gait Posture 38(2):165 (PMID 23755883): stiff-knee often from inadequate push-off,
  not rectus-femoris spasticity (needs EMG to attribute cause; skeleton sees the reduced-flexion effect only).
- Bowden et al. 2006 Stroke 37(3):872 (PMID 16456121): paretic propulsion ~16% (high severity) / 36% / 49% of total;
  KINETIC, needs force plates, NOT skeleton-recoverable.
- Patterson et al. 2008 Arch Phys Med Rehabil 89(2):304 (PMID 18226655): ~55% of community stroke survivors have
  temporal asymmetry, ~33% spatial; temporal asymmetry correlates with speed and motor recovery.
- Patterson, Gage, Brooks, Black, McIlroy 2010 Gait Posture 31(2):241 (PMID 19932621): CANONICAL symmetry-index
  methods paper; recommends the Symmetry Ratio on step length, swing time, stance time. THIS is the validated stroke
  biomarker and it is fully skeleton-recoverable.
- Skeleton validity: Stenum et al. 2021 PLoS Comput Biol 17(4):e1008935 (PMID 33891585) temporal MAE 0.02 s/step,
  step-length MAE 0.049 m, sagittal hip/knee/ankle MAE 4.0/5.6/7.4 deg; Wren et al. 2023 Gait Posture 104:9
  (PMID 37285635) markerless RMSD <6 deg. Circumduction/crouch are the least accurate (~11 deg, Horsak 2023 PMID 37738945).

PARKINSON'S (SNpc dopaminergic loss -> basal-ganglia loss of automaticity; asymmetric onset):
- Redgrave et al. 2010 Nat Rev Neurosci 11(11):760 (PMID 20944662): posterior-putamen dopamine loss -> loss of habitual/
  automatic control; Wu, Hallett, Chan 2015 Neurobiol Dis 82:226 (PMID 26102020): loss of automaticity -> reduced arm
  swing, decreased stride length, freezing.
- Riederer & Sian-Hulsmann 2012 J Neural Transm 119(8):953 (PMID 22367437): asymmetric/unilateral onset = contralateral
  nigrostriatal degeneration (why PD also loads a lateralized axis early).
- Morris et al. 1994 Brain 117(5):1169 (PMID 7953597) and 1996 Brain 119(2):551 (PMID 8800948): reduced stride length is
  the fundamental deficit, higher cadence is compensatory, cues normalize stride length (automaticity evidence).
- Hausdorff et al. 1998 Mov Disord 13(3):428 (PMID 9613733): PD gait-timing variability ~2x controls, correlates with
  severity, mean timing similar (variability and mean under DIFFERENT control). Hausdorff 2007 Hum Mov Sci 26(4):555
  (PMID 17618701): variability/fractal dynamics carry meaning (foundational variability-as-biomarker).
- Schaafsma et al. 2003 J Neurol Sci 212(1-2):47 (PMID 12809998): stride-time CV 8.8% fallers vs 4.2% non-fallers
  (p<0.009); levodopa-responsive; independent of cardinal signs. CONCRETE CV anchors.
- Hausdorff et al. 2003 Exp Brain Res 149(2):187 (PMID 12610686): stride-time variability higher in freezers.
- Jordan, Challis, Newell 2007 Gait Posture 26(1):128 (PMID 16982195): variability is SPEED-dependent (interpret CV
  relative to speed). MDS-UPDRS: Goetz et al. 2008 Mov Disord 23(15):2129 (PMID 19025984).
- Skeleton validity for variability: Stenum 2021 (PMID 33891585) temporal MAE 0.02 s/step supports stride-time CV recovery.

CEREBRAL PALSY (non-progressive developing-brain injury / PVL / corticospinal -> spasticity):
- Rosenbaum et al. 2007 Dev Med Child Neurol Suppl 109:8 (PMID 17370477): CP consensus definition; ~80-85% spastic.
- Volpe 2009 Lancet Neurol 8(1):110 (PMID 19081519); Back et al. 2007 Stroke 38(2 Suppl):724 (PMID 17261726): PVL,
  periventricular leg-corticospinal fibers -> legs>arms; bilateral PVL -> diplegia, unilateral lesion -> hemiplegia.
  (This is the within-CP unilateral-vs-bilateral distinction = the lateralized-vs-symmetric axis inside one condition.)
- Lance spasticity definition via Trompetto et al. 2014 BioMed Res Int 2014:354906 (PMID 25530960).
- Crouch cutoff: de Morais Filho et al. 2010 J Pediatr Orthop B 19(3):226 (PMID 20300011) min stance knee flexion >=30 deg.
- Rodda & Graham 2001 Eur J Neurol 8(Suppl5):98 (PMID 11851738); Rodda et al. 2004 JBJS Br 86(2):251 (PMID 15046442):
  sagittal diplegia patterns (true equinus/jump/apparent equinus/crouch/asymmetric).
- Winters, Gage, Hicks 1987 JBJS Am 69(3):437 (PMID 3818706): hemiplegia Types I-IV (increasingly proximal).
- Sutherland & Davids 1993 Clin Orthop Relat Res 288:139 (PMID 8458127): kinematic-chain coupling; Gait Deviation Index
  Schwartz & Rozumalski 2008 Gait Posture 28(3):351 (PMID 18565753). Anterior pelvic tilt de Morais Filho 2018 Gait
  Posture 63:165 (PMID 29753172).
- Skeleton validity in CP: Costa 2026 Gait Posture (PMID 42107175) sagittal RMSD <6 deg but transverse >10 deg; Poomulna
  2025 (PMID 39490268) GDI 6.9 pts lower markerless. Sagittal crouch/equinus recoverable; transverse rotation/kinetics NOT.
  ML on CP kinematics: Zhang 2019 Comput Biol Med 106:33 (PMID 30665140, ANN 93.5%); Lan 2024 (PMID 38350399, LSTM 0.77-0.99).

MYOPATHY (primary muscle disease -> symmetric proximal weakness; NOT central):
- Nagy & Veerapaneni StatPearls Myopathy (NBK562290): primary muscle disorder, proximal (pelvic>shoulder) weakness,
  no sensory loss; EMG most sensitive. Barohn et al. 2014 Neurol Clin 32(3) (PMID 25037080): symmetric proximal
  (limb-girdle) is THE characteristic distribution; symmetry is the discriminator vs lateralized UMN.
- Vandekerckhove et al. 2022 Front Hum Neurosci 16:861136 (PMID 35721358): DMD vs typically-developing (Table 2):
  anterior pelvic tilt 16.4 deg [14.3-19.0] vs 11.6 deg [8.7-15.3] (p=0.005); reduced velocity; wider step width
  (0.28 vs 0.16); reduced hip extension; PRESERVED cadence (2.25 vs 2.21 steps/s, NS). CONCRETE anchors.
- Vandekerckhove et al. 2025 J Neuroeng Rehabil 22 (PMID 41034979): hip-extensor weakness -> anterior pelvic tilt
  (rb=-0.812) and posterior trunk lean; hip-abductor weakness -> wider step width (waddling mechanism).
- Xiong et al. 2023 Biomed Eng Online 22 (PMID 37525241): DMD spatiotemporal parameters show NO significant left-right
  asymmetry vs controls (only higher-order synergy asymmetry differs). KEY: at the skeleton level myopathy reads as
  SYMMETRIC, the opposite of hemiplegic stroke/CP. "Low stride-time variability" is argued mechanistically (no rhythm-
  generator lesion) + preserved cadence, NOT from a verified DMD CV number (flag before quoting a CV value).
- Baptista et al. 2014 Braz J Phys Ther 18(2) (PMID 24838810): DMD pelvic anteversion ~2x controls. Bushby et al. 2010
  Lancet Neurol 9(1):77 (PMID 19945913) DMD care standard. EMG-only coactivation: Ropars et al. 2016 PLoS One 11(9)
  e0161938 (PMID 27622734) (NOT skeleton-recoverable).

THE DISCRIMINATIVE AXIS, now literature-grounded (put this in the portfolio index):
- Lateralized deficit (skeleton shows left-right asymmetry): stroke (corticospinal decussation), hemiplegic CP
  (unilateral PVL/lesion), PD early (contralateral nigrostriatal). Validated biomarker = symmetry ratio (Patterson 2010).
- Rhythm/variability deficit (skeleton shows high stride-time CV): PD (basal-ganglia loss of automaticity). Validated
  biomarker = stride-time CV (Hausdorff; Schaafsma 8.8 vs 4.2%).
- Symmetric, rhythm-preserved, posturally abnormal (skeleton shows anterior pelvic tilt/trunk lean, LOW asymmetry,
  preserved cadence): myopathy (primary muscle). Discriminator = symmetry + low variability (Xiong 2023; Vandekerckhove).
- What skeletons CANNOT recover (be honest in limitations): kinetics/propulsion (Bowden force plates), EMG/spasticity/
  coactivation (Ropars), transverse-plane rotation (Costa/Wishaupt), etiologic muscle diagnosis (biopsy/CK/genetics).
