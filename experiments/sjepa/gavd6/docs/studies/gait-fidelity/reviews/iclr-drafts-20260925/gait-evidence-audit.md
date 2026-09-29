# Gait fidelity: independent evidence ledger, 2026-09-25

## Strongest defensible claim

A controlled synthetic movement-measurement benchmark reveals that optimizing a scalar change measure can worsen the reconstructed knee-angle trajectory, and that the tested JEPA residual-coupling intervention has not yet established improved movement-response recovery. The contribution presently supported is an interpretable evaluation framework and a controlled negative/inconclusive empirical finding—not a demonstrated superior JEPA method, validated world model, clinical measurement system, or fall-risk predictor.

Suggested paper thesis: **Preserving a movement summary and preserving its underlying trajectory are distinct learning objectives; in this controlled study, neither paired latent supervision nor a denser readout objective has yet shown a clear improvement on its primary comparison.** The shared benchmark exposes failure penalties, readout-weight sensitivity, and a strong zero-response reference that would be obscured by pose-error reporting alone.

## Dataset and actual completed experiment

- Reused AMASS motion-capture/body-model data, rendered synthetic RGB, fresh off-the-shelf pose extraction. No newly recruited participants; no GAVD/clinical evaluation in these outputs.
- Actual training: **112 people; 692 raw-motion hashes; 1,645 source windows; 315,840 condition-expanded observation rows**. `outputs/iclr/walking-core/ledger.json:1224` and its `sampling_coverage` fields. Every fit sees all112people; 2,000-update phases expose approximately9.2% of available endpoint rows, with95.99–96.05% source-window coverage. Thus this is a deliberately limited training regime, not exhaustive training to convergence.
- Development: **14 people; 155 source windows**. `outputs/iclr/readout-repair/development/evaluation/summary.json:22`; twelve people from BioMotionLab_NTroje, two from KIT. This common retained evaluation set is reused in all three stages; do not count successive stages as independent replications.
- The planned cohort is larger: `outputs/iclr/walking-core/cohort/summary.json` gives113train/15dev/14confirmation people and2171/214/199windows before preparation/exclusions. Do not report planned counts as trained counts.
- Each window: **128frames at25Hz, about5.1s; 12body joints; (x,y), estimator confidence, observed/missing indicator, timestamps**. Fixed body12 contains shoulders/elbows/wrists/hips/knees/ankles. Ground-truth projected trajectories retained separately. Model inputs never include reference validity, intervention magnitude, side, or pair identity. `synthetic_training_v2/models.py:16,93–123`; `gait_fidelity/training.py:288`.
- Conditions: original/mirrored physical motion; 45°oblique/90°side camera; clear/15%occlusion; correct/global-swapped/temporary-swapped joint names; local knee edits0/5/10/15°nominal; 15°held from training; RTMPose-M,HRNet-W32,ViTPose-Base; ViTPose held from training. `walking-core/config.json:data/preparation`; preparation code:135,464–538. Original baseline and exact no-change copy both present; nominal3Dedit level is not equal to observed2Dresponse.
- Raw training315840=1645×192condition rows. Development55800=155×360rows per fit (including baseline and no-change duplicate); primary movement contrasts33480 per fit (nonzero intervention levels), not33480independentpeople. `jepa-response/evaluation/coverage.csv`.
- Geometry/admission checks are technical, not human gait/clinical validation. Person reservations existed; new confirmation exposure audit is **ready:false**,14people review_required (`readout-repair/exposure-audit.json`). No independent confirmation is complete.

## Model/learning facts

- Small local **offline** Transformer restoration model, not pretrained V-JEPA2 and not an action-conditioned/future-rollout world model. Whole128framewindow is available. 4encoderlayers,4attentionheads,width96;4frames/joint/token=384tokens. Same nonlinear coordinate readout (10280trainableparameters), predicting corrections at observed joints and absolute positions at missing ones. `synthetic_training_v2/models.py:93,126,202`; configs.
- JEPA = Joint Embedding Predictive Architecture: student maps noisy masked estimated poses to feature vectors; predictor targets features computed by an exponential-moving-average teacher from corresponding clean projected poses. Training base is centered teacher-student crossentropy plus variance/invariance/covariance regularization (VICReg) on translated views. Teacher receives privileged synthetic references, so do not call this pure self-supervision from unlabeled real video. Explain as reference-paired representation learning.
- Graph/time mask: nominal50% available joint/4-frame tokens hidden during pretraining; natural missing observations are also output queries. Readout and inference do not add this mask. Independent endpoint masks; support for the auxiliary requires both querytokens and all referenceframes valid in both endpoints. `response_calibration.py:73–118`; `response_objectives.py:19`.
- Normalize each input window using observed context only: median origin and isotropic P95–P5 coordinate-span norm, excluding artificial mask values before estimating normalization. Reference coordinates transformed with same input-derived origin/scale. `training.py:288`.
- Core:5model families×2objectives×3seeds=30fittedmodels;9pretrainingphases. Families directend-to-end; frozeninitialized; coordinatepretrained; referencepairedJEPA; shuffled-referenceJEPA. Sixnonneuralcontrols separately, includingunchangedandfilter0identical. Core does not include every implementedfullrecipe (e.g.SmoothNet-style does not appear in core table). `walking-core/plan.json`; `report.md:9`.
- Followup:3representationobjectives×2readouts×3seeds=18newmodels,9pretrainingphases. NewobjectivesJEPA residualdifference, matchedendpoint residual, coordinatedifference. Retains core controls withoutretraining. `jepa-response/plan.json`;report.
- Let e_s=H(p_s/τ_s−stopgrad((t_s−c)/τ_t)), H subtracts meanfeaturechannel. Auxiliary delta loss=||e1−e0||²/(2D); matchedendpoint loss=(||e0||²+||e1||²)/(2D). Their difference=−e0·e1/D, isolatingcross-state residualcoupling conditional onsharedbase,support,batches,masks,coefficient. Bothλ=0.012646811 from32seed17trainingbatches, setlargestinitialauxgradient to10%base gradient. CoordinateΔλ3.793892. `response_objectives.py:10–16,38–60`; `diagnostics/loss-calibration.json:61`.
- Training:AdamW? Confirm optimizer code if spellingout. LR0.0003,batch16endpoints(8pairs),2000pretrain+2000frozenreadout,4000directupdates;seeds17,29,43,float32. Equalpeople→motions→windows→pairedconditiondraws. Directvsfrozencomparisonispracticalbenchmark, notcleanattributionofpretraining;directoptimizesencoder4000updateswhilefrozenencoderonly2000pretrainupdates.
- Repair:2frozenJEPArepresentations×2readoutobjectives×3seeds=12newreadouts,0newpretraining;15retainedcomparatorfits=27evalfits/9methods. Scalar_low coefficient0.1 vsoriginal1.0. Dense objective matches framewiseknee-angle changes atsamephysical timestamps acrossstates; it is not velocityloss. Densecoefficientcalibratedperrepresentation/seedtomatchinitialscalargradientmagnitude; geometrypenaltyheld1. `repair_objectives.py:13–36`; `repair_training.py:295`;ledgercalibrations.

## Measurement definitions and validity

- θ_L(t),θ_R(t):unsigned interior2Dangleat hip–knee–ankle, degrees, inimageplane. For eachlegE=P95(θ)−P5(θ). A=E_R−E_L. e.g.right50°,left40°givesA=+10°;bothlegs+10°leavesAunchanged, motivatingwaveform/leg-specificchecks. `measurements.py:14,60`.
- MovementresponseΔA=A(state1)−A(state0);error=|predictedΔA−referenceΔA|, lowerbetter. Measuresmagnitude/signofchangeinright-left excursiondifference, not isolatedone-legamplitude. Lateralidentity stress includesphysicalmirroringandjointlabelswapconditions.
- Directionaccuracy restrictedtoreference|ΔA|>1°; missingpredictioncountsincorrect. Near50%cannotbeunqualifiedlycalledchancewithoutcheckingclassbalance. `jepa-response/report.md`.
- Waveformerror:meankneeangleabsoluteerroracrossvalidtimestampsandbothlegs, lowerbetter; usesactualanglecurve not justexcursion. Sourceimplementation `evaluation.py`.
- CoordinateNLE:normalizedlandmarkerror, averagedpointdistance dividedreferenceboundingboxdiagonal; dimensionless. Do notconvert todegrees orclaimclinicalmmaccuracy.
- Referenceeligibilityfixedbeforelookingatpredictions, commonacrosssourcefamilyconditions. Min2pxlimbsegments;≥16frames and≥80%ofwindow referencecoverage. Failureonanyeligibleframereceives360°Aerror,720°response/nuisancecontrast,180°waveform. These are protocol worstcasebounds, not measuredphysicalangles.
- Means:conditionswithinwindow→windowswithinrawmotion→motionswithinperson→equalperson/seedweight. Allplotsmustnotpseudoreplicateframes/conditions asindependentpeople.
- Core/followup95%crossedbootstrap independentlyresamplespairedpeopleandseeds,2000draws; only3seeds—descriptive developmentintervals. Repairprimaryusespairedper-personmeansaveragedoverthreefixedseeds withStudent-tinterval14people; crossedbootstrap secondary. Do notmix CIdefinitionswithoutlabeling.

## Principal quantitative results

Allanglesdegrees, lowererrorbetter. Pooledacrossthreeextractors unlessstated.

| Method | Response error | Waveform error | Direction accuracy |
|---|---:|---:|---:|
| Direct coordinate-only |7.544155|12.073284|0.660911|
| Core paired JEPA +scalarreadout |10.066234|22.553313|0.489394|
| Followup delta JEPA +scalarreadout |9.988035|21.764814|0.484416|
| Followup endpoint JEPA +scalarreadout |10.361088|22.031103|0.478449|
| Followup delta JEPA +coordinate-onlyreadout |10.944775|17.376973|0.493139|
| Followup endpoint JEPA +coordinate-onlyreadout |10.232657|16.898832|0.475292|
| Zero-change prediction diagnostic |5.810830|unavailable|unavailable|

Source:`jepa-response/evaluation/per-person.csv`,42rowspermethodequalmean;reportsround4dp. **ALL16learnedvariantsresponseerror>zero-change5.81083°**. This diagnostic alwayspredictsΔA=0; itdoesnotreconstructposes butsetsaminimumbarforshowingreliablechangerecovery. Directsuccessconditionalresponse~5.09°is belowzero,buttotalfailure-inclusive7.54isnot. Deltamodelsuccessfulcontribution6.2096itselfalreadyexceedszero.

1. Core declaredcandidateJEPA+scalar vsdirect+scalar:responseimprovement0.688084°,crossed95%CI[−0.640212,1.976709]. Nuisance+3.866182°[2.117165,5.538073],waveform−3.272474°[−4.699395,−1.944364]. `walking-core/evaluation/comparisons.json:2`.
2. Followupprimarydelta vsendpoint withscalarreadout:responseimprovement0.373054°,crossed95%CI[−1.110272,1.760337]. Nuisance1.910905°[0.292120,3.467633]secondary; waveform0.266289°[−1.238896,2.627342]. `jepa-response/evaluation/comparisons.json:2`.
3. Deltaresponsegain0.373054=0.276676reducedfailurepenalty+0.096378successful-outputcontribution. Totaldelta9.988035=3.778439failure+6.209596successful;endpoint10.361088=4.055115+6.305974. Rawfailurecount550/100440delta reference-eligiblepairsacross3seeds;rawfrequencydiffersfromhierarchicalaverage. Do notreframeallmeanimprovement asbetterkinematics.
4. **Addingoriginalscalar paired-changeobjective worsens waveformerrorin ALL8matchedmodel/representationfamilies** (core5+followup3), descriptively. Delta17.376973→21.764814 (+4.387841°;pairedcomparisonimprovement−4.387841,crossedCI[−5.922412,−2.764010]);direct12.073284→19.280839;endpoint16.898832→22.031103. Source`jepa-response/evaluation/readout-control-comparisons.json` andper-personmeans. This isstrongobjective-tradeoffevidenceinthisregime, notuniversaltheoremthatmeasurementlossesfail.
5. RepairPRIMARY isViTPoseonly, notpooled:delta dense waveform19.194663 vsscalar-low19.473519;improvement0.278856°,person-t95%CI[−0.194914,0.752625]. Response12.978403vs13.107513;improvement0.129110°[−2.686467,2.944686]. `readout-repair/development/evaluation/comparisons.json:primary`; meansfrom`per-person-by-extractor.csv`,filtervitpose_base.
6. Delta dense vsoriginalscalarViTPosewaveformimprovement3.976253°[2.762050,5.190456]secondary; LOW-WEIGHTSCALARaloneimproves3.697397°[2.584569,4.810225]. Denseadditionbeyondlowweightisthusunresolvedprimary. Densevsoriginalscalarresponse worsens1.050455°[−4.466639,2.365730]inimprovementdirection;absenceclearharmdoesnotshownoninferiority.
7. Repairprimaryresponseimprovement0.129110=0.127976failurecontribution+0.001133successcontribution. Waveformimprovement0.278856=0.119692failure+0.159164success. Densevsbasewaveform−0.068963°[−0.473908,0.335981], anddensevsdirect−6.172813°[−6.864923,−5.480703]secondaryViTPose.
8. Endpointdense vsscalar-lowwaveform+0.718163°[0.387453,1.048872]secondaryunadjusted; do notpromotethis toprimaryafterseeingresults. Itsresponse−0.103616°[−2.062516,1.855285]unresolved.

## Mechanism and claims to avoid

- Densechanges temporal supervision AND gradient support. Evenafavorablecomparisonwouldnotisolate sparsequantilegradientsasuniquecause. Initializationgradientmatchdoesnotmatchoptimizerupdatesovertraining. Scalarweightreductionisa majoralternativeexplanation forrepairrelative tooriginalscalar.
- BothJEPAvariantsreceiveprivilegedcleanreferencefeatures andpairedmovementwindows. Noevidence oflearningbiomechanicaldynamics orcausalworldmodel. Controlledkinematiceditsnotdynamicallyvalidated, notpathology/simulatedMSorPD.
- Featurediagnosticsglobalflag `positive_training_person_probe` ispotentiallymisleading: positiveprobecomesfromteacher receivingcleanreferences. ALL21deployment-encoder ridgeprobesareworsethanzerobenchmark. Exampledelta seed17:MSE48.136vszero20.822;teacherMSE8.001. Diagnosticsaretraining-personfeasibilityonly, notdevelopmentgeneralization,andfailureofonefixedlinearprobe doesnotproveunrecoverability. Do notselect/stopmodelsusingtheseprobes.
- Small14persondevelopmentsetdominatedby12BioMotionLabpeople; repeatedtreadmillmotions andsyntheticcamera/occlusionconditionsarenotnaturalexternaldomain diversity. NoGAVDresult,clinicalvalidation,3Danatomicalmeasurementaccuracy,fallriskoutcomes,realhomecameraoptimization,orindependentconfirmation.
- Camera placement relevance isa potentialfutureapplication ofpairedobservationevaluation,notdemonstratedoptimalcameraresults.
- `outputs/iclr`containsreport/config/ledger/aggregateCSVdata anddiagnostics, nooriginalposearrays/rawreconstructions. These live atHAICpathsrecordedinthereceipts. If actualdatavisualizationsareunavailable, explicitlylabelschematics andstateartifactgap rather thanpresentingsyntheticillustrationasexperimentalsample.
- Currentcodehashesmatchfrozenresponseobjectives,models,measurementfunctions;coretraining/preparationhaveevolvedsincecorefreeze. Primaryresultsaregroundedinfrozenledger/output;avoidclaimingallcurrentimplementationdetailsareverifiedhistorical withoutreceiptcorroboration.

## Narrative recommendation

Leadwithmeasurementtask andconcretelegchangeexample→pairedsyntheticreference benchmark→whyplaincoordinateaccuracyinsufficient→three-stagecontrolledinvestigation(core tradeoff; latentcoupling; readoutrepair)→primaryuncertaintyandzerochangebenchmark→scientificlessonandboundednextsteps. Fullwriteupcanincludecompactappendixreproducibilitytable. Shortversionshouldspendoneminimaldiagramorrealexample,oneresultstable/contrastfigure, notmultiplearchitecturefigures crowdedintotwopages.
