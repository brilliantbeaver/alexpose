"""Full-data notebook cells. Shared split and training logic lives in sjepa/."""


def split_cells(md, code):
    return [md(
        "## Keep each source video together\n\n"
        "We use the same frozen five-fold registry in notebooks 02–06. A source video may have "
        "several clips; each clip may yield overlapping windows. All of those relatives stay "
        "together. Each round uses about 60% of sources for training, 20% for validation, and 20% "
        "for testing. Only notebook 06 evaluates test clips.\n\n"
        "The splitter runs on one row per source, with condition labels used to balance source "
        "counts. It never splits windows. The loader checks the full cache, reviewed exclusions, "
        "and registry checksum. A changed cache requires a new registry and new checkpoints. "
        "See [the full method](docs/11-full-data-splits.md)."
    ), code(
        "from IPython.display import display",
        "import pandas as pd",
        "from sjepa.splits import load_full_registry, partition_records, split_summary",
        "records, registry = load_full_registry(EXP_DIR)",
        "FOLD = 0  # teaching example; notebook 06 independently trains all five folds",
        "train_recs, val_recs, test_recs = partition_records(records, registry, FOLD)",
        "display(pd.DataFrame(split_summary(records, registry)))",
        "print('usable clips:', len(records), '| excluded raw clips:', len(registry['inventory']['exclusions']))",
        "print('registry:', registry['registry_sha256'])",
    )]


def nb_03(md, code, badge, boot):
    c = [badge("03_sjepa_model_and_pretrain_normal.ipynb"), md(
        "# 03 - Build S-JEPA and pretrain on training sources\n\n"
        "S-JEPA predicts hidden skeleton features. Its view encoder reads visible joints, its "
        "predictor receives joint and time positions, and its target encoder supplies targets. "
        "The target encoder is a slow moving average of the view encoder.\n\n"
        "The filename is historical: we train on all three conditions in the training partition, "
        "with no condition labels in the loss. Unlabeled test motion would still leak information, "
        "so neither validation nor test windows enter training. Run notebook 01 first."
    )]
    c += boot(need_torch=True)
    c += split_cells(md, code)
    c += [code(
        "from IPython.display import SVG, display",
        "display(SVG(filename=str(IMAGES_DIR / 'sjepa_two_lane.svg')))",
        "from sjepa.config import get_config, describe",
        "from sjepa.models import build_model, pick_device",
        "from sjepa.splits import fold_run_dir",
        "cfg = get_config(); device = pick_device()",
        "RUN_DIR = fold_run_dir(EXP_DIR, registry, cfg, FOLD)",
        "model = build_model(cfg, device=device, repaired=True)",
        "print(describe(cfg)); print('device:', device)",
        "print('checkpoint directory:', RUN_DIR)",
    ), md(
        "## Train only after splitting\n\n"
        "The training helper checks the clip and source lists before making windows. It samples "
        "source videos uniformly, so sources with more windows do not dominate the updates. "
        "The usual budget is 800 updates; smoke mode uses 4 to check execution only. "
        "Stochastic masks let every joint appear as context and as a target."
    ), code(
        "from sjepa.full_experiment import train_checkpoint",
        "SMOKE = cfg.profile.endswith('smoke')",
        "UPDATES = 4 if SMOKE else 800",
        "state = train_checkpoint(model, train_recs, cfg, registry, FOLD, 'ssl',",
        "                         UPDATES, device, RUN_DIR / 'ssl.pt')",
        "print('training clips:', len(train_recs), '| training sources:', len({r.source_id for r in train_recs}))",
        "print('final effective rank:', state.eff_rank[-1])",
    ), md(
        "## Inspect training diagnostics\n\n"
        "Loss alone cannot show that useful features were learned. Effective rank measures how "
        "many directions the features use; teacher drift measures how far teacher weights moved "
        "from student weights. These are training checks, not test accuracy or proof of gait learning."
    ), code(
        "import matplotlib.pyplot as plt",
        "fig, ax = plt.subplots(1, 3, figsize=(12, 3))",
        "for a, values, title in zip(ax, [state.losses, state.eff_rank, state.teacher_drift],",
        "                            ['training loss', 'effective rank', 'teacher drift']):",
        "    a.plot(values); a.set_title(title); a.set_xlabel('update')",
        "plt.tight_layout(); plt.show()",
        "print('Saved a checkpoint tied to this cache, registry, fold, and configuration.')",
    )]
    from notebook_03_tutorial import add_tutorial
    return add_tutorial(c, md, code)


def nb_04(md, code, badge, boot):
    c = [badge("04_progressive_finetune_ms_pd_vicreg.ipynb"), md(
        "# 04 - Compare training budgets using validation sources\n\n"
        "We compare the original label-free checkpoint with additional label-free training. "
        "For each checkpoint, a frozen encoder produces features and a supervised linear head "
        "learns the condition labels. The head and its scaler fit training clips only. "
        "Validation sources choose the training budget; outer test sources stay untouched.\n\n"
        "The filename is historical. This notebook does not use class-aware VICReg. Using labels "
        "in a training loss would make that stage supervised; it would not by itself be test leakage."
    )]
    c += boot(need_torch=True)
    c += split_cells(md, code)
    c += [code(
        "from sjepa.config import get_config",
        "from sjepa.models import build_model, pick_device",
        "from sjepa.splits import fold_run_dir, load_partition_checkpoint",
        "from sjepa.full_experiment import train_checkpoint, embed_records, fit_probe, score_records",
        "cfg = get_config(); device = pick_device()",
        "RUN_DIR = fold_run_dir(EXP_DIR, registry, cfg, FOLD)",
        "model = build_model(cfg, device=device, repaired=True)",
        "load_partition_checkpoint(RUN_DIR / 'ssl.pt', model, cfg, registry, FOLD, 'ssl', device)",
    ), md(
        "## Continue learning from the same training sources\n\n"
        "The usual extra budget is 400 updates; smoke mode uses 2. We keep the learned model and "
        "teacher weights but start a fresh optimizer, schedule, and centering state. This is an "
        "additional training stage, not an exact resume of the earlier optimizer."
    ), code(
        "MORE = 2 if cfg.profile.endswith('smoke') else 400",
        "state = train_checkpoint(model, train_recs, cfg, registry, FOLD, 'continued',",
        "                         MORE, device, RUN_DIR / 'continued.pt')",
    ), md(
        "## Fit on training clips; compare on validation clips\n\n"
        "Each clip gets one vector by averaging its windows and a fixed, seeded token readout. "
        "The head is logistic regression with C=1 and balanced class weights. For selection, "
        "each validation source has total weight one, divided among its clips. We choose the "
        "higher macro-F1; a tie keeps the original checkpoint. Nothing is refit on validation."
    ), code(
        "validation_scores = {}",
        "for stage in ['ssl', 'continued']:",
        "    m = build_model(cfg, device=device, repaired=True)",
        "    load_partition_checkpoint(RUN_DIR / f'{stage}.pt', m, cfg, registry, FOLD, stage, device)",
        "    probe = fit_probe(embed_records(m, train_recs, cfg, device), train_recs)",
        "    pred = probe.predict(embed_records(m, val_recs, cfg, device))",
        "    validation_scores[stage] = score_records(val_recs, pred, equal_source=True).macro_f1",
        "selected = 'continued' if validation_scores['continued'] > validation_scores['ssl'] else 'ssl'",
        "print('Validation source-weighted macro-F1:', validation_scores)",
        "print('Selected stage:', selected, '| outer test clips have not been evaluated')",
    ), md(
        "## What the comparison means\n\n"
        "This is one small validation example. Its scores help choose between two budgets and "
        "are not final performance estimates. Notebook 06 repeats the complete train-and-select "
        "procedure from fresh model weights inside each outer fold. Do not change the procedure "
        "after seeing its test results and then present those same results as an untouched test."
    )]
    return c


def nb_05(md, code, badge, boot):
    c = [badge("05_representation_visualization.ipynb"), md(
        "# 05 - Inspect training representations\n\n"
        "These pictures use only the teaching fold's training clips. A plot can suggest a pattern, "
        "but it cannot measure performance on unseen sources. Camera conditions and pose-detector "
        "confidence may explain a cluster, so we compare learned features with visibility features. "
        "Validation and test clips are excluded from embeddings, projection fitting, and silhouette scores."
    )]
    c += boot(need_torch=True)
    c += split_cells(md, code)
    c += [code(
        "import numpy as np",
        "from sjepa.config import get_config",
        "from sjepa.models import build_model, pick_device",
        "from sjepa.splits import fold_run_dir, load_partition_checkpoint",
        "from sjepa.full_experiment import embed_records, nuisance_features",
        "cfg = get_config(); device = pick_device()",
        "RUN_DIR = fold_run_dir(EXP_DIR, registry, cfg, FOLD)",
        "representations = {}",
        "for stage in ['ssl', 'continued']:",
        "    model = build_model(cfg, device=device, repaired=True)",
        "    load_partition_checkpoint(RUN_DIR / f'{stage}.pt', model, cfg, registry, FOLD, stage, device)",
        "    representations[stage] = embed_records(model, train_recs, cfg, device)",
        "representations['visibility'] = nuisance_features(train_recs)",
        "y = [r.label for r in train_recs]",
        "np.savez(RUN_DIR / 'training_embeddings.npz', **representations, labels=np.array(y),",
        "         clips=np.array([r.clip_name for r in train_recs]),",
        "         sources=np.array([r.source_id for r in train_recs]),",
        "         registry_sha256=registry['registry_sha256'], partition='train')",
        "print('training clips plotted:', len(y))",
    ), md(
        "## Project training points to two dimensions\n\n"
        "t-SNE and UMAP can distort distances and create apparent clusters. We standardize each "
        "feature using these training clips, then fit the projection on the same training clips. "
        "Colors show dataset labels. Multiple points from one source are related observations."
    ), code(
        "from sklearn.manifold import TSNE",
        "from sklearn.preprocessing import StandardScaler",
        "from sjepa.viz import scatter_2d",
        "import matplotlib.pyplot as plt",
        "scaled = {name: StandardScaler().fit_transform(E) for name, E in representations.items()}",
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))",
        "for ax, (name, E) in zip(axes, scaled.items()):",
        "    xy = TSNE(n_components=2, perplexity=min(15, len(E)-1), random_state=42, init='pca').fit_transform(E)",
        "    scatter_2d(xy, y, ax, f'training only: {name}')",
        "plt.tight_layout(); plt.savefig(RUN_DIR / 'training_tsne.png', dpi=130); plt.show()",
    ), code(
        "try:",
        "    import umap",
        "except ImportError:",
        "    print('Install umap-learn to enable the optional UMAP view.')",
        "else:",
        "    fig, axes = plt.subplots(1, 3, figsize=(15, 4))",
        "    for ax, (name, E) in zip(axes, scaled.items()):",
        "        xy = umap.UMAP(n_neighbors=min(15, len(E)-1), min_dist=0.3, random_state=42).fit_transform(E)",
        "        scatter_2d(xy, y, ax, f'training UMAP: {name}')",
        "    plt.tight_layout(); plt.show()",
    ), code(
        "from sjepa.eval import silhouette",
        "for name, E in scaled.items():",
        "    print(f'{name}: training-only silhouette = {silhouette(E, y):.3f}')",
    ), md(
        "A training silhouette is descriptive. It is not an out-of-fold score or proof that the "
        "encoder learned a clinical feature. Keep the evaluation choices fixed before notebook 06. "
        "Changing them after exploring this collection makes the study exploratory."
    )]
    return c


def nb_06(md, code, badge, boot):
    c = [badge("06_capstone_rf_vs_sjepa.ipynb"), md(
        "# 06 - Compare models with five source-grouped test folds\n\n"
        "This notebook runs the full-data experiment from fresh model weights in each fold. "
        "Every usable clip receives exactly one out-of-fold test prediction from a model that "
        "did not train or select its settings on that source. The older g1 results belong to a "
        "different dataset and are not loaded here.\n\n"
        "We compare Random Forest, a validation-selected S-JEPA linear probe, two fixed linear "
        "controls (visibility and mean pose), and the most common training label. All systems "
        "use identical training and test clips."
    )]
    c += boot(need_torch=True)
    c += split_cells(md, code)
    c += [md(
        "## Fix the procedure before testing\n\n"
        "For each outer fold: train S-JEPA for 800 updates, fit a training-only probe, and score "
        "validation. Continue for 400 updates and repeat. Choose by validation source-weighted "
        "macro-F1, with ties going to the original stage. Then evaluate the chosen model on test "
        "clips. We do not refit on validation. RF uses 100 trees, depth 5, and balanced classes. "
        "Both controls use the same fixed linear-head settings as S-JEPA.\n\n"
        "The inner validation set is the first split of a four-fold splitter within the outer "
        "development sources. We use one inner holdout, not all four inner folds. Smoke mode "
        "still checks all five outer folds, but uses 4+2 updates and a tiny model; its scores "
        "are execution checks. A normal run can take substantially longer."
    ), code(
        "from sjepa.config import get_config",
        "from sjepa.models import pick_device",
        "from sjepa.full_experiment import run_cross_validation, new_evaluation_dir",
        "cfg = get_config(); device = pick_device()",
        "SMOKE = cfg.profile.endswith('smoke')",
        "UPDATES, MORE = (4, 2) if SMOKE else (800, 400)",
        "OUTPUT_DIR = new_evaluation_dir(EXP_DIR, registry, cfg)",
        "print('output:', OUTPUT_DIR, '| smoke execution check:', SMOKE)",
        "results = run_cross_validation(records, registry, cfg, device, UPDATES, MORE, OUTPUT_DIR)",
    ), md(
        "## Read both ways of weighting the test predictions\n\n"
        "Macro-F1 gives the three conditions equal importance. Clip-weighted scoring gives each "
        "clip one vote. Source-weighted scoring gives each source a total weight of one, shared "
        "among its clips. This prevents a source with 13 clips from counting 13 times as much as "
        "a source with one clip. It still scores clip predictions; it is not a person-level diagnosis.\n\n"
        "The pooled score uses all out-of-fold predictions. The fold mean and standard deviation "
        "describe variation across rounds; they are not a confidence interval because training "
        "sets overlap. Do not choose a control or a model using this final table."
    ), code(
        "rows = []",
        "for name, scores in results['metrics'].items():",
        "    variation = scores['source_weighted_fold_mean_std']['macro_f1']",
        "    rows.append({'system': name, 'pooled source macro-F1': scores['source_weighted']['macro_f1'],",
        "                 'pooled clip macro-F1': scores['clip_weighted']['macro_f1'],",
        "                 'fold source macro-F1 mean': variation['mean'], 'fold SD': variation['std']})",
        "display(pd.DataFrame(rows))",
        "print('Saved one test prediction per clip to', OUTPUT_DIR / 'oof.json')",
        "if SMOKE: print('SMOKE CHECK ONLY: these scores do not establish model quality.')",
    ), code(
        "import matplotlib.pyplot as plt, seaborn as sns",
        "from sjepa.splits import LABELS",
        "fig, axes = plt.subplots(1, 2, figsize=(10, 4))",
        "for ax, name in zip(axes, ['rf', 'sjepa']):",
        "    cm = results['metrics'][name]['source_weighted']['confusion']",
        "    sns.heatmap(cm, annot=True, fmt='.1f', cmap='Blues', cbar=False,",
        "                xticklabels=LABELS, yticklabels=LABELS, ax=ax)",
        "    ax.set_title(f'{name}: source-weighted OOF'); ax.set_xlabel('predicted'); ax.set_ylabel('true')",
        "plt.tight_layout(); plt.show()",
    ), md(
        "## Limits of this test\n\n"
        "A source ID identifies a recording, not a verified participant. The same person or "
        "reposted footage may occur under different IDs. Acquisition conditions may also track "
        "the labels. Source grouping prevents known within-video overlap but does not remove "
        "those problems. There is no separate external test cohort. These are development "
        "estimates on a small, previously inspected collection.\n\n"
        "For the audit, exact counts, exclusions, commands, and statistical references, read "
        "[docs/11-full-data-splits.md](docs/11-full-data-splits.md)."
    )]
    return c
