"""Motion-sensitive frozen summaries and source-separated readout diagnostics.

Temporal statistics are in feature units per prepared token. They do not
recover the original recording clock or turn this completion task into a forecast.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from laterality.config import canonical_json_digest
from laterality.metrics import source_weights, weighted_mae, weighted_r2
from .comparative_evaluation import aggregate_predictions, fit_source_readout, paired_source_bootstrap
from .comparative_training import evaluation_implementation_digest
from .masked_learning import PROBE_PAIRS, raw_pose_features

RIDGE_ALPHAS = (0.01, 0.1, 1., 10., 100., 1000., 10000.)
REPRESENTATIONS = tuple(f"{encoder}__{summary}" for encoder in
    ("pretrained_online", "pretrained_teacher", "initial_online") for summary in ("mean", "mean_motion")) + ("direct_pose", "training_mean")


def bilateral_summaries(tokens, valid):
    """Preserve five bilateral means; separately append SD and absolute increments.

Use common left/right support and consecutive valid transitions only. Missing
support produces zeros, as in the existing mean summary. Motion summaries append
support fractions so a zero change is distinguishable from an unobserved change.
"""
    z, v = np.asarray(tokens, float), np.asarray(valid)
    if z.ndim != 4 or z.shape[2] != 33 or v.shape != z.shape[:3] or v.dtype != bool:
        raise ValueError("Expected [batch, blocks, 33, channels] and boolean validity")
    if not np.isfinite(z[v]).all():
        raise ValueError("Observed features must be finite")
    means, motion, supports = [], [], []
    for left, right in PROBE_PAIRS:
        common = v[:, :, left] & v[:, :, right]
        denominator = np.maximum(common.sum(1), 1)[:, None]
        sides = [np.where(common[..., None], z[:, :, j], 0) for j in (left, right)]
        mu = [side.sum(1) / denominator for side in sides]
        means.extend((mu[0] - mu[1], mu[0] + mu[1]))
        sd = [np.sqrt(np.where(common[..., None], (side - m[:, None]) ** 2, 0).sum(1) / denominator)
              for side, m in zip(sides, mu)]
        transitions = common[:, 1:] & common[:, :-1]
        delta = [np.where(transitions[..., None], np.abs(np.diff(side, axis=1)), 0).sum(1)
                 / np.maximum(transitions.sum(1), 1)[:, None] for side in sides]
        motion.extend((sd[0] - sd[1], sd[0] + sd[1], delta[0] - delta[1], delta[0] + delta[1]))
        supports.extend((common.mean(1)[:, None], (transitions.sum(1) / max(z.shape[1] - 1, 1))[:, None]))
    mean = np.concatenate(means, axis=1)
    # Treat roundoff-size means as zero; the synthetic control must not amplify
    # numerical integration residues into an apparent movement signal.
    mean[np.abs(mean) < 1e-12] = 0
    return {"mean": mean, "mean_motion": np.concatenate((mean, *motion, *supports), axis=1)}


def encode_motion_summaries(encoder, dataset, *, batch_size=32):
    encoder.eval()
    device = next(encoder.parameters()).device
    outputs = {"mean": [], "mean_motion": []}
    with torch.no_grad():
        for start in range(0, len(dataset.xyz), batch_size):
            valid = dataset.valid[start:start + batch_size]
            patches = valid.reshape(len(valid), encoder.segments, encoder.segment_length, 33).all(2)
            if not patches.any((1, 2)).all():
                raise ValueError("Cannot encode a clip without observed tokens")
            xyz = torch.as_tensor(np.where(valid[..., None], dataset.xyz[start:start + batch_size], 0),
                                  dtype=torch.float32, device=device)
            z = encoder(xyz, torch.as_tensor(patches, device=device)).reshape(
                len(valid), encoder.segments, 33, encoder.embed_dim).cpu().numpy()
            for name, values in bilateral_summaries(z, patches).items():
                outputs[name].append(values)
    return {name: np.concatenate(values) for name, values in outputs.items()}


def evaluate_motion_readouts(result, dataset, settings, *, alphas=RIDGE_ALPHAS, include_predictor=False):
    """The ridge grid selects only the readout, on outer-training sources.

Every declared summary is reported. Outer-test results never select a summary,
mask, encoder checkpoint, regularizer or training budget.
"""
    if (set(result["identity"]["train_sources"]) != set(dataset.train_sources)
            or set(result["identity"]["test_sources"]) != set(dataset.test_sources)
            or result["identity"]["settings"]["seed"] != settings.seed
            or result["identity"]["settings"]["fold"] != dataset.fold
            or settings.fold != dataset.fold):
        raise ValueError("Readout data, seed or source roles disagree with the trained comparison")
    train, test = dataset.train_rows, dataset.test_rows
    rows, selections, diagnostics = [], [], []
    initial = next(iter(result["runs"].values()))["initial_model"]
    # The legacy loader restores the initial model on CPU. Use the same device
    # on fresh and cached evaluations so numerical backend changes cannot alter
    # the supposedly identical initial control.
    device = next(next(iter(result["runs"].values()))["model"].parameters()).device
    initial.to(device)
    initial_features = encode_motion_summaries(initial.view_encoder, dataset)
    direct = raw_pose_features(dataset)
    mean_target = float(np.average(dataset.targets[train], weights=source_weights(dataset.source_ids[train])))
    for condition, run in result["runs"].items():
        features = {f"initial_online__{k}": v for k, v in initial_features.items()}
        for name, encoder in (("pretrained_online", run["model"].view_encoder),
                              ("pretrained_teacher", run["model"].target_encoder)):
            features.update({f"{name}__{k}": v for k, v in encode_motion_summaries(encoder, dataset).items()})
        features["direct_pose"] = direct
        for representation in REPRESENTATIONS:
            alpha = np.nan
            if representation == "training_mean":
                predicted = np.full(len(test), mean_target)
            else:
                readout = fit_source_readout(features[representation], dataset.targets, dataset.source_ids,
                    train_sources=dataset.train_sources, test_sources=dataset.test_sources, alphas=alphas)
                predicted = readout.predict(features[representation][test])
                alpha = readout.selected_alpha
                for record in readout.validation.to_dict("records"):
                    selections.append({**record, "condition": condition, "representation": representation,
                        "selected": record["alpha"] == alpha, "selected_alpha": alpha,
                        "at_grid_boundary": alpha in (min(alphas), max(alphas)),
                        "fold": dataset.fold, "seed": settings.seed})
                diagnostics.append({"condition": condition, "representation": representation,
                    "fold": dataset.fold, "seed": settings.seed, **readout.training_feature_diagnostics})
            for i, row in enumerate(test):
                rows.append({"condition": condition, "representation": representation,
                    "sequence_id": str(dataset.sequence_ids[row]), "source_id": str(dataset.source_ids[row]),
                    "fold": dataset.fold, "seed": settings.seed, "target": float(dataset.targets[row]),
                    "prediction": float(predicted[i]), "available": True, "observation": "unaltered",
                    "selected_alpha": alpha, "synthetic": dataset.synthetic,
                    "cohort_digest": dataset.cohort_digest, "split_digest": dataset.split_digest,
                    "comparison_id": "motion_structured_readouts/v1",
                    "checkpoint": "initial" if representation.startswith("initial") else "final"})
    predictor = pd.DataFrame()
    if include_predictor:
        from .comparative_evaluation import make_evaluation_mask_bank, predictor_diagnostics
        valid = dataset.valid.reshape(len(dataset.xyz), -1, settings.segment_length, 33).all(2)
        bank = make_evaluation_mask_bank(valid, seed=1801)
        predictor = pd.concat([
            *[predictor_diagnostics(run["model"], dataset, bank, condition=name)
              for name, run in result["runs"].items()],
            predictor_diagnostics(initial, dataset, bank, condition="initial"),
        ], ignore_index=True)
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity = {"training": canonical_json_digest(result["identity"]), "alphas": list(alphas),
                "readout_implementation": implementation, "dependencies": evaluation_implementation_digest(),
                "summaries": ["mean", "mean_motion"], "selection": "source-separated ridge only",
                "include_predictor": include_predictor, "evaluation_mask_seed": 1801}
    return {"predictions": pd.DataFrame(rows), "selection": pd.DataFrame(selections),
            "diagnostics": pd.DataFrame(diagnostics), "predictor_diagnostics": predictor, "identity": identity}


def evaluate_retained_comparison(directory, *, output_dir=None):
    """Reanalyse one compatible Notebook 12 job without encoder training.

Recompute the expected identity from current data, code, runtime and saved
configuration before loading. A manifest is not its own compatibility proof.
Original files are read-only; new readouts have a separate identity/destination.
"""
    from laterality.config import SUITE_ROOT
    from .comparative_masks import MaskBudget, MaskPolicy
    from .comparative_training import _save_tables, comparison_identity, load_comparison
    from .masked_learning import LearningSettings, load_learning_dataset
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    declared = manifest["identity"]
    if "study_schema" in declared:
        raise ValueError("This entry point is for retained Notebook 12 comparisons")
    settings = LearningSettings(**declared["settings"])
    dataset = load_learning_dataset(real=not declared["synthetic"], fold=settings.fold)
    expected = comparison_identity(dataset, settings,
        {n: MaskPolicy(**p) for n, p in declared["conditions"].items()},
        {n: MaskBudget(**b) for n, b in declared["budgets"].items()},
        declared["matched_to"], tuple(declared["checkpoint_steps"]))
    result = load_comparison(directory, expected)
    evaluation = evaluate_motion_readouts(result, dataset, settings, include_predictor=True)
    destination = Path(output_dir) if output_dir else SUITE_ROOT / "artifacts/motion_structured/retained_readouts"
    _save_tables({k: evaluation[k] for k in ("predictions", "selection", "diagnostics", "predictor_diagnostics")},
                 evaluation["identity"], destination)
    return evaluation


def aggregate_motion_study(predictions, expected, plan):
    """Strict coverage per declared experiment, then source-balanced pooled scores."""
    if set(predictions.experiment.unique()) != set(plan["experiments"]):
        raise ValueError("Missing or undeclared experiments")
    per_seed, summary, intervals = [], [], []
    for experiment in plan["experiments"]:
        table = predictions[predictions.experiment == experiment]
        conditions = tuple(plan["arms"][experiment])
        aggregate = aggregate_predictions(table, expected, seeds=plan["seeds"], conditions=conditions,
                                          representations=REPRESENTATIONS)
        per_seed.append(aggregate["per_seed"].assign(experiment=experiment))
        summary.append(aggregate["summary"].assign(experiment=experiment))
        for condition in conditions:
            if condition != "uniform":
                intervals.append({"experiment": experiment, **paired_source_bootstrap(table,
                    first=condition, reference="uniform", representation="pretrained_teacher__mean_motion")})
    return {"per_seed": pd.concat(per_seed, ignore_index=True),
            "summary": pd.concat(summary, ignore_index=True), "paired_intervals": pd.DataFrame(intervals)}


def pooling_positive_control():
    """Known oscillation amplitudes; labels and sources are synthetic by design."""
    rng = np.random.default_rng(1801)
    n, blocks = 48, 16
    left, right = rng.uniform(0.2, 1.5, size=(2, n))
    wave = np.sin(2 * np.pi * np.arange(blocks) / blocks)
    z = np.zeros((n, blocks, 33, 1))
    for l, r in PROBE_PAIRS:
        z[:, :, l, 0] = left[:, None] * wave
        z[:, :, r, 0] = right[:, None] * wave
    features = bilateral_summaries(z, np.ones(z.shape[:3], dtype=bool))
    sources = np.array([f"generated_source_{i // 2:02d}" for i in range(n)])
    train_sources, test_sources = tuple(sorted(set(sources))[:18]), tuple(sorted(set(sources))[18:])
    target = left - right
    test = np.isin(sources, test_sources)
    scores = []
    for name, x in features.items():
        readout = fit_source_readout(x, target, sources, train_sources=train_sources,
                                     test_sources=test_sources, alphas=RIDGE_ALPHAS)
        pred = readout.predict(x[test])
        w = source_weights(sources[test])
        scores.append({"summary": name, "r2": weighted_r2(target[test], pred, w),
                       "mae": weighted_mae(target[test], pred, w), "alpha": readout.selected_alpha,
                       "near_constant": readout.training_feature_diagnostics["near_constant"],
                       "test_sources": len(test_sources), "test_clips": int(test.sum())})
    return pd.DataFrame(scores), {"tokens": z, "wave": wave, "target": target}
