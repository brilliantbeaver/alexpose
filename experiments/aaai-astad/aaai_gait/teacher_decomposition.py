"""Optional diagnostic separating predictor and contextual-teacher changes.

This module does not participate in the main study's model or run identity.
It evaluates already fitted models without updating the teacher or its center.
Both inputs must retain the same coordinate normalization and window ordering.
The four losses are computational interventions, not causal clinical effects.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .context_prediction import (
    BRANCHES, READOUT_JOINTS, _array, _restricted_encode, make_mask_bank,
)
from .model import JOINT_INDICES, _evaluation, token_validity

FACTORS = ("p0y0", "p1y0", "p0y1", "p1y1")
KEYS = ("window_index", "joint", "patch_index", "mask_id", "target_type")
LOSSES = tuple("loss_" + branch for branch in BRANCHES)


@torch.no_grad()
def score_teacher_decomposition(bundle, original_x, edited_x, *, mask_seed=17,
                                mask_repeats=16, batch_size=16,
                                joints=READOUT_JOINTS):
    """Return four matched latent-loss tables keyed p0y0/p1y0/p0y1/p1y1.

    p0 and p1 use original and edited visible context respectively. y0 and y1
    are the original and edited *complete-input* frozen teacher distributions.
    Each score is KL(teacher || prediction), using the original student and
    teacher temperatures. All branches exclude hidden measurements before any
    encoder layer, through the existing matched-context implementation.

    Uniform masks are identical across inputs, factors, windows and batch sizes.
    A requested target is valid only if it is observed and has at least two
    visible own-joint patches in *both* inputs. There is no redraw or imputation.
    Requested but ineligible rows remain present with all three losses NaN.

    To analyze nonlinear profile distances, aggregate each of these four tables
    on one common physical-patch selection, score all four with the SAME frozen
    normal reference, and only then form the decomposition. Summing raw-loss
    components cannot recover a nonlinear profile-distance decomposition.
    """
    original = _array(original_x, bundle.config)
    edited = _array(edited_x, bundle.config)
    if original.shape != edited.shape:
        raise ValueError("Original and edited windows must have identical shapes and ordering")
    joints = tuple(joints)
    if (not isinstance(batch_size, (int, np.integer)) or isinstance(batch_size, (bool, np.bool_))
            or batch_size < 1 or not joints or len(joints) != len(set(joints))
            or not set(joints).issubset(JOINT_INDICES)):
        raise ValueError("Use positive integer batches and distinct whitelisted joints")
    if (not isinstance(mask_repeats, (int, np.integer)) or isinstance(mask_repeats, (bool, np.bool_))
            or not 1 <= mask_repeats <= 16):
        raise ValueError("Use one through sixteen fixed scoring masks")
    device = next(bundle.parameters()).device
    bank = make_mask_bank(bundle.config, mask_seed, mask_repeats).to(device)
    center_before = bundle.backbone.target_center.detach().clone()
    records = {name: [] for name in FACTORS}
    counts = dict(requested_targets=0, valid_targets=0,
                  original_eligible_targets=0, edited_eligible_targets=0)
    with _evaluation(bundle):
        for start in range(0, len(original), batch_size):
            batches = [torch.as_tensor(a[start:start+batch_size], device=device) for a in (original, edited)]
            targets = [bundle.targets(batch) for batch in batches]
            observed = [token_validity(batch, bundle.backbone.cfg) for batch in batches]
            for mask_id, one_mask in enumerate(bank):
                mask = one_mask.expand(len(batches[0]), -1)
                memory = [_restricted_encode(bundle.backbone.view_encoder, batch, (~mask) & valid)
                          for batch, valid in zip(batches, observed, strict=True)]
                for joint in joints:
                    slot = JOINT_INDICES.index(joint)
                    slots = torch.full((len(batches[0]),), slot, dtype=torch.long, device=device)
                    predictions = [bundle.predict_joint(batch, mask, slots, targets=target, full_memory=mem)
                                   for batch, target, mem in zip(batches, targets, memory, strict=True)]
                    first, second = predictions
                    common = first["eligible"] & second["eligible"]
                    rows, patches = first["requested"].nonzero(as_tuple=True)
                    common_rows, common_patches = common.nonzero(as_tuple=True)
                    ok = common[rows, patches].cpu().numpy()
                    original_ok = first["eligible"][rows, patches].cpu().numpy()
                    edited_ok = second["eligible"][rows, patches].cpu().numpy()
                    target_valid = (first["target_valid"] & second["target_valid"])[rows, patches].cpu().numpy()
                    own_counts = torch.minimum(first["own_visible_count"], second["own_visible_count"])[rows].cpu().numpy()
                    reason = np.where(~target_valid, "target_unobserved_in_pair",
                                      np.where(own_counts < 2, "insufficient_own_context_in_pair", ""))
                    metadata = dict(
                        window_index=rows.cpu().numpy().astype(np.int32) + start,
                        joint=np.full(len(rows), joint, np.int16),
                        patch_index=patches.cpu().numpy().astype(np.int16),
                        mask_id=np.full(len(rows), mask_id, np.int16), target_type="latent",
                        valid=ok, target_valid=target_valid, own_visible_count=own_counts.astype(np.int16),
                        original_valid=original_ok, edited_valid=edited_ok,
                        original_own_visible_count=first["own_visible_count"][rows].cpu().numpy(),
                        edited_own_visible_count=second["own_visible_count"][rows].cpu().numpy(),
                        exclusion_reason=reason,
                    )
                    losses = {name: {column: np.full(len(rows), np.nan, np.float64) for column in LOSSES}
                              for name in FACTORS}
                    if len(common_rows):
                        token_indices = common_patches * 16 + slot
                        expected = [target["latent"][common_rows, token_indices] for target in targets]
                        for branch in BRANCHES:
                            # predict_joint flattens eligible indices in row-major
                            # order. Restrict each flat array to the same intersection.
                            log_predictions = [F.log_softmax(
                                result["predictions"]["latent_" + branch][common[result["eligible"]]] /
                                bundle.backbone.cfg.student_temperature, dim=-1) for result in predictions]
                            for name, pi, yi in (("p0y0",0,0),("p1y0",1,0),("p0y1",0,1),("p1y1",1,1)):
                                teacher = expected[yi]
                                error = (teacher.exp() * (teacher - log_predictions[pi])).sum(-1).clamp_min(0.)
                                if not torch.isfinite(error).all():
                                    raise FloatingPointError("Nonfinite teacher-decomposition loss on an eligible target")
                                losses[name]["loss_" + branch][ok] = error.cpu().numpy()
                    for name in FACTORS:
                        records[name].append(pd.DataFrame({**metadata, **losses[name]}))
                    counts["requested_targets"] += len(rows)
                    counts["valid_targets"] += int(ok.sum())
                    counts["original_eligible_targets"] += int(original_ok.sum())
                    counts["edited_eligible_targets"] += int(edited_ok.sum())
    if not torch.equal(center_before, bundle.backbone.target_center):
        raise RuntimeError("The teacher center changed during a frozen decomposition")
    provenance = dict(**counts, mask_seed=int(mask_seed), mask_repeats=int(mask_repeats),
        joint_indices=list(joints), original_array_sha256=hashlib.sha256(original.tobytes()).hexdigest(),
        edited_array_sha256=hashlib.sha256(edited.tobytes()).hexdigest(),
        teacher_center_sha256=hashlib.sha256(center_before.cpu().numpy().tobytes()).hexdigest(),
        mask_bank_sha256=hashlib.sha256(bank.cpu().numpy().tobytes()).hexdigest(),
        metric="KL(complete-input frozen teacher || restricted-context predictor)",
        normal_reference_rule="Score four separately aggregated profiles against one fixed reference before decomposing distance")
    tables = {}
    for name, parts in records.items():
        table = pd.concat(parts, ignore_index=True).sort_values(list(KEYS)).reset_index(drop=True)
        table.attrs.update(provenance, factor=name)
        tables[name] = table
    # Validate paired support and the additive identity before returning artifacts.
    decompose_teacher_losses(tables)
    return tables


def decompose_teacher_losses(tables):
    """Return additive raw-loss changes on identical eligible target identities.

    For each branch, predictor = p1y0 - p0y0, teacher = p0y1 - p0y0,
    interaction = p1y1 - p1y0 - p0y1 + p0y0, and joint = p1y1 - p0y0.
    The joint change equals predictor + teacher + interaction. These are signed
    computational components; none is a probability or a biological attribution.
    """
    if set(tables) != set(FACTORS):
        raise ValueError("Exactly p0y0, p1y0, p0y1 and p1y1 tables are required")
    normalized = {}
    for name in FACTORS:
        table = tables[name].copy()
        if not set(KEYS + LOSSES + ("valid",)).issubset(table):
            raise ValueError("Each factor needs latent target keys, three losses and common eligibility")
        if table[list(KEYS)].isna().any().any() or table.duplicated(list(KEYS)).any():
            raise ValueError("Target identities must be nonmissing and unique within each factor")
        if not table.target_type.eq("latent").all():
            raise ValueError("Teacher decomposition uses latent KL units only")
        if table.valid.isna().any() or not table.valid.isin([True,False]).all():
            raise ValueError("Common eligibility must be boolean")
        table['valid'] = table.valid.astype(bool)
        table = table.sort_values(list(KEYS)).reset_index(drop=True)
        normalized[name] = table
    reference = normalized["p0y0"]
    for table in normalized.values():
        try:pd.testing.assert_frame_equal(reference[list(KEYS)+["valid"]],table[list(KEYS)+["valid"]],check_dtype=False)
        except AssertionError as error:
            raise ValueError("All four factors must have identical target identities and eligibility") from error
        if not np.isfinite(table.loc[table.valid,list(LOSSES)].to_numpy(float)).all():
            raise FloatingPointError("An eligible decomposition target has a nonfinite loss")
    result = reference[list(KEYS)+["valid"]].copy()
    valid = result.valid.to_numpy(bool)
    for branch in BRANCHES:
        a,b,c,d = [normalized[name]["loss_"+branch].to_numpy(float) for name in FACTORS]
        predictor,teacher,interaction,joint = b-a,c-a,d-b-c+a,d-a
        if not np.allclose(joint[valid],(predictor+teacher+interaction)[valid],atol=1e-10,rtol=1e-10):
            raise FloatingPointError("Teacher/predictor decomposition failed its additive identity")
        for component,values in (("predictor",predictor),("teacher",teacher),("interaction",interaction),("joint",joint)):
            result[component+"_"+branch] = np.where(valid,values,np.nan)
    result.attrs.update(reference.attrs)
    result.attrs["decomposition"] = "joint = predictor + teacher + interaction, in raw per-target KL units"
    result.attrs.pop("factor",None)
    return result
