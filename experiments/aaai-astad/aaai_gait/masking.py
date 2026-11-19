"""Value-independent U/J/A masks and frozen evaluation banks.

The existing Torch ``uniform_mask`` is re-exported without changing its default.
Validity and target selection are separate from artificial input hiding.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import torch

from .model import JOINT_INDICES, uniform_mask

POLICIES = ("U", "J", "A")
ANATOMICAL_GROUPS = ((11, 12, 23, 24), (13, 15), (14, 16),
                     (25, 27, 29, 31), (26, 28, 30, 32))
TOKEN_COUNT = 128
MASK_COUNT = 64


def stream_seed(seed: int, name: str) -> int:
    """Named streams do not depend on another sampler's draw consumption."""
    raw = f"iteration02:{int(seed)}:{name}".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def rng_stream(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng(stream_seed(seed, name))


def core_masks(policy: str) -> np.ndarray:
    """Sixteen equiprobable disjoint eight-token cores, in declared order."""
    if policy not in ("J", "A"):
        raise ValueError("A core policy must be J or A")
    cores = []
    if policy == "J":
        for joint in range(16):
            core = np.zeros((8, 16), dtype=bool)
            core[:, joint] = True
            cores.append(core.reshape(-1))
    else:
        for group in ANATOMICAL_GROUPS:
            width = len(group)
            duration = 8 // width
            slots = [JOINT_INDICES.index(joint) for joint in group]
            for start in range(0, 8, duration):
                core = np.zeros((8, 16), dtype=bool)
                core[start:start + duration, slots] = True
                cores.append(core.reshape(-1))
    result = np.stack(cores)
    if result.shape != (16, 128) or not np.all(result.sum(0) == 1):
        raise AssertionError("The declared core partition is invalid")
    return result


def draw_masks(policy: str, count: int, *, rng: np.random.Generator) -> np.ndarray:
    """Draw before inspecting coordinates or validity; exactly 64 hidden cells."""
    if policy not in POLICIES or count < 1:
        raise ValueError("Need U/J/A and a positive mask count")
    result = np.zeros((count, 128), dtype=bool)
    cores = None if policy == "U" else core_masks(policy)
    for i in range(count):
        if cores is None:
            result[i, rng.choice(128, 64, replace=False)] = True
        else:
            result[i] = cores[int(rng.integers(16))]
            free = np.flatnonzero(~result[i])
            result[i, rng.choice(free, 56, replace=False)] = True
    return result


@dataclass(frozen=True)
class MaskBank:
    hidden: np.ndarray
    targets: np.ndarray
    policy: str
    seed: int
    producer: str

    def __post_init__(self):
        hidden = np.asarray(self.hidden)
        targets = np.asarray(self.targets)
        if hidden.dtype != bool or targets.dtype != bool or hidden.shape != targets.shape:
            raise ValueError("Masks and target selectors must be Boolean arrays of identical shape")
        if hidden.ndim != 2 or hidden.shape[1] != 128 or len(hidden) < 1:
            raise ValueError("Expected a nonempty [masks,128] bank")
        if not np.all(hidden.sum(1) == 64) or np.any(targets & ~hidden):
            raise ValueError("Need 64 hidden tokens and targets contained in hidden inputs")
        for name, value in (("hidden", hidden), ("targets", targets)):
            value = value.copy()
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def sha256(self) -> str:
        header = json.dumps({"policy": self.policy, "seed": int(self.seed),
                             "producer": self.producer, "shape": list(self.hidden.shape)},
                            sort_keys=True).encode()
        return hashlib.sha256(header + self.hidden.tobytes() + self.targets.tobytes()).hexdigest()

    @property
    def metadata(self) -> dict:
        return {"policy": self.policy, "seed": int(self.seed), "producer": self.producer,
                "mask_count": len(self.hidden), "hidden_per_mask": 64,
                "targets_per_mask": self.targets.sum(1).tolist(), "sha256": self.sha256}


def make_mask_bank(policy: str = "U", repeats: int = 16, seed: int | None = None) -> MaskBank:
    """U is byte-identical to the historical Torch bank for the same arguments."""
    seed = (117 if policy == "E_leg" else 17) if seed is None else seed
    if policy == "U":
        hidden = uniform_mask(repeats, 128, .5,
                              generator=torch.Generator().manual_seed(seed)).numpy()
        return MaskBank(hidden, hidden, policy, seed, "historical_torch_uniform_mask")
    rng = rng_stream(seed, "evaluation_masks")
    if policy in ("J", "A"):
        hidden = draw_masks(policy, repeats, rng=rng)
        return MaskBank(hidden, hidden, policy, seed, "numpy_pcg64_named_stream_v1")
    if policy != "E_leg" or repeats != 16:
        raise ValueError("E_leg requires exactly16 masks; other policies are U/J/A")
    hidden, targets = [], []
    for joints in ((25, 27, 29, 31), (26, 28, 30, 32)):
        slots = [JOINT_INDICES.index(joint) for joint in joints]
        for start in (0, 2, 4, 6):
            for _ in range(2):
                core = np.zeros((8, 16), dtype=bool)
                core[start:start + 2, slots] = True
                core = core.reshape(-1)
                mask = core.copy()
                mask[rng.choice(np.flatnonzero(~core), 56, replace=False)] = True
                hidden.append(mask)
                targets.append(core)
    return MaskBank(np.stack(hidden), np.stack(targets), policy, seed,
                    "numpy_pcg64_named_stream_core_only_targets_v1")
