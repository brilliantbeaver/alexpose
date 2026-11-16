"""Deterministic synthetic fixtures, never evidence about clinical recognition."""
from __future__ import annotations
import numpy as np
from .masking import JOINT_IDS


def make_demo(n=40, frames=16, seed=7, conditions=False):
    """Return xy, validity, labels and synthetic IDs for runnable lessons.

    Coordinates are dimensionless image-plane values with pelvis at zero. These
    motions are pedagogical oscillations, not simulations of medical conditions.
    When conditions=True, names merely exercise a five-class software interface.
    """
    rng = np.random.default_rng(seed)
    base = np.array([[-.3,-1],[.3,-1],[-.5,-.55],[.5,-.55],[-.6,-.1],[.6,-.1],
                     [-.22,0],[.22,0],[-.22,.7],[.22,.7],[-.22,1.4],[.22,1.4],
                     [-.30,1.46],[.14,1.46],[-.05,1.5],[.39,1.5]], dtype=np.float32)
    x = np.broadcast_to(base,(n,frames,len(JOINT_IDS),2)).copy()
    names = ("normal", "parkinsons", "stroke", "cerebral palsy", "myopathic")
    labels = np.array([names[i % len(names)] if conditions else "normal" for i in range(n)])
    phase = np.linspace(0, 2*np.pi, frames, endpoint=False)
    for i in range(n):
        shift, speed, amp = rng.uniform(-1,1), rng.uniform(.8,1.2), rng.uniform(.08,.2)
        for j in range(len(JOINT_IDS)):
            side_phase = np.pi * (j % 2)
            strength = .3 if j < 8 else 1.0
            x[i,:,j,0] += amp * strength * np.sin(phase*speed + shift + side_phase)
            x[i,:,j,1] += .04 * strength * np.cos(phase*speed + shift + side_phase)
        x[i] += rng.normal(0,.006,x[i].shape)
    return {"x":x,"valid":np.ones(x.shape[:-1],dtype=bool),"labels":labels,
            "source_ids":np.array([f"synthetic-source-{i:04}" for i in range(n)]),
            "evidence_type":"synthetic software fixture"}
