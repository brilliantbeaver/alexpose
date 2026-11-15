"""Synthetic gait-shaped arrays for software checks, never clinical evidence."""
import numpy as np
import pandas as pd

LABELS = ['normal', 'parkinsons', 'stroke', 'cerebral palsy', 'myopathic']

def synthetic_corpus(seed=42, frames=32, sources_per_class=6):
    """A tiny, clearly artificial cohort with source-disjoint predefined splits.

    Conditions deliberately change a few parameters. They are NOT simulations
    of disease physiology, and scores on these arrays have no medical meaning.
    """
    rng = np.random.default_rng(seed)
    base = np.array([[-.3,-1],[.3,-1],[-.4,-.6],[.4,-.6],[-.45,-.25],[.45,-.25],
                     [-.2,0],[.2,0],[-.22,.6],[.22,.6],[-.2,1.2],[.2,1.2],
                     [-.24,1.22],[.24,1.22],[-.12,1.25],[.32,1.25]],np.float32)
    xs, rows = [], []
    for label_id, label in enumerate(LABELS):
        for source in range(sources_per_class):
            split = 'train' if source < sources_per_class-2 else ('val' if source == sources_per_class-2 else 'test')
            phase = rng.uniform(0, 2*np.pi)
            amplitude = .13 * (1 - .11*label_id)
            for sequence in range(2):
                t = np.arange(frames)/15
                x = np.repeat(base[None],frames,axis=0)
                wave = np.sin(2*np.pi*(1.1-.06*label_id)*t + phase + sequence*.2)
                x[:,[8,10,12,14],0] += amplitude*wave[:,None]
                x[:,[9,11,13,15],0] -= amplitude*wave[:,None]*(1 if label_id < 2 else .5)
                x[:,[2,4],0] -= .6*amplitude*wave[:,None]
                x[:,[3,5],0] += .6*amplitude*wave[:,None]
                x[:,:,0] += (label_id*.015)*np.sin(2*np.pi*.5*t)[:,None]
                x += rng.normal(0,.004,x.shape)
                xs.append(x)
                rows.append(dict(label=label,source_id=f'synthetic-{label_id}-{source}',
                                 sequence_id=f'synthetic-{label_id}-{source}-{sequence}',
                                 split=split,window_start=0,fps=15.,synthetic=True))
    return np.asarray(xs,dtype=np.float32),pd.DataFrame(rows)
