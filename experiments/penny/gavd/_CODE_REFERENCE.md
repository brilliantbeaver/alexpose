# Verbatim code snippets to reuse in the gavd/ series

Copy these into the relevant notebooks so the full-dataset series stays
byte-for-byte consistent with `gait/skeleton-jepa/tutorials/`. Adapt only the
CONFIG values that a notebook legitimately needs to change (documented inline).

--------------------------------------------------------------------------------
## SKELETON CONSTANTS (define once per notebook that draws or masks skeletons)

```python
# 33 BLAZEPOSE_33 landmark names in index order (source: ambient/pose/keypoint_data.py)
LANDMARK_NAMES = [
    "NOSE","LEFT_EYE_INNER","LEFT_EYE","LEFT_EYE_OUTER","RIGHT_EYE_INNER",
    "RIGHT_EYE","RIGHT_EYE_OUTER","LEFT_EAR","RIGHT_EAR","MOUTH_LEFT",
    "MOUTH_RIGHT","LEFT_SHOULDER","RIGHT_SHOULDER","LEFT_ELBOW","RIGHT_ELBOW",
    "LEFT_WRIST","RIGHT_WRIST","LEFT_PINKY","RIGHT_PINKY","LEFT_INDEX",
    "RIGHT_INDEX","LEFT_THUMB","RIGHT_THUMB","LEFT_HIP","RIGHT_HIP",
    "LEFT_KNEE","RIGHT_KNEE","LEFT_ANKLE","RIGHT_ANKLE","LEFT_HEEL",
    "RIGHT_HEEL","LEFT_FOOT_INDEX","RIGHT_FOOT_INDEX",
]

# 35 skeleton edges as (i, j) index pairs
EDGES = [
    (0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,9),(0,10),(9,10),          # face
    (11,12),(11,23),(12,24),(23,24),                                  # torso
    (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),                  # left arm
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),                  # right arm
    (23,25),(25,27),(27,29),(27,31),(29,31),                          # left leg
    (24,26),(26,28),(28,30),(28,32),(30,32),                          # right leg
]

# 6 semantic groups (for limb-based block masking)
GROUPS = {
    "face":      [0,1,2,3,4,5,6,7,8,9,10],
    "left_arm":  [11,13,15,17,19,21],
    "right_arm": [12,14,16,18,20,22],
    "torso":     [11,12,23,24],
    "left_leg":  [23,25,27,29,31],
    "right_leg": [24,26,28,30,32],
}
```

--------------------------------------------------------------------------------
## ANIMATION HELPER (verbatim from tutorials/01; use in every notebook)

Requires `import numpy as np` and `import matplotlib.pyplot as plt` earlier.

```python
# Inline animation helper: animate_skeleton
# Uses matplotlib.animation and to_jshtml so it works in Jupyter and Colab
# with no ffmpeg or system dependencies.
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def animate_skeleton(seq, edges, title="Walking skeleton", mask=None, fps=8):
    """Animate a skeleton sequence inline.
    seq: (T, 33, C) array, C >= 2 (uses x=seq[...,0], y=seq[...,1]).
    edges: list of (i, j) index pairs.
    mask: optional (T, n_groups) or (T, 33) boolean array to grey out hidden joints.
    Returns an IPython.display.HTML wrapping the JS animation.
    """
    T = seq.shape[0]
    x_all = seq[:, :, 0]
    y_all = seq[:, :, 1]
    groups = [
        list(range(11)),
        [11, 13, 15, 17, 19, 21],
        [12, 14, 16, 18, 20, 22],
        [11, 12, 23, 24],
        [23, 25, 27, 29, 31],
        [24, 26, 28, 30, 32],
    ]
    colors = ["#8b5cf6", "#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#ec4899"]
    grey = "#cbd5e1"
    joint_mask = None
    if mask is not None:
        if mask.shape == (T, len(groups)):
            joint_mask = np.zeros((T, 33), dtype=bool)
            for t in range(T):
                for g_idx, grp in enumerate(groups):
                    if mask[t, g_idx]:
                        joint_mask[t, grp] = True
        else:
            joint_mask = mask
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    margin = max(x_max - x_min, y_max - y_min) * 0.1
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_max + margin, y_min - margin)
    def draw_frame(t):
        ax.clear(); ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_max + margin, y_min - margin)
        ax.set_title(f"{title} (frame {t}/{T})")
        x = x_all[t]; y = y_all[t]
        for g_idx, grp in enumerate(groups):
            grp_edges = [(i, j) for (i, j) in edges if i in grp and j in grp]
            for (i, j) in grp_edges:
                hidden = joint_mask is not None and (joint_mask[t, i] or joint_mask[t, j])
                c = grey if hidden else colors[g_idx]
                ax.plot([x[i], x[j]], [y[i], y[j]], color=c, linewidth=2, alpha=0.7)
        for g_idx, grp in enumerate(groups):
            x_g = [x[i] for i in grp]; y_g = [y[i] for i in grp]
            if joint_mask is not None:
                c_g = [grey if joint_mask[t, i] else colors[g_idx] for i in grp]
            else:
                c_g = [colors[g_idx]] * len(grp)
            ax.scatter(x_g, y_g, c=c_g, s=40, zorder=3, edgecolors='white', linewidths=0.5)
    anim = FuncAnimation(fig, draw_frame, frames=T, interval=1000/fps, repeat=True)
    plt.close(fig)
    return HTML(anim.to_jshtml())

print("animate_skeleton helper defined (uses to_jshtml, works in Colab with no ffmpeg).")
```

--------------------------------------------------------------------------------
## SYNTHETIC WALKING SKELETON (SMOKE-mode data generator)

```python
def synthesize_walking_skeleton(T=16, n_joints=33, n_channels=3, seed=0, gait_bias=0.0):
    """Make a plausible synthetic (T, 33, 3) walking skeleton for SMOKE mode.
    A standing skeleton whose legs and arms swing sinusoidally out of phase.
    gait_bias adds a small left/right asymmetry so we can fake distinct classes.
    """
    rng = np.random.RandomState(seed)
    # Neutral standing pose in a rough (x, y) layout, z small.
    base = np.zeros((n_joints, 3), dtype=np.float32)
    # Full (x, y) layout for all 33 landmarks, laid out as a person seen head-on.
    # y grows downward (head near 0.16, feet near 0.97). x has the midline at 0.50,
    # with the left side (odd joint indices) left of it and the right side right of it.
    # Shoulders are wider than the hips, the arms hang OUTSIDE the hips down to about
    # hip height, and the head sits just above the shoulders, so the figure reads as a
    # real walking body instead of collapsing onto one vertical line. Every joint gets
    # an explicit (x, y): nothing falls back to a fragile parity default.
    xs = {
        0:0.500,                                            # nose
        1:0.485, 2:0.475, 3:0.465, 4:0.515, 5:0.525, 6:0.535,  # eyes (left then right)
        7:0.455, 8:0.545,                                   # ears
        9:0.485, 10:0.515,                                  # mouth
        11:0.415, 12:0.585,                                 # shoulders (wide)
        13:0.395, 14:0.605,                                 # elbows (arms hang outside)
        15:0.405, 16:0.595,                                 # wrists
        17:0.395, 18:0.605, 19:0.405, 20:0.595, 21:0.420, 22:0.580,  # hands track their wrist
        23:0.455, 24:0.545,                                 # hips (narrower than shoulders)
        25:0.450, 26:0.550,                                 # knees
        27:0.448, 28:0.552,                                 # ankles
        29:0.448, 30:0.552, 31:0.455, 32:0.545,             # heels, foot tips
    }
    ys = {
        0:0.16,                                             # nose
        1:0.145, 2:0.145, 3:0.145, 4:0.145, 5:0.145, 6:0.145,  # eyes
        7:0.155, 8:0.155,                                   # ears
        9:0.185, 10:0.185,                                  # mouth (short neck to shoulders)
        11:0.24, 12:0.24,                                   # shoulders
        13:0.38, 14:0.38,                                   # elbows
        15:0.51, 16:0.51,                                   # wrists (about hip height)
        17:0.545, 18:0.545, 19:0.545, 20:0.545, 21:0.535, 22:0.535,  # hands (just past wrists)
        23:0.50, 24:0.50,                                   # hips
        25:0.71, 26:0.71,                                   # knees
        27:0.92, 28:0.92,                                   # ankles
        29:0.94, 30:0.94, 31:0.965, 32:0.965,               # heels, foot tips
    }
    for j in range(n_joints):
        base[j, 0] = xs[j]
        base[j, 1] = ys[j]
        base[j, 2] = 0.0
    seq = np.repeat(base[None], T, axis=0)
    t = np.linspace(0, 2 * np.pi, T, endpoint=False)
    swing = 0.06 * np.sin(t)
    # Legs swing out of phase; arms opposite to legs; add tiny per-clip asymmetry.
    for k, amp in [(25, 1.0), (27, 1.3), (31, 1.4), (13, -0.8), (15, -1.0)]:   # left side
        seq[:, k, 0] += swing * amp * (1.0 + gait_bias)
    for k, amp in [(26, -1.0), (28, -1.3), (32, -1.4), (14, 0.8), (16, 1.0)]:  # right side
        seq[:, k, 0] += swing * amp * (1.0 - gait_bias)
    seq += rng.randn(T, n_joints, 3).astype(np.float32) * 0.004
    return seq.astype(np.float32)
```

--------------------------------------------------------------------------------
## NORMALIZE (pelvis-center + torso-scale), used in 03/04/05

```python
def normalize_skeleton_seq(seq):
    """seq: (T, 33, 3). Center on pelvis midpoint (mean of hips 23,24) and scale
    by torso length (shoulder midpoint to hip midpoint) so absolute camera
    position and scale do not matter. Returns a new (T, 33, 3) array."""
    seq = seq.astype(np.float32).copy()
    hips = (seq[:, 23, :] + seq[:, 24, :]) / 2.0          # (T, 3)
    shoulders = (seq[:, 11, :] + seq[:, 12, :]) / 2.0     # (T, 3)
    seq = seq - hips[:, None, :]                          # pelvis-center
    torso = np.linalg.norm(shoulders - hips, axis=1)      # (T,)
    scale = np.median(torso[torso > 1e-6]) if np.any(torso > 1e-6) else 1.0
    return (seq / (scale + 1e-6)).astype(np.float32)
```

--------------------------------------------------------------------------------
## JEPA PIECES (verbatim from tutorials/03 and 04). Requires torch/nn/F/copy.

```python
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextEncoder(nn.Module):
    """Maps tokens (B, T*n_joints, C) to embeddings (B, T*n_joints, D) with a small
    transformer, after adding a learned time embedding and a learned joint embedding so the
    model knows which frame and which landmark each token is. Tokens must be in row-major
    (t, j) order (token n = t*n_joints + j). Build with the same T and n_joints the encoder
    was trained with. NOTE: this is the ONE gavd/ divergence from tutorials/03, which omits
    the positional embeddings; without them the encoder is permutation-invariant and a pooled
    clip embedding cannot represent gait dynamics (see _SHARED_FACTS.md)."""
    def __init__(self, input_dim=3, embed_dim=64, n_layers=2, n_heads=4, T=32, n_joints=33):
        super().__init__()
        self.T = T
        self.n_joints = n_joints
        self.embed_dim = embed_dim
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.time_embed = nn.Parameter(torch.randn(T, embed_dim) * 0.1)
        self.joint_embed = nn.Parameter(torch.randn(n_joints, embed_dim) * 0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 2,
            batch_first=True, dropout=0.0, activation="gelu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    def forward(self, x):
        h = self.input_proj(x)                                              # (B, N, D)
        pos = self.time_embed[:, None, :] + self.joint_embed[None, :, :]    # (T, n_joints, D)
        pos = pos.reshape(self.T * self.n_joints, self.embed_dim)           # (N, D)
        return self.transformer(h + pos[None, :, :])

def make_target_encoder(context_encoder):
    target = copy.deepcopy(context_encoder)
    for p in target.parameters():
        p.requires_grad = False
    return target

@torch.no_grad()
def ema_update(target_encoder, context_encoder, m):
    for p_tgt, p_ctx in zip(target_encoder.parameters(), context_encoder.parameters()):
        p_tgt.data.mul_(m).add_(p_ctx.data, alpha=(1.0 - m))

class Predictor(nn.Module):
    """Guesses hidden token embeddings from context + position info. (B, N, D_in) -> (B, N, D_out)."""
    def __init__(self, input_dim=64, output_dim=64, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))
    def forward(self, x):
        return self.net(x)

def vicreg_loss(pred, target, cfg, context=None):
    """CORRECTED gavd/ loss (use THIS in 04 and 05, not the tutorials/03 symmetric version):
    L2 between the prediction and the LayerNorm-normalized target, plus light VICReg variance
    + covariance on the ONLINE side only (the context embedding when given, else the
    prediction), never on the stop-gradient target. See _SHARED_FACTS.md for why."""
    B, N, D = pred.shape
    assert target.shape == pred.shape
    if B < 2:
        raise ValueError("VICReg needs batch size >= 2 for variance/covariance")
    lam_sim, lam_var, lam_cov = cfg["VICREG_SIM"], cfg["VICREG_VAR"], cfg["VICREG_COV"]
    gamma, eps = cfg["VAR_TARGET"], cfg["EPS"]
    tgt_norm = F.layer_norm(target, (D,))
    pred_flat = pred.reshape(-1, D); tgt_flat = tgt_norm.reshape(-1, D)
    sim_loss = F.mse_loss(pred_flat, tgt_flat)
    online = context if context is not None else pred
    of = online.reshape(-1, D)
    def variance_term(x):
        std = torch.sqrt(x.var(dim=0, unbiased=True) + eps)
        return F.relu(gamma - std).mean()
    def covariance_term(x):
        xc = x - x.mean(dim=0, keepdim=True)
        cov = (xc.T @ xc) / max(x.size(0) - 1, 1)
        cov = cov - torch.diag(torch.diag(cov))
        return cov.pow(2).sum() / x.size(1)
    total = lam_sim * sim_loss + lam_var * variance_term(of) + lam_cov * covariance_term(of)
    return total, {"sim": sim_loss.detach().item()}

def make_block_mask(T, groups, style, ratio, rng):
    """Spatiotemporal block mask -> (T, n_groups) bool, True = hidden. style in {limb,time}."""
    group_names = list(groups.keys()); n_groups = len(group_names)
    mask = np.zeros((T, n_groups), dtype=bool)
    if style == "limb":
        window_len = min(max(1, int(ratio * T * n_groups)), T)
        limb_groups = [i for i, name in enumerate(group_names)
                       if name in ["left_arm","right_arm","left_leg","right_leg"]]
        if not limb_groups:
            limb_groups = list(range(n_groups))
        group_idx = rng.choice(limb_groups)
        start_t = rng.randint(0, max(1, T - window_len + 1))
        mask[start_t:min(start_t + window_len, T), group_idx] = True
    elif style == "time":
        window_len = min(max(1, int(ratio * T)), T)
        start_t = rng.randint(0, max(1, T - window_len + 1))
        mask[start_t:min(start_t + window_len, T), :] = True
    else:
        raise ValueError(f"Unknown style: {style}")
    return mask
```

--------------------------------------------------------------------------------
## CONFIG defaults (JEPA hyperparameters) for the gavd/ series (04 and 05)

Use the corrected, light weights that go with the corrected vicreg_loss above. Also carry
T and N_JOINTS so the positional-embedding encoder can be rebuilt in 05.

```python
"T": 32, "N_JOINTS": 33, "C": 3, "EMBED_DIM": 64, "EMA_M": 0.996,
"VICREG_SIM": 25.0, "VICREG_VAR": 0.5, "VICREG_COV": 0.04,
"VAR_TARGET": 0.5, "EPS": 1e-4, "SEED": 42,
```

(The concept tutorials/03 still uses the older 25/25/1, gamma=1 symmetric loss on its short
toy run, where the divergence never shows. Do not copy those into the gavd/ series.)
