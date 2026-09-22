#!/usr/bin/env python3
"""Read-only sampler audit. Run with gavd6/.venv/bin/python; no training occurs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True, help='gavd6 checkout')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
root = args.root.resolve()
ms = root.parent.parent / 'multiple-sclerosis'
sys.path.insert(0, str(root / 'src'))
sys.path.insert(0, str(ms))

import numpy as np
import torch
from sjepa.masking_v2 import mask_bank_stats, sample_target_mask
from gavd6_sjepa.research_directions.synthetic_training_v2.models import ModelConfig
from gavd6_sjepa.research_directions.synthetic_training_v2.training import _mask

stats = mask_bank_stats(33, 8, n_masks=512, seed=0)
sources = [
    ms / '02_anatomical_mask_and_tokenization.ipynb',
    ms / 'sjepa/masking_v2.py', ms / 'sjepa/config.py',
    ms / 'sjepa/models.py', ms / 'sjepa/tokenizer.py',
    ms / 'sjepa/data.py', ms / 'sjepa/train_v2.py', ms / 'sjepa/augment.py',
    root / 'src/gavd6_sjepa/research_directions/synthetic_training_v2/models.py',
    root / 'src/gavd6_sjepa/research_directions/synthetic_training_v2/training.py',
]
r = {
    'audit_date': '2026-09-21',
    'scope': 'Deterministic sampler and implementation checks; no model training or efficacy claim.',
    'environment': {'python': sys.version, 'numpy': np.__version__, 'torch': torch.__version__},
    'source_sha256': {str(p.relative_to(root.parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    'mask_bank': {'sampler': 'multiple-sclerosis/sjepa/masking_v2.py', 'seed': 0,
        'n_masks': 512, 'joints': 33, 'time_blocks': 8, 'configured_target_ratio': .6,
        'clinical_bias': 1.5, 'max_time_span_frac': .75},
    'realized_target_fraction': float(stats.mean_target_frac),
    'minimum_context_somewhere_in_window_fraction': float(stats.joint_visible_frac.min()),
    'minimum_target_somewhere_in_window_fraction': float(stats.joint_target_frac.min()),
    'left_hip_context_frequency_by_block': stats.token_visible_frac[:, 23].tolist(),
    'right_hip_context_frequency_by_block': stats.token_visible_frac[:, 24].tolist(),
    'left_hip_whole_window_hidden_fraction': float(stats.joint_always_target_frac[23]),
    'right_hip_whole_window_hidden_fraction': float(stats.joint_always_target_frac[24]),
    'joint_order': 'BlazePose-33 integer indices 0 through 32',
    'per_joint': {
        'context_somewhere_in_window_fraction': stats.joint_visible_frac.tolist(),
        'target_somewhere_in_window_fraction': stats.joint_target_frac.tolist(),
        'target_token_fraction': stats.joint_target_token_frac.tolist(),
        'context_token_fraction': stats.joint_visible_token_frac.tolist(),
        'whole_window_hidden_fraction': stats.joint_always_target_frac.tolist(),
    },
    'exact_slot_context_fraction_time_by_joint': stats.token_visible_frac.tolist(),
}
try:
    sample_target_mask(12, 16, np.random.default_rng(0))
except Exception as exc:
    r['direct_body12_call'] = f'{type(exc).__name__}: {exc}'
cfg = ModelConfig()
m = _mask(cfg, torch.ones((2, cfg.window_size, 12), dtype=torch.bool),
          .5, np.random.default_rng(17), 'cpu')
r['synthetic_training_v2_complete_input_example'] = {
    'seed': 17, 'shape': list(m.shape),
    'fully_hidden_blocks_per_window': m.all(-1).sum(1).tolist(),
    'partly_hidden_blocks_per_window': (m.any(-1) & ~m.all(-1)).sum(1).tolist(),
    'masked_fraction': float(m.float().mean()),
    'mask_bits': m.tolist(),
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(r, indent=2) + '\n')
print(json.dumps({'output': str(args.output), 'realized_target_fraction': r['realized_target_fraction'],
                  'sources_hashed': len(sources), 'slot_table_shape': list(stats.token_visible_frac.shape)}, indent=2))
