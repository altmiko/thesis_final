"""
Default hyperparameters for all 8 per-class β-VAEs.
8-class labels follow sklearn alphabetical encoding:
Benign=0, BruteForce=1, DDoS=2, DoS=3, Mirai=4, Recon=5, Spoofing=6, Web=7
"""
from __future__ import annotations

CLASSES: list[str] = ['Benign', 'BruteForce', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web']

CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}
ID_TO_CLASS: dict[int, str] = {i: name for i, name in enumerate(CLASSES)}

DEFAULT_CONFIG: dict = {
    'classes': CLASSES,
    'latent_dim': {cls: 16 for cls in CLASSES},
    # Anti-collapse pair. β=0.5 reduces the pressure that drives latent KL to
    # zero, and free_bits_lambda gives each latent dim a 0.1-nat KL allowance
    # before the KL term penalizes it. Together they keep information spread
    # across all 16 dims instead of packing it into a few, which previously left
    # most classes with >8 collapsed dims (failing the no-excess-collapse gate).
    # Note: the collapse diagnostic still uses the real measured per-dim KL with
    # a 0.01 threshold, so a dim with KL just above the 0.1 allowance (e.g. DoS
    # min KL ≈ 0.057) is correctly counted as non-collapsed.
    'beta_target': {cls: 0.5 for cls in CLASSES},
    'free_bits_lambda': 0.1,
    'protocol_embed_dim': 4,
    'encoder_hidden': [128, 64],
    'decoder_hidden': [64, 128],
    'max_epochs': 200,
    'batch_size': 512,
    'num_workers': 0,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'warmup_frac': 0.3,
    # β warms up over this many epochs (independent of max_epochs). With early
    # stopping, the old warmup_frac * max_epochs schedule (≈60 epochs) meant β
    # rarely reached its target before training ended. Set to None to fall back
    # to the warmup_frac schedule.
    'beta_warmup_epochs': 10,
    # Continuous reconstruction likelihood: 'gaussian' (L2 NLL, original) or
    # 'laplace' (L1 NLL, more robust to heavy-tailed flow features).
    'continuous_likelihood': 'gaussian',
    'early_stop_patience': 10,
    'grad_clip': 5.0,
    'use_protocol_class_weights': True,
    'protocol_class_weight_power': 0.5,
    'protocol_loss_weight': 2.0,
    'constraint_loss_weight': 0.1,
    'physics_constraint_loss_weight': 0.0,
    'continuous_feature_loss_weights': {},
    'binary_feature_loss_weights': {},
    'normalize_feature_loss_weights': True,
    'use_structured_continuous_decoder': True,
    'use_structured_physics_decoder': False,
    'structured_continuous_mode': 'full',
    'structured_std_floor': 0.01,
    'latent_logvar_floor': -6.0,
    'latent_logvar_ceiling': 6.0,
    'continuous_logvar_floor': -7.0,
    'continuous_logvar_ceiling': 2.0,
    'continuous_nll_per_sample_cap': None,
}
