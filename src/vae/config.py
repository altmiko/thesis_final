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
    'beta_target': {cls: 1.0 for cls in CLASSES},
    'protocol_embed_dim': 4,
    'encoder_hidden': [128, 64],
    'decoder_hidden': [64, 128],
    'max_epochs': 200,
    'batch_size': 512,
    'num_workers': 0,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'warmup_frac': 0.3,
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
