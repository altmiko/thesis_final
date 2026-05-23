import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
import os

np.random.seed(42)
sys.path.insert(0, 'D:/thesis_final/src')
from src.preprocessing.feature_groups import FEATURE_NAMES, BINARY_FEATURES, INTEGER_FEATURES, CATEGORY_MAP

plt.rcParams.update({'font.size': 9, 'figure.dpi': 150})

df = pd.read_parquet('D:/thesis_final/data/processed/raw_loaded.parquet')
df['category'] = df['Label'].map(CATEGORY_MAP)
print(f"Loaded: {df.shape}")

FIGURES_DIR = 'D:/thesis_final/figures'
os.makedirs(FIGURES_DIR, exist_ok=True)

CAT_COLORS = {
    'DDoS': '#e74c3c', 'DoS': '#e67e22', 'Mirai': '#9b59b6',
    'Benign': '#2ecc71', 'Spoofing': '#3498db', 'Recon': '#f1c40f',
    'Web': '#1abc9c', 'BruteForce': '#34495e',
}

# ── F1: Class distribution bar chart ─────────────────────────────────
print("Generating F1...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

# Panel A: 34 classes
vc = df['Label'].value_counts().sort_values(ascending=True)
colors_a = [CAT_COLORS.get(CATEGORY_MAP.get(l, ''), '#999') for l in vc.index]
bars = ax1.barh(range(len(vc)), vc.values, color=colors_a)
ax1.set_yticks(range(len(vc)))
ax1.set_yticklabels(vc.index, fontsize=7)
ax1.set_xscale('log')
ax1.set_xlabel('Count (log scale)')
ax1.set_title('A) Per-Class Distribution (34 classes)')
for i, (label, count) in enumerate(vc.items()):
    ax1.text(count * 1.1, i, f'{count:,}', va='center', fontsize=6)

# Panel B: 8 categories
cat_vc = df['category'].value_counts().sort_values(ascending=True)
colors_b = [CAT_COLORS.get(c, '#999') for c in cat_vc.index]
ax2.barh(range(len(cat_vc)), cat_vc.values, color=colors_b)
ax2.set_yticks(range(len(cat_vc)))
ax2.set_yticklabels(cat_vc.index, fontsize=9)
ax2.set_xscale('log')
ax2.set_xlabel('Count (log scale)')
ax2.set_title('B) Per-Category Distribution (8 categories)')
for i, (cat, count) in enumerate(cat_vc.items()):
    ax2.text(count * 1.1, i, f'{count:,}', va='center', fontsize=8)

plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/F1_class_distribution.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("F1 saved.")

# ── F2: Log-binned histogram overlays ────────────────────────────────
print("Generating F2...")
kde_features = ['Rate', 'IAT', 'AVG', 'Tot size', 'Number', 'Header_Length']
kde_classes = ['BENIGN', 'DDOS-ICMP_FLOOD', 'DOS-SYN_FLOOD', 'RECON-PORTSCAN', 'MIRAI-GREETH_FLOOD']
kde_colors = ['#2ecc71', '#e74c3c', '#e67e22', '#f1c40f', '#9b59b6']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for idx, feat in enumerate(kde_features):
    ax = axes[idx // 3][idx % 3]

    # Build one global (per-feature) positive range and bins so all class
    # overlays share the exact same x-axis extent and bucketization.
    pooled = df[feat].to_numpy(dtype=np.float64)
    pooled = pooled[np.isfinite(pooled)]
    pooled = pooled[pooled > 0]
    if pooled.size == 0:
        ax.text(0.5, 0.5, f'No positive values for {feat}',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(feat, fontsize=10)
        continue

    x_min = float(np.quantile(pooled, 0.001))
    x_max = float(np.quantile(pooled, 0.999))
    x_min = max(x_min, 1e-8)
    x_max = max(x_max, x_min * 10)
    bins = np.logspace(np.log10(x_min), np.log10(x_max), 60)

    for cls, color in zip(kde_classes, kde_colors):
        data = df.loc[df['Label'] == cls, feat].to_numpy(dtype=np.float64)
        data = data[np.isfinite(data)]
        data = data[(data > 0) & (data >= x_min) & (data <= x_max)]
        if data.size == 0:
            continue
        if len(data) > 50000:
            data = np.random.choice(data, 50000, replace=False)
        ax.hist(data, bins=bins, density=True, histtype='step',
                linewidth=1.4, label=cls, color=color)

    ax.set_xscale('log')
    ax.set_xlim(x_min, x_max)
    ax.set_title(feat, fontsize=10)
    ax.legend(fontsize=6)
axes[0][0].legend(fontsize=7, loc='upper right')
plt.suptitle('Feature Distributions (Log-binned Histograms): Benign vs Key Attack Types', fontsize=12)
plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/F2_feature_distributions.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("F2 saved.")

# ── F3: Correlation heatmap ──────────────────────────────────────────
print("Generating F3...")
numeric_feats = [f for f in FEATURE_NAMES if f not in []]
corr = df[numeric_feats].corr(method='spearman').abs()

mask = np.triu(np.ones_like(corr, dtype=bool))
fig, ax = plt.subplots(figsize=(14, 12))

feat_types = []
for f in numeric_feats:
    if f in BINARY_FEATURES:
        feat_types.append('binary')
    elif f in INTEGER_FEATURES:
        feat_types.append('integer')
    else:
        feat_types.append('float')

sns.heatmap(corr, mask=mask, cmap='YlOrRd', vmin=0, vmax=1,
            xticklabels=numeric_feats, yticklabels=numeric_feats,
            ax=ax, square=True)
ax.tick_params(axis='both', labelsize=6)
ax.set_title('Spearman Absolute Correlation Heatmap', fontsize=12)

# Colored sidebar for feature types
type_colors = {'binary': '#3498db', 'integer': '#e74c3c', 'float': '#2ecc71'}
for i, ft in enumerate(feat_types):
    ax.add_patch(plt.Rectangle((-1.5, i), 1, 1, color=type_colors[ft], clip_on=False))
    ax.add_patch(plt.Rectangle((i, len(numeric_feats) + 0.2), 1, 1, color=type_colors[ft], clip_on=False))

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=t) for t, c in type_colors.items()]
ax.legend(handles=legend_elements, loc='upper left', fontsize=8, title='Feature Type')

plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/F3_correlation_heatmap.pdf', dpi=150, bbox_inches='tight')
plt.close()

corr.to_csv('D:/thesis_final/tables/F3_correlation_matrix.csv')
print("F3 saved.")
print("\nF1-F3 complete. Proceeding to F4-F7...")
