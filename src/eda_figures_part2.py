import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import sys
import os

np.random.seed(42)
sys.path.insert(0, 'D:/thesis_final/src')
from feature_groups import FEATURE_NAMES, CATEGORY_MAP

FIGURES_DIR = 'D:/thesis_final/figures'
TABLES_DIR = 'D:/thesis_final/tables'
PROC_DIR = 'D:/thesis_final/data/processed'
APPENDIX_FIGURES_DIR = f'{FIGURES_DIR}/appendix'

df = pd.read_parquet(f'{PROC_DIR}/raw_loaded.parquet')
df['category'] = df['Label'].map(CATEGORY_MAP)
print(f"Loaded: {df.shape}")

CAT_COLORS = {
    'DDoS': '#e74c3c', 'DoS': '#e67e22', 'Mirai': '#9b59b6',
    'Benign': '#2ecc71', 'Spoofing': '#3498db', 'Recon': '#f1c40f',
    'Web': '#1abc9c', 'BruteForce': '#34495e',
}
CAT_ORDER = ['DDoS', 'DoS', 'Mirai', 'Benign', 'Spoofing', 'Recon', 'Web', 'BruteForce']

X_all = df[FEATURE_NAMES].values

# ── F4: PCA ──────────────────────────────────────────────────────────
print("Generating F4 (PCA)...")
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

scaler_viz = RobustScaler()
X_scaled = scaler_viz.fit_transform(X_all)
X_scaled = np.clip(X_scaled, -10, 10)

pca = PCA(n_components=2, random_state=42)
X_pca_all = pca.fit_transform(X_scaled)

print(f"  Variance explained: PC1={pca.explained_variance_ratio_[0]:.4f}, PC2={pca.explained_variance_ratio_[1]:.4f}")

# Sample for plotting
sample_idx = []
for cls in df['Label'].unique():
    cls_idx = np.where(df['Label'].values == cls)[0]
    n = min(len(cls_idx), 3000)
    chosen = np.random.choice(cls_idx, n, replace=False)
    sample_idx.extend(chosen)
sample_idx = np.array(sample_idx)

fig, ax = plt.subplots(figsize=(12, 10))
for cat in CAT_ORDER:
    mask = df['category'].values[sample_idx] == cat
    if mask.sum() > 0:
        ax.scatter(X_pca_all[sample_idx[mask], 0], X_pca_all[sample_idx[mask], 1],
                   c=CAT_COLORS[cat], label=cat, alpha=0.3, s=3, rasterized=True)
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
ax.set_title('PCA Projection (2D) by Attack Category')
ax.legend(markerscale=5, fontsize=9)
plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/F4a_pca_by_category.pdf', dpi=150, bbox_inches='tight')
plt.close()

# PCA loadings
loadings = pd.DataFrame(pca.components_.T, index=FEATURE_NAMES, columns=['PC1', 'PC2'])
loadings.to_csv(f'{TABLES_DIR}/F4_pca_loadings.csv')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for i, pc in enumerate(['PC1', 'PC2']):
    top = loadings[pc].abs().nlargest(10)
    vals = loadings.loc[top.index, pc]
    colors = ['#e74c3c' if v < 0 else '#2ecc71' for v in vals]
    axes[i].barh(range(len(vals)), vals.values, color=colors)
    axes[i].set_yticks(range(len(vals)))
    axes[i].set_yticklabels(vals.index, fontsize=8)
    axes[i].set_title(f'{pc} Top 10 Loadings ({pca.explained_variance_ratio_[i]:.2%} var)')
    axes[i].axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/F4b_pca_loadings.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("F4 saved.")

# Top 5 loadings per component for logging
for pc in ['PC1', 'PC2']:
    top5 = loadings[pc].abs().nlargest(5)
    print(f"  {pc} top 5: {list(top5.index)}")

# ── F5: t-SNE ────────────────────────────────────────────────────────
print("Generating F5 (t-SNE)... this may take a few minutes")
from sklearn.manifold import TSNE

tsne_idx = []
for cls in df['Label'].unique():
    cls_idx = np.where(df['Label'].values == cls)[0]
    n = min(len(cls_idx), 2000)
    chosen = np.random.choice(cls_idx, n, replace=False)
    tsne_idx.extend(chosen)
tsne_idx = np.array(tsne_idx)
print(f"  t-SNE sample: {len(tsne_idx)} points")

n_pca_components = min(39, 50)
pca50 = PCA(n_components=n_pca_components, random_state=42)
X_pca50 = pca50.fit_transform(X_scaled[tsne_idx])

tsne = TSNE(n_components=2, perplexity=40, learning_rate='auto',
            init='pca', random_state=42, n_jobs=-1)
X_tsne = tsne.fit_transform(X_pca50)
print(f"  t-SNE done. KL divergence: {tsne.kl_divergence_:.4f}")

fig, ax = plt.subplots(figsize=(12, 10))
cats_tsne = df['category'].values[tsne_idx]
for cat in CAT_ORDER:
    mask = cats_tsne == cat
    if mask.sum() > 0:
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=CAT_COLORS[cat], label=cat, alpha=0.4, s=5, rasterized=True)
ax.set_title('t-SNE Projection (2D) by Attack Category')
ax.legend(markerscale=4, fontsize=9)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/F5_tsne_by_category.pdf', dpi=150, bbox_inches='tight')
plt.close()

np.savez(f'{PROC_DIR}/tsne_coords.npz', coords=X_tsne,
         labels=df['Label'].values[tsne_idx],
         categories=cats_tsne)
print("F5 saved.")

# ── F6: UMAP ─────────────────────────────────────────────────────────
print("Generating F6 (UMAP)...")
import umap

umap_idx = []
for cls in df['Label'].unique():
    cls_idx = np.where(df['Label'].values == cls)[0]
    n = min(len(cls_idx), 10000)
    chosen = np.random.choice(cls_idx, n, replace=False)
    umap_idx.extend(chosen)
umap_idx = np.array(umap_idx)
print(f"  UMAP sample: {len(umap_idx)} points")

reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='euclidean',
                     random_state=42, n_jobs=-1)
X_umap = reducer.fit_transform(X_scaled[umap_idx])

fig, ax = plt.subplots(figsize=(12, 10))
cats_umap = df['category'].values[umap_idx]
for cat in CAT_ORDER:
    mask = cats_umap == cat
    if mask.sum() > 0:
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                   c=CAT_COLORS[cat], label=cat, alpha=0.3, s=3, rasterized=True)
ax.set_title('UMAP Projection (2D) by Attack Category')
ax.legend(markerscale=5, fontsize=9)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
plt.tight_layout()
os.makedirs(APPENDIX_FIGURES_DIR, exist_ok=True)
fig.savefig(f'{FIGURES_DIR}/F6_umap_by_category.pdf', dpi=150, bbox_inches='tight')
fig.savefig(f'{APPENDIX_FIGURES_DIR}/F6_umap_by_category.pdf', dpi=150, bbox_inches='tight')
plt.close()

# Keep global UMAP as appendix material; retain F6b in main figures.
if os.path.exists(f'{FIGURES_DIR}/F6_umap_by_category.pdf'):
    os.remove(f'{FIGURES_DIR}/F6_umap_by_category.pdf')

np.savez(f'{PROC_DIR}/umap_coords.npz', coords=X_umap,
         labels=df['Label'].values[umap_idx],
         categories=cats_umap)

# F6b: UMAP zoomed to Benign + Mirai
mirai_benign_mask = np.isin(cats_umap, ['Benign', 'Mirai'])
fig, ax = plt.subplots(figsize=(10, 8))
for cat in ['Benign', 'Mirai']:
    mask = cats_umap == cat
    if mask.sum() > 0:
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                   c=CAT_COLORS[cat], label=cat, alpha=0.4, s=5, rasterized=True)
ax.set_title('UMAP: Benign vs Mirai (zoomed)')
ax.legend(markerscale=4, fontsize=10)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/F6b_umap_mirai_vs_benign.pdf', dpi=150, bbox_inches='tight')
plt.close()
print("F6 saved.")

# ── F7: NetDiffuser feature categorization ───────────────────────────
print("Generating F7 (NetDiffuser)...")
from netdiffuser_categorization import categorize_features
from scipy.cluster.hierarchy import dendrogram

nd_sample = df.sample(n=min(50000, len(df)), random_state=42)
result = categorize_features(nd_sample, FEATURE_NAMES)

print(f"  Discrete: {len(result['discrete'])} features")
print(f"  Relative: {len(result['relative'])} features")
print(f"  Best cut height: {result['best_cut']:.4f}")

with open(f'{PROC_DIR}/netdiffuser_categorization.json', 'w') as f:
    serializable = {k: v for k, v in result.items() if k != 'linkage_matrix'}
    serializable['discrete'] = result['discrete']
    serializable['relative'] = result['relative']
    json.dump(serializable, f, indent=2)

# F7a: Dendrogram
fig, ax = plt.subplots(figsize=(14, 8))
from scipy.cluster.hierarchy import fcluster
labels_nd = fcluster(result['linkage_matrix'], t=result['best_cut'], criterion='distance')
label_sizes = pd.Series(labels_nd).value_counts().to_dict()
leaf_colors = {}
for i, feat in enumerate(FEATURE_NAMES):
    if label_sizes[labels_nd[i]] == 1:
        leaf_colors[feat] = '#2ecc71'
    else:
        leaf_colors[feat] = '#e74c3c'

dend = dendrogram(result['linkage_matrix'], labels=FEATURE_NAMES,
                  leaf_rotation=90, leaf_font_size=7, ax=ax)
ax.axhline(y=result['best_cut'], color='blue', linestyle='--',
           linewidth=1.5, label=f"Cut height = {result['best_cut']:.3f}")
ax.set_title('NetDiffuser Feature Categorization (Hierarchical Clustering)')
ax.set_ylabel('Distance')
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/F7a_dendrogram.pdf', dpi=150, bbox_inches='tight')
plt.close()

# F7b: CH scores vs cut height
fig, ax = plt.subplots(figsize=(10, 6))
h_grid = result['h_grid']
ch_scores = result['ch_scores']
ax.plot(h_grid, ch_scores, 'b-o', markersize=4)
ax.axvline(x=result['best_cut'], color='red', linestyle='--',
           label=f"Best cut = {result['best_cut']:.3f}")
ax.set_xlabel('Cut Height')
ax.set_ylabel('Calinski-Harabasz Score')
ax.set_title('CH Score vs Cut Height')
ax.legend()
plt.tight_layout()
fig.savefig(f'{FIGURES_DIR}/F7b_ch_scores.pdf', dpi=150, bbox_inches='tight')
plt.close()

# T6: Feature categorization table
with open(f'{TABLES_DIR}/T6_feature_categorization.md', 'w') as f:
    f.write("# T6 - NetDiffuser Feature Categorization\n\n")
    f.write("| Discrete Features | Relative Features |\n")
    f.write("|---|---|\n")
    max_len = max(len(result['discrete']), len(result['relative']))
    for i in range(max_len):
        d = result['discrete'][i] if i < len(result['discrete']) else ''
        r = result['relative'][i] if i < len(result['relative']) else ''
        f.write(f"| {d} | {r} |\n")

# Also save as CSV
t6_data = {'discrete': result['discrete'], 'relative': result['relative']}
max_len = max(len(t6_data['discrete']), len(t6_data['relative']))
t6_rows = []
for i in range(max_len):
    t6_rows.append({
        'discrete': result['discrete'][i] if i < len(result['discrete']) else '',
        'relative': result['relative'][i] if i < len(result['relative']) else '',
    })
pd.DataFrame(t6_rows).to_csv(f'{TABLES_DIR}/T6_feature_categorization.csv', index=False)
print("F7 saved.")

# ── Final EDA sanity print ───────────────────────────────────────────
with open(f'{PROC_DIR}/clean_data_validity_report.json') as f:
    validity = json.load(f)

print("\n" + "=" * 50)
print("         EDA COMPLETE")
print("=" * 50)
print(f"Schema:       Modified Schema A (39 features)")
print(f"Rows used:    {len(df):,}  (Mode: SAMPLE)")
print(f"Figures generated: 7 (F1, F2, F3, F4a/b, F5, F6a/b, F7a/b)")
print(f"Tables generated:  6 (T1, T2a/b, T3, T4, T5, T6)")
print(f"Clean data validity: {validity['validity_rate']:.4%} overall (see T4)")
print(f"NetDiffuser: {len(result['discrete'])} Discrete, {len(result['relative'])} Relative features")
print(f"t-SNE sample: {len(tsne_idx)} points | UMAP sample: {len(umap_idx)} points")
print("=" * 50)
