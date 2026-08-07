import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import calinski_harabasz_score


def categorize_features(df: pd.DataFrame, feature_cols: list,
                        method='spearman', h_grid_points=100):
    """
    NetDiffuser Algorithm 1: partition features into Discrete vs Relative.
    """
    corr = df[feature_cols].corr(method=method).abs().values.copy()  # .copy(): numpy 2.x returns a read-only view; fill_diagonal needs it writable
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.maximum(2 * (1 - corr), 0.0))

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method='average')

    # Constrained search region requested for publication stability.
    h_grid = np.linspace(0.1, 1.0, h_grid_points)

    ch_scores = []
    nontrivial_flags = []
    features_as_rows = df[feature_cols].values.T
    for h in h_grid:
        labels = fcluster(Z, t=h, criterion='distance')
        counts = pd.Series(labels).value_counts()
        n_nontrivial_clusters = int((counts >= 2).sum())
        is_nontrivial = n_nontrivial_clusters >= 3
        nontrivial_flags.append(is_nontrivial)

        if len(set(labels)) < 2 or len(set(labels)) >= len(feature_cols):
            ch_scores.append(np.nan)
        else:
            try:
                ch = calinski_harabasz_score(features_as_rows, labels)
            except Exception:
                ch = np.nan
            ch_scores.append(ch)
    ch_scores = np.array(ch_scores)
    nontrivial_flags = np.array(nontrivial_flags, dtype=bool)

    # Prefer local maxima satisfying the non-trivial clustering constraint.
    local_maxima = []
    for i in range(1, len(ch_scores) - 1):
        if np.isnan(ch_scores[i - 1]) or np.isnan(ch_scores[i]) or np.isnan(ch_scores[i + 1]):
            continue
        if ch_scores[i] >= ch_scores[i - 1] and ch_scores[i] >= ch_scores[i + 1] and nontrivial_flags[i]:
            local_maxima.append(i)

    if local_maxima:
        best_i = max(local_maxima, key=lambda i: ch_scores[i])
    else:
        valid_nontrivial = np.where(nontrivial_flags & ~np.isnan(ch_scores))[0]
        if len(valid_nontrivial) > 0:
            best_i = int(valid_nontrivial[np.nanargmax(ch_scores[valid_nontrivial])])
        else:
            valid_any = np.where(~np.isnan(ch_scores))[0]
            best_i = int(valid_any[np.nanargmax(ch_scores[valid_any])])

    best_h = float(h_grid[best_i])
    labels = fcluster(Z, t=best_h, criterion='distance')

    label_sizes = pd.Series(labels).value_counts().to_dict()
    relative = [feature_cols[i] for i, L in enumerate(labels) if label_sizes[L] > 1]
    discrete = [feature_cols[i] for i, L in enumerate(labels) if label_sizes[L] == 1]

    return {
        'discrete': discrete,
        'relative': relative,
        'linkage_matrix': Z,
        'best_cut': best_h,
        'best_index': int(best_i),
        'best_is_nontrivial': bool(nontrivial_flags[best_i]),
        'ch_scores': ch_scores.tolist(),
        'nontrivial_flags': nontrivial_flags.tolist(),
        'h_grid': h_grid.tolist(),
        'correlation_matrix': corr.tolist(),
    }
