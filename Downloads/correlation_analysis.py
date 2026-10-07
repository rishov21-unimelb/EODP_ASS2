import os
import re
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

# run after code.py, this is the remaining preprocesing stuff and then also correlation analysis.

np.random.seed(42)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

BASE = os.environ.get("EODP_BASE", r"C:\Users\shaun\EODP_ASS2")   #change as needed
IN_PATH = os.path.join(BASE, "outputs", "listings_processed.csv")   
OUT_DIR = os.path.join(BASE, "outputs") + os.sep
os.makedirs(OUT_DIR, exist_ok=True)

LOG = []
def log(msg=""):
    print(msg)
    LOG.append(str(msg))

df = pd.read_csv(IN_PATH)
log(f"loaded processed data: {df.shape}")
need = ["price_clean", "accommodates", 'bedrooms_imputed', 'bathrooms_num', 'dist_cbd_km',
        'review_scores_rating', 'price_tier_code', 'room_type', 'property_type_group', 'distance_band']
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns {missing} - rerun code.py")


#correlation analyis

log("\n" + "=" * 70)
log("Correlation analysis (Pearson, Spearman, MI, NMI)")
log("=" * 70)

corr_vars = ['price_clean', 'accommodates', 'bedrooms_imputed', 'bathrooms_num', 'dist_cbd_km', 'review_scores_rating']
log(f"Variable: {corr_vars} (price_clean = continuous proxy for the price-tier target)")
log("Jutification: capacity (accommodates/bedrooms/bathrooms), location (dist_cbd_km), and quality (rating) are the main drivers of tier membership.")

def discretise(s, bins=10):
    if s.nunique() <= bins:
        return pd.factorize(s)[0]
    return pd.qcut(s, q=bins, labels=False, duplicates='drop').astype(int)

def pair_metrics(a, b):
    m = a.notna() & b.notna()
    x, y = a[m], b[m]
    dx, dy = discretise(x), discretise(y)
    return {
        'n': int(m.sum()),
        'pearson': stats.pearsonr(x, y)[0],
        'spearman': stats.spearmanr(x, y)[0],
        'MI': mutual_info_score(dx, dy),
        'NMI': normalized_mutual_info_score(dx, dy)
    }

rows = []
for i, a in enumerate(corr_vars):
    for b in corr_vars[i + 1:]:
        rows.append({'var1': a, 'var2': b, **pair_metrics(df[a], df[b])})
corr_tbl = pd.DataFrame(rows)
log(f"\nAll {len(corr_tbl)} pairs x 4 methods:")
log(corr_tbl.round(3).to_string(index=False))
log("Method: Pearson = linear, Spearman = monotonic (rank), MI/NMi = any dependence, estimated on 10 equal frequency bins "
    "so depend on the binning. MI is unbounded (nats). NMI is scaled to 0-1. MI/NMI have no sign.")

#heatmap matricies
def to_matrix(col):
    M = pd.DataFrame(np.eye(len(corr_vars)) if col in ('pearson', 'spearman') else np.nan, 
                     index=corr_vars, columns=corr_vars)
    for _, r in corr_tbl.iterrows():
        M.loc[r['var1'], r['var2']] = r[col]
        M.loc[r['var2'], r['var1']] = r[col]
    if col == 'NMI':
        for v in corr_vars:
            M.loc[v, v] = 1.0
    return M

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
for ax, (col, vmin, vmax, cmap) in zip(axes.ravel(), [
        ('pearson', -1, 1, 'coolwarm'), ('spearman', -1, 1, 'coolwarm'),
        ('MI', 0, None, 'viridis'), ('NMI', 0, 1, 'viridis')]):
    sns.heatmap(to_matrix(col), annot=True, fmt='.2f', cmap=cmap, vmin=vmin, vmax=vmax, ax=ax,
                cbar_kws={'shrink': .8})
    ax.set_title(col.capitalize() if col in ('pearson', 'spearman') else col)
plt.tight_layout()
plt.savefig(OUT_DIR + 'correlation_heatmaps.png', dpi=150)
plt.close()


#values to be discussed
corr_tbl['abs_gap_pear_spear'] = (corr_tbl['pearson'] - corr_tbl['spearman']).abs()
corr_tbl['rank_pearson'] = corr_tbl['pearson'].abs().rank(ascending=False)
corr_tbl['rank_NMI'] = corr_tbl['NMI'].rank(ascending=False)
corr_tbl['rank_gap'] = (corr_tbl['rank_pearson'] - corr_tbl['rank_NMI']).abs()
log("\nPairs where Pearson - Spearman is largest (skew/outliers/non-linearity):")
log(corr_tbl.nlargest(3, 'abs_gap_pear_spear')[['var1', 'var2', 'pearson', 'spearman']].round(3).to_string(index=False))
log("\nPairs where Pearson-rank and NMI-rank disagree most:")
log(corr_tbl.nlargest(3, 'rank_gap')[['var1', 'var2', 'pearson', 'NMI', 'rank_pearson', 'rank_NMI']].round(3).to_string(index=False))
pred_pairs = corr_tbl[(corr_tbl.var1 != 'price_clean') & (corr_tbl.var2 != 'price_clean')]
hi = pred_pairs[pred_pairs[['pearson', 'spearman']].abs().max(axis=1) > 0.7]
log("\nPredictor-predictor pairs with |r| > 0.7 (Pearson or Spearman):")
log(hi[['var1', 'var2', 'pearson', 'spearman', 'NMI']].round(3).to_string(index=False) if len(hi) else "none")
tgt = corr_tbl[(corr_tbl.var1 == 'price_clean') | (corr_tbl.var2 == 'price_clean')].copy()
tgt['predictor'] = np.where(tgt.var1 == 'price_clean', tgt.var2, tgt.var1)
log("\nTarget - Predictors ranked by Spearman with price_clean:")
log(tgt.reindex(tgt['spearman'].abs().sort_values(ascending=False).index)
    [['predictor', 'pearson', 'spearman', 'MI', 'NMI', 'n']].round(3).to_string(index=False))

log("\nTier target - Predictors vs price_tier_code (0=Budget,1=Mid,2=Premium):")
tier_rows = []
y_code = df['price_tier_code']
for v in ['accommodates', 'bedrooms_imputed', 'bathrooms_num', 'dist_cbd_km', 'review_scores_rating']:
    m = df[v].notna()
    tier_rows.append({'predictor': v, 'type': 'numeric',
                      'spearman': stats.spearmanr(df.loc[m, v], y_code[m])[0],
                      'MI': mutual_info_score(discretise(df.loc[m, v]), y_code[m]),
                      'NMI': normalized_mutual_info_score(discretise(df.loc[m, v]), y_code[m])})
for v in ['room_type', 'property_type_group', 'distance_band']:
    codes = pd.factorize(df[v])[0]
    tier_rows.append({'predictor': v, 'type': 'nominal/ordinal-cat', 'spearman': np.nan,
                      'MI': mutual_info_score(codes, y_code), 'NMI': normalized_mutual_info_score(codes, y_code)})
    
tier_tbl = pd.DataFrame(tier_rows)
log(tier_tbl.round(3).to_string(index=False))
log("Pearson/Spearman are not applicable to nominal predictors (room_type, property_type_group): their category codes have no order. " \
    "Only MI/NMI are reported there.")

corr_tbl.to_csv(OUT_DIR + 'correlation_table.csv', index=False)
tier_tbl.to_csv(OUT_DIR + 'correlation_vs_tier_table.csv', index=False)
with open(OUT_DIR + 'analysis_log_part2.txt', 'w') as f:
    f.write("\n".join(LOG))
log("\nSaved: correlation_heatmaps.png, correlation_table.csv, correlation_vs_tier_table.csv, "
    "analysis_log_part2.txt")