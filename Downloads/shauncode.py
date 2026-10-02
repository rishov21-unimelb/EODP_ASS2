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
        'mi': mutual_info_score(dx, dy),
        'nmi': normalized_mutual_info_score(dx, dy)
    }

rows = []
for i, a in enumerate(corr_vars):
    for b in corr_vars[i + 1:]:
        rows.append({'var1': a, 'var2': b, **pair_metrics(df[a], df[b])})
corr_tbl = pd.DataFrame(rows)
log(f"\nAll {len(corr_tbl)} pairs x 4 methods:")
log(corr_tbl.round(3).to_string(index=False))
log("Method: Pearson = linear, Spearman = monotonic (rank), MI/NMi = any dependence, estimated on 10 equal frequency bins"
    "so depend on the binning. MI is unbounded (nats). NMI is scaled to 0-1. MI/NMI have no sign.")

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
