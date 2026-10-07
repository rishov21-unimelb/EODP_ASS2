import os
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

np.random.seed(42)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
 
BASE = os.environ.get("EODP_BASE", r"C:\Users\shaun\EODP_ASS2")   # change as needed
IN_PATH = os.path.join(BASE, "outputs", "listings_processed.csv")
OUT_DIR = os.path.join(BASE, "outputs") + os.sep
os.makedirs(OUT_DIR, exist_ok=True)
 
LOG = []
def log(msg=""):
    print(msg)
    LOG.append(str(msg))
 
df = pd.read_csv(IN_PATH, low_memory=False)
log(f"Loaded processed data: {df.shape}")
need = ['id', 'price_clean', 'price_tier', 'accommodates', 'bedrooms_imputed', 'beds_imputed', 'bathrooms_num',
        'dist_cbd_km', 'review_scores_rating_imp', 'has_rating', 'minimum_nights', 'number_of_reviews',
        'host_listings_count', 'availability_365', 'host_is_superhost', 'property_type_group']
missing = [c for c in need if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns {missing} - rerun Code.py")
 
def discretise(s, bins=10):
    """Equal-frequency bins (or raw values if <= `bins` unique) for MI/NMI - same as the correlation script."""
    if s.nunique() <= bins:
        return pd.factorize(s)[0]
    return pd.qcut(s, q=bins, labels=False, duplicates='drop').astype(int)


#feature selection
#filtering MI vss embedded decision tree importance
log("\n" + "=" * 70)
log("Feature selection (filter: MI/NMI, embedded: Decision Tree importance)")
log("=" * 70)
 
# encoding and split as supervised_ml_code.py so the numbers agree across sections
fs = df.copy()
fs['superhost_flag'] = (fs['host_is_superhost'] == 't').astype(int)
fs['log_minimum_nights'] = np.log1p(fs['minimum_nights'])
fs['log_number_of_reviews'] = np.log1p(fs['number_of_reviews'])
fs['log_host_listings_count'] = np.log1p(fs['host_listings_count'])
NUM_COLS = ['accommodates', 'bedrooms_imputed', 'beds_imputed', 'bathrooms_num',
            'dist_cbd_km', 'review_scores_rating_imp', 'has_rating',
            'log_number_of_reviews', 'log_minimum_nights', 'log_host_listings_count',
            'availability_365', 'superhost_flag']
dummies = pd.get_dummies(fs['property_type_group'], prefix='ptype').astype(int)
DUMMY_COLS = list(dummies.columns)
fs = pd.concat([fs, dummies], axis=1)
X_COLS = NUM_COLS + DUMMY_COLS
FEATURE_COLS = {c: [c] for c in NUM_COLS}

#7 one hot cols are scored as a single feature for MI/NMI, but individually for the decision tree. 
#So we add a single entry for the group of 7 one-hot columns.
FEATURE_COLS['property_type_group'] = DUMMY_COLS    

X_all, y_all = fs[X_COLS], fs['price_tier']
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all, random_state=42)
mn_median = X_tr['log_minimum_nights'].median()        # training median only, no test leakage
X_tr = X_tr.fillna({'log_minimum_nights': mn_median})
X_all = X_all.fillna({'log_minimum_nights': mn_median})
log(f"Feature selection uses the TRAINING split only (n={len(X_tr)} of {len(X_all)}), "
    f"so the test set plays no part in ranking features.")
log(f"Feature pool: {len(NUM_COLS)} numeric + property_type_group ({len(DUMMY_COLS)} one-hot columns) "
    f"= {len(FEATURE_COLS)} features. price_clean is excluded (it defines the target).")

#filter MI of each feature with price tier
filter_rows = []
for f in NUM_COLS:
    dx = discretise(X_tr[f])
    filter_rows.append({'feature': f, 'MI': mutual_info_score(dx, y_tr),
                        'NMI': normalized_mutual_info_score(dx, y_tr)})
dx = discretise(fs.loc[X_tr.index, 'property_type_group'])
filter_rows.append({'feature': 'property_type_group', 'MI': mutual_info_score(dx, y_tr),
                    'NMI': normalized_mutual_info_score(dx, y_tr)})
filter_tbl = pd.DataFrame(filter_rows).sort_values('MI', ascending=False).reset_index(drop=True)
filter_tbl['filter_rank'] = np.arange(1, len(filter_tbl) + 1)

#embedded decision tree 
res_path = OUT_DIR + 'supervised_results.json'
sl_results = None
if os.path.exists(res_path):
    with open(res_path) as f:
        sl_results = json.load(f)
    dt_params = sl_results['dt_params']
    log(f"Tree settings read from supervised_results.json: {dt_params}")
else:
    log(f"supervised_results.json not found, run supervised_ml_code.py first to get the decision tree parameters.")

tree = DecisionTreeClassifier(random_state=42, **dt_params).fit(X_tr[X_COLS], y_tr)
imp = pd.Series(tree.feature_importances_, index=X_COLS)
emb = imp[NUM_COLS].copy()
emb['property_type_group'] = imp[DUMMY_COLS].sum()
embedded_tbl = emb.sort_values(ascending=False).rename('tree_importance').rename_axis('feature').reset_index()
embedded_tbl['embedded_rank'] = np.arange(1, len(embedded_tbl) + 1)
if sl_results is not None and 'dt_feature_importance' in sl_results:
    theirs = pd.Series(sl_results['dt_feature_importance'])
    diff = (emb.reindex(theirs.index) - theirs).abs().max()
    log(f"Check vs supervised section: max |importance difference| = {diff:.6f} (should be ~0)")


merged = filter_tbl.merge(embedded_tbl, on='feature')
log("\nFull ranking (filter MI vs embedded tree importance):")
log(merged.round(4).to_string(index=False))
top_f = filter_tbl['feature'].head(3).tolist()
top_e = embedded_tbl['feature'].head(3).tolist()
log(f"\nTOP-3 FILTER (MI):        {top_f}")
log(f"TOP-3 EMBEDDED (DTree):   {top_e}")
log(f"Overlap: {sorted(set(top_f) & set(top_e))}; only filter: {sorted(set(top_f) - set(top_e))}; "
    f"only embedded: {sorted(set(top_e) - set(top_f))}")
for f in sorted(set(top_f) ^ set(top_e)):
    r = merged[merged.feature == f].iloc[0]
    log(f"[Disagreement] {f}: filter rank {int(r.filter_rank)} (MI={r.MI:.3f}, NMI={r.NMI:.3f}) vs "
        f"embedded rank {int(r.embedded_rank)} (importance={r.tree_importance:.3f})")
merged['rank_gap'] = merged['embedded_rank'] - merged['filter_rank']
log("\nFeatures the filter ranks much higher than the tree:")
log(merged.nlargest(2, 'rank_gap')[['feature', 'filter_rank', 'embedded_rank', 'MI', 'tree_importance']].round(3).to_string(index=False))
log("Features the tree ranks much higher than the filter:")
log(merged.nsmallest(2, 'rank_gap')[['feature', 'filter_rank', 'embedded_rank', 'MI', 'tree_importance']].round(3).to_string(index=False))
 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
filter_tbl.sort_values('MI').plot.barh(x='feature', y='MI', ax=axes[0], color='#4C72B0', legend=False)
axes[0].set_title('Filter: mutual information with price tier')
embedded_tbl.sort_values('tree_importance').plot.barh(x='feature', y='tree_importance', ax=axes[1],
                                                      color="#557CA8", legend=False)

axes[1].set_title('Embedded: Decision Tree importance')
plt.tight_layout()
plt.savefig(OUT_DIR + 'feature_selection.png', dpi=150)
plt.close()