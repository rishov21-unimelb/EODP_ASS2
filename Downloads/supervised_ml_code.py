import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay)
 
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
 
BASE = os.environ.get("EODP_BASE", r"C:\Users\shaun\EODP_ASS2")   # change for your own path
IN_PATH = os.path.join(BASE, "outputs", "listings_processed.csv")
OUT_DIR = os.path.join(BASE, "outputs") + os.sep
os.makedirs(OUT_DIR, exist_ok=True)
 
LOG = []
def log(msg=""):
    print(msg)
    LOG.append(str(msg))
 
RESULTS = {}   # every number quoted in the report -> supervised_results.json
TIER_LABELS = ['Budget', 'Mid', 'Premium']
 
# Target and features
# Target: price_tier (Budget / Mid / Premium tertiles of price)
df = pd.read_csv(IN_PATH, low_memory=False)
log(f"Loaded processed data: {df.shape}")
 
# superhost is stored as 't'/'f' -> 1/0
df['superhost_flag'] = (df['host_is_superhost'] == 't').astype(int)
 
# rescaling: log transform for heavily right-skewed counts
df['log_minimum_nights'] = np.log1p(df['minimum_nights'])
df['log_number_of_reviews'] = np.log1p(df['number_of_reviews'])
df['log_host_listings_count'] = np.log1p(df['host_listings_count'])
 
NUM_COLS = ['accommodates', 'bedrooms_imputed', 'beds_imputed', 'bathrooms_num',
            'dist_cbd_km', 'review_scores_rating_imp', 'has_rating',
            'log_number_of_reviews', 'log_minimum_nights', 'log_host_listings_count',
            'availability_365', 'superhost_flag']
 
# one-hot encoding for the nominal property_type_group
dummies = pd.get_dummies(df['property_type_group'], prefix='ptype').astype(int)
DUMMY_COLS = list(dummies.columns)
df = pd.concat([df, dummies], axis=1)
 
X_COLS = NUM_COLS + DUMMY_COLS
y_COL = 'price_tier'
 
# instant_bookable is 100% missing in this scrape, so instant_bookable_flag is always 0 -> not used
log(f"instant_bookable non-missing values: {df['instant_bookable'].notna().sum()} -> flag not used")
log(f"Predictors: {len(NUM_COLS)} numeric + {len(DUMMY_COLS)} one-hot property-type columns")
 
X = df[X_COLS]
y = df[y_COL]
 
class_counts = y.value_counts().reindex(TIER_LABELS)
log(f"\nTarget class counts (n={len(df)}):\n{class_counts.to_string()}")
RESULTS['n_total'] = int(len(df))
RESULTS['class_counts'] = class_counts.astype(int).to_dict()
 
# Train/test split: stratified 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                    random_state=RANDOM_STATE)
 
# minimum_nights has 9 missing values, fill with the TRAINING median so the test set is not used
train_median_mn = X_train['log_minimum_nights'].median()
X_train = X_train.fillna({'log_minimum_nights': train_median_mn})
X_test = X_test.fillna({'log_minimum_nights': train_median_mn})
 
train_share = (y_train.value_counts(normalize=True).reindex(TIER_LABELS) * 100).round(2)
test_share = (y_test.value_counts(normalize=True).reindex(TIER_LABELS) * 100).round(2)
_, _, _, y_test_random = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
random_share = (y_test_random.value_counts(normalize=True).reindex(TIER_LABELS) * 100).round(2)
 
log(f"\nTrain n={len(X_train)}, test n={len(X_test)}")
log(f"Train tier shares (%):  {train_share.to_dict()}")
log(f"Test tier shares (%):   {test_share.to_dict()}")
log(f"Unstratified split test shares (%), same seed: {random_share.to_dict()}")
RESULTS['split'] = {'n_train': int(len(X_train)), 'n_test': int(len(X_test)),
                    'train_share_pct': train_share.to_dict(), 'test_share_pct': test_share.to_dict(),
                    'unstratified_test_share_pct': random_share.to_dict()}
 
# Helper functions

# builds a KNN or Decision Tree from a dict of hyperparameters
def make_model(name, params):
    if name == 'knn':
        return KNeighborsClassifier(**params)
    return DecisionTreeClassifier(random_state=RANDOM_STATE, **params)

# KNN is distance-based, so features are standardised.
def fit_and_predict(name, params, X_tr, y_tr, X_ev, cols=None):
    cols = X_COLS if cols is None else cols
    X_tr, X_ev = X_tr[cols], X_ev[cols]
    if name == 'knn':
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_ev = scaler.transform(X_tr), scaler.transform(X_ev)
    model = make_model(name, params)
    model.fit(X_tr, y_tr)
    return model, model.predict(X_ev)
 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# 5-fold stratified CV on the TRAINING set
def cv_macro_f1(name, params, cols=None):
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        _, y_pred = fit_and_predict(name, params, X_tr, y_tr, X_val, cols)
        scores.append(f1_score(y_val, y_pred, average='macro'))
    return np.mean(scores), np.std(scores)
 

# Hyperparameter tuning (5-fold stratified CV, metric = macro-F1)
K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51, 75, 101]
WEIGHTS = ['uniform', 'distance']
DEPTHS = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None]
LEAVES = [1, 5, 10, 20, 50, 100]
 
knn_rows = []
for k in K_VALUES:
    for w in WEIGHTS:
        mean, std = cv_macro_f1('knn', {'n_neighbors': k, 'weights': w})
        knn_rows.append({'n_neighbors': k, 'weights': w, 'cv_macro_f1': mean, 'cv_std': std})
knn_cv = pd.DataFrame(knn_rows)
knn_cv.round(4).to_csv(OUT_DIR + 'tuning_knn_all_values.csv', index=False)
 
dt_rows = []
for d in DEPTHS:
    for leaf in LEAVES:
        params = {'criterion': 'entropy', 'max_depth': d, 'min_samples_leaf': leaf}
        mean, std = cv_macro_f1('dt', params)
        dt_rows.append({'max_depth': 'None' if d is None else d, 'min_samples_leaf': leaf,
                        'cv_macro_f1': mean, 'cv_std': std})
dt_cv = pd.DataFrame(dt_rows)
dt_cv.round(4).to_csv(OUT_DIR + 'tuning_dt_all_values.csv', index=False)
 
# every value tried, as tables for the report
log("\n[KNN] CV macro-F1 at every value tried (rows = k, columns = weights):")
log(knn_cv.groupby(['n_neighbors', 'weights'])['cv_macro_f1'].mean().unstack().round(4).to_string())
log("\n[DT] CV macro-F1 at every value tried (rows = max_depth, columns = min_samples_leaf):")
depth_order = [str(d) for d in DEPTHS]
dt_table = dt_cv.assign(max_depth=dt_cv['max_depth'].astype(str)).groupby(
    ['max_depth', 'min_samples_leaf'])['cv_macro_f1'].mean().unstack().reindex(depth_order)
log(dt_table.round(4).to_string())
 
best_knn_row = knn_cv.loc[knn_cv['cv_macro_f1'].idxmax()]
best_dt_row = dt_cv.loc[dt_cv['cv_macro_f1'].idxmax()]
KNN_PARAMS = {'n_neighbors': int(best_knn_row['n_neighbors']), 'weights': best_knn_row['weights']}
best_depth = None if best_dt_row['max_depth'] == 'None' else int(best_dt_row['max_depth'])
DT_PARAMS = {'criterion': 'entropy', 'max_depth': best_depth,
             'min_samples_leaf': int(best_dt_row['min_samples_leaf'])}
log(f"\n[KNN] chosen: {KNN_PARAMS}, CV macro-F1 = {best_knn_row['cv_macro_f1']:.4f} "
    f"(fold std {best_knn_row['cv_std']:.4f})")
log(f"[DT]  chosen: {DT_PARAMS}, CV macro-F1 = {best_dt_row['cv_macro_f1']:.4f} "
    f"(fold std {best_dt_row['cv_std']:.4f})")
RESULTS['knn_params'] = KNN_PARAMS
RESULTS['knn_cv_macro_f1'] = round(float(best_knn_row['cv_macro_f1']), 4)
RESULTS['dt_params'] = DT_PARAMS
RESULTS['dt_cv_macro_f1'] = round(float(best_dt_row['cv_macro_f1']), 4)
 

# Validation curves (figure for the report)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for w, marker in [('uniform', 'o'), ('distance', 's')]:
    sub = knn_cv[knn_cv['weights'] == w]
    axes[0].plot(sub['n_neighbors'], sub['cv_macro_f1'], marker=marker, label=f'weights={w}')
axes[0].axvline(KNN_PARAMS['n_neighbors'], color='grey', linestyle=':',
                label=f"chosen k={KNN_PARAMS['n_neighbors']}")
axes[0].set_xscale('log')
axes[0].set_xlabel('n_neighbors k (log scale)')
axes[0].set_ylabel('Macro-F1 (5-fold stratified CV)')
axes[0].set_title('KNN: CV macro-F1 vs k')
axes[0].legend(fontsize=8)
 
for leaf in LEAVES:
    s = dt_table[leaf]
    axes[1].plot(depth_order, s.values, marker='o', label=f'min_samples_leaf={leaf}')
axes[1].set_xlabel('max_depth')
axes[1].set_ylabel('Macro-F1 (5-fold stratified CV)')
axes[1].set_title('Decision Tree (entropy): CV macro-F1 vs max_depth')
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR + 'sl_validation_curves.png', dpi=150)
plt.close()
 

# Effect of every hyperparameter changed from its sklearn default
# For each one, CV macro-F1 with ONLY that hyperparameter put back to its default
# (the others stay at the chosen values), compared with the chosen value
 
def knn_cv_at(k, w):
    return float(knn_cv[(knn_cv['n_neighbors'] == k) & (knn_cv['weights'] == w)]['cv_macro_f1'].iloc[0])
 
def dt_cv_at(d, leaf):
    d = 'None' if d is None else d
    return float(dt_cv[(dt_cv['max_depth'] == d) & (dt_cv['min_samples_leaf'] == leaf)]['cv_macro_f1'].iloc[0])
 
k_best, w_best = KNN_PARAMS['n_neighbors'], KNN_PARAMS['weights']
d_best, leaf_best = DT_PARAMS['max_depth'], DT_PARAMS['min_samples_leaf']
gini_score, _ = cv_macro_f1('dt', {'criterion': 'gini', 'max_depth': d_best, 'min_samples_leaf': leaf_best})
 
hp_effects = [
    {'model': 'KNN', 'hyperparameter': 'n_neighbors', 'default': '5', 'chosen': str(k_best),
     'cv_f1_default': knn_cv_at(5, w_best), 'cv_f1_chosen': knn_cv_at(k_best, w_best)},
    {'model': 'KNN', 'hyperparameter': 'weights', 'default': 'uniform', 'chosen': w_best,
     'cv_f1_default': knn_cv_at(k_best, 'uniform'), 'cv_f1_chosen': knn_cv_at(k_best, w_best)},
    {'model': 'DT', 'hyperparameter': 'criterion', 'default': 'gini', 'chosen': 'entropy',
     'cv_f1_default': gini_score, 'cv_f1_chosen': dt_cv_at(d_best, leaf_best)},
    {'model': 'DT', 'hyperparameter': 'max_depth', 'default': 'None', 'chosen': str(d_best),
     'cv_f1_default': dt_cv_at(None, leaf_best), 'cv_f1_chosen': dt_cv_at(d_best, leaf_best)},
    {'model': 'DT', 'hyperparameter': 'min_samples_leaf', 'default': '1', 'chosen': str(leaf_best),
     'cv_f1_default': dt_cv_at(d_best, 1), 'cv_f1_chosen': dt_cv_at(d_best, leaf_best)},
]
hp_df = pd.DataFrame(hp_effects)
hp_df = hp_df[hp_df['default'] != hp_df['chosen']]          # only report ones actually changed
hp_df['change'] = hp_df['cv_f1_chosen'] - hp_df['cv_f1_default']
log("\nEffect of each hyperparameter changed from its default (CV macro-F1):")
log(hp_df.round(4).to_string(index=False))
RESULTS['hyperparameter_effects'] = hp_df.round(4).to_dict(orient='records')


# Final models: fit on the whole training set, evaluate ONCE on the test set
# Metrics:
#  - macro-F1
#  - per-class precision, recall and F1
#  - accuracy
# Baseline: 0R (Zero-R) = always predict the most common tier in the training set
 
knn_model, knn_pred = fit_and_predict('knn', KNN_PARAMS, X_train, y_train, X_test)
dt_model, dt_pred = fit_and_predict('dt', DT_PARAMS, X_train, y_train, X_test)
 
majority_tier = y_train.value_counts().idxmax()
zero_r_pred = np.array([majority_tier] * len(y_test))
log(f"\n0R baseline predicts '{majority_tier}' for every listing "
    f"(training counts: {y_train.value_counts().reindex(TIER_LABELS).to_dict()})")
 
def evaluate(name, y_true, y_pred):
    row = {'model': name,
           'accuracy': accuracy_score(y_true, y_pred),
           'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0)}
    p = precision_score(y_true, y_pred, labels=TIER_LABELS, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=TIER_LABELS, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, labels=TIER_LABELS, average=None, zero_division=0)
    for i, t in enumerate(TIER_LABELS):
        row[f'{t}_precision'], row[f'{t}_recall'], row[f'{t}_f1'] = p[i], r[i], f[i]
    return row
 
preds = {'0R baseline': zero_r_pred, 'KNN (tuned)': knn_pred, 'DT (tuned)': dt_pred}
metrics = pd.DataFrame([evaluate(n, y_test, p) for n, p in preds.items()]).set_index('model')
metrics.round(4).to_csv(OUT_DIR + 'sl_test_metrics.csv')
 
log(f"\nTEST-SET RESULTS (n={len(y_test)}):")
log(metrics[['accuracy', 'macro_f1']].round(4).to_string())
for t in TIER_LABELS:
    log(f"\n{t}: precision / recall / F1")
    log(metrics[[f'{t}_precision', f'{t}_recall', f'{t}_f1']].round(4).to_string())
RESULTS['test_metrics'] = metrics.round(4).reset_index().to_dict(orient='records')
 
# overfitting check: training score vs CV vs test
train_f1_dt = f1_score(y_train, dt_model.predict(X_train), average='macro')
gap = {'DT': {'train': round(float(train_f1_dt), 4), 'cv': RESULTS['dt_cv_macro_f1'],
              'test': round(float(metrics.loc['DT (tuned)', 'macro_f1']), 4)},
       'KNN': {'cv': RESULTS['knn_cv_macro_f1'], 'test': round(float(metrics.loc['KNN (tuned)', 'macro_f1']), 4)}}
log(f"\nTrain / CV / test macro-F1: {gap}")
log("(KNN train score is not reported: with weights='distance' every training listing is its own "
    "nearest neighbour at distance 0, so it always scores 1.0 on training data.)")
RESULTS['train_cv_test'] = gap
 
# improvement over 0R in absolute terms (percentage points)
improve = []
for m in ['KNN (tuned)', 'DT (tuned)']:
    improve.append({'model': m,
                    'macro_f1_gain_pts': round((metrics.loc[m, 'macro_f1'] - metrics.loc['0R baseline', 'macro_f1']) * 100, 2),
                    'accuracy_gain_pts': round((metrics.loc[m, 'accuracy'] - metrics.loc['0R baseline', 'accuracy']) * 100, 2)})
log("\nAbsolute improvement over 0R (percentage points):")
log(pd.DataFrame(improve).to_string(index=False))
RESULTS['improvement_over_0R'] = improve
 

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
for ax, name in zip(axes, ['KNN (tuned)', 'DT (tuned)']):
    cm = confusion_matrix(y_test, preds[name], labels=TIER_LABELS)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=TIER_LABELS).plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"{name}, macro-F1 = {metrics.loc[name, 'macro_f1']:.3f}")
    RESULTS[f'confusion_matrix_{name}'] = cm.tolist()
plt.tight_layout()
plt.savefig(OUT_DIR + 'sl_confusion_matrices.png', dpi=150)
plt.close()
 
# Uncertainty: bootstrap with out-of-bag testing
# draw b bootstrap samples from the TRAINING data (same size, with replacement); train on each; evaluate on its OOB rows (~37% of the data); 
# report the mean and spread of the b scores
B = 200
rng = np.random.default_rng(RANDOM_STATE)
n_train = len(X_train)
boot_rows = []
for b in range(B):
    in_bag = rng.integers(0, n_train, size=n_train)            # sample row positions with replacement
    oob = np.setdiff1d(np.arange(n_train), in_bag)             # rows never drawn = OOB test rows
    X_bs, y_bs = X_train.iloc[in_bag], y_train.iloc[in_bag]
    X_oob, y_oob = X_train.iloc[oob], y_train.iloc[oob]
    _, p_knn = fit_and_predict('knn', KNN_PARAMS, X_bs, y_bs, X_oob)
    _, p_dt = fit_and_predict('dt', DT_PARAMS, X_bs, y_bs, X_oob)
    f_knn = f1_score(y_oob, p_knn, average='macro')
    f_dt = f1_score(y_oob, p_dt, average='macro')
    boot_rows.append({'knn': f_knn, 'dt': f_dt, 'oob_share': len(oob) / n_train})
boot = pd.DataFrame(boot_rows)
boot.round(4).to_csv(OUT_DIR + 'sl_bootstrap_scores.csv', index=False)
 
log(f"\nBootstrap ({B} samples of the training set, scored on OOB rows; "
    f"mean OOB share {boot['oob_share'].mean() * 100:.1f}%):")
ci = {}
for col, label in [('knn', 'KNN (tuned)'), ('dt', 'DT (tuned)')]:
    lo, hi = np.percentile(boot[col], [2.5, 97.5])
    ci[label] = {'mean': round(boot[col].mean(), 4), 'std': round(boot[col].std(), 4),
                 'ci95_low': round(lo, 4), 'ci95_high': round(hi, 4)}
    log(f"  {label}: mean {ci[label]['mean']:.4f}, std {ci[label]['std']:.4f}, "
        f"95% interval [{lo:.4f}, {hi:.4f}]")
RESULTS['bootstrap_oob_macro_f1'] = ci
 
# Feature influence
# (a) Decision Tree: built-in feature importance = total entropy reduction contributed by each feature's splits. The 7 one-hot columns are added back
#     together into one property_type_group value.
# (b) Both models: drop one feature at a time, re-run the same 5-fold CV, and measure how much
#     macro-F1 falls. KNN has no built-in importance, so this is the only way to compare the two
#     models on the same footing. property_type_group's 7 one-hot columns are dropped together.
 
dt_imp = pd.Series(dt_model.feature_importances_, index=X_COLS)
dt_imp_grouped = dt_imp[NUM_COLS].copy()
dt_imp_grouped['property_type_group'] = dt_imp[DUMMY_COLS].sum()
dt_imp_grouped = dt_imp_grouped.sort_values(ascending=False)
log("\n[DT] built-in feature importance (information gain share, sums to 1):")
log(dt_imp_grouped.round(4).to_string())
RESULTS['dt_feature_importance'] = dt_imp_grouped.round(4).to_dict()
 
FEATURE_GROUPS = {c: [c] for c in NUM_COLS}
FEATURE_GROUPS['property_type_group'] = DUMMY_COLS
drop_rows = []
for feat, cols_to_drop in FEATURE_GROUPS.items():
    keep = [c for c in X_COLS if c not in cols_to_drop]
    knn_drop, _ = cv_macro_f1('knn', KNN_PARAMS, keep)
    dt_drop, _ = cv_macro_f1('dt', DT_PARAMS, keep)
    drop_rows.append({'feature': feat,
                      'knn_f1_drop': best_knn_row['cv_macro_f1'] - knn_drop,   # unrounded, so it
                      'dt_f1_drop': best_dt_row['cv_macro_f1'] - dt_drop})     # matches section 9
drop_df = pd.DataFrame(drop_rows).set_index('feature').sort_values('dt_f1_drop', ascending=False)
drop_df['knn_rank'] = drop_df['knn_f1_drop'].rank(ascending=False).astype(int)
drop_df['dt_rank'] = drop_df['dt_f1_drop'].rank(ascending=False).astype(int)
drop_df.round(4).to_csv(OUT_DIR + 'sl_drop_one_feature.csv')
log("\nDrop-one-feature: fall in CV macro-F1 when the feature is removed (bigger = more relied on):")
log(drop_df.round(4).to_string())
RESULTS['drop_one_feature'] = drop_df.round(4).reset_index().to_dict(orient='records')
 
fig, ax = plt.subplots(figsize=(9, 5.5))
order = drop_df.index[::-1]
ypos = np.arange(len(order))
ax.barh(ypos - 0.2, drop_df.loc[order, 'knn_f1_drop'], height=0.4, label='KNN (tuned)')
ax.barh(ypos + 0.2, drop_df.loc[order, 'dt_f1_drop'], height=0.4, label='DT (tuned)')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_yticks(ypos)
ax.set_yticklabels(order)
ax.set_xlabel('Fall in 5-fold CV macro-F1 when the feature is removed')
ax.set_title('Feature influence: drop-one-feature comparison')
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR + 'sl_drop_one_feature.png', dpi=150)
plt.close()
 
# Feature-set comparison
# Same tuned settings, same 5-fold stratified CV, only the feature set changes.
#  - distance_band rows: preprocessing task 1 discretised distance into Inner/Middle/Outer bands.
#    Compare the bands (one-hot) with continuous dist_cbd_km and with no location at all,
#    to measure that step's effect on model performance.
#  - drop rows: the correlation analysis found accommodates and bedrooms_imputed strongly
#    related (Pearson 0.850), and beds_imputed carries the same size information. Test whether
#    removing one of the redundant size features changes performance.
 
band_dummies = pd.get_dummies(df['distance_band'], prefix='band').astype(int)
BAND_COLS = list(band_dummies.columns)
X_train = X_train.join(band_dummies.loc[X_train.index])     # extra columns, only used when listed below
 
FEATURE_SETS = {
    'current (continuous dist_cbd_km)': X_COLS,
    'distance_band instead of dist_cbd_km': [c for c in X_COLS if c != 'dist_cbd_km'] + BAND_COLS,
    'no location feature': [c for c in X_COLS if c != 'dist_cbd_km'],
    'drop beds_imputed': [c for c in X_COLS if c != 'beds_imputed'],
    'drop accommodates': [c for c in X_COLS if c != 'accommodates'],
}
variant_rows = []
for name, cols in FEATURE_SETS.items():
    knn_f1, knn_sd = cv_macro_f1('knn', KNN_PARAMS, cols)
    dt_f1, dt_sd = cv_macro_f1('dt', DT_PARAMS, cols)
    variant_rows.append({'feature_set': name, 'knn_cv_macro_f1': knn_f1, 'knn_cv_std': knn_sd,
                         'dt_cv_macro_f1': dt_f1, 'dt_cv_std': dt_sd})
variants = pd.DataFrame(variant_rows).set_index('feature_set')
variants['knn_change_vs_current'] = variants['knn_cv_macro_f1'] - variants.loc['current (continuous dist_cbd_km)', 'knn_cv_macro_f1']
variants['dt_change_vs_current'] = variants['dt_cv_macro_f1'] - variants.loc['current (continuous dist_cbd_km)', 'dt_cv_macro_f1']
variants.round(4).to_csv(OUT_DIR + 'sl_feature_variants.csv')
log("\nFeature-set comparison (5-fold stratified CV macro-F1, tuned settings):")
log(variants.round(4).to_string())
RESULTS['feature_variants'] = variants.round(4).reset_index().to_dict(orient='records')
 
# save everything
def to_builtin(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return str(o)
 
with open(OUT_DIR + 'supervised_results.json', 'w') as f:
    json.dump(RESULTS, f, indent=2, default=to_builtin)
with open(OUT_DIR + 'supervised_log.txt', 'w') as f:
    f.write("\n".join(LOG))
log("\nSaved supervised_results.json, supervised_log.txt, CSVs and figures to " + OUT_DIR)
