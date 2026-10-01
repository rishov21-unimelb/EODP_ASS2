import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from pathlib import Path

pd.set_option('display.max_columns', None)
np.random.seed(42)

LOG = []
def log(msg=""):
    print(msg)
    LOG.append(str(msg))


BASE = Path(os.environ.get("EODP_BASE", r"C:\Users\shaun\EODP_ASS2")) #change for your own path 
OUT = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
OUT_STR = str(OUT) + os.sep

df = pd.read_csv(BASE / "listings.csv")
n_raw = len(df)
log(f"Raw dataset shape: {df.shape}")


df['price_clean'] = (df['price'].str.replace('$', '', regex=False)
                                 .str.replace(',', '', regex=False)
                                 .str.strip())
df['price_clean'] = pd.to_numeric(df['price_clean'], errors='coerce')

n_missing_price = df['price_clean'].isna().sum()
log(f"\n[Basic cleaning] price missing: {n_missing_price} ({n_missing_price/n_raw*100:.2f}%)")

# drop rows with no price - cannot assign a price tier without it
df = df[df['price_clean'].notna()].copy()
n_after_price_drop = len(df)

# outlier cap: prices > $1500/night are treated as implausible for short-term
# Melbourne rentals; the actual 99th percentile is logged below so the cut-off is evidence-based
p99_price = df['price_clean'].quantile(0.99)
log(f"[Basic cleaning] 99th percentile of valid prices: ${p99_price:.2f}")
outlier_cutoff = 1500
n_outliers = (df['price_clean'] > outlier_cutoff).sum()
log(f"[Basic cleaning] price outliers (> ${outlier_cutoff}): {n_outliers} "
    f"({n_outliers/n_after_price_drop*100:.2f}% of price-valid rows)")
df = df[df['price_clean'] <= outlier_cutoff].copy()
n_clean = len(df)
log(f"[Basic cleaning] Final clean sample size: {n_clean} "
    f"({n_clean/n_raw*100:.1f}% of raw {n_raw} rows retained)")

df['log_price'] = np.log(df['price_clean'])



# TASK 1: DISTANCE-FROM-CBD DISCRETISATION

log("\n" + "="*70)
log("TASK 1: Distance-from-CBD discretisation")
log("="*70)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

CBD_LAT, CBD_LON = -37.8136, 144.9631
df['dist_cbd_km'] = haversine(df['latitude'], df['longitude'], CBD_LAT, CBD_LON)

# Alternative approach A: quantile-based tertiles (equal-count bins)
q33, q66 = df['dist_cbd_km'].quantile([0.33, 0.66])
log(f"Alternative considered - quantile tertiles: cutoffs at {q33:.2f}km / {q66:.2f}km "
    f"(33rd/66th percentile)")

# Alternative approach B (chosen): fixed, interpretable urban-structure bins
bins = [-0.01, 5, 15, np.inf]
labels = ['Inner (<5km)', 'Middle (5-15km)', 'Outer (>15km)']
df['distance_band'] = pd.cut(df['dist_cbd_km'], bins=bins, labels=labels)
band_counts = df['distance_band'].value_counts().reindex(labels)
log(f"\nChosen approach - fixed geographic bins (Inner<5km / Middle 5-15km / Outer>15km):")
log(band_counts.to_string())
log(f"Group balance: {(band_counts/len(df)*100).round(1).to_dict()}")
log("Reasoning: fixed bins align with Melbourne's recognised inner-city / middle-"
    "ring / outer-suburb structure (train zone boundaries), making bands "
    "interpretable to a reader, whereas quantile tertiles would force exactly "
    "33.3% into each bin regardless of whether that reflects a real geographic "
    "break.")

# Impact: relationship between distance and price
corr_before = df[['dist_cbd_km','price_clean']].corr().iloc[0,1]
band_means = df.groupby('distance_band', observed=True)['price_clean'].mean()
f_stat, p_val = stats.f_oneway(*[g['price_clean'].values for _, g in df.groupby('distance_band', observed=True)])
log(f"\n[Impact] Pearson correlation(dist_cbd_km, price): r = {corr_before:.3f}")
log(f"[Impact] Mean price by distance band:\n{band_means.round(2).to_string()}")
log(f"[Impact] One-way ANOVA across bands: F = {f_stat:.2f}, p = {p_val:.2e}")
pct_diff = (band_means['Inner (<5km)'] - band_means['Outer (>15km)']) / band_means['Outer (>15km)'] * 100
log(f"[Impact] Inner-band mean price is {pct_diff:.1f}% higher than outer-band mean price "
    f"(${band_means['Inner (<5km)']:.2f} vs ${band_means['Outer (>15km)']:.2f})")


# TASK 2: MISSING-DATA STRATEGY FOR BEDROOMS/BEDS (ref. room_type)

log("\n" + "="*70)
log("TASK 2: Missing-data strategy for bedrooms/beds")
log("="*70)

miss_bed_by_type = df.groupby('room_type')['bedrooms'].apply(lambda x: x.isna().mean()*100)
miss_beds_by_type = df.groupby('room_type')['beds'].apply(lambda x: x.isna().mean()*100)
log("Missingness of 'bedrooms' (%) by room_type:")
log(miss_bed_by_type.round(1).to_string())
log("\nMissingness of 'beds' (%) by room_type:")
log(miss_beds_by_type.round(1).to_string())

n_bedrooms_missing = df['bedrooms'].isna().sum()
n_beds_missing = df['beds'].isna().sum()
log(f"\nOverall: bedrooms missing = {n_bedrooms_missing} ({n_bedrooms_missing/len(df)*100:.1f}%), "
    f"beds missing = {n_beds_missing} ({n_beds_missing/len(df)*100:.1f}%)")

# alternative 1: drop rows with missing bedrooms/beds
n_would_drop = df[df['bedrooms'].isna() | df['beds'].isna()].shape[0]
log(f"\nAlternative considered - drop rows with missing bedrooms/beds: "
    f"would remove {n_would_drop} rows ({n_would_drop/len(df)*100:.1f}% of clean sample) - rejected, "
    f"too much data loss and drop is concentrated in Private/Shared rooms "
    f"({miss_bed_by_type.get('Private room',0):.1f}% missing), biasing the remaining sample "
    f"toward entire-home listings.")

# alternative 2: global median imputation (ignoring room_type)
global_median_bed = df['bedrooms'].median()
private_observed_mode = df.loc[df['room_type']=='Private room','bedrooms'].dropna().mode()[0]
log(f"Alternative considered - global median imputation: global median bedrooms = "
    f"{global_median_bed:.0f}, but observed Private-room bedrooms are overwhelmingly "
    f"{private_observed_mode:.0f} "
    f"({(df.loc[df['room_type']=='Private room','bedrooms'].dropna()==private_observed_mode).mean()*100:.1f}% "
    f"of non-missing Private-room rows) - global median would systematically "
    f"overstate bedroom counts for private/shared rooms.")

# CHOSEN: group-based imputation
df['bedrooms_imputed'] = df['bedrooms'].copy()
df['beds_imputed'] = df['beds'].copy()

# Private room / Shared room / Hotel room -> definitionally ~1 bedroom
mask_single_room = df['room_type'].isin(['Private room', 'Shared room', 'Hotel room'])
df.loc[mask_single_room & df['bedrooms_imputed'].isna(), 'bedrooms_imputed'] = 1

# Entire home/apt -> median bedrooms within accommodates group
entire_mask = df['room_type'] == 'Entire home/apt'
med_bed_by_acc = df[entire_mask].groupby('accommodates')['bedrooms'].median()
def impute_bedrooms(row):
    if pd.notna(row['bedrooms_imputed']):
        return row['bedrooms_imputed']
    m = med_bed_by_acc.get(row['accommodates'], np.nan)
    return m if pd.notna(m) else df['bedrooms'].median()
df['bedrooms_imputed'] = df.apply(impute_bedrooms, axis=1)

# beds: median within (room_type, accommodates), fallback overall median
med_beds_grp = df.groupby(['room_type','accommodates'])['beds'].median()
def impute_beds(row):
    if pd.notna(row['beds_imputed']):
        return row['beds_imputed']
    m = med_beds_grp.get((row['room_type'], row['accommodates']), np.nan)
    return m if pd.notna(m) else df['beds'].median()
df['beds_imputed'] = df.apply(impute_beds, axis=1)

log(f"\n[Chosen] Private/Shared/Hotel room -> impute bedrooms = 1 "
    f"({mask_single_room.sum()} rows in this group)")
log(f"[Chosen] Entire home/apt -> impute bedrooms/beds by median within accommodates group")
log(f"[Impact] bedrooms missing before: {n_bedrooms_missing} -> after: {df['bedrooms_imputed'].isna().sum()}")
log(f"[Impact] beds missing before: {n_beds_missing} -> after: {df['beds_imputed'].isna().sum()}")

corr_bed_before = df[['bedrooms','price_clean']].corr().iloc[0,1]
corr_bed_after = df[['bedrooms_imputed','price_clean']].corr().iloc[0,1]
log(f"[Impact] corr(bedrooms, price) before imputation (n={df['bedrooms'].notna().sum()}): {corr_bed_before:.3f}")
log(f"[Impact] corr(bedrooms, price) after imputation (n={len(df)}): {corr_bed_after:.3f}")

mean_bed_before = df['bedrooms'].mean()
mean_bed_after = df['bedrooms_imputed'].mean()
log(f"[Impact] mean bedrooms before (observed only): {mean_bed_before:.3f} -> "
    f"after imputation (full sample): {mean_bed_after:.3f}")


# TASK 3: PROPERTY-TYPE CONSOLIDATION

log("\n" + "="*70)
log("TASK 3: Property-type consolidation")
log("="*70)

vc = df['property_type'].value_counts()
n_categories_before = vc.shape[0]
n_rare = (vc < 10).sum()
log(f"Raw property_type categories: {n_categories_before}, of which {n_rare} have < 10 listings")

# Alternative considered: keep top-10 categories + "Other"
top10 = set(vc.head(10).index)
n_in_other_alt = df[~df['property_type'].isin(top10)].shape[0]
log(f"\nAlternative considered - top-10 + 'Other' bucket: would dump "
    f"{n_in_other_alt} rows ({n_in_other_alt/len(df)*100:.1f}%) with very different "
    f"housing styles (e.g. tiny homes, farm stays, boats, hostels) into one "
    f"undifferentiated 'Other' group, hiding physically meaningful differences "
    f"relevant to price. Rejected.")

# CHOSEN: two-step grouping = (entire place vs room) x (building type).
# Raw property_type strings such as "Private room in home" vs "Entire home" differ hugely in
# price, so building type alone would throw that signal away (see eta-squared below).
NICHE_KEYS = ['tiny home', 'boat', 'houseboat', 'cabin', 'camper', 'rv', 'treehouse', 'yurt',
              'dome', 'tent', 'barn', 'container', 'farm stay', 'campsite', 'island',
              'religious building']

def building_type(pt):
    p = pt.lower()
    if any(k in p for k in ['hotel', 'aparthotel', 'bed and breakfast', 'bnb', 'hostel']):
        return 'Hotel/B&B/Hostel'
    # word-boundary match: plain substring 'rv' wrongly matched "serviced apartment"
    if any(re.search(r'\b' + re.escape(k) + r'\b', p) for k in NICHE_KEYS):
        return 'Unique/Niche'
    if any(k in p for k in ['home', 'house', 'townhouse', 'cottage', 'villa', 'bungalow', 'chalet']):
        return 'House'
    if any(k in p for k in ['rental unit', 'condo', 'loft', 'serviced apartment',
                             'guest suite', 'guesthouse']):
        return 'Apartment/Unit'
    return 'Other'

def consolidate(row):
    b = building_type(row['property_type'])
    if row['room_type'] == 'Entire home/apt':
        return 'Entire ' + (b if b in ('House', 'Apartment/Unit') else 'Unique/Other')
    return 'Room in ' + (b if b in ('House', 'Apartment/Unit', 'Hotel/B&B/Hostel') else 'Unique/Other')

df['property_type_group'] = df.apply(consolidate, axis=1)
grp_counts = df['property_type_group'].value_counts()
n_categories_after = grp_counts.shape[0]
log(f"\n[Chosen] Consolidated into {n_categories_after} groups:")
log(grp_counts.to_string())

# impact: category stability (min group size) + variance explained (eta^2) comparison
min_group_before = vc.min()
min_group_after = grp_counts.min()
log(f"\n[Impact] Smallest category size before: {min_group_before} listings -> after: {min_group_after} listings")

# eta squared for price ~ property_type_group  (one-way ANOVA effect size)
def eta_squared(groups):
    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g)*(g.mean()-grand_mean)**2 for g in groups)
    ss_total = sum((all_vals-grand_mean)**2)
    return ss_between/ss_total

groups_after = [g['price_clean'].values for _, g in df.groupby('property_type_group')]
eta2_after = eta_squared(groups_after)

# for 'before', restrict to categories with >=10 rows to allow a fair ANOVA (rare cats unstable)
valid_cats = vc[vc>=10].index
groups_before = [g['price_clean'].values for _, g in df[df['property_type'].isin(valid_cats)].groupby('property_type')]
eta2_before = eta_squared(groups_before)

log(f"[Impact] eta-squared (variance in price explained), raw property_type "
    f"(categories with >=10 listings, n={len(valid_cats)} categories): {eta2_before:.3f}")
log(f"[Impact] eta-squared, consolidated property_type_group "
    f"({n_categories_after} categories): {eta2_after:.3f}")
df['_building_only'] = df['property_type'].apply(building_type)
eta2_building_only = eta_squared([g['price_clean'].values for _, g in df.groupby('_building_only')])
df = df.drop(columns='_building_only')
log(f"[Impact] Alternative considered - building type only (no entire/room split, "
    f"{df['property_type'].apply(building_type).nunique()} groups): eta-squared = {eta2_building_only:.3f} "
    f"-> rejected, it discards the entire-vs-room price gap.")
log(f"[Impact] Chosen grouping retains {eta2_after/eta2_before*100:.0f}% of the raw field's "
    f"eta-squared ({eta2_after:.3f} / {eta2_before:.3f}) with {n_categories_after} groups instead of "
    f"{len(valid_cats)} (+{n_categories_before - len(valid_cats)} categories with <10 listings).")
log("Mean price by group:\n" + df.groupby('property_type_group')['price_clean'].mean().round(2).to_string())

# ============================================================
# CANDIDATES NOT SELECTED (named for completeness, per assignment requirement
# of naming >= 6 candidates)
# ============================================================
log("\n" + "="*70)
log("CANDIDATES CONSIDERED BUT NOT IMPLEMENTED")
log("="*70)
_mn = df['minimum_nights']
_hl = df['host_listings_count']
log(f"""
4. Review-recency feature (days_since_last_review): useful for gauging listing
   activity/staleness, but not a strong theoretical driver of nightly *price*
   itself - it reflects booking demand history, not the listing's price
   positioning. {df['last_review'].isna().mean()*100:.1f}% of clean rows have no last_review date, so the
   feature would also be undefined for those listings.

5. Minimum-stay discretisation (short/medium/long): minimum_nights is heavily
   skewed (median={_mn.median():.0f}, mean={_mn.mean():.2f}, max={_mn.max():.0f}) and dominated by
   short-stay listings ({(_mn <= 3).mean()*100:.1f}% require <=3 nights; only {(_mn >= 30).mean()*100:.1f}% require 30+);
   it speaks to booking policy rather than the physical/locational attributes
   that most plausibly separate price tiers.

6. Host-listings-count flag (single- vs multi-listing host): potentially
   interesting (professional/multi-property hosts vs individual hosts) but
   host_listings_count is extremely skewed (median={_hl.median():.0f}, max={_hl.max():.0f}),
   so it may reflect Airbnb-wide rather than Melbourne-only counts and would
   need separate cleaning before it could be trusted as a price-tier predictor.
""")
log("Final 3 selected (distance-from-CBD, bedrooms/beds missing-data strategy, "
    "property-type consolidation) were chosen because they are the most direct, "
    "well-evidenced structural drivers of short-term rental price (location, "
    "size/capacity, housing type) and are all required as usable numeric/"
    "categorical features before any later price-tier analysis (covered "
    "elsewhere in the assignment, not in this script) can be done.")

#rest of the preprocessing -shaun
log("\n" + "=" * 70)
log("Additional preprocessing (target, bathrooms, rating)")
log("=" * 70)

# price tiers -------
# Tertiles give three equal-sized classes. Fixed dollar cut-offs are the
# alternative: they are easier to read but give unbalanced classes in a right-skewed market.
q1, q2 = df['price_clean'].quantile([1/3, 2/3])
tier_labels = ['Budget', 'Mid', 'Premium']
df['price_tier'] = pd.qcut(df['price_clean'], q=3, labels=tier_labels)
df['price_tier_code'] = df['price_tier'].cat.codes          # 0,1,2 (ordinal)
tier_counts = df['price_tier'].value_counts().reindex(tier_labels)
log(f"[Target] Tertile cut-offs: Budget <= ${q1:.2f} < Mid <= ${q2:.2f} < Premium")
log(f"[Target] Tier counts:\n{tier_counts.to_string()}")
log(f"[Target] Tier shares (%): {(tier_counts / len(df) * 100).round(1).to_dict()}")
fixed = pd.cut(df['price_clean'], [0, 100, 250, np.inf], labels=tier_labels)
log(f"[Target] Alternative (fixed $100/$250 cut-offs) would give shares (%): "
    f"{(fixed.value_counts(normalize=True).reindex(tier_labels) * 100).round(1).to_dict()}")
log("NOTE: price_clean / log_price define the target, so they are EXCLUDED from the "
    "modelling predictors (target leakage). price_clean is used as the continuous "
    "proxy only inside the correlation analysis.")

# Bathrooms: numeric from bathrooms_text -------
def parse_bath(s):
    if pd.isna(s):
        return np.nan
    s = str(s).lower()
    m = re.search(r'(\d+(?:\.\d+)?)', s)
    if m:
        return float(m.group(1))
    if 'half' in s:
        return 0.5
    return np.nan

if 'bathrooms' in df.columns and df['bathrooms'].notna().any():
    df['bathrooms_num'] = df['bathrooms'].astype(float)
    src = "existing 'bathrooms' column"
else:
    df['bathrooms_num'] = np.nan
    src = "bathrooms_text parse"
if 'bathrooms_text' in df.columns:
    parsed = df['bathrooms_text'].apply(parse_bath)
    df['bathrooms_num'] = df['bathrooms_num'].fillna(parsed)
n_bath_miss = df['bathrooms_num'].isna().sum()
log(f"\n[Bathrooms] source: {src}; still missing: {n_bath_miss} "
    f"({n_bath_miss / len(df) * 100:.2f}%)")
bath_med = df.groupby('room_type')['bathrooms_num'].transform('median')
df['bathrooms_num'] = df['bathrooms_num'].fillna(bath_med).fillna(df['bathrooms_num'].median())
log(f"[Bathrooms] after median-by-room_type imputation, missing: {df['bathrooms_num'].isna().sum()}")

# review_scores_rating: missing + indicator -------
n_rating_miss = df['review_scores_rating'].isna().sum()
log(f"\n[Rating] review_scores_rating missing: {n_rating_miss} "
    f"({n_rating_miss / len(df) * 100:.1f}%)")
miss_tier = df.groupby('price_tier', observed=True)['review_scores_rating'].apply(lambda x: x.isna().mean() * 100)
log(f"[Rating] % missing by price tier (informative missingness check):\n{miss_tier.round(1).to_string()}")
df['has_rating'] = df['review_scores_rating'].notna().astype(int)
r_before = df[['review_scores_rating', 'price_clean']].corr().iloc[0, 1]
df['review_scores_rating_imp'] = df['review_scores_rating'].fillna(df['review_scores_rating'].median())
r_after = df[['review_scores_rating_imp', 'price_clean']].corr().iloc[0, 1]
log(f"[Rating] corr(rating, price): observed-only {r_before:.3f} -> after median imputation {r_after:.3f}")
log("Correlation analysis (Code_part2.py) uses pairwise deletion on the raw column; "
    "the imputed column + has_rating flag are kept for later modelling.")

# Other features: skew handling + binary encoding ----
for c in ['minimum_nights', 'host_listings_count', 'number_of_reviews', 'availability_365']:
    if c in df.columns:
        sk = df[c].skew()
        log(f"[Skew] {c}: skewness = {sk:.2f}, max = {df[c].max():.0f}, median = {df[c].median():.0f}")
if 'instant_bookable' in df.columns:
    df['instant_bookable_flag'] = (df['instant_bookable'].astype(str).str.lower() == 't').astype(int)


# CHARTS (for the 3.2 tasks only)

sns.set_style("whitegrid")

# 1. Price distribution before/after cleaning + log
fig, axes = plt.subplots(1,2, figsize=(11,4))
sns.histplot(df['price_clean'], bins=50, ax=axes[0], color='#4C72B0')
axes[0].set_title('Cleaned Price Distribution')
axes[0].set_xlabel('Price ($/night)')
sns.histplot(df['log_price'], bins=50, ax=axes[1], color='#55A868')
axes[1].set_title('Log(Price) Distribution')
axes[1].set_xlabel('log(Price)')
plt.tight_layout()
plt.savefig(OUT_STR + 'price_distribution.png', dpi=150)
plt.close()

# 2. Mean price by distance band
fig, ax = plt.subplots(figsize=(6,4))
band_means.reindex(labels).plot(kind='bar', ax=ax, color='#C44E52')
ax.set_ylabel('Mean price ($)')
ax.set_title('Mean Price by Distance-from-CBD Band')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(OUT_STR + 'price_by_distance.png', dpi=150)
plt.close()

log("\nCharts saved to " + str(OUT))

# save the log to a text file and the processed dataframe
with open(OUT / "analysis_log.txt", 'w') as f:
    f.write("\n".join(LOG))

df.to_csv(OUT / "listings_processed.csv", index=False)
log("\nProcessed dataset saved.")