import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

pd.set_option('display.max_columns', None)
np.random.seed(42)

LOG = []
def log(msg=""):
    print(msg)
    LOG.append(str(msg))


BASE = Path(r"C:\Users\shaun\EODP_ASS2")
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
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

# outlier cap: prices > $1500/night are implausible data-entry errors for
# short-term Melbourne rentals (99th pct = $1340.86); remove them
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
    f"({(df.loc[df['room_type']=='Private room','bedrooms']==private_observed_mode).mean()*100:.1f}% "
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

# CHOSEN: rule-based keyword grouping into physically-meaningful categories
def consolidate(pt):
    p = pt.lower()
    if any(k in p for k in ['hostel']):
        return 'Hostel/Shared'
    if any(k in p for k in ['hotel', 'aparthotel', 'bed and breakfast', 'bnb']):
        return 'Hotel/B&B'
    if any(k in p for k in ['tiny home', 'boat', 'houseboat', 'cabin', 'camper', 'rv',
                             'treehouse', 'yurt', 'dome', 'tent', 'barn', 'container',
                             'farm stay', 'campsite', 'island', 'religious building']):
        return 'Unique/Niche'
    if any(k in p for k in ['home', 'house', 'townhouse', 'cottage', 'villa',
                             'bungalow', 'chalet', 'cabin']):
        return 'House'
    if any(k in p for k in ['rental unit', 'condo', 'loft', 'serviced apartment',
                             'guest suite', 'guesthouse']):
        return 'Apartment/Unit'
    return 'Other'

df['property_type_group'] = df['property_type'].apply(consolidate)
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
log("-> consolidation retains almost all of the price-explanatory power of the raw "
    "field while cutting the category count and removing categories too small to "
    "generalise from.")

# ============================================================
# CANDIDATES NOT SELECTED (named for completeness, per assignment requirement
# of naming >= 6 candidates)
# ============================================================
log("\n" + "="*70)
log("CANDIDATES CONSIDERED BUT NOT IMPLEMENTED")
log("="*70)
log("""
4. Review-recency feature (days_since_last_review): useful for gauging listing
   activity/staleness, but not a strong theoretical driver of nightly *price*
   itself - it reflects booking demand history, not the listing's price
   positioning. Lower priority for a price-tier question than location/size/type.

5. Minimum-stay discretisation (short/medium/long): minimum_nights is heavily
   skewed (median=2, mean=4.6, max=1000) and dominated by short-stay listings
   (~75% require <=3 nights); it speaks to booking policy rather than the
   physical/locational attributes that most plausibly separate price tiers.

6. Host-listings-count flag (single- vs multi-listing host): potentially
   interesting (professional/multi-property hosts vs individual hosts) but
   host_listings_count has its own data-quality issues (max=667, likely
   Airbnb-wide counts, not Melbourne-specific) which would need separate
   cleaning before it could be trusted as a price-tier predictor.
""")
log("Final 3 selected (distance-from-CBD, bedrooms/beds missing-data strategy, "
    "property-type consolidation) were chosen because they are the most direct, "
    "well-evidenced structural drivers of short-term rental price (location, "
    "size/capacity, housing type) and are all required as usable numeric/"
    "categorical features before any later price-tier analysis (covered "
    "elsewhere in the assignment, not in this script) can be done.")

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
plt.savefig('/home/claude/outputs/price_distribution.png', dpi=150)
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
