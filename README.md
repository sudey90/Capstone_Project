# Capstone_Project

## Data Findings

Based on the processed shipment dataset (1,200 observations), the analysis revealed the following key insights:

1. **Service Type Delay Rates:** **Express** shipments experience delays most frequently, recording a delay rate of **0.4897** (119 delayed out of 243 shipments), compared to Standard shipments at **0.2056** and Economy shipments at **0.0857**.
2. **Hub Tier Progression:** Delay rates increase sharply from lower to higher tiers, climbing from **0.0400** in Tier 1 hubs to **0.2150** in Tier 2, and peaking at **0.4400** in Tier 3 hubs.
3. **Distance Band Impact:** Delivery delays escalate significantly with transit distance, starting at **0.0219** for the 0–200 km band, rising through **0.1368** (201–400 km) and **0.4364** (401–600 km), and reaching **0.8523** for distances of 601+ km.
4. **Top Correlations with Delivery Hours:** Ranking the features by absolute correlation with `delivery_hours` (excluding itself), the top three columns are `promised_hours` (**0.8372**), `distance_km` (**0.4013**), and `n_stops` (**0.3247**).
5. **Declared Value Transformation:** `declared_value_inr` was log-transformed because its right-skewed raw distribution (mean of **1,533.05** vs. median of **1,121.50**) compresses after transformation into a more symmetric distribution (mean of **7.0280** vs. median of **7.0220**), reducing the undue leverage of high-value outliers in modeling.