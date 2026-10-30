# Capstone_Project

## Data Foundation and Delay Evidence

Based on the processed shipment dataset (1,200 observations), the analysis revealed the following key insights:

1. **Service Type Delay Rates:** **Express** shipments experience delays most frequently, recording a delay rate of **0.4897** (119 delayed out of 243 shipments), compared to Standard shipments at **0.2056** and Economy shipments at **0.0857**.
2. **Hub Tier Progression:** Delay rates increase sharply from lower to higher tiers, climbing from **0.0400** in Tier 1 hubs to **0.2150** in Tier 2, and peaking at **0.4400** in Tier 3 hubs.
3. **Distance Band Impact:** Delivery delays escalate significantly with transit distance, starting at **0.0219** for the 0–200 km band, rising through **0.1368** (201–400 km) and **0.4364** (401–600 km), and reaching **0.8523** for distances of 601+ km.
4. **Top Correlations with Delivery Hours:** Ranking the features by absolute correlation with `delivery_hours` (excluding itself), the top three columns are `promised_hours` (**0.8372**), `distance_km` (**0.4013**), and `n_stops` (**0.3247**).
5. **Declared Value Transformation:** `declared_value_inr` was log-transformed because its right-skewed raw distribution (mean of **1,533.05** vs. median of **1,121.50**) compresses after transformation into a more symmetric distribution (mean of **7.0280** vs. median of **7.0220**), reducing the undue leverage of high-value outliers in modeling.

## Delivery Hours Regression

### 1. Baseline performance
The baseline model predicted the average training delivery time for all validation shipments. It achieved an R² of **-0.021** and RMSE of **18.855 hours**, indicating poor predictive performance.

### 2. Decision Tree depth comparison
Decision Tree depth 6 achieved the lowest validation RMSE of **6.280 hours**. Depth 1 showed high bias (underfitting), while unlimited depth showed high variance (overfitting), with training RMSE of 0 and validation RMSE of 7.789 hours.

### 3. Model comparison and learning rate
Four regression models were evaluated. **Linear Regression** achieved the lowest validation RMSE of **4.891 hours** with R² of 0.931, making it the selected model.

For Gradient Boosting, learning rate **0.1** performed best (R² = 0.931). A rate of 0.01 learned too slowly, while 1.0 resulted in overly aggressive updates.

### 4. Cross-validation
Five-fold cross-validation was performed on the two best models.

- Linear Regression: Mean R² = 0.926, Std = 0.00475
- Gradient Boosting: Mean R² = 0.935, Std = 0.00442

Both models showed consistent performance across folds.

### 5. Final test performance
Linear Regression was selected based on its lowest validation RMSE, even though Gradient Boosting achieved a slightly higher cross-validation R².

Final test results:

- **MAE:** 3.567 hours
- **MSE:** 20.462
- **RMSE:** 4.524 hours
- **R²:** 0.949

The final model explains approximately 94.9% of the variation in delivery hours on unseen test data.