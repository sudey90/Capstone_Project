"""PART B: DELIVERY-HOURS REGRESSION

Uses the exact 17 FEATURES in the assignment and predicts delivery_hours.
Input: data/modelling_table.csv
"""

# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# STEP 2: SET UP PROJECT FOLDERS AND READ PART A DATA
# ============================================================
PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
MODEL_FOLDER = PROJECT_FOLDER / "models"

OUTPUT_FOLDER.mkdir(exist_ok=True)
MODEL_FOLDER.mkdir(exist_ok=True)

df = pd.read_csv(DATA_FOLDER / "modelling_table.csv")
print("Dataset loaded successfully.")
print("Number of shipments:", len(df))


# ============================================================
# STEP 3: SELECT THE EXACT 17 FEATURES IN THE REQUIRED ORDER
# ============================================================
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]
TARGET = "delivery_hours"

for column in FEATURES + [TARGET]:
    if column not in df.columns:
        raise ValueError("Required column is missing: " + column)

X = df[FEATURES].copy()
y = df[TARGET].copy()

if X.isna().any().any():
    raise ValueError("One or more of the 17 FEATURES contain missing values.")
if y.isna().any():
    raise ValueError("delivery_hours contains missing values.")

# Keep the full feature-engineered Part A table for Parts C and D.
df.to_csv(DATA_FOLDER / "modeling_features.csv", index=False)
pd.DataFrame({"feature": FEATURES}).to_csv(
    OUTPUT_FOLDER / "part_b_feature_columns.csv", index=False
)


# ============================================================
# STEP 4: SPLIT INTO TRAINING, VALIDATION AND TEST SETS
# ============================================================
X_train, X_temporary, y_train, y_temporary = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_validation, X_test, y_validation, y_test = train_test_split(
    X_temporary, y_temporary, test_size=0.50, random_state=42
)

print("\nData split completed:")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_validation))
print("Test rows:", len(X_test))

if (len(X_train), len(X_validation), len(X_test)) != (840, 180, 180):
    raise ValueError("Split sizes must be exactly 840 / 180 / 180.")


# ============================================================
# STEP 5: SCALING CHECK - FIT ONLY ON TRAINING ROWS
# ============================================================
# Fit the scaler on training data only. Then use that fitted scaler
# to transform validation and test data. Never fit on validation/test.
scaler_check = StandardScaler()
X_train_scaled = scaler_check.fit_transform(X_train)
X_validation_scaled = scaler_check.transform(X_validation)
X_test_scaled = scaler_check.transform(X_test)

train_scaled_max_abs_mean = np.abs(X_train_scaled.mean(axis=0)).max()
test_scaled_max_abs_mean = np.abs(X_test_scaled.mean(axis=0)).max()

checks_df = pd.DataFrame([
    {"metric": "train_rows", "value": len(X_train)},
    {"metric": "val_rows", "value": len(X_validation)},
    {"metric": "test_rows", "value": len(X_test)},
    {"metric": "train_scaled_max_abs_mean", "value": train_scaled_max_abs_mean},
    {"metric": "test_scaled_max_abs_mean", "value": test_scaled_max_abs_mean},
])
checks_df.to_csv(OUTPUT_FOLDER / "part_b_checks.csv", index=False)
print("\nScaling checks:")
print(checks_df.to_string(index=False))


# ============================================================
# STEP 6: CREATE A FUNCTION TO CALCULATE REGRESSION METRICS
# ============================================================
def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(actual, predicted),
    }


# ============================================================
# STEP 7: BASELINE - ALWAYS PREDICT THE TRAINING MEAN
# ============================================================
training_mean = y_train.mean()
baseline_predictions = np.repeat(training_mean, len(y_validation))
baseline_metrics = calculate_metrics(y_validation, baseline_predictions)
print("\nBaseline results (validation data):")
print(pd.Series(baseline_metrics).round(3))


# ============================================================
# STEP 8: DECISION TREE DEPTH DIAGNOSTIC
# ============================================================
depths_to_try = [1, 2, 3, 4, 6, 8, 12, None]
tree_rows = []
tree_models = {}

for depth in depths_to_try:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)

    train_prediction = tree.predict(X_train)
    validation_prediction = tree.predict(X_validation)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_prediction))
    validation_rmse = np.sqrt(mean_squared_error(y_validation, validation_prediction))

    depth_label = "unlimited" if depth is None else depth
    tree_rows.append({
        "max_depth": depth_label,
        "train_rmse": train_rmse,
        "val_rmse": validation_rmse,
    })
    tree_models[depth_label] = tree

tree_depth_df = pd.DataFrame(tree_rows)
tree_depth_df.to_csv(OUTPUT_FOLDER / "part_b_tree_depth.csv", index=False)

# idxmin returns the first minimum. Since depths are in ascending order,
# a tie is resolved in favour of the smaller depth.
best_tree_index = tree_depth_df["val_rmse"].idxmin()
best_tree_depth = tree_depth_df.loc[best_tree_index, "max_depth"]
best_tree = tree_models[best_tree_depth]

print("\nDecision Tree depth comparison:")
print(tree_depth_df.round(3).to_string(index=False))
print("Best tree depth:", best_tree_depth)


# ============================================================
# STEP 9: TRAIN THE FOUR REQUIRED CANDIDATE MODELS
# ============================================================
# Linear Regression uses the specifically required scaler pipeline.
# Tree models do not need feature scaling.
models = {
    "linear_regression": Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),
    "decision_tree": best_tree,
    "random_forest": RandomForestRegressor(
        n_estimators=100, random_state=42
    ),
    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100, learning_rate=0.1, random_state=42
    ),
}

validation_metrics_by_model = {}
comparison_rows = []

for model_name, model in models.items():
    model.fit(X_train, y_train)
    validation_predictions = model.predict(X_validation)
    metrics = calculate_metrics(y_validation, validation_predictions)
    validation_metrics_by_model[model_name] = metrics

    comparison_rows.append({
        "model": model_name,
        "validation_MAE": metrics["MAE"],
        "validation_RMSE": metrics["RMSE"],
        "validation_R2": metrics["R2"],
    })

comparison_df = pd.DataFrame(comparison_rows).sort_values("validation_RMSE")
comparison_df.to_csv(OUTPUT_FOLDER / "part_b_model_comparison.csv", index=False)
print("\nFour-model validation comparison:")
print(comparison_df.round(3).to_string(index=False))


# ============================================================
# STEP 10: GRADIENT BOOSTING LEARNING-RATE EXPERIMENT
# ============================================================
learning_rate_rows = []
for rate in [0.01, 0.1, 1.0]:
    gb_model = GradientBoostingRegressor(
        n_estimators=100, learning_rate=rate, random_state=42
    )
    gb_model.fit(X_train, y_train)
    predictions = gb_model.predict(X_validation)
    metrics = calculate_metrics(y_validation, predictions)

    learning_rate_rows.append({
        "learning_rate": rate,
        "val_rmse": metrics["RMSE"],
        "val_r2": metrics["R2"],
    })

learning_rate_df = pd.DataFrame(learning_rate_rows)
learning_rate_df.to_csv(
    OUTPUT_FOLDER / "part_b_gb_learning_rate.csv", index=False
)
print("\nGradient Boosting learning-rate comparison:")
print(learning_rate_df.round(3).to_string(index=False))


# ============================================================
# STEP 11: SELECT THE BEST OF THE FOUR CANDIDATES
# ============================================================
# The assignment says to select the lowest validation RMSE among the
# four candidates. Do not add a duplicate 'Best Gradient Boosting' model.
best_model_name = comparison_df.iloc[0]["model"]
best_model = models[best_model_name]
print("\nSelected model based on validation RMSE:", best_model_name)


# ============================================================
# STEP 12: FIVE-FOLD CROSS-VALIDATION ON TRAIN + VALIDATION
# ============================================================
X_train_validation = pd.concat([X_train, X_validation])
y_train_validation = pd.concat([y_train, y_validation])

top_two_model_names = comparison_df.head(2)["model"].tolist()

# One KFold object is shared by both models.
folds = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rows = []

for model_name in top_two_model_names:
    # No scoring argument: sklearn uses the estimator's default R-squared.
    scores = cross_val_score(
        models[model_name],
        X_train_validation,
        y_train_validation,
        cv=folds,
    )
    cv_rows.append({
        "model": model_name,
        "mean_r2": scores.mean(),
        "std_r2": scores.std(),
    })

cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(OUTPUT_FOLDER / "part_b_cv.csv", index=False)
print("\nFive-fold cross-validation (default R-squared scoring):")
print(cv_df.round(3).to_string(index=False))


# ============================================================
# STEP 13: TEST THE WINNING MODEL ONCE
# ============================================================
# The winner is already fitted on the 840 training rows.
# Do not refit it on training + validation for this assignment.
test_predictions = best_model.predict(X_test)
test_metrics = calculate_metrics(y_test, test_predictions)
print("\nFinal test results:")
print(pd.Series(test_metrics).round(3))


# ============================================================
# STEP 14: WRITE THE REQUIRED SIX ROWS TO part_b_metrics.csv
# ============================================================
metrics_rows = [{
    "model": "mean_baseline",
    "split": "val",
    **baseline_metrics,
}]

for model_name in [
    "linear_regression", "decision_tree", "random_forest", "gradient_boosting"
]:
    metrics_rows.append({
        "model": model_name,
        "split": "val",
        **validation_metrics_by_model[model_name],
    })

metrics_rows.append({
    "model": best_model_name,
    "split": "test",
    **test_metrics,
})

metrics_df = pd.DataFrame(
    metrics_rows,
    columns=["model", "split", "MAE", "MSE", "RMSE", "R2"],
)
metrics_df.to_csv(OUTPUT_FOLDER / "part_b_metrics.csv", index=False)
print("\nRequired Part B metrics:")
print(metrics_df.round(3).to_string(index=False))


# ============================================================
# STEP 15: SAVE TEST PREDICTIONS
# ============================================================
test_predictions_df = pd.DataFrame({
    "actual_delivery_hours": y_test,
    "predicted_delivery_hours": test_predictions,
    "prediction_error_hours": y_test - test_predictions,
    "absolute_error_hours": np.abs(y_test - test_predictions),
})
test_predictions_df.to_csv(
    OUTPUT_FOLDER / "part_b_test_predictions.csv", index_label="row_index"
)


# ============================================================
# STEP 16: SAVE THE TRAINED MODEL FOR LATER USE
# ============================================================
model_information = {
    "model": best_model,
    "feature_columns": FEATURES,
    "target_column": TARGET,
    "selected_model": best_model_name,
}
joblib.dump(model_information, MODEL_FOLDER / "delivery_hours_model.pkl")


# ============================================================
# STEP 17: FINAL SUMMARY
# ============================================================
print("\n" + "=" * 58)
print("PART B COMPLETED")
print("=" * 58)
print("Selected regression model:", best_model_name)
print("Training / validation / test:", len(X_train), len(X_validation), len(X_test))
print("Final test performance:")
for metric_name, metric_value in test_metrics.items():
    print(metric_name + ":", round(metric_value, 4))
print("\nOutputs saved in:", OUTPUT_FOLDER)
print("Reusable dataset:", DATA_FOLDER / "modeling_features.csv")
print("Trained model:", MODEL_FOLDER / "delivery_hours_model.pkl")