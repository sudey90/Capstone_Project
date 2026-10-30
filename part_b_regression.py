"""
PART B: DELIVERY-HOURS REGRESSION

Goal:
Predict how many hours a shipment will take to reach the customer.

Input:
    data/modelling_table.csv  (created in Part A)

Important assumptions:
    n_stops is the planned number of delivery stops.
    rain_flag is the weather forecast available at dispatch time.

Files created for later tasks:
    data/modeling_features.csv
    models/delivery_hours_model.pkl

Other results are saved in the outputs folder.
"""

# ============================================================
# STEP 1: IMPORT THE LIBRARIES
# ============================================================

from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# STEP 2: SET UP PROJECT FOLDERS
# ============================================================

PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
MODEL_FOLDER = PROJECT_FOLDER / "models"

OUTPUT_FOLDER.mkdir(exist_ok=True)
MODEL_FOLDER.mkdir(exist_ok=True)

# Read the cleaned dataset created in Part A.
df = pd.read_csv(DATA_FOLDER / "modelling_table.csv")

print("Dataset loaded successfully.")
print("Number of shipments:", len(df))


# ============================================================
# STEP 3: PREPARE THE DATE AND TIME FEATURES
# ============================================================

# Convert dispatch_time from text into a proper date/time value.
df["dispatch_time"] = pd.to_datetime(df["dispatch_time"], errors="coerce")

if df["dispatch_time"].isna().any():
    raise ValueError("Some dispatch_time values are missing or invalid.")

# Extract useful information from the dispatch date.
df["dispatch_day"] = df["dispatch_time"].dt.day
df["dispatch_month"] = df["dispatch_time"].dt.month
df["dispatch_weekday"] = df["dispatch_time"].dt.dayofweek

# A clock repeats every 24 hours. Sine and cosine represent this cycle:
# 23:00 and 00:00 are therefore treated as neighbouring times.
hour = df["dispatch_time"].dt.hour + df["dispatch_time"].dt.minute / 60
df["dispatch_hour_sin"] = np.sin(2 * np.pi * hour / 24)
df["dispatch_hour_cos"] = np.cos(2 * np.pi * hour / 24)


# ============================================================
# STEP 4: CHOOSE THE INPUT COLUMNS AND THE ANSWER TO PREDICT
# ============================================================

# These columns contain information available when the shipment is dispatched.
# We do NOT include delivery_hours here because that is what we want to predict.

feature_columns = [
    "service_type",
    "payment_mode",
    "parcel_weight_kg",
    "distance_km",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "hub_city",
    "region",
    "hub_tier",
    "hub_daily_capacity",
    "promised_hours",
    "is_cod",
    "is_weekend",
    "dispatch_day",
    "dispatch_month",
    "dispatch_weekday",
    "dispatch_hour_sin",
    "dispatch_hour_cos",
    "log_declared_value",
]

# Check that all required input columns exist.
for column in feature_columns:
    if column not in df.columns:
        raise ValueError("Required column is missing: " + column)

# X = information given to the model.
# y = correct delivery time that the model must learn to predict.
X = df[feature_columns].copy()
y = df["delivery_hours"].copy()

if y.isna().any():
    raise ValueError("delivery_hours contains missing values.")

# Save the feature-engineered dataset for Parts C and D.
# It keeps delivery_hours and is_delayed because later tasks need them.
df.to_csv(DATA_FOLDER / "modeling_features.csv", index=False)

# Save the list of features used by the regression model.
pd.DataFrame({"feature": feature_columns}).to_csv(
    OUTPUT_FOLDER / "part_b_feature_columns.csv", index=False
)


# ============================================================
# STEP 5: SPLIT THE DATA INTO TRAINING, VALIDATION AND TEST SETS
# ============================================================

# Training data (70%): teaches the models.
# Validation data (15%): helps us compare and select a model.
# Test data (15%): gives the final performance on unseen shipments.

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("\nData split completed:")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_validation))
print("Test rows:", len(X_test))


# ============================================================
# STEP 6: PREPARE CATEGORICAL AND NUMERIC COLUMNS
# ============================================================

# Categorical columns contain names or labels.
# Example: Express, Standard, Economy, Kolkata, etc.
categorical_columns = [
    "service_type",
    "payment_mode",
    "hub_city",
    "region",
]

# All remaining input columns are numeric.
numeric_columns = [
    column for column in feature_columns
    if column not in categorical_columns
]

# Numeric preparation:
# 1. Fill missing numeric values using the median.
# 2. Standardize numeric values so they are on comparable scales.
numeric_preparation = Pipeline(steps=[
    ("fill_missing", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

# Categorical preparation:
# 1. Fill missing labels using the most common value.
# 2. Convert labels into numeric columns (One-Hot Encoding).
categorical_preparation = Pipeline(steps=[
    ("fill_missing", SimpleImputer(strategy="most_frequent")),
    ("convert_to_numbers", OneHotEncoder(handle_unknown="ignore")),
])

# Apply the correct preparation to each type of column.
preparation = ColumnTransformer(transformers=[
    ("numeric", numeric_preparation, numeric_columns),
    ("categorical", categorical_preparation, categorical_columns),
])


# ============================================================
# STEP 7: CREATE A SMALL HELPER TO BUILD A MODEL PIPELINE
# ============================================================

# A pipeline performs data preparation first and model training second.
# This also ensures that preprocessing is learned only from the training
# portion of each split or cross-validation fold.

def create_model(regression_algorithm):
    return Pipeline(steps=[
        ("preparation", clone(preparation)),
        ("regression_model", regression_algorithm),
    ])


# ============================================================
# STEP 8: CREATE A FUNCTION TO CALCULATE MODEL PERFORMANCE
# ============================================================

def calculate_metrics(actual_values, predicted_values):
    mae = mean_absolute_error(actual_values, predicted_values)
    mse = mean_squared_error(actual_values, predicted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual_values, predicted_values)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


# ============================================================
# STEP 9: CREATE A SIMPLE BASELINE
# ============================================================

# Baseline means a very simple model for comparison.
# It always predicts the average delivery time from the training data.

average_delivery_time = y_train.mean()

baseline_predictions = np.repeat(average_delivery_time, len(y_validation))
baseline_results = calculate_metrics(y_validation, baseline_predictions)

print("\nBaseline results (validation data):")
print(pd.Series(baseline_results).round(3))


# ============================================================
# STEP 10: FIND A SUITABLE DECISION TREE DEPTH
# ============================================================

# A shallow tree may be too simple.
# A very deep tree may memorize the training data.
# We compare several depths and choose the one with the lowest validation RMSE.

tree_depths = [1, 2, 3, 4, 6, 8, 12, None]

tree_results = []
tree_models = {}

for depth in tree_depths:

    tree = create_model(
        DecisionTreeRegressor(max_depth=depth, random_state=42)
    )

    tree.fit(X_train, y_train)

    train_predictions = tree.predict(X_train)
    validation_predictions = tree.predict(X_validation)

    train_rmse = calculate_metrics(y_train, train_predictions)["RMSE"]
    validation_metrics = calculate_metrics(y_validation, validation_predictions)

    depth_name = str(depth)
    tree_models[depth_name] = tree

    tree_results.append({
        "tree_depth": depth_name,
        "training_RMSE": train_rmse,
        "validation_RMSE": validation_metrics["RMSE"],
        "validation_MAE": validation_metrics["MAE"],
        "validation_R2": validation_metrics["R2"],
    })

tree_results_df = pd.DataFrame(tree_results)
tree_results_df.to_csv(OUTPUT_FOLDER / "part_b_tree_depth.csv", index=False)

# Select the tree with the lowest validation RMSE.
best_tree_row = tree_results_df["validation_RMSE"].idxmin()
best_tree_depth = tree_results_df.loc[best_tree_row, "tree_depth"]
best_tree = tree_models[best_tree_depth]

print("\nDecision Tree depth comparison:")
print(tree_results_df.round(3).to_string(index=False))
print("Best tree depth:", best_tree_depth)


# ============================================================
# STEP 11: TRAIN AND COMPARE FOUR REGRESSION MODELS
# ============================================================

# We compare different algorithms to find which predicts delivery time best.

models = {
    "Linear Regression": create_model(LinearRegression()),
    "Decision Tree": best_tree,
    "Random Forest": create_model(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    ),
    "Gradient Boosting": create_model(
        GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        )
    ),
}

model_results = []

for model_name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_validation)
    results = calculate_metrics(y_validation, predictions)

    model_results.append({
        "model": model_name,
        "validation_MAE": results["MAE"],
        "validation_RMSE": results["RMSE"],
        "validation_R2": results["R2"],
    })

model_results_df = pd.DataFrame(model_results)
model_results_df = model_results_df.sort_values("validation_RMSE")
model_results_df.to_csv(OUTPUT_FOLDER / "part_b_model_comparison.csv", index=False)

print("\nRegression model comparison:")
print(model_results_df.round(3).to_string(index=False))


# ============================================================
# STEP 12: TRY DIFFERENT GRADIENT BOOSTING LEARNING RATES
# ============================================================

# Learning rate controls how much each new tree contributes.
# We try three values and select the one with the lowest validation RMSE.

learning_rates = [0.01, 0.1, 1.0]
learning_rate_results = []
gradient_boosting_models = {}

for rate in learning_rates:

    gb_model = create_model(
        GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=rate,
            random_state=42,
        )
    )

    gb_model.fit(X_train, y_train)

    predictions = gb_model.predict(X_validation)
    results = calculate_metrics(y_validation, predictions)

    learning_rate_results.append({
        "learning_rate": rate,
        "validation_RMSE": results["RMSE"],
        "validation_MAE": results["MAE"],
        "validation_R2": results["R2"],
    })

    gradient_boosting_models[str(rate)] = gb_model

learning_rate_df = pd.DataFrame(learning_rate_results)
learning_rate_df = learning_rate_df.sort_values("validation_RMSE")
learning_rate_df.to_csv(
    OUTPUT_FOLDER / "part_b_gb_learning_rate.csv", index=False
)

best_rate = str(learning_rate_df.iloc[0]["learning_rate"])
best_gradient_boosting = gradient_boosting_models[best_rate]

print("\nGradient Boosting learning-rate comparison:")
print(learning_rate_df.round(3).to_string(index=False))


# Add the best Gradient Boosting version to the model comparison.
models["Best Gradient Boosting"] = best_gradient_boosting

# Compare all candidate models using validation RMSE.
final_comparison = []

for model_name, model in models.items():

    predictions = model.predict(X_validation)
    results = calculate_metrics(y_validation, predictions)

    final_comparison.append({
        "model": model_name,
        "validation_RMSE": results["RMSE"],
        "validation_MAE": results["MAE"],
        "validation_R2": results["R2"],
    })

final_comparison_df = pd.DataFrame(final_comparison)
final_comparison_df = final_comparison_df.sort_values("validation_RMSE")

best_model_name = final_comparison_df.iloc[0]["model"]
best_model = models[best_model_name]

print("\nFinal model selection:")
print(final_comparison_df.round(3).to_string(index=False))
print("Selected model:", best_model_name)


# ============================================================
# STEP 13: CHECK THE BEST TWO MODELS USING 5-FOLD CROSS-VALIDATION
# ============================================================

# Cross-validation checks whether model performance is consistent
# across five different splits of the training + validation data.
# The test data is not used here.

X_train_validation = pd.concat([X_train, X_validation])
y_train_validation = pd.concat([y_train, y_validation])

top_two_models = final_comparison_df.head(2)["model"].tolist()

five_folds = KFold(n_splits=5, shuffle=True, random_state=42)
cross_validation_results = []

for model_name in top_two_models:

    model = models[model_name]

    # sklearn returns negative RMSE, so multiply by -1 to get normal RMSE.
    scores = cross_val_score(
        model,
        X_train_validation,
        y_train_validation,
        cv=five_folds,
        scoring="neg_root_mean_squared_error",
    )

    rmse_scores = -scores

    cross_validation_results.append({
        "model": model_name,
        "average_CV_RMSE": rmse_scores.mean(),
        "variation_CV_RMSE": rmse_scores.std(),
    })

cross_validation_df = pd.DataFrame(cross_validation_results)
cross_validation_df.to_csv(OUTPUT_FOLDER / "part_b_cv.csv", index=False)

print("\nFive-fold cross-validation:")
print(cross_validation_df.round(3).to_string(index=False))


# ============================================================
# STEP 14: TRAIN THE SELECTED MODEL USING TRAINING + VALIDATION DATA
# ============================================================

# We have finished choosing the model.
# Now we allow it to learn from both training and validation rows.
# The test data is still kept separate.

final_model = clone(best_model)
final_model.fit(X_train_validation, y_train_validation)


# ============================================================
# STEP 15: EVALUATE THE FINAL MODEL ON TEST DATA
# ============================================================

# This is the final check using data the model has not seen before.

test_predictions = final_model.predict(X_test)
test_results = calculate_metrics(y_test, test_predictions)

print("\nFinal test results:")
print(pd.Series(test_results).round(3))


# Save the final model metrics.
metrics_to_save = pd.DataFrame([
    {
        "dataset": "Validation baseline",
        "model": "Average delivery time",
        **baseline_results,
    },
    {
        "dataset": "Test",
        "model": best_model_name,
        **test_results,
    },
])

metrics_to_save.to_csv(OUTPUT_FOLDER / "part_b_metrics.csv", index=False)

# Save actual and predicted delivery times for each test shipment.
test_predictions_df = pd.DataFrame({
    "actual_delivery_hours": y_test,
    "predicted_delivery_hours": test_predictions,
    "prediction_error_hours": y_test - test_predictions,
    "absolute_error_hours": np.abs(y_test - test_predictions),
})

test_predictions_df.to_csv(
    OUTPUT_FOLDER / "part_b_test_predictions.csv",
    index_label="row_index",
)


# ============================================================
# STEP 16: SAVE THE FINAL MODEL FOR FUTURE PREDICTIONS
# ============================================================

# Save the complete pipeline, not just the regression algorithm.
# This means future data will receive the same preprocessing automatically.

model_information = {
    "model": final_model,
    "feature_columns": feature_columns,
    "target_column": "delivery_hours",
    "selected_model": best_model_name,
}

joblib.dump(
    model_information,
    MODEL_FOLDER / "delivery_hours_model.pkl",
)


# ============================================================
# STEP 17: PRINT A FINAL SUMMARY
# ============================================================

print("\n" + "=" * 55)
print("PART B COMPLETED SUCCESSFULLY")
print("=" * 55)

print("Selected regression model:", best_model_name)
print("Training rows:", len(X_train))
print("Validation rows:", len(X_validation))
print("Test rows:", len(X_test))

print("\nFinal test performance:")
for metric_name, metric_value in test_results.items():
    print(metric_name + ":", round(metric_value, 4))

print("\nReusable dataset saved at:")
print(DATA_FOLDER / "modeling_features.csv")

print("\nTrained model saved at:")
print(MODEL_FOLDER / "delivery_hours_model.pkl")

print("\nAll Part B output files are saved in:")
print(OUTPUT_FOLDER)