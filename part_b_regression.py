
# ==========================================
# PART B: DELIVERY-HOURS REGRESSION
# ==========================================
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# STEP 1: Set the file locations

project_folder = Path(__file__).parent
data_folder = project_folder / "data"
output_folder = project_folder / "outputs"

output_folder.mkdir(parents=True, exist_ok=True)


# STEP 2: Read the Part A modelling table

df = pd.read_csv(data_folder / "modelling_table.csv")

# These are the exact 17 features in the required order.
FEATURES = [
    "service_type",
    "payment_mode",
    "parcel_weight_kg",
    "distance_km",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "dispatch_time",
    "hub_city",
    "region",
    "hub_tier",
    "hub_daily_capacity",
    "promised_hours",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "log_declared_value",
]

# X contains the information used to make predictions.
X = df[FEATURES].copy()

# y contains the answer we want the model to learn.
y = df["delivery_hours"].copy()

# Convert dispatch_time into numeric information.
# The separate dispatch_hour feature is already available,
# so we do not need the full timestamp as a model input.
dispatch_dates = pd.to_datetime(X["dispatch_time"])

X["dispatch_time"] = (
    dispatch_dates.astype("int64") / 1_000_000_000
)


# STEP 3: Split the data into training, validation and test

# First, keep 70% for training and 30% temporarily aside.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)

# Split the temporary 30% equally into validation and test.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)

print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))


# STEP 4: Convert text categories into numeric columns

# Learn the category columns from training data only.
X_train = pd.get_dummies(X_train, dtype=int)

# Give validation and test the same columns as training.
# An unseen category becomes all zeros for its missing columns.
X_val = pd.get_dummies(X_val, dtype=int)
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

X_test = pd.get_dummies(X_test, dtype=int)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)


# STEP 5: Check scaling without data leakage

# Learn the scaling values using training data only.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Use those same scaling values for validation and test.
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# The largest absolute column mean should be near zero
# for scaled training data.
train_scaled_max_abs_mean = abs(
    X_train_scaled.mean(axis=0)
).max()

# Test data should not normally have column means of zero.
test_scaled_max_abs_mean = abs(
    X_test_scaled.mean(axis=0)
).max()

checks = pd.DataFrame({
    "metric": [
        "train_rows",
        "val_rows",
        "test_rows",
        "train_scaled_max_abs_mean",
        "test_scaled_max_abs_mean",
    ],
    "value": [
        len(X_train),
        len(X_val),
        len(X_test),
        train_scaled_max_abs_mean,
        test_scaled_max_abs_mean,
    ],
})

checks.to_csv(output_folder / "part_b_checks.csv", index=False)


# STEP 6: Create a function to calculate model scores

def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = mse ** 0.5
    r2 = r2_score(actual, predicted)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


# Store each model's validation scores here.
metric_rows = []


# STEP 7: Calculate the simple mean baseline

# Predict the average training delivery time for every
# validation shipment.
baseline_prediction = [y_train.mean()] * len(y_val)

baseline_scores = calculate_metrics(y_val, baseline_prediction)

metric_rows.append({
    "model": "mean_baseline",
    "split": "val",
    **baseline_scores,
})


# STEP 8: Test different decision-tree depths

depths = [1, 2, 3, 4, 6, 8, 12, None]

depth_rows = []
tree_models = {}

for depth in depths:

    tree = DecisionTreeRegressor(
        max_depth=depth,
        random_state=42
    )

    # Train the tree using training rows only.
    tree.fit(X_train, y_train)

    # Check its performance on training and validation data.
    train_prediction = tree.predict(X_train)
    val_prediction = tree.predict(X_val)

    train_rmse = mean_squared_error(
        y_train, train_prediction
    ) ** 0.5

    val_rmse = mean_squared_error(
        y_val, val_prediction
    ) ** 0.5

    depth_name = "unlimited" if depth is None else depth

    depth_rows.append({
        "max_depth": depth_name,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
    })

    tree_models[depth_name] = tree


depth_df = pd.DataFrame(depth_rows)
depth_df.to_csv(
    output_folder / "part_b_tree_depth.csv",
    index=False
)

# Choose the depth with the lowest validation RMSE.
# If there is a tie, prefer the smaller depth.
best_depth_row = min(
    depth_rows,
    key=lambda row: row["val_rmse"]
)

best_depth = best_depth_row["max_depth"]
best_tree = tree_models[best_depth]


# STEP 9: Create the four candidate models

models = {
    "linear_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ]),

    "decision_tree": best_tree,

    "random_forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),

    "gradient_boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    ),
}


# Train each candidate and score validation data only.
for model_name, model in models.items():

    model.fit(X_train, y_train)

    val_prediction = model.predict(X_val)
    scores = calculate_metrics(y_val, val_prediction)

    metric_rows.append({
        "model": model_name,
        "split": "val",
        **scores,
    })


# STEP 10: Try three gradient-boosting learning rates

learning_rates = [0.01, 0.1, 1.0]
learning_rate_rows = []

for lr in learning_rates:

    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=lr,
        random_state=42
    )

    gb_model.fit(X_train, y_train)

    val_prediction = gb_model.predict(X_val)

    learning_rate_rows.append({
        "learning_rate": lr,
        "val_rmse": mean_squared_error(
            y_val, val_prediction
        ) ** 0.5,
        "val_r2": r2_score(y_val, val_prediction),
    })


pd.DataFrame(learning_rate_rows).to_csv(
    output_folder / "part_b_gb_learning_rate.csv",
    index=False
)


# STEP 11: Choose the best candidate by validation RMSE

# Only the four candidate models are compared here.
candidate_rows = [
    row for row in metric_rows
    if row["model"] in models
]

candidate_rows.sort(key=lambda row: row["RMSE"])

# The first candidate has the lowest validation RMSE.
winning_model_name = candidate_rows[0]["model"]
winning_model = models[winning_model_name]


# STEP 12: Cross-validation for the two best candidates

# Use training plus validation data, not test data.
X_train_val = pd.concat([X_train, X_val], axis=0)
y_train_val = pd.concat([y_train, y_val], axis=0)

# One shared KFold object for both models.
kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_rows = []

for row in candidate_rows[:2]:

    model_name = row["model"]
    model = models[model_name]

    # Default scoring for regressors is R-squared.
    cv_scores = cross_val_score(
        model,
        X_train_val,
        y_train_val,
        cv=kf
    )

    cv_rows.append({
        "model": model_name,
        "mean_r2": cv_scores.mean(),
        "std_r2": cv_scores.std(),
    })


pd.DataFrame(cv_rows).to_csv(
    output_folder / "part_b_cv.csv",
    index=False
)


# STEP 13: Evaluate the winning model on the test set ONCE

# The winning model was already fitted on X_train.
# Do not fit it again and do not use test results to
# change the model choice.
test_prediction = winning_model.predict(X_test)

test_scores = calculate_metrics(y_test, test_prediction)

metric_rows.append({
    "model": winning_model_name,
    "split": "test",
    **test_scores,
})


# STEP 14: Save the model metrics

metrics_df = pd.DataFrame(
    metric_rows,
    columns=["model", "split", "MAE", "MSE", "RMSE", "R2"]
)

metrics_df.to_csv(
    output_folder / "part_b_metrics.csv",
    index=False
)


# STEP 15: Print a summary

print("\n====================================")
print("PART B FINISHED")
print("====================================")

print("\nBest decision-tree depth:", best_depth)
print("Best model on validation:", winning_model_name)

print("\nValidation model scores:")
print(
    metrics_df[metrics_df["split"] == "val"][
        ["model", "MAE", "RMSE", "R2"]
    ].to_string(index=False)
)

print("\nFinal test scores:")
print(
    metrics_df[metrics_df["split"] == "test"][
        ["model", "MAE", "RMSE", "R2"]
    ].to_string(index=False)
)

print("\nAll Part B output files have been saved.")