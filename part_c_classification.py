
"""
PART C: DELAY CLASSIFICATION AND ALERT THRESHOLD

Objective:
Predict whether a shipment will be delayed.

Input:
    data/modelling_table.csv

Outputs:
    outputs/part_c_models.csv
    outputs/part_c_feature_scores.csv
    outputs/part_c_selectk.csv
    outputs/part_c_tuning.csv
    outputs/part_c_thresholds.csv
    outputs/part_c_final_test.csv
    outputs/part_c_final_config.csv
"""

# ============================================================
# STEP 1: IMPORT THE REQUIRED LIBRARIES
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# STEP 2: SET UP PROJECT FOLDERS
# ============================================================

PROJECT_FOLDER = Path(__file__).resolve().parent

DATA_FOLDER = PROJECT_FOLDER / "data"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"

OUTPUT_FOLDER.mkdir(exist_ok=True)

# Read the dataset created in Part A.
df = pd.read_csv(DATA_FOLDER / "modelling_table.csv")

print("Dataset loaded successfully.")
print("Total shipments:", len(df))


# ============================================================
# STEP 3: SELECT THE EXACT 17 INPUT FEATURES
# ============================================================

FEATURES = [
    "promised_hours",
    "distance_km",
    "parcel_weight_kg",
    "log_declared_value",
    "n_stops",
    "rain_flag",
    "sale_event_flag",
    "is_cod",
    "is_weekend",
    "dispatch_hour",
    "hub_tier",
    "hub_daily_capacity",
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]

# This is the column we want the model to predict.
TARGET = "is_delayed"

# Five columns that contain bookkeeping or random identifier information.
NOISE_COLUMNS = [
    "label_print_batch",
    "scanner_build_no",
    "tape_roll_no",
    "sticker_serial_mod",
    "shelf_slot_no"
]

# Check whether all required columns exist.
for column in FEATURES + [TARGET]:

    if column not in df.columns:
        raise ValueError("Missing required column: " + column)

# X contains the 17 input features.
X = df[FEATURES].copy()

# y contains the answer: 0 = on time, 1 = delayed.
y = df[TARGET].astype(int).copy()

# Check for missing values.
if X.isna().any().any() or y.isna().any():
    raise ValueError("Missing values found. Please check Part A.")


# ============================================================
# STEP 4: SPLIT DATA INTO TRAINING, VALIDATION AND TEST
# ============================================================

# First split: 70% training and 30% temporary data.
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Second split: divide the temporary data equally.
# This gives us 15% validation and 15% test data.
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)

print("\nDATA SPLIT RESULTS")
print("------------------")

print("Training rows:", len(X_train))
print("Training delayed:", int(y_train.sum()))

print("Validation rows:", len(X_val))
print("Validation delayed:", int(y_val.sum()))

print("Test rows:", len(X_test))
print("Test delayed:", int(y_test.sum()))

# Check the expected results.
assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)

assert (
    int(y_train.sum()),
    int(y_val.sum()),
    int(y_test.sum())
) == (195, 42, 41)


# ============================================================
# STEP 5: COMPARE FOUR CLASSIFICATION MODELS
# ============================================================

# We will store the results of all four models here.
model_results = []

# ------------------------------------------------------------
# MODEL 1: BLIND BASELINE
# ------------------------------------------------------------

# Predict that every shipment will be on time.
blind_predictions = np.zeros(len(y_val), dtype=int)

tn, fp, fn, tp = confusion_matrix(
    y_val, blind_predictions, labels=[0, 1]
).ravel()

model_results.append({
    "model": "blind_all_on_time",
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
    "accuracy": accuracy_score(y_val, blind_predictions),
    "precision": precision_score(
        y_val, blind_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_val, blind_predictions, zero_division=0
    ),
    "f1": f1_score(
        y_val, blind_predictions, zero_division=0
    )
})


# ------------------------------------------------------------
# MODEL 2: LOGISTIC REGRESSION
# ------------------------------------------------------------

# Scaling makes numerical features comparable.
# Pipeline first scales the data and then trains Logistic Regression.
logistic_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_val)

tn, fp, fn, tp = confusion_matrix(
    y_val, logistic_predictions, labels=[0, 1]
).ravel()

model_results.append({
    "model": "logistic_regression",
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
    "accuracy": accuracy_score(y_val, logistic_predictions),
    "precision": precision_score(
        y_val, logistic_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_val, logistic_predictions, zero_division=0
    ),
    "f1": f1_score(
        y_val, logistic_predictions, zero_division=0
    )
})


# ------------------------------------------------------------
# MODEL 3: DECISION TREE
# ------------------------------------------------------------

# Try different tree depths.
tree_depths = [2, 3, 4, 5, 6, 8, 10]

tree_results = []

for depth in tree_depths:

    tree_model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        random_state=42
    )

    tree_model.fit(X_train, y_train)

    tree_predictions = tree_model.predict(X_val)

    tree_f1 = f1_score(
        y_val, tree_predictions, zero_division=0
    )

    tree_results.append({
        "depth": depth,
        "f1": tree_f1,
        "model": tree_model
    })

# Sort by highest F1.
# If two depths have the same F1, choose the smaller depth.
tree_results = sorted(
    tree_results,
    key=lambda result: (-result["f1"], result["depth"])
)

best_tree = tree_results[0]["model"]
best_depth = tree_results[0]["depth"]

tree_predictions = best_tree.predict(X_val)

tn, fp, fn, tp = confusion_matrix(
    y_val, tree_predictions, labels=[0, 1]
).ravel()

model_results.append({
    "model": "decision_tree_depth_" + str(best_depth),
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
    "accuracy": accuracy_score(y_val, tree_predictions),
    "precision": precision_score(
        y_val, tree_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_val, tree_predictions, zero_division=0
    ),
    "f1": f1_score(
        y_val, tree_predictions, zero_division=0
    )
})


# ------------------------------------------------------------
# MODEL 4: DEFAULT RANDOM FOREST
# ------------------------------------------------------------

forest_17 = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

forest_17.fit(X_train, y_train)

forest_predictions = forest_17.predict(X_val)

tn, fp, fn, tp = confusion_matrix(
    y_val, forest_predictions, labels=[0, 1]
).ravel()

model_results.append({
    "model": "random_forest_default",
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
    "accuracy": accuracy_score(y_val, forest_predictions),
    "precision": precision_score(
        y_val, forest_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_val, forest_predictions, zero_division=0
    ),
    "f1": f1_score(
        y_val, forest_predictions, zero_division=0
    )
})

# Save all four model results.
models_df = pd.DataFrame(model_results)

models_df.to_csv(
    OUTPUT_FOLDER / "part_c_models.csv",
    index=False
)

print("\nMODEL COMPARISON")
print(models_df.round(4).to_string(index=False))


# ============================================================
# STEP 6: IDENTIFY IMPORTANT AND NOISY FEATURES
# ============================================================

# First, get the F-score of all 17 features.
# Higher F-score means a stronger relationship with the target.
selector_all = SelectKBest(
    score_func=f_classif,
    k="all"
)

selector_all.fit(X_train, y_train)

f_scores = selector_all.scores_

# Get Random Forest importance for all 17 features.
rf_importances = forest_17.feature_importances_

# Create a table containing both scores.
feature_scores = pd.DataFrame({
    "feature": FEATURES,
    "f_score": f_scores,
    "rf_importance": rf_importances
})

# Rank features by F-score.
# Rank 1 means the highest F-score.
feature_scores["f_rank"] = feature_scores["f_score"].rank(
    ascending=False,
    method="min"
).astype(int)

# Identify the five noise columns.
feature_scores["is_noise"] = feature_scores["feature"].isin(
    NOISE_COLUMNS
)

# Arrange the columns in the required order.
feature_scores = feature_scores[
    ["feature", "f_score", "f_rank", "rf_importance", "is_noise"]
]

# Sort the table by feature rank.
feature_scores = feature_scores.sort_values(
    "f_rank"
).reset_index(drop=True)

feature_scores.to_csv(
    OUTPUT_FOLDER / "part_c_feature_scores.csv",
    index=False
)

# Calculate the combined Random Forest importance of noise columns.
noise_importance = feature_scores.loc[
    feature_scores["is_noise"], "rf_importance"
].sum()

print("\nFEATURE IMPORTANCE")
print(feature_scores.round(4).to_string(index=False))

print("\nCombined noise importance:", round(noise_importance, 4))


# ============================================================
# STEP 7: COMPARE DIFFERENT FEATURE COUNTS
# ============================================================

# Test whether 5, 7, 10, 12 or 17 features give the best result.
k_values = [5, 7, 10, 12, 17]

selectk_results = []

for k in k_values:

    selectk_model = Pipeline([
        ("selector", SelectKBest(f_classif, k=k)),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    selectk_model.fit(X_train, y_train)

    predictions = selectk_model.predict(X_val)

    validation_f1 = f1_score(
        y_val, predictions, zero_division=0
    )

    selectk_results.append({
        "k": k,
        "val_f1": validation_f1
    })

selectk_df = pd.DataFrame(selectk_results)

selectk_df.to_csv(
    OUTPUT_FOLDER / "part_c_selectk.csv",
    index=False
)

print("\nFEATURE COUNT COMPARISON")
print(selectk_df.round(4).to_string(index=False))

# From this point onwards, we will use seven features.
K = 7


# ============================================================
# STEP 8: TUNE THE RANDOM FOREST MODEL
# ============================================================

# Select the best seven features using training data only.
selector_7 = SelectKBest(
    score_func=f_classif,
    k=7
)

X_train_7 = selector_7.fit_transform(X_train, y_train)

# Use the same selected columns for validation.
X_val_7 = selector_7.transform(X_val)

# Define the different settings we want to test.
parameter_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 5, 10]
}

# ------------------------------------------------------------
# GRID SEARCH
# ------------------------------------------------------------

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=parameter_grid,
    cv=5
)

grid_search.fit(X_train_7, y_train)

grid_predictions = grid_search.predict(X_val_7)

grid_f1 = f1_score(
    y_val, grid_predictions, zero_division=0
)

print("\nGRID SEARCH")
print("Best parameters:", grid_search.best_params_)
print("Best CV accuracy:", grid_search.best_score_)
print("Validation F1:", grid_f1)


# ------------------------------------------------------------
# RANDOMIZED SEARCH
# ------------------------------------------------------------

random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=parameter_grid,
    n_iter=9,
    cv=5,
    random_state=42
)

random_search.fit(X_train_7, y_train)

random_predictions = random_search.predict(X_val_7)

random_f1 = f1_score(
    y_val, random_predictions, zero_division=0
)

print("\nRANDOMIZED SEARCH")
print("Best parameters:", random_search.best_params_)
print("Best CV accuracy:", random_search.best_score_)
print("Validation F1:", random_f1)


# ------------------------------------------------------------
# DEFAULT RANDOM FOREST USING THE SAME SEVEN FEATURES
# ------------------------------------------------------------

default_forest_7 = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

default_forest_7.fit(X_train_7, y_train)

default_predictions = default_forest_7.predict(X_val_7)

default_f1 = f1_score(
    y_val, default_predictions, zero_division=0
)

print("\nDEFAULT RANDOM FOREST")
print("Validation F1:", default_f1)


# Save the results of all three approaches.
tuning_results = pd.DataFrame([
    {
        "search": "grid_search",
        "candidates": 27,
        "cv_folds": 5,
        "total_fits": 135,
        "best_params": str(grid_search.best_params_),
        "best_cv_accuracy": grid_search.best_score_,
        "val_f1": grid_f1
    },
    {
        "search": "randomized_search",
        "candidates": 9,
        "cv_folds": 5,
        "total_fits": 45,
        "best_params": str(random_search.best_params_),
        "best_cv_accuracy": random_search.best_score_,
        "val_f1": random_f1
    },
    {
        "search": "default_forest_same_7_features",
        "candidates": 1,
        "cv_folds": 0,
        "total_fits": 1,
        "best_params": "defaults",
        "best_cv_accuracy": np.nan,
        "val_f1": default_f1
    }
])

tuning_results.to_csv(
    OUTPUT_FOLDER / "part_c_tuning.csv",
    index=False
)

print("\nTUNING COMPARISON")
print(tuning_results.round(4).to_string(index=False))


# ============================================================
# STEP 9: CHOOSE THE BEST FINAL MODEL
# ============================================================

# Priority when validation F1 values are tied:
# 1. Default forest
# 2. Grid search
# 3. Randomized search

priority = {
    "default_forest_same_7_features": 1,
    "grid_search": 2,
    "randomized_search": 3
}

# Sort by highest validation F1 and then by the priority above.
tuning_results = tuning_results.sort_values(
    by=["val_f1"],
    ascending=False,
    kind="stable"
)

# Apply the specified tie-breaking rule.
tuning_results["priority"] = tuning_results["search"].map(priority)

tuning_results = tuning_results.sort_values(
    by=["val_f1", "priority"],
    ascending=[False, True]
)

winning_search = tuning_results.iloc[0]["search"]

# Get the parameters of the winning model.
if winning_search == "grid_search":

    chosen_parameters = grid_search.best_params_

elif winning_search == "randomized_search":

    chosen_parameters = random_search.best_params_

else:

    chosen_parameters = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1
    }

print("\nWINNING MODEL:", winning_search)
print("Chosen parameters:", chosen_parameters)


# Build the final pipeline.
# The selector is fitted on training data only.
final_model = Pipeline([
    ("selector", SelectKBest(f_classif, k=7)),
    ("model", RandomForestClassifier(
        random_state=42,
        **chosen_parameters
    ))
])

final_model.fit(X_train, y_train)


# ============================================================
# STEP 10: SELECT THE ALERT THRESHOLD
# ============================================================

# Get the probability that each validation shipment will be delayed.
validation_probabilities = final_model.predict_proba(X_val)[:, 1]

# Use exact decimal values.
threshold_values = [
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9
]

threshold_results = []

for threshold in threshold_values:

    # Predict delayed when probability >= threshold.
    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    threshold_results.append({
        "threshold": threshold,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_val, predictions, zero_division=0
        )
    })

threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    OUTPUT_FOLDER / "part_c_thresholds.csv",
    index=False
)

print("\nTHRESHOLD COMPARISON")
print(threshold_df.round(4).to_string(index=False))

# Find thresholds where recall is at least 80%.
eligible_thresholds = threshold_df[
    threshold_df["recall"] >= 0.80
]

# Select the highest eligible threshold.
if len(eligible_thresholds) > 0:

    chosen_threshold = eligible_thresholds["threshold"].max()

else:

    chosen_threshold = 0.1

# Get the validation metrics at the selected threshold.
chosen_row = threshold_df[
    threshold_df["threshold"] == chosen_threshold
].iloc[0]

print("\nCHOSEN ALERT THRESHOLD:", chosen_threshold)

print("Validation precision:", round(chosen_row["precision"], 4))
print("Validation recall:", round(chosen_row["recall"], 4))
print("Validation F1:", round(chosen_row["f1"], 4))


# ============================================================
# STEP 11: FINAL TEST EVALUATION
# ============================================================

# IMPORTANT:
# We are now evaluating the test data for the first and only time.

test_probabilities = final_model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= chosen_threshold
).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test, test_predictions, labels=[0, 1]
).ravel()

final_test_results = pd.DataFrame([{
    "threshold": chosen_threshold,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
    "accuracy": accuracy_score(y_test, test_predictions),
    "precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_test, test_predictions, zero_division=0
    ),
    "f1": f1_score(
        y_test, test_predictions, zero_division=0
    )
}])

final_test_results.to_csv(
    OUTPUT_FOLDER / "part_c_final_test.csv",
    index=False
)


# ============================================================
# STEP 12: SAVE THE FINAL MODEL CONFIGURATION
# ============================================================

final_configuration = pd.DataFrame([{
    "selector_k": 7,
    "n_estimators": chosen_parameters["n_estimators"],
    "max_depth": (
        "unlimited"
        if chosen_parameters["max_depth"] is None
        else chosen_parameters["max_depth"]
    ),
    "min_samples_leaf": chosen_parameters["min_samples_leaf"],
    "threshold": chosen_threshold,
    "chosen_from": winning_search
}])

final_configuration.to_csv(
    OUTPUT_FOLDER / "part_c_final_config.csv",
    index=False
)


# ============================================================
# STEP 13: DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PART C COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFINAL TEST RESULTS")
print(final_test_results.round(4).to_string(index=False))

print("\nFINAL CONFIGURATION")
print(final_configuration.to_string(index=False))

print("\nAll Part C output files are saved in:")
print(OUTPUT_FOLDER)