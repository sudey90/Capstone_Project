
# ============================================
# PART D: PACKAGE THE FINAL CLASSIFICATION MODEL
# ============================================

# STEP 1: IMPORT REQUIRED LIBRARIES

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.shipment_prep import clean_shipments


# ============================================
# STEP 2: SET UP PROJECT FOLDERS
# ============================================

project_folder = Path(__file__).resolve().parent

data_folder = project_folder / "data"
output_folder = project_folder / "outputs"
model_folder = project_folder / "models"

output_folder.mkdir(exist_ok=True)
model_folder.mkdir(exist_ok=True)


# ============================================
# STEP 3: READ THE PART C CONFIGURATION
# ============================================

config = pd.read_csv(
    output_folder / "part_c_final_config.csv"
)

# There is one row containing our final settings.
config = config.iloc[0]

# Read the settings selected in Part C.
selector_k = int(config["selector_k"])
n_estimators = int(config["n_estimators"])
min_samples_leaf = int(config["min_samples_leaf"])
threshold = float(config["threshold"])

# Convert unlimited depth into Python None.
if str(config["max_depth"]).lower() == "unlimited":
    max_depth = None
else:
    max_depth = int(config["max_depth"])

print("FINAL MODEL CONFIGURATION")
print("-------------------------")
print("Selected features:", selector_k)
print("Number of trees:", n_estimators)
print("Maximum depth:", max_depth)
print("Minimum samples per leaf:", min_samples_leaf)
print("Alert threshold:", threshold)


# ============================================
# STEP 4: DEFINE THE SAME 17 FEATURES FROM PART C
# ============================================

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

TARGET = "is_delayed"


# ============================================
# STEP 5: LOAD THE HISTORICAL DATA
# ============================================

df = pd.read_csv(
    data_folder / "modelling_table.csv"
)

X = df[FEATURES].copy()
y = df[TARGET].astype(int).copy()

print("\nTOTAL HISTORICAL SHIPMENTS:", len(df))


# ============================================
# STEP 6: RECREATE THE SAME DATA SPLIT AS PART C
# ============================================

# First split:
# 70% training and 30% temporary data.

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# Second split:
# Divide the temporary data equally into
# 15% validation and 15% test data.

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nDATA SPLIT")
print("----------")
print("Training rows:", len(X_train))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))

# Check that the split matches Part C.
assert len(X_train) == 840
assert len(X_val) == 180
assert len(X_test) == 180


# ============================================
# STEP 7: BUILD THE FINAL MODEL
# ============================================

# First, select the best seven features.
# Then train the Random Forest using those features.

final_model = Pipeline([
    (
        "selector",
        SelectKBest(
            score_func=f_classif,
            k=selector_k
        )
    ),
    (
        "model",
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )
    )
])


# ============================================
# STEP 8: TRAIN THE FINAL MODEL
# ============================================

# Train using only the training data.
final_model.fit(X_train, y_train)

print("\nFinal model trained successfully.")


# ============================================
# STEP 9: SAVE THE TRAINED MODEL
# ============================================

model_path = model_folder / "delay_model.pkl"

joblib.dump(final_model, model_path)

print("Saved model:", model_path)


# ============================================
# STEP 10: SAVE THE ALERT THRESHOLD
# ============================================

threshold_df = pd.DataFrame({
    "threshold": [threshold]
})

threshold_df.to_csv(
    model_folder / "delay_threshold.csv",
    index=False
)


# ============================================
# STEP 11: SAVE THE FEATURE ORDER
# ============================================

feature_df = pd.DataFrame({
    "feature": FEATURES
})

feature_df.to_csv(
    model_folder / "feature_order.csv",
    index=False
)


# ============================================
# STEP 12: SAVE THE PART A IMPUTATION MEDIANS
# ============================================

# Read the original raw shipment data.
raw = pd.read_csv(
    data_folder / "raw_shipments.csv"
)

# Run the same cleaning function used in Part A.
# This calculates the original three median values.
clean_df, medians = clean_shipments(raw)

impute_df = pd.DataFrame({
    "column": list(medians.keys()),
    "median": list(medians.values())
})

impute_df.to_csv(
    model_folder / "impute_values.csv",
    index=False
)

print("\nMODEL SUPPORT FILES SAVED.")


# ============================================
# STEP 13: RELOAD THE SAVED MODEL
# ============================================

# Load the model from the saved file.
loaded_model = joblib.load(model_path)

print("Saved model loaded successfully.")


# ============================================
# STEP 14: COMPARE MODEL PREDICTIONS
# ============================================

# Get delayed-shipment probabilities
# from the original model.

original_probabilities = final_model.predict_proba(
    X_test
)[:, 1]

# Get delayed-shipment probabilities
# from the reloaded model.

reloaded_probabilities = loaded_model.predict_proba(
    X_test
)[:, 1]

# Find the largest difference between both sets
# of probabilities.

max_difference = np.max(
    np.abs(
        original_probabilities - reloaded_probabilities
    )
)

print("\nMODEL RELOAD CHECK")
print("------------------")
print("Test rows scored:", len(X_test))
print("Maximum probability difference:", max_difference)


# ============================================
# STEP 15: SAVE THE RELOAD CHECK RESULTS
# ============================================

reload_check = pd.DataFrame({
    "metric": [
        "test_rows_scored",
        "reload_max_abs_probability_diff",
        "threshold"
    ],
    "value": [
        len(X_test),
        max_difference,
        threshold
    ]
})

reload_check.to_csv(
    output_folder / "part_d_reload_check.csv",
    index=False
)


# ============================================
# STEP 16: FINAL VALIDATION
# ============================================

# The original and reloaded model should give
# practically identical probabilities.

assert len(X_test) == 180
assert max_difference <= 1e-9

# Check that all four model files exist.

assert (model_folder / "delay_model.pkl").exists()
assert (model_folder / "delay_threshold.csv").exists()
assert (model_folder / "feature_order.csv").exists()
assert (model_folder / "impute_values.csv").exists()

print("\n====================================")
print("PART D MODEL PACKAGING COMPLETED")
print("====================================")

print("Test rows verified:", len(X_test))
print("Maximum probability difference:", max_difference)
print("Alert threshold:", threshold)

print("\nAll model files saved successfully.")
print("All reload checks passed.")