
# ============================================
# PART D: PREDICT DELAYS FOR NEW SHIPMENTS
# ============================================


# ============================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# ============================================

from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from src.shipment_prep import clean_shipments, build_features


# ============================================
# STEP 2: SET UP PROJECT FOLDERS
# ============================================

project_folder = Path(__file__).resolve().parent

data_folder = project_folder / "data"
output_folder = project_folder / "outputs"
model_folder = project_folder / "models"

output_folder.mkdir(exist_ok=True)


# ============================================
# STEP 3: LOAD THE NEW SHIPMENT DATA
# ============================================

new_shipments = pd.read_csv(
    data_folder / "new_shipments.csv"
)

# Keep the original shipment order.
original_shipment_ids = new_shipments["shipment_id"].tolist()

print("NEW SHIPMENT DATA")
print("-----------------")
print("Total new shipments:", len(new_shipments))


# ============================================
# STEP 4: LOAD THE HISTORICAL MEDIANS
# ============================================

# These are the median values calculated in Part A
# and saved during model packaging.

impute_values = pd.read_csv(
    model_folder / "impute_values.csv"
)

# Convert the CSV into a dictionary.
# Example: distance_km -> 314.95

medians = dict(
    zip(
        impute_values["column"],
        impute_values["median"]
    )
)

print("\nHISTORICAL MEDIANS")
print("------------------")
print(medians)


# ============================================
# STEP 5: CLEAN THE NEW SHIPMENT DATA
# ============================================

# Use the same cleaning function from Part A.
# Pass the historical medians so that missing
# values are filled consistently.

clean_df, used_medians = clean_shipments(
    new_shipments,
    medians=medians
)

print("\nNEW SHIPMENTS CLEANED SUCCESSFULLY.")


# ============================================
# STEP 6: LOAD HUB INFORMATION
# ============================================

hubs = pd.read_csv(
    data_folder / "hubs.csv"
)

# Use the same feature-building function from Part A.
# This attaches hub details and creates features
# such as promised_hours, is_cod and dispatch_hour.

features_df = build_features(
    clean_df,
    hubs
)

print("Hub information and features added.")


# ============================================
# STEP 7: LOAD THE FEATURE ORDER
# ============================================

feature_order_df = pd.read_csv(
    model_folder / "feature_order.csv"
)

# Get the 17 feature names in the saved order.
feature_order = feature_order_df["feature"].tolist()

# Check that all required features exist.
for column in feature_order:

    if column not in features_df.columns:
        raise ValueError(
            "Missing required feature: " + column
        )

# Arrange the columns in exactly the same order
# used when the model was trained.

X_new = features_df[feature_order].copy()

# Check that there are no missing feature values.
if X_new.isna().any().any():
    raise ValueError("Missing feature values found.")


# ============================================
# STEP 8: LOAD THE SAVED MODEL
# ============================================

model = joblib.load(
    model_folder / "delay_model.pkl"
)

print("\nSAVED MODEL LOADED SUCCESSFULLY.")


# ============================================
# STEP 9: LOAD THE ALERT THRESHOLD
# ============================================

threshold_df = pd.read_csv(
    model_folder / "delay_threshold.csv"
)

threshold = float(
    threshold_df.loc[0, "threshold"]
)

print("Alert threshold:", threshold)


# ============================================
# STEP 10: PREDICT DELAY PROBABILITIES
# ============================================

# predict_proba gives the probability of both classes.
# Column 0 = probability of on-time delivery.
# Column 1 = probability of delayed delivery.

delay_probabilities = model.predict_proba(
    X_new
)[:, 1]

# Round probabilities to four decimal places.

delay_probabilities = np.round(
    delay_probabilities,
    4
)

# Flag a shipment when its delay probability
# is greater than or equal to the threshold.

flagged = (
    delay_probabilities >= threshold
).astype(int)

print("\nDELAY PREDICTIONS COMPLETED.")


# ============================================
# STEP 11: LOAD HUB SEGMENTS
# ============================================

hub_segments = pd.read_csv(
    output_folder / "hub_segments.csv"
)

# Keep only the columns required for joining.

hub_segments = hub_segments[
    ["hub_id", "segment"]
]

# Attach the segment to each new shipment.

result = clean_df[
    ["shipment_id", "hub_id"]
].merge(
    hub_segments,
    on="hub_id",
    how="left",
    validate="many_to_one",
    sort=False
)

# Check that every shipment received a segment.

if result["segment"].isna().any():
    raise ValueError("Some shipments do not have a hub segment.")


# ============================================
# STEP 12: ADD PREDICTION RESULTS
# ============================================

result["delay_probability"] = delay_probabilities
result["flagged"] = flagged


# ============================================
# STEP 13: RECOMMEND AN OPERATIONAL ACTION
# ============================================

# Rules:
#
# 1. Flagged + Escalate hub:
#    Escalate to hub manager
#
# 2. Flagged + Standard or Watch hub:
#    Proactive customer call
#
# 3. Not flagged:
#    No action

result["action"] = "No action"

result.loc[
    (result["flagged"] == 1) &
    (result["segment"] == "Escalate"),
    "action"
] = "Escalate to hub manager"

result.loc[
    (result["flagged"] == 1) &
    (result["segment"] != "Escalate"),
    "action"
] = "Proactive customer call"


# ============================================
# STEP 14: KEEP ONLY THE REQUIRED COLUMNS
# ============================================

result = result[
    [
        "shipment_id",
        "hub_id",
        "segment",
        "delay_probability",
        "flagged",
        "action"
    ]
]


# ============================================
# STEP 15: RESTORE ORIGINAL SHIPMENT ORDER
# ============================================

# Keep the same order as new_shipments.csv.

result["original_order"] = range(len(result))

result = result.sort_values(
    "original_order"
).drop(
    columns=["original_order"]
).reset_index(drop=True)


# ============================================
# STEP 16: SAVE THE FINAL PREDICTIONS
# ============================================

output_path = (
    output_folder / "new_shipment_actions.csv"
)

result.to_csv(
    output_path,
    index=False
)


# ============================================
# STEP 17: DISPLAY SUMMARY
# ============================================

print("\nNEW SHIPMENT ACTION SUMMARY")
print("----------------------------")

print("Total shipments:", len(result))

print(
    "Flagged shipments:",
    int(result["flagged"].sum())
)

print(
    "Not flagged:",
    int((result["flagged"] == 0).sum())
)

print("\nRECOMMENDED ACTIONS")

print(
    result["action"].value_counts().to_string()
)


# ============================================
# STEP 18: FINAL VALIDATION
# ============================================

# There should be one output row per new shipment.

assert len(result) == len(new_shipments)

# Every shipment ID should be unique.

assert result["shipment_id"].is_unique

# There should be no missing values.

assert not result.isna().any().any()

# Check that the original shipment order is maintained.

assert (
    result["shipment_id"].tolist()
    == original_shipment_ids
)

# Check that probabilities are between 0 and 1.

assert result["delay_probability"].between(
    0, 1
).all()

# Check that flagged values follow the threshold.

assert (
    result["flagged"].to_numpy()
    == (
        result["delay_probability"].to_numpy()
        >= threshold
    ).astype(int)
).all()

# Check that all actions follow the required rules.

assert (
    result.loc[
        result["flagged"] == 0, "action"
    ] == "No action"
).all()

assert (
    result.loc[
        (result["flagged"] == 1) &
        (result["segment"] == "Escalate"),
        "action"
    ] == "Escalate to hub manager"
).all()

assert (
    result.loc[
        (result["flagged"] == 1) &
        (result["segment"] != "Escalate"),
        "action"
    ] == "Proactive customer call"
).all()


# ============================================
# STEP 19: DISPLAY FINAL RESULTS
# ============================================

print("\n====================================")
print("NEW SHIPMENT PREDICTION COMPLETED")
print("====================================")

print("Total rows:", len(result))
print("Unique shipment IDs:", result["shipment_id"].nunique())
print("Missing values:", int(result.isna().sum().sum()))

print("\nOutput saved to:")
print(output_path)

print("\nAll prediction checks passed.")