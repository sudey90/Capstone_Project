
# ============================================
# PART A: DATA CLEANING AND DELAY ANALYSIS
# ============================================

import pandas as pd
from pathlib import Path

from src.shipment_prep import clean_shipments, build_features


# STEP 1: Find the project folders

project_folder = Path(__file__).resolve().parent

data_folder = project_folder / "data"
output_folder = project_folder / "outputs"

# Create the outputs folder if it does not exist
output_folder.mkdir(exist_ok=True)


# STEP 2: Read the CSV files

raw = pd.read_csv(data_folder / "raw_shipments.csv")
hubs = pd.read_csv(data_folder / "hubs.csv")

print("Raw rows:", len(raw))


# STEP 3: Count duplicate rows

# Remove exact duplicate rows
raw_without_duplicates = raw.drop_duplicates()

duplicate_rows_removed = len(raw) - len(raw_without_duplicates)

print("Duplicate rows removed:", duplicate_rows_removed)


# STEP 4: Count missing values BEFORE cleaning

missing_distance_before = raw_without_duplicates["distance_km"].isna().sum()

missing_weight_before = raw_without_duplicates["parcel_weight_kg"].isna().sum()

missing_value_before = raw_without_duplicates["declared_value"].isna().sum()

missing_payment_before = raw_without_duplicates["payment_mode"].isna().sum()

# Count weights greater than 100 kg
sentinel_rows = (
    raw_without_duplicates["parcel_weight_kg"] > 100
).sum()

# Count different service type spellings before cleaning
service_types_before = raw_without_duplicates["service_type"].nunique()


# STEP 5: Clean the shipment data

# This function is already present in shipment_prep.py
clean, medians = clean_shipments(raw)

print("Clean rows:", len(clean))
print("Medians:", medians)

service_types_after = clean["service_type"].nunique()


# STEP 6: Add hub information and new features

model_df = build_features(clean, hubs)

print("Rows after hub merge:", len(model_df))


# STEP 7: Create the delay target

# 1 means delayed
# 0 means delivered within the promised time

model_df["is_delayed"] = (
    model_df["delivery_hours"] > model_df["promised_hours"]
).astype(int)


# STEP 8: Check the required columns

# These are the 17 feature columns specified in the assignment
features = [
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

# Add the two columns that contain the actual delivery result
features_and_target = features + [
    "delivery_hours",
    "is_delayed",
]

# Count all missing cells in these 19 columns
null_count = model_df[features_and_target].isna().sum().sum()

print("Missing values in features and target:", null_count)

# Stop if a required value is missing
assert null_count == 0, "Missing values found in required columns"

assert len(model_df) == 1200, "Expected 1200 rows"

assert model_df["shipment_id"].is_unique, "Duplicate shipment IDs found"


# STEP 9: Save the modelling table

model_df.to_csv(
    data_folder / "modelling_table.csv",
    index=False
)

print("Modelling table saved.")


# STEP 10: Calculate delay statistics

delayed_count = model_df["is_delayed"].sum()

delay_rate = model_df["is_delayed"].mean()

print("Delayed shipments:", delayed_count)
print("Delay rate:", round(delay_rate, 4))


# STEP 11: Calculate delivery-hour outliers

q1 = model_df["delivery_hours"].quantile(0.25)
q3 = model_df["delivery_hours"].quantile(0.75)

iqr = q3 - q1

lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outlier_count = (
    (model_df["delivery_hours"] < lower_bound)
    | (model_df["delivery_hours"] > upper_bound)
).sum()


# STEP 12: Calculate declared-value statistics

value_mean = model_df["declared_value_inr"].mean()
value_median = model_df["declared_value_inr"].median()

log_value_mean = model_df["log_declared_value"].mean()
log_value_median = model_df["log_declared_value"].median()


# STEP 13: Create the checks file

# Each row contains one metric name and its calculated value

checks = pd.DataFrame({
    "metric": [
        "raw_rows",
        "duplicate_rows_removed",
        "rows_after_dedup",
        "missing_distance_km_before",
        "missing_parcel_weight_before",
        "sentinel_weight_rows",
        "missing_declared_value_before",
        "missing_payment_mode_before",
        "service_type_spellings_before",
        "service_type_spellings_after",
        "median_distance_km",
        "median_parcel_weight_kg",
        "median_declared_value_inr",
        "rows_after_hub_merge",
        "nulls_in_features_and_target",
        "delayed_count",
        "delay_rate",
        "hours_q1",
        "hours_q3",
        "hours_lower_bound",
        "hours_upper_bound",
        "hours_outlier_count",
        "declared_value_mean_inr",
        "declared_value_median_inr",
        "log_declared_value_mean",
        "log_declared_value_median",
    ],

    "value": [
        len(raw),
        duplicate_rows_removed,
        len(raw_without_duplicates),
        missing_distance_before,
        missing_weight_before,
        sentinel_rows,
        missing_value_before,
        missing_payment_before,
        service_types_before,
        service_types_after,
        medians["distance_km"],
        medians["parcel_weight_kg"],
        medians["declared_value_inr"],
        len(model_df),
        null_count,
        delayed_count,
        round(delay_rate, 4),
        round(q1, 2),
        round(q3, 2),
        round(lower_bound, 2),
        round(upper_bound, 2),
        outlier_count,
        round(value_mean, 2),
        round(value_median, 2),
        round(log_value_mean, 3),
        round(log_value_median, 3),
    ]
})

checks.to_csv(
    output_folder / "part_a_checks.csv",
    index=False
)

print("Checks file saved.")


# STEP 14: Create the delay evidence table

# Create the distance groups
model_df["distance_band"] = pd.cut(
    model_df["distance_km"],
    bins=[0, 200, 400, 600, 2000],
    labels=["0-200", "201-400", "401-600", "601+"],
    right=True
)

# Create an empty table to collect the results
delay_results = pd.DataFrame(
    columns=[
        "dimension",
        "group",
        "shipments",
        "delayed",
        "delay_rate",
        "mean_hours",
    ]
)


# Analyse service type, hub tier and distance band
for column in ["service_type", "hub_tier", "distance_band"]:

    grouped = model_df.groupby(column, observed=True)

    # Count shipments in each group
    shipments = grouped["shipment_id"].count()

    # Count delayed shipments in each group
    delayed = grouped["is_delayed"].sum()

    # Calculate the percentage of delayed shipments
    rates = grouped["is_delayed"].mean()

    # Calculate average delivery hours
    average_hours = grouped["delivery_hours"].mean()

    # Put the results into a table
    result = pd.DataFrame({
        "dimension": column,
        "group": shipments.index.astype(str),
        "shipments": shipments.values,
        "delayed": delayed.values,
        "delay_rate": rates.values,
        "mean_hours": average_hours.values,
    })

    # Add these results to the main table
    delay_results = pd.concat(
        [delay_results, result],
        ignore_index=True
    )


# Round delay rates to four decimal places
delay_results["delay_rate"] = delay_results["delay_rate"].round(4)

# Save the file
delay_results.to_csv(
    output_folder / "part_a_delay_by_group.csv",
    index=False
)

print("Delay evidence file saved.")


# STEP 15: Create the correlation matrix

# Select only numerical columns
numeric_data = model_df.select_dtypes(include="number")

# Calculate correlations
correlations = numeric_data.corr()

# Save row names in the first column
correlations.to_csv(
    output_folder / "part_a_correlations.csv",
    index_label="column"
)

print("Correlation file saved.")


# STEP 16: Finish

print()
print("====================================")
print("PART A SCRIPT FINISHED")
print("====================================")
print("Modelling table:", len(model_df), "rows")
print("Delayed shipments:", delayed_count)
print("Delay rate:", round(delay_rate, 4))
print("Please compare the CSV outputs with the reference values.")