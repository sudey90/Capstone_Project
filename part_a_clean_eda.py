
from pathlib import Path

import pandas as pd

from src.shipment_prep import clean_shipments, build_features


# --------------------------------------------------
# STEP 1: Define project folders
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# STEP 2: Load CSV files
# --------------------------------------------------

raw = pd.read_csv(DATA_DIR / "raw_shipments.csv")
hubs = pd.read_csv(DATA_DIR / "hubs.csv")

print("=" * 45)
print("PART A: DATA CLEANING AND EDA")
print("=" * 45)

print("Raw rows:", len(raw))


# --------------------------------------------------
# STEP 3: Clean shipment data
# --------------------------------------------------

clean, medians = clean_shipments(raw)

print("Clean rows:", len(clean))
print("Medians:", medians)


# --------------------------------------------------
# STEP 4: Merge hubs and build features
# --------------------------------------------------

model_df = build_features(clean, hubs)

print("Rows after hub merge:", len(model_df))
print("Feature columns:", model_df.columns.tolist())


# --------------------------------------------------
# STEP 5: Validate the modelling table
# --------------------------------------------------

assert len(raw) == 1224, "Unexpected raw row count"
assert len(clean) == 1200, "Unexpected clean row count"
assert len(model_df) == 1200, "Unexpected merged row count"

assert model_df["shipment_id"].is_unique, (
    "Duplicate shipment IDs found"
)

assert model_df["parcel_weight_kg"].notna().all(), (
    "Missing parcel weights remain"
)

assert model_df["delivery_hours"].notna().all(), (
    "Missing delivery_hours values found"
)

assert model_df["promised_hours"].notna().all(), (
    "Missing promised_hours values found"
)

print("All validation checks passed.")


# --------------------------------------------------
# STEP 6: Create the modelling table
# --------------------------------------------------

model_df.to_csv(
    DATA_DIR / "modelling_table.csv",
    index=False
)

print("Saved: data/modelling_table.csv")


# --------------------------------------------------
# STEP 7: Create the delay target
# --------------------------------------------------

# A parcel is delayed when actual delivery time
# exceeds its promised delivery time.

model_df["is_delayed"] = (
    model_df["delivery_hours"]
    > model_df["promised_hours"]
).astype(int)

model_df["delay_hours"] = (
    model_df["delivery_hours"]
    - model_df["promised_hours"]
).clip(lower=0)


# --------------------------------------------------
# STEP 8: Analyse delays by service type
# --------------------------------------------------

delay_by_service = (
    model_df.groupby("service_type")
    .agg(
        shipment_count=("shipment_id", "count"),
        avg_delivery_hours=("delivery_hours", "mean"),
        avg_promised_hours=("promised_hours", "mean"),
        delayed_shipments=("is_delayed", "sum"),
        avg_delay_hours=("delay_hours", "mean"),
    )
    .reset_index()
)

delay_by_service["delay_rate_pct"] = (
    delay_by_service["delayed_shipments"]
    / delay_by_service["shipment_count"]
    * 100
)

delay_by_service.to_csv(
    OUTPUT_DIR / "delay_by_service.csv",
    index=False
)

print("\nDelay analysis by service:")
print(delay_by_service.round(2))


# --------------------------------------------------
# STEP 9: Calculate numerical correlations
# --------------------------------------------------

numeric_df = model_df.select_dtypes(
    include="number"
)

correlation_matrix = numeric_df.corr()

correlation_matrix.to_csv(
    OUTPUT_DIR / "correlations.csv"
)

print("\nCorrelations with delivery_hours:")

print(
    correlation_matrix["delivery_hours"]
    .sort_values(ascending=False)
    .round(3)
)


# --------------------------------------------------
# STEP 10: Save a project validation summary
# --------------------------------------------------

summary = pd.DataFrame(
    [
        {
            "check": "Raw shipment rows",
            "value": len(raw),
        },
        {
            "check": "Clean shipment rows",
            "value": len(clean),
        },
        {
            "check": "Merged modelling rows",
            "value": len(model_df),
        },
        {
            "check": "Unique shipment IDs",
            "value": model_df["shipment_id"].nunique(),
        },
        {
            "check": "Missing parcel weights",
            "value": model_df["parcel_weight_kg"].isna().sum(),
        },
        {
            "check": "Missing delivery hours",
            "value": model_df["delivery_hours"].isna().sum(),
        },
    ]
)

summary.to_csv(
    OUTPUT_DIR / "part_a_validation.csv",
    index=False
)

print("\nPart A completed successfully.")
print("Check the data/ and outputs/ folders.")
print("=" * 45)