import numpy as np
import pandas as pd

PROMISED_HOURS = {
    "Express": 36,
    "Standard": 60,
    "Economy": 96,
}

def clean_shipments(raw, medians=None):
    """Clean raw shipment data and return the cleaned data and medians."""

    # Step 1: Remove exact duplicate rows.
    clean_df = raw.drop_duplicates().copy()

    # Step 2: Standardize service types.
    clean_df["service_type"] = (
        clean_df["service_type"].astype("string").str.strip().str.title()
    )

    # Step 3: Replace missing payment modes.
    clean_df["payment_mode"] = clean_df["payment_mode"].fillna("Unknown")

    # Step 4: Convert declared value text into numeric INR.
    clean_df["declared_value_inr"] = pd.to_numeric(
        clean_df["declared_value"]
        .astype("string")
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip(),
        errors="coerce",
    )

    clean_df = clean_df.drop(columns=["declared_value"])

    # Step 5: Convert dispatch time to datetime.
    clean_df["dispatch_time"] = pd.to_datetime(
        clean_df["dispatch_time"], errors="coerce"
    )

    # Step 6: Treat scanner-fault weights above 100 kg as missing.
    clean_df.loc[
        clean_df["parcel_weight_kg"] > 100, "parcel_weight_kg"
    ] = np.nan

    # Step 7: Calculate or reuse historical medians.
    median_columns = [
        "distance_km",
        "parcel_weight_kg",
        "declared_value_inr",
    ]

    if medians is None:
        medians = {
            col: float(clean_df[col].median())
            for col in median_columns
        }
    else:
        medians = {
            col: float(medians[col])
            for col in median_columns
        }

    # Step 8: Fill missing values using the chosen medians.
    for col in median_columns:
        clean_df[col] = clean_df[col].astype("Float64")
        clean_df[col] = clean_df[col].fillna(medians[col])

    return clean_df, medians


def build_features(clean, hubs):
    """Attach hub information and create shipment features."""

    # Left merge: preserve every shipment row.
    features = clean.merge(
        hubs[
            [
                "hub_id",
                "hub_city",
                "region",
                "hub_tier",
                "hub_daily_capacity",
            ]
        ],
        on="hub_id",
        how="left",
        validate="many_to_one",
    )

    # Promised delivery time, based on service type.
    features["promised_hours"] = (
        features["service_type"].map(PROMISED_HOURS).astype("int64")
    )

    # Payment, weekend and dispatch-hour features.
    features["is_cod"] = (
        features["payment_mode"].eq("COD").astype("int64")
    )

    features["is_weekend"] = (
        features["dispatch_time"].dt.dayofweek.ge(5).astype("int64")
    )

    features["dispatch_hour"] = (
        features["dispatch_time"].dt.hour.astype("int64")
    )

    # Natural logarithm of declared value in INR.
    features["log_declared_value"] = np.log(
        features["declared_value_inr"]
    )

    return features