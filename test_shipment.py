
import pandas as pd

from src.shipment_prep import clean_shipments, build_features

raw = pd.read_csv("data/raw_shipments.csv")
hubs = pd.read_csv("data/hubs.csv")

clean, medians = clean_shipments(raw)
features = build_features(clean, hubs)

print("Raw rows:", len(raw))
print("Clean rows:", len(clean))
print("Rows after hub merge:", len(features))
print("Medians:", medians)
print("Service types:", sorted(clean["service_type"].unique()))
print("Missing weights:", clean["parcel_weight_kg"].isna().sum())
print("Feature columns:", features.columns.tolist())

new_raw = pd.read_csv("data/new_shipments.csv")
new_clean, _ = clean_shipments(new_raw, medians=medians)
new_features = build_features(new_clean, hubs)

print("New shipment rows:", len(new_features))