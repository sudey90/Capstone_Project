
# ============================================
# PART D: HUB SEGMENTATION
# ============================================

import pandas as pd
import numpy as np

from pathlib import Path
from itertools import combinations

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# STEP 1: Find the project folders

project_folder = Path(__file__).resolve().parent

data_folder = project_folder / "data"
output_folder = project_folder / "outputs"

output_folder.mkdir(exist_ok=True)


# STEP 2: Read the historical shipment data and hub data

df = pd.read_csv(data_folder / "modelling_table.csv")
hubs = pd.read_csv(data_folder / "hubs.csv")

print("Total historical shipments:", len(df))
print("Total hubs:", hubs["hub_id"].nunique())


# STEP 3: Create one profile for each hub

# Each hub has 50 historical shipments.
# We calculate the average performance of each hub.

hub_profile = df.groupby("hub_id").agg(
    shipments=("shipment_id", "count"),
    mean_delivery_hours=("delivery_hours", "mean"),
    delay_rate=("is_delayed", "mean"),
    mean_distance_km=("distance_km", "mean"),
    mean_n_stops=("n_stops", "mean")
).reset_index()

# Keep hubs in ascending order: HUB01 to HUB24.
hub_profile = hub_profile.sort_values("hub_id").reset_index(drop=True)

# Save the hub profile.
hub_profile.to_csv(
    output_folder / "hub_profile.csv",
    index=False
)

print("\nHub profile created:")
print(hub_profile.head())


# STEP 4: Scale the four measurements

# These four columns have different units and ranges.
# Scaling puts them on a comparable scale before clustering.

profile_columns = [
    "mean_delivery_hours",
    "delay_rate",
    "mean_distance_km",
    "mean_n_stops"
]

scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    hub_profile[profile_columns]
)


# STEP 5: Compare different numbers of clusters

# The assignment asks us to compare K = 2, 3, 4 and 5.

k_values = [2, 3, 4, 5]

separation_results = []

for k in k_values:

    # Create and train the K-Means model.
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(scaled_data)

    # Count the number of hubs in each cluster.
    cluster_sizes = (
        pd.Series(labels)
        .value_counts()
        .sort_values(ascending=False)
        .tolist()
    )

    # Calculate distances between every pair of hubs.
    within_distances = []
    across_distances = []

    for i, j in combinations(range(len(scaled_data)), 2):

        # Euclidean distance between two hubs.
        distance = np.linalg.norm(
            scaled_data[i] - scaled_data[j]
        )

        if labels[i] == labels[j]:
            within_distances.append(distance)
        else:
            across_distances.append(distance)

    # Average distance within the same cluster.
    mean_within = np.mean(within_distances)

    # Average distance between different clusters.
    mean_across = np.mean(across_distances)

    # Higher ratio means better separation.
    separation_ratio = mean_across / mean_within

    separation_results.append({
        "K": k,
        "across_over_within": separation_ratio,
        "cluster_sizes": "-".join(
            str(size) for size in cluster_sizes
        )
    })


# Save the separation comparison.
separation_df = pd.DataFrame(separation_results)

separation_df.to_csv(
    output_folder / "part_d_separation.csv",
    index=False
)

print("\nK-Means separation comparison:")
print(separation_df.round(3).to_string(index=False))


# STEP 6: Create the final three hub segments

# K = 3 is fixed because the business needs exactly
# three operational playbooks.

final_kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

hub_profile["cluster"] = final_kmeans.fit_predict(
    scaled_data
)


# STEP 7: Give meaningful names to the three clusters

# First calculate the average delay rate of each cluster.
cluster_delay_rates = hub_profile.groupby(
    "cluster"
)["delay_rate"].mean()

# Sort clusters from lowest delay rate to highest.
ordered_clusters = cluster_delay_rates.sort_values().index.tolist()

# Give the clusters their business names.
segment_names = {
    ordered_clusters[0]: "Standard",
    ordered_clusters[1]: "Watch",
    ordered_clusters[2]: "Escalate"
}

hub_profile["segment"] = hub_profile["cluster"].map(
    segment_names
)


# STEP 8: Attach hub city and tier information

hub_segments = hub_profile[
    ["hub_id", "segment"]
].merge(
    hubs[["hub_id", "hub_city", "hub_tier"]],
    on="hub_id",
    how="left"
)

# Keep the required column order.
hub_segments = hub_segments[
    ["hub_id", "segment", "hub_city", "hub_tier"]
]

hub_segments = hub_segments.sort_values(
    "hub_id"
).reset_index(drop=True)

hub_segments.to_csv(
    output_folder / "hub_segments.csv",
    index=False
)

print("\nHub segment assignments:")
print(hub_segments.to_string(index=False))


# STEP 9: Create the segment performance profile

# Combine the hub profiles using their segment names.
segment_profile = hub_profile.groupby("segment").agg(
    hubs=("hub_id", "count"),
    mean_delivery_hours=("mean_delivery_hours", "mean"),
    delay_rate=("delay_rate", "mean"),
    mean_distance_km=("mean_distance_km", "mean"),
    mean_n_stops=("mean_n_stops", "mean")
).reset_index()

# Keep the business order.
segment_order = ["Standard", "Watch", "Escalate"]

segment_profile["segment"] = pd.Categorical(
    segment_profile["segment"],
    categories=segment_order,
    ordered=True
)

segment_profile = segment_profile.sort_values(
    "segment"
).reset_index(drop=True)

segment_profile["segment"] = segment_profile["segment"].astype(str)

segment_profile.to_csv(
    output_folder / "segment_profile.csv",
    index=False
)

print("\nSegment performance profile:")
print(segment_profile.round(4).to_string(index=False))


# STEP 10: Find tier-versus-segment mismatches

# Expected relationship between tier and segment.
expected_segment = {
    1: "Standard",
    2: "Watch",
    3: "Escalate"
}

# Find the segment expected from each hub's existing tier.
hub_segments["expected_segment"] = (
    hub_segments["hub_tier"].map(expected_segment)
)

# Keep only hubs where the actual segment differs from the expected one.
mismatches = hub_segments[
    hub_segments["segment"] != hub_segments["expected_segment"]
].copy()

# Remove the temporary expected_segment column.
mismatches = mismatches[
    ["hub_id", "segment", "hub_city", "hub_tier"]
]

mismatches.to_csv(
    output_folder / "tier_vs_segment_mismatch.csv",
    index=False
)

print("\nTier-versus-segment mismatches:")
print(mismatches.to_string(index=False))


# STEP 11: Check the required results

assert len(hub_profile) == 24
assert hub_profile["shipments"].eq(50).all()

assert len(separation_df) == 4
assert len(hub_segments) == 24
assert len(segment_profile) == 3
assert len(mismatches) == 4

print("\n====================================")
print("PART D HUB SEGMENTATION COMPLETED")
print("====================================")

print("Hub profiles:", len(hub_profile))
print("Hub segments:", len(hub_segments))
print("Number of segments:", len(segment_profile))
print("Tier mismatches:", len(mismatches))

print("\nAll basic checks passed.")