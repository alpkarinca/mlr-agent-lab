#!/usr/bin/env python3
"""
Exploratory Data Analysis of data/leaves.csv
Generates summary statistics, duplicates audit, anomaly detection,
and a publication-quality scatter plot saved to ml-study/leaves_scatter.png.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = os.path.abspath(os.path.join(script_dir, ".."))
    csv_path = os.path.join(workspace_root, "data", "leaves.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    # Load data
    df = pd.read_csv(csv_path)
    print("=" * 60)
    print("1. DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing values:\n{df.isnull().sum().to_dict()}")

    print("\n" + "=" * 60)
    print("2. CLASS DISTRIBUTION (LEAF TYPE COUNTS)")
    print("=" * 60)
    counts = df["Tree type"].value_counts()
    percentages = df["Tree type"].value_counts(normalize=True) * 100
    for tree_type in counts.index:
        print(f"  - {tree_type:<8}: {counts[tree_type]:>4} leaves ({percentages[tree_type]:.2f}%)")

    print("\n" + "=" * 60)
    print("3. FEATURE SUMMARY STATISTICS")
    print("=" * 60)
    stats_by_type = df.groupby("Tree type").agg(
        height_mean=("Leaf height", "mean"),
        height_std=("Leaf height", "std"),
        height_min=("Leaf height", "min"),
        height_max=("Leaf height", "max"),
        width_mean=("Leaf width", "mean"),
        width_std=("Leaf width", "std"),
        width_min=("Leaf width", "min"),
        width_max=("Leaf width", "max"),
    )
    print(stats_by_type.round(2))

    print("\n" + "=" * 60)
    print("4. DUPLICATES ANALYSIS")
    print("=" * 60)
    exact_duplicates = df.duplicated().sum()
    all_duplicate_rows = df.duplicated(keep=False).sum()
    unique_rows = len(df.drop_duplicates())
    print(f"Exact duplicate rows (repeated occurrences): {exact_duplicates}")
    print(f"Total rows involved in duplicates: {all_duplicate_rows}")
    print(f"Unique distinct rows: {unique_rows}")

    # Check for dimension collisions with differing tree types
    conflicts = []
    for (h, w), group in df.groupby(["Leaf height", "Leaf width"]):
        if group["Tree type"].nunique() > 1:
            conflicts.append((h, w, group["Tree type"].tolist(), list(group.index)))

    print(f"Conflicting coordinate pairs (same height & width, different label): {len(conflicts)}")
    for h, w, types, indices in conflicts:
        print(f"  - Height={h}, Width={w}: labeled as {types} (indices: {indices})")

    print("\n" + "=" * 60)
    print("5. UNUSUAL OBSERVATIONS & OUTLIERS")
    print("=" * 60)
    # Cherry outliers with extreme width (normal width is ~3.5 to 7.0)
    cherry_outliers = df[(df["Tree type"] == "cherry") & (df["Leaf width"] > 12)]
    print(f"Cherry leaves with extreme width (>12 cm) [likely mislabeled maple]:")
    for idx, row in cherry_outliers.iterrows():
        print(f"  Index {idx}: Height={row['Leaf height']}, Width={row['Leaf width']}, Type={row['Tree type']}")

    # Maple outliers with high aspect ratio or tiny width (normal maple width is ~10 to 25)
    maple_outliers = df[(df["Tree type"] == "maple") & (df["Leaf width"] < 6)]
    print(f"Maple leaves with narrow width (<6 cm) [likely mislabeled cherry]:")
    for idx, row in maple_outliers.iterrows():
        print(f"  Index {idx}: Height={row['Leaf height']}, Width={row['Leaf width']}, Type={row['Tree type']}")

    # Walnut outliers with broad width (>12 cm)
    walnut_outliers = df[(df["Tree type"] == "walnut") & (df["Leaf width"] > 12)]
    print(f"Walnut leaves with unusually large width (>12 cm):")
    for idx, row in walnut_outliers.iterrows():
        print(f"  Index {idx}: Height={row['Leaf height']}, Width={row['Leaf width']}, Type={row['Tree type']}")

    # Generate Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

    # Style definitions
    colors = {
        "maple": "#2E7D32",   # Forest Green
        "cherry": "#C2185B",  # Deep Rose / Magenta
        "walnut": "#E65100"   # Amber / Orange
    }
    markers = {
        "maple": "o",
        "cherry": "s",
        "walnut": "^"
    }

    # Plot base data by category
    for tree_type in ["maple", "cherry", "walnut"]:
        subset = df[df["Tree type"] == tree_type]
        ax.scatter(
            subset["Leaf width"],
            subset["Leaf height"],
            label=f"{tree_type.capitalize()} (n={len(subset)})",
            color=colors[tree_type],
            marker=markers[tree_type],
            alpha=0.65,
            edgecolors="black",
            linewidths=0.5,
            s=55,
            zorder=3
        )

    # Highlight identical duplicate points
    dup_df = df[df.duplicated(keep=False)]
    ax.scatter(
        dup_df["Leaf width"],
        dup_df["Leaf height"],
        s=120,
        facecolors="none",
        edgecolors="#2979FF",
        linewidths=1.5,
        linestyle="--",
        label=f"Duplicate points ({all_duplicate_rows} rows)",
        zorder=4
    )

    # Highlight extreme anomalies / mislabeled points
    anomalies = pd.concat([cherry_outliers, maple_outliers])
    for idx, row in anomalies.iterrows():
        ax.annotate(
            f"Anomaly #{idx}\n({row['Tree type']})",
            xy=(row["Leaf width"], row["Leaf height"]),
            xytext=(row["Leaf width"] + 0.8, row["Leaf height"] + 0.8),
            arrowprops=dict(facecolor="black", shrink=0.08, width=1, headwidth=5),
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFEBEE", edgecolor="#D32F2F", alpha=0.9)
        )

    # Highlight conflicting points
    for h, w, types, _ in conflicts:
        ax.plot(w, h, marker="x", color="black", markersize=10, markeredgewidth=2, zorder=5)

    ax.plot([], [], marker="x", color="black", linestyle="None", markersize=8, markeredgewidth=1.5,
            label=f"Conflicting label coordinates ({len(conflicts)})")

    ax.set_title("Leaf Dimensions by Tree Type (Height vs. Width)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Leaf Width (cm)", fontsize=12, labelpad=8)
    ax.set_ylabel("Leaf Height (cm)", fontsize=12, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(frameon=True, facecolor="white", edgecolor="#CCCCCC", fontsize=10, loc="upper left")

    plt.tight_layout()
    output_png = os.path.join(script_dir, "leaves_scatter.png")
    fig.savefig(output_png)
    plt.close()
    print(f"\nScatter plot saved to: {output_png}")

if __name__ == "__main__":
    main()
