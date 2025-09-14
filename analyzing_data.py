"""
Analyzing Data with Pandas and Visualizing Results with Matplotlib
Author: FRANC PAUL APIYO
Description:
    - Load and explore the Iris dataset
    - Perform basic statistical analysis
    - Visualize patterns using Matplotlib (and Seaborn for style)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

def main():
    print("=== Data Analysis with Pandas and Matplotlib ===\n")

    # ----------------------------
    # Task 1: Load and Explore Data
    # ----------------------------
    try:
        # Load Iris dataset from sklearn
        iris = load_iris(as_frame=True)
        df = iris.frame
        print("Dataset loaded successfully.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Inspect first few rows
    print("\nFirst 5 rows of dataset:")
    print(df.head())

    # Check dataset info
    print("\nDataset Info:")
    print(df.info())

    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Clean data (fill/drop if necessary — Iris is already clean)
    df = df.dropna()

    # ----------------------------
    # Task 2: Basic Data Analysis
    # ----------------------------
    print("\nBasic Statistics:")
    print(df.describe())

    # Grouping: mean of features by species
    grouped = df.groupby("target").mean()
    print("\nMean values grouped by species:")
    print(grouped)

    # Map target integers to species names
    df["species"] = df["target"].map(dict(zip(range(3), iris.target_names)))

    # Interesting finding:
    print("\nObservation: Petal dimensions (length & width) clearly separate species.")
    print("Versicolor and virginica have longer petals compared to setosa.")

    # ----------------------------
    # Task 3: Data Visualization
    # ----------------------------
    sns.set(style="whitegrid")

    # 1. Line chart - sepal length trend (not time-series, but index as proxy)
    plt.figure(figsize=(8, 5))
    plt.plot(df.index, df["sepal length (cm)"], label="Sepal Length", color="blue")
    plt.title("Sepal Length Trend Across Samples")
    plt.xlabel("Sample Index")
    plt.ylabel("Sepal Length (cm)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Bar chart - average petal length per species
    plt.figure(figsize=(8, 5))
    sns.barplot(x="species", y="petal length (cm)", data=df, ci=None, palette="muted")
    plt.title("Average Petal Length per Species")
    plt.xlabel("Species")
    plt.ylabel("Petal Length (cm)")
    plt.tight_layout()
    plt.show()

    # 3. Histogram - distribution of sepal width
    plt.figure(figsize=(8, 5))
    plt.hist(df["sepal width (cm)"], bins=15, color="green", alpha=0.7, edgecolor="black")
    plt.title("Distribution of Sepal Width")
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # 4. Scatter plot - sepal length vs petal length
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="sepal length (cm)", 
        y="petal length (cm)", 
        hue="species", 
        data=df, 
        palette="deep"
    )
    plt.title("Sepal Length vs Petal Length by Species")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend(title="Species")
    plt.tight_layout()
    plt.show()

    print("\n=== Analysis complete. Visualizations displayed. ===")

if __name__ == "__main__":
    main()
