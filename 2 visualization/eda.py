import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris  # FIXED

# Load dataset
iris_data = load_iris()

# Create DataFrame
iris_df = pd.DataFrame(
    iris_data.data,
    columns=["Sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]
)

# Add species column
iris_df["Species"] = iris_data.target
iris_df["Species"] = iris_df["Species"].replace({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

# Basic checks
print("Sample record")
print(iris_df.head())

print("\nMissing value check")
print(iris_df.isna().sum())

print("\nStatistical Summary")
print(iris_df.describe())

# Histograms for each numeric column
for col in iris_df.columns[:-1]:
    plt.figure()
    plt.hist(iris_df[col], bins=15)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {col}")
    plt.show()

# Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=iris_df.iloc[:, :-1])
plt.title("Box plot of Iris features")
plt.show()

# Correlation heatmap
correlation = iris_df.iloc[:, :-1].corr()  # FIXED
plt.figure(figsize=(7, 5))
sns.heatmap(correlation, annot=True, cmap="coolwarm")
plt.title("Feature correlation heatmap")
plt.show()

# Feature distribution by species
for feature in iris_df.columns[:-1]:
    plt.figure(figsize=(8, 4))
    sns.histplot(
        data=iris_df,
        x=feature,
        hue="Species",
        bins=15,
        kde=True
    )
    plt.title(f"Distribution of {feature} by species")  # FIXED
    plt.show()
