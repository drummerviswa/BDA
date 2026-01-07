# Importing necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load Iris dataset
from sklearn.datasets import load_iris

# Load data into pandas DataFrame
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Rename species column for clarity
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Check for any missing values or duplicates
print(f"Missing values: \n{df.isnull().sum()}")
print(f"Duplicate entries: {df.duplicated().sum()}")

# Clean data: drop duplicates if any
df_cleaned = df.drop_duplicates()

# Descriptive Statistics
print("\nDescriptive Statistics:")
print(df_cleaned.describe())

# Correlation Matrix (for the numeric columns)
correlation = df_cleaned.drop('species', axis=1).corr()
print("\nCorrelation Matrix:")
print(correlation)

# Plotting Histogram for each feature
df_cleaned.drop('species', axis=1).hist(figsize=(10, 8), bins=20)
plt.suptitle('Histograms of Features')
plt.tight_layout()
plt.show()

# Plotting Box Plot for each feature
plt.figure(figsize=(10, 8))
sns.boxplot(data=df_cleaned.drop('species', axis=1))
plt.title('Box Plot of Features')
plt.tight_layout()
plt.show()

# Correlation Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
plt.title('Correlation Matrix Heatmap')
plt.tight_layout()
plt.show()

# Plotting Histograms and Box Plots for each feature individually
for feature in df_cleaned.columns[:-1]:
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df_cleaned[feature], kde=True, color='skyblue')
    plt.title(f'Histogram of {feature}')
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_cleaned[feature], color='lightcoral')
    plt.title(f'Box Plot of {feature}')
    
    plt.tight_layout()
    plt.show()
