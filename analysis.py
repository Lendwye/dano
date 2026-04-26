import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os

os.makedirs('plots', exist_ok=True)

df = pd.read_csv('data_cleaned.csv')

# --- 1. EDA ---
print("--- 1. EDA ---")
print("Info:", df.info())
print("\nDescribe:", df.describe())

# Check missing values (should be none as per clean_data.py)
print("\nMissing values:\n", df.isnull().sum())

# Define columns of interest based on the assignment:
# markup, cashback, order total, categories, demographics
print("\nColumns:", df.columns)

# Visualizing correlations
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png')
plt.close()

# Let's explore markup and cashback distribution
if 'markup' in df.columns and 'cashback' in df.columns:
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.countplot(x='markup', data=df)
    plt.title('Markup Distribution')
    
    plt.subplot(1, 2, 2)
    sns.histplot(df['cashback'], bins=20)
    plt.title('Cashback Distribution')
    plt.tight_layout()
    plt.savefig('plots/markup_cashback_dist.png')
    plt.close()

# --- 2. Regression Model ---
# Hypothesis: Higher markup decreases basket size (or order total), but higher cashback moderates it.
# y = a + b1*markup + b2*cashback + b3*(markup*cashback) + controls

# Let's see the column names first to build the model correctly.
print("\nHead:\n", df.head())
