import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('plots', exist_ok=True)
df = pd.read_csv('data_cleaned.csv')

# --- EDA & Visualizations ---
# 1. Markup and Cashback vs GMV
plt.figure(figsize=(10, 6))
sns.barplot(x='markup', y='gmv_without_markup', hue='cb_percent', data=df)
plt.title('GMV without Markup by Markup Level and Cashback Percent')
plt.savefig('plots/gmv_vs_markup_cb.png')
plt.close()

# 2. Total items vs markup
df['total_items'] = df['done_food_good_cnt'] + df['bread_and_cake_good_cnt'] + df['fruits_and_vegetables_good_cnt'] + df['water_sodas_and_drinks_good_cnt'] + df['sweets_good_cnt']
plt.figure(figsize=(10, 6))
sns.barplot(x='markup', y='total_items', data=df)
plt.title('Total Items in Basket by Markup Level')
plt.savefig('plots/items_vs_markup.png')
plt.close()

# --- Math Model ---
print("--- 1. Base Model ---")
# Dependent variable: gmv_without_markup (log transformation might be better to handle outliers, let's just use raw for now or log(x+1))
df['log_gmv'] = np.log1p(df['gmv_without_markup'])
# We will use log_gmv as dependent variable
base_model = smf.ols('log_gmv ~ markup * cb_percent', data=df).fit()
print(base_model.summary())

print("\n--- 2. Model with Controls ---")
# Controls: gender_cd, age_group, income_group, lifestyle___super_cnt (as proxy for experience)
# Clean up categorical for formula
df['gender_cd'] = df['gender_cd'].astype(str)
df['age_group'] = df['age_group'].astype(str).str.replace('-', '_').str.replace('+', '_plus')
df['income_group'] = df['income_group'].astype(str)

control_model = smf.ols('log_gmv ~ markup * cb_percent + C(gender_cd) + C(age_group) + C(income_group) + lifestyle___super_cnt', data=df).fit()
print(control_model.summary())

print("\n--- 3. Robustness: Experience Interaction ---")
# Assignment asks: "Различается ли чувствительность к наценке у клиентов с разным опытом использования сервиса?"
# Let's create an experience dummy or use lifestyle___super_cnt directly in interaction
df['is_experienced'] = (df['lifestyle___super_cnt'] > df['lifestyle___super_cnt'].median()).astype(int)

exp_model = smf.ols('log_gmv ~ markup * is_experienced + cb_percent', data=df).fit()
print(exp_model.summary())

# Save coefficients to a csv or just print
with open('model_results.txt', 'w') as f:
    f.write("BASE MODEL:\n")
    f.write(base_model.summary().as_text())
    f.write("\n\nCONTROL MODEL:\n")
    f.write(control_model.summary().as_text())
    f.write("\n\nEXPERIENCE MODEL:\n")
    f.write(exp_model.summary().as_text())
