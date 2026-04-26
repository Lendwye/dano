import pandas as pd
import numpy as np

def clean_data(input_file, output_file):
    print(f"Reading data from {input_file}...")
    df = pd.read_csv(input_file, decimal=',')
    print(f"Original shape: {df.shape}")
    df.drop_duplicates(inplace=True)
    print(f"Shape after dropping duplicates: {df.shape}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df[col] = df[col].fillna('unknown')

    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df['order_date'] = df['order_date'].ffill()
    
    print("Data cleaning completed. Sample of the cleaned data:")
    print(df.head())
    print(f"Saving cleaned data to {output_file}...")
    df.to_csv(output_file, index=False)
    print("Done!")

if __name__ == '__main__':
    clean_data('data.csv', 'data_cleaned.csv')
