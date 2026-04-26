import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_and_prep_data(filepath):
    df = pd.read_csv(filepath)
    df['gmv_with_markup'] = df['gmv_with_markup'].astype(str).str.replace(',', '.').astype(float)
    
    lower_bound = df['gmv_with_markup'].quantile(0.01)
    upper_bound = df['gmv_with_markup'].quantile(0.99)
    
    df_filtered = df[(df['gmv_with_markup'] >= lower_bound) & (df['gmv_with_markup'] <= upper_bound)]
    
    return df_filtered

def create_impact_matrix(df):
    matrix = df.groupby(['markup', 'cb_percent'])['gmv_with_markup'].mean().unstack()
    return matrix

def plot_matrix(matrix, output_dir='plots', filename='markup_cashback_matrix.png'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="RdYlGn", 
                cbar_kws={'label': 'Средний размер корзины (GMV)'})
    plt.title('Влияние наценки и кэшбека на средний размер корзины')
    plt.xlabel('Кэшбек (%)')
    plt.ylabel('Наценка (%)')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    print(f"График сохранен в {output_path}")

if __name__ == "__main__":
    filepath = "data.csv"
    
    df = load_and_prep_data(filepath)
    matrix = create_impact_matrix(df)
    
    print(matrix.round(2))
    
    plot_matrix(matrix)
