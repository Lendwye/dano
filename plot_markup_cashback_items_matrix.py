import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_and_prep_data(filepath):
    df = pd.read_csv(filepath)
    
    cnt_columns = [
        'done_food_good_cnt',
        'bread_and_cake_good_cnt',
        'fruits_and_vegetables_good_cnt',
        'water_sodas_and_drinks_good_cnt',
        'sweets_good_cnt'
    ]
    
    df['total_good_cnt'] = df[cnt_columns].sum(axis=1)
    
    lower_bound = df['total_good_cnt'].quantile(0.01)
    upper_bound = df['total_good_cnt'].quantile(0.99)
    
    df_filtered = df[(df['total_good_cnt'] >= lower_bound) & (df['total_good_cnt'] <= upper_bound)]
    
    return df_filtered

def create_impact_matrix(df):
    matrix = df.groupby(['markup', 'cb_percent'])['total_good_cnt'].mean().unstack()
    return matrix

def plot_matrix(matrix, output_dir='plots', filename='markup_cashback_items_matrix.png'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlGn", 
                cbar_kws={'label': 'Среднее количество товаров'})
    plt.title('Влияние наценки и кэшбека на количество товаров в корзине')
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
