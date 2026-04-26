import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def load_and_prep_data(filepath):
    df = pd.read_csv(filepath)
    
    df['gmv_with_markup'] = df['gmv_with_markup'].astype(str).str.replace(',', '.').astype(float)
    df['revenue'] = df['revenue'].astype(str).str.replace(',', '.').astype(float)
    
    cnt_columns = [
        'done_food_good_cnt',
        'bread_and_cake_good_cnt',
        'fruits_and_vegetables_good_cnt',
        'water_sodas_and_drinks_good_cnt',
        'sweets_good_cnt'
    ]
    df['total_good_cnt'] = df[cnt_columns].sum(axis=1)
    
    metrics = ['gmv_with_markup', 'total_good_cnt', 'revenue']
    
    for metric in metrics:
        lower_bound = df[metric].quantile(0.01)
        upper_bound = df[metric].quantile(0.99)
        df = df[(df[metric] >= lower_bound) & (df[metric] <= upper_bound)]
        
    return df

def create_and_plot_correlation(df, output_dir='plots', filename='correlation_matrix.png'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    columns_to_correlate = ['markup', 'cb_percent', 'gmv_with_markup', 'total_good_cnt', 'revenue']
    corr_matrix = df[columns_to_correlate].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0,
                cbar_kws={'label': 'Коэффициент корреляции Пирсона'})
    plt.title('Матрица корреляции наценки, кэшбека и ключевых метрик')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path)
    print(f"График сохранен в {output_path}")

if __name__ == "__main__":
    filepath = "data.csv"
    
    df = load_and_prep_data(filepath)
    create_and_plot_correlation(df)
