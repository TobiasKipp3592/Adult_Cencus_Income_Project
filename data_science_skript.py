import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

def preprocess_data(df):
    """Bereinigt die Daten: wandelt Variablen 'income' und 'sex' um"""
    
    df['income'] = df['income'].map({'>50K': 1, '<=50K': 0}).astype('int64')
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0}).astype('int64')

    return df



def generate_plots(df, heatmap_file="heatmap.png", pairplot_file="pairplot.png"):
    """
    Generates a heatmap and a pairplot from a DataFrame and saves them as images.
    If the files already exist, they are loaded instead of being recreated and displayed.

    Args:
        df (pd.DataFrame): The input DataFrame.
        heatmap_file (str): Filename for the heatmap image.
        pairplot_file (str): Filename for the pairplot image.
    """
    df_numeric = df.select_dtypes(include=["int64", "float64"])

    # Heatmap
    if os.path.exists(heatmap_file):
        print(f"File '{heatmap_file}' already exists. Loading...")
        img = plt.imread(heatmap_file)
        plt.imshow(img)
        plt.axis("off") 
        plt.show()
    else:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df_numeric.corr(), cmap="plasma", vmax=0.8)
        plt.title("Correlation Heatmap")
        plt.savefig(heatmap_file)
        plt.show()
        print(f"Heatmap saved as '{heatmap_file}'")

    # Pairplot
    if os.path.exists(pairplot_file):
        print(f"File '{pairplot_file}' already exists. Loading...")
        img = plt.imread(pairplot_file)
        plt.imshow(img)
        plt.axis("off")
        plt.show()
    else:
        pairplot = sns.pairplot(df_numeric)
        pairplot.savefig(pairplot_file)
        plt.show()
        print(f"Pairplot saved as '{pairplot_file}'")