import pandas as pd

def preprocess_data(df):
    """Bereinigt die Daten: wandelt Variablen 'income' und 'sex' um"""
    
    df['income'] = df['income'].map({'>50K': 1, '<=50K': 0}).astype('int64')
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0}).astype('int64')

    return df