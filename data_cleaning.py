import pandas as pd

def fill_missing_values(df):
    """
    Ersetzt fehlende Werte in bestimmten Spalten durch 'Unknown'.

    Args:
        df (pd.DataFrame): Eingabe-DataFrame.

    Returns:
        pd.DataFrame: Bereinigter DataFrame.
    """
    df["workclass"] = df["workclass"].fillna("Unknown")
    df["occupation"] = df["occupation"].fillna("Unknown")
    df["native.country"] = df["native.country"].fillna("Unknown")
    
    return df


def rename_columns(df):
    """
    Ersetzt "." durch "_" in bestimmten Spaltennamen.

    Args:
        df (pd.DataFrame): Eingabe-DataFrame.

    Returns:
        pd.DataFrame: DataFrame mit umbenannten Spalten.
    """
    columns_to_rename = [
        "education.num", "marital.status", "capital.gain",
        "capital.loss", "hours.per.week", "native.country"
    ]
    
    df.rename(columns={col: col.replace(".", "_") for col in columns_to_rename}, inplace=True)
    
    return df