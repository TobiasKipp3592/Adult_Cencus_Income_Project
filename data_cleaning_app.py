import pandas as pd

def fill_missing_values(df):
    """
    Ersetzt fehlende Werte in bestimmten Spalten durch 'Unknown' und 'Amer-Indian-Eskimo' in der Spalte 'race' durch 'Indigenous'.

    Args:
        df (pd.DataFrame): Eingabe-DataFrame.

    Returns:
        pd.DataFrame: Bereinigter DataFrame.
    """
    df = df.copy()  # Vermeidung von Änderungen am Original-DataFrame
    df["workclass"] = df["workclass"].fillna("Unknown")
    df["occupation"] = df["occupation"].fillna("Unknown")
    df["native.country"] = df["native.country"].fillna("Unknown")
    df["race"] = df["race"].replace("Amer-Indian-Eskimo", "Indigenous")
    
    return df

def rename_columns(df):
    """
    Ersetzt "." durch "_" in bestimmten Spaltennamen.

    Args:
        df (pd.DataFrame): Eingabe-DataFrame.

    Returns:
        pd.DataFrame: DataFrame mit umbenannten Spalten.
    """
    df = df.copy()
    columns_to_rename = [
        "education.num", "marital.status", "capital.gain",
        "capital.loss", "hours.per.week", "native.country"
    ]
    
    df.rename(columns={col: col.replace(".", "_") for col in columns_to_rename}, inplace=True)
    
    return df

def clean_data(df):
    """
    Führt alle Bereinigungsschritte aus und gibt das bereinigte DataFrame zurück.

    Args:
        df (pd.DataFrame): Eingabe-DataFrame.

    Returns:
        pd.DataFrame: Bereinigter DataFrame.
    """
    df = fill_missing_values(df)
    df = rename_columns(df)
    return df  # cleaned_df

# Beispiel für die Anwendung:
# df = pd.read_csv("data.csv")  # Beispiel für das Laden eines DataFrames
# cleaned_df = clean_data(df)
# print(cleaned_df.head())
