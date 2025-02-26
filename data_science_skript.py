import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def preprocess_data(df):
    """Bereinigt die Daten: wandelt Variablen 'income' und 'sex' um"""
    
    df['income'] = df['income'].map({'>50K': 1, '<=50K': 0}).astype('int64')
    df['sex'] = df['sex'].map({'Male': 1, 'Female': 0}).astype('int64')

    return df



def generate_plots(df, heatmap_file="data_science\heatmap.png", pairplot_file="data_science\pairplot.png"):
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

        plt.show()
    else:
        pairplot = sns.pairplot(df_numeric, height=3, aspect=1.2)
        pairplot.savefig(pairplot_file, dpi=300)
        plt.show()
        print(f"Pairplot saved as '{pairplot_file}'")



def train_model(model, features_train, target_train, features_test, target_test):
    """
    Trains a given machine learning model and evaluates its performance on the test set.

    Args:
        model (sklearn-compatible estimator): The machine learning model to be trained.
    X_train (pandas.DataFrame or numpy.ndarray): Training features.
    y_train (pandas.Series or numpy.ndarray): Target values for training.
    X_test (pandas.DataFrame or numpy.ndarray): Test features.
    y_test (pandas.Series or numpy.ndarray): True target values for evaluation.

    Returns:
    dict
        A dictionary containing evaluation metrics:
        - "Accuracy": Model accuracy score.
        - "Precision": Weighted precision score.
        - "Recall": Weighted recall score.
        - "F1 Score": Weighted F1 score.
        - "ROC-AUC": Area under the ROC curve (binary classification).
    """
    model.fit(features_train, target_train)
    target_pred = model.predict(features_test)
    
    metrics = {
        "Accuracy": accuracy_score(target_test, target_pred),
        "Precision": precision_score(target_test, target_pred, average="weighted"),
        "Recall": recall_score(target_test, target_pred, average="weighted"),
        "F1 Score": f1_score(target_test, target_pred, average="weighted"),
        "ROC-AUC": float(roc_auc_score(target_test, target_pred))
    }
    return metrics