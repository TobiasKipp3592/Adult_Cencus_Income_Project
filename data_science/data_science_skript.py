import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from fairlearn.metrics import equalized_odds_difference, demographic_parity_difference, demographic_parity_ratio


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



def train_and_predict(model, features_train, target_train, features_test, target_test):
    """
    Trains a given machine learning model and evaluates its performance on the test set.

    Args:
        model (sklearn-compatible estimator): The machine learning model to be trained.
        features_train (pandas.DataFrame or numpy.ndarray): Training features.
        target_train (pandas.Series or numpy.ndarray): Target values for training.
        features_test (pandas.DataFrame or numpy.ndarray): Test features.
        target_test (pandas.Series or numpy.ndarray): True target values for evaluation.

    Returns:
        numpy.ndarray: Predicted values for the test set.
    """

    model.fit(features_train, target_train)
    target_pred = model.predict(features_test)
    return target_pred



def evaluate_model(target_test, target_pred):
    """
    Evaluates the model's performance using standard classification metrics.

    Args:
        target_test (pandas.Series or numpy.ndarray): True target values for evaluation.
        target_pred (numpy.ndarray): Predicted values from the model.

    Returns_
        dict: dictionary containing evaluation metrics:
            - "Accuracy": Model accuracy score.
            - "Precision": Weighted precision score.
            - "Recall": Weighted recall score.
            - "F1 Score": Weighted F1 score.
            - "ROC-AUC": Area under the ROC curve (binary classification).
    """
    
    metrics = {
        "Accuracy": accuracy_score(target_test, target_pred),
        "Precision": precision_score(target_test, target_pred, average="weighted"),
        "Recall": recall_score(target_test, target_pred, average="weighted"),
        "F1 Score": f1_score(target_test, target_pred, average="weighted"),
        "ROC-AUC": float(roc_auc_score(target_test, target_pred))
    }
    return metrics



def evaluate_discrimination(target_test, target_pred, features_test):
    """
    Evaluates the discrimination metrics of the model, focusing on fairness across different demographic groups.

    Args:
        target_test (pandas.Series or numpy.ndarray): True target values for evaluation.
        target_pred (numpy.ndarray): Predicted values from the model.
        features_test (pandas.DataFrame or numpy.ndarray): Test features.

    Returns:
        dict: A dictionary containing fairness-related evaluation metrics:
            - "ROC-AUC-male": ROC-AUC score for the male subgroup.
            - "ROC-AUC-female": ROC-AUC score for the female subgroup.
            - "ROC-AUC-white": ROC-AUC score for the white subgroup.
            - "ROC-AUC-non_white": ROC-AUC score for the non-white subgroup.
            - "EOD 'sex'": Equalized Odds Difference for the 'sex' attribute.
            - "EOD 'race'": Equalized Odds Difference for the 'race' attribute.
            - "DPR 'sex'": Demographic Parity Ratio for the 'sex' attribute.
            - "DPR 'race'": Demographic Parity Ratio for the 'race' attribute.
            - "DPD 'sex'": Demographic Parity Difference for the 'sex' attribute.
            - "DPD 'race'": Demographic Parity Difference for the 'race' attribute.

    Notes:
        - **ROC-AUC subgroup analysis**: Measures the model's ability to discriminate between classes within specific demographic subgroups.
        - **Equalized Odds Difference (EOD)**: Measures disparities in **true positive rates (TPR) and false positive rates (FPR)** across groups.
        - **Demographic Parity Ratio (DPR)**: Ratio of positive predictions between groups.
        - **Demographic Parity Difference (DPD)**: Difference in positive predictions between groups.

    """

    fairness_metrics = {
        "roc_auc": {},
        "eod": {},
        "dpr": {},
        "dpd": {}
    }

    mask_male = features_test["sex"] == 1
    mask_female = features_test["sex"] == 0
    fairness_metrics["roc_auc"]["male"] = roc_auc_score(target_test[mask_male], target_pred[mask_male])
    fairness_metrics["roc_auc"]["female"] = roc_auc_score(target_test[mask_female], target_pred[mask_female])
    
    mask_white = features_test["race"] == "White"
    mask_non_white = features_test["race"] != "White"
    fairness_metrics["roc_auc"]["white"] = roc_auc_score(target_test[mask_white], target_pred[mask_white])
    fairness_metrics["roc_auc"]["non_white"] = roc_auc_score(target_test[mask_non_white], target_pred[mask_non_white])
    
    fairness_metrics["eod"]["sex"] = equalized_odds_difference(target_test, target_pred, sensitive_features=features_test["sex"])
    fairness_metrics["eod"]["race"] = equalized_odds_difference(target_test, target_pred, sensitive_features=features_test["race"])
    
    fairness_metrics["dpr"]["sex"] = demographic_parity_ratio(target_test, target_pred, sensitive_features=features_test["sex"])
    fairness_metrics["dpr"]["race"] = demographic_parity_ratio(target_test, target_pred, sensitive_features=features_test["race"])
    
    fairness_metrics["dpd"]["sex"] = demographic_parity_difference(target_test, target_pred, sensitive_features=features_test["sex"])
    fairness_metrics["dpd"]["race"] = demographic_parity_difference(target_test, target_pred, sensitive_features=features_test["race"])
    
    return fairness_metrics
    
