#%%
# ================================
#       Imports
# ================================
import sys
import os
from math import sqrt
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

import phik
from phik.report import plot_correlation_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, brier_score_loss
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib


#%%
# =================================================
#       Chemin absolu de la racine du projet
# =================================================
current_file = Path(__file__).absolute()
project_root = current_file.parent.parent.parent  # Remonte 3 niveaux
sys.path.insert(0, str(project_root))


#%%
# =================================================================
#       Splits initial df into train/ val and test X and y dfs
# =================================================================

def prepare_train_val_test(
    df,
    patient_col="subject_id",
    time_col="intime",
    target_col="readmit_72h",
    test_size=0.20,
    val_size=0.20,
    drop_id_cols=True,
    drop_time_cols=True,
    random_state=42
):
    """
    Performs a leakage-proof patient-level temporal split and
    returns X_train, X_val, X_test, y_train, y_val, y_test.

    Combines:
      - temporal_patient_split()
      - build_feature_target_sets()

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """

    # ---------------------------------------
    # Ensure datetime and sort
    # ---------------------------------------
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)

    # ---------------------------------------
    # 1. First ICU admission per patient
    # ---------------------------------------
    patient_first_admit = (
        df.groupby(patient_col)[time_col]
        .min()
        .reset_index()
        .rename(columns={time_col: "first_admit"})
        .sort_values("first_admit")
    )

    # ---------------------------------------
    # 2. Temporal patient-based split
    # ---------------------------------------
    n_patients = len(patient_first_admit)
    test_cut = int((1 - test_size) * n_patients)

    trainval_patient_ids = patient_first_admit.iloc[:test_cut][patient_col]
    test_patient_ids     = patient_first_admit.iloc[test_cut:][patient_col]

    df_trainval = df[df[patient_col].isin(trainval_patient_ids)]
    df_test     = df[df[patient_col].isin(test_patient_ids)]

    # ---------------------------------------
    # 3. Random patient-level split into train/val
    # ---------------------------------------
    shuffled = trainval_patient_ids.sample(frac=1, random_state=random_state)
    cut_val = int((1 - val_size) * len(shuffled))

    train_ids = shuffled.iloc[:cut_val]
    val_ids   = shuffled.iloc[cut_val:]

    df_train = df[df[patient_col].isin(train_ids)]
    df_val   = df[df[patient_col].isin(val_ids)]

    # ---------------------------------------
    # Sanity check: No leakage
    # ---------------------------------------
    assert len(set(df_train[patient_col]) & set(df_val[patient_col])) == 0
    assert len(set(df_train[patient_col]) & set(df_test[patient_col])) == 0
    assert len(set(df_val[patient_col]) & set(df_test[patient_col])) == 0

    # ---------------------------------------
    # 4. Build X/y splits
    # ---------------------------------------
    cols_to_drop = [target_col]

    if drop_id_cols:
        for c in ["subject_id", "hadm_id", "stay_id"]:
            if c in df.columns:
                cols_to_drop.append(c)

    if drop_time_cols:
        for c in ["intime", "outtime"]:
            if c in df.columns:
                cols_to_drop.append(c)

    y_train = df_train[target_col].copy()
    y_val   = df_val[target_col].copy()
    y_test  = df_test[target_col].copy()

    X_train = df_train.drop(columns=cols_to_drop).copy()
    X_val   = df_val.drop(columns=cols_to_drop).copy()
    X_test  = df_test.drop(columns=cols_to_drop).copy()

    return X_train, X_val, X_test, y_train, y_val, y_test

# =============================================================================
#       Plot distribution of key variables according to pt readmitted or not
# =============================================================================

def plot_readmit_distribution(df, variable, bins=20, figsize=(12,4), binary=False):
    """
    Improved: now includes counts + proportions above bars for binary variables.
    """

    if variable not in df.columns:
        raise ValueError(f"Column '{variable}' not found in dataframe.")
    if "readmit_72h" not in df.columns:
        raise ValueError("readmit_72h column missing from dataframe.")

    readmitted = df[df["readmit_72h"] == 1][variable].dropna()
    not_readmitted = df[df["readmit_72h"] == 0][variable].dropna()

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    #-------------------------------------------------------
    #                BINARY VARIABLE
    #-------------------------------------------------------
    if binary:

        # ---------- LEFT: Readmitted ----------
        ax = axes[0]
        sns.countplot(x=readmitted.astype(int), ax=ax,
                      palette=["#e57373", "#81c784"])
        ax.set_xticks([0, 1])
        ax.set_title(f"{variable} (Readmitted) n={len(readmitted)}")
        ax.set_xlabel(variable)
        ax.set_ylabel("Count")

        # Add counts + proportions
        total = len(readmitted)
        for p in ax.patches:
            count = int(p.get_height())
            proportion = count / total if total else 0
            ax.annotate(f"{count}\n({proportion:.1%})",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha='center', va='bottom', fontsize=11)

        # ---------- RIGHT: Not Readmitted ----------
        ax = axes[1]
        sns.countplot(x=not_readmitted.astype(int), ax=ax,
                      palette=["#e57373", "#81c784"])
        ax.set_xticks([0, 1])
        ax.set_title(f"{variable} (Not Readmitted) n={len(not_readmitted)}")
        ax.set_xlabel(variable)
        ax.set_ylabel("")

        total = len(not_readmitted)
        for p in ax.patches:
            count = int(p.get_height())
            proportion = count / total if total else 0
            ax.annotate(f"{count}\n({proportion:.1%})",
                        (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha='center', va='bottom', fontsize=11)

    #-------------------------------------------------------
    #            CONTINUOUS VARIABLE
    #-------------------------------------------------------
    else:
        sns.histplot(readmitted, bins=bins, kde=False, color="green", ax=axes[0])
        axes[0].set_title(f"{variable}: Readmitted (n={len(readmitted)})")
        axes[0].set_xlabel(variable)
        axes[0].set_ylabel("Count")

        sns.histplot(not_readmitted, bins=bins, kde=False, color="red", ax=axes[1])
        axes[1].set_title(f"{variable}: Not Readmitted (n={len(not_readmitted)})")
        axes[1].set_xlabel(variable)
        axes[1].set_ylabel("")

    plt.tight_layout()
    plt.show()

# =============================================================================
#       Plot distribution of key variables according to missingness
# =============================================================================

def plot_missing_distribution(df, variable, outcome="readmit_72h", figsize=(12,4)):
    """
    Improved: Adds count + proportions above each bar.
    """

    if variable not in df.columns:
        raise ValueError(f"Column '{variable}' not found.")
    if outcome not in df.columns:
        raise ValueError(f"Outcome '{outcome}' missing.")

    missing_group = df[df[variable].isna()]
    present_group = df[df[variable].notna()]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # -------------------- Missing Group --------------------
    ax = axes[0]
    sns.countplot(x=missing_group[outcome], palette="Blues", ax=ax)

    ax.set_title(f"{variable}: MISSING (n={len(missing_group)})")
    ax.set_xlabel(outcome)
    ax.set_ylabel("Count")
    ax.set_xticklabels(["No Readmit", "Readmit"])

    # Add counts + proportions
    total = len(missing_group)
    for p in ax.patches:
        count = int(p.get_height())
        proportion = count / total if total else 0
        ax.annotate(f"{count}\n({proportion:.1%})",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=11)

    # -------------------- Present Group --------------------
    ax = axes[1]
    sns.countplot(x=present_group[outcome], palette="Greens", ax=ax)

    ax.set_title(f"{variable}: PRESENT (n={len(present_group)})")
    ax.set_xlabel(outcome)
    ax.set_ylabel("")
    ax.set_xticklabels(["No Readmit", "Readmit"])

    total = len(present_group)
    for p in ax.patches:
        count = int(p.get_height())
        proportion = count / total if total else 0
        ax.annotate(f"{count}\n({proportion:.1%})",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.show()

# ===================================================
#       Plot nullity distribution
# ===================================================

def plot_nullity_correlation(df, figsize=(12,10), cmap="coolwarm", annot=False):
    """
    Computes and plots the nullity correlation matrix between all variables.

    Args:
        df: pandas DataFrame
        figsize: size of the heatmap
        cmap: colormap
        annot: show correlation numbers (can get busy with many variables)
    """

    # 1 = missing, 0 = not missing
    null_df = df.isna().astype(int)

    # Correlation matrix of missingness
    corr = null_df.corr()

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr,
        cmap=cmap,
        annot=annot,
        square=True,
        cbar=True,
        xticklabels=True,
        yticklabels=True
    )
    plt.title("Nullity Correlation Matrix", fontsize=16)
    plt.tight_layout()
    plt.show()

    return corr

# ===================================================
#       Plot variable correlation heatmap
# ===================================================

def plot_phik_heatmap(df, figsize=(18,16), cmap="coolwarm"):
    
    df_phik = df.copy()

    # ---------------------------------------------------------
    # 1. Convert datetime columns to numerical timestamps
    # ---------------------------------------------------------
    datetime_cols = df_phik.select_dtypes(include=["datetime", "datetimetz"]).columns
    for col in datetime_cols:
        df_phik[col] = df_phik[col].astype("int64") // 10**9

    # ---------------------------------------------------------
    # 2. Drop columns that cannot be processed (0–2 unique values)
    # ---------------------------------------------------------
    too_few_unique = [
        col for col in df_phik.columns 
        if df_phik[col].nunique(dropna=True) <= 2
    ]

    print(f"Dropping columns with ≤2 unique values (cannot compute phi_k):")
    print(too_few_unique)

    df_phik = df_phik.drop(columns=too_few_unique)

    # ---------------------------------------------------------
    # 3. Drop columns that are almost entire NaN
    # ---------------------------------------------------------
    nan_ratio = df_phik.isna().mean()
    mostly_missing = nan_ratio[nan_ratio > 0.95].index.tolist()

    if mostly_missing:
        print(f"Dropping columns with >95% missing values:")
        print(mostly_missing)

    df_phik = df_phik.drop(columns=mostly_missing)

    # ---------------------------------------------------------
    # 4. Compute phi_k correlation matrix safely
    # ---------------------------------------------------------
    phik_corr = df_phik.phik_matrix(interval_cols="auto")

    # ---------------------------------------------------------
    # 5. Plot heatmap
    # ---------------------------------------------------------
    plt.figure(figsize=figsize)
    sns.heatmap(
        phik_corr,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        cbar=True,
        square=True,
        linewidths=0.5
    )
    plt.title("Φₖ (Phi-k) Correlation Heatmap — Safe ICU Version", fontsize=18)
    plt.tight_layout()
    plt.show()

    return phik_corr

# ===================================================
#       Plot outliers
# ===================================================

def plot_variable_with_outliers(df, variable, figsize=(12,6), bins=20):
    """
    Plots distribution of a continuous variable, marks Q1, median, Q3,
    and outlier thresholds (Q1-1.5*IQR, Q3+1.5*IQR).
    Outliers displayed in red.

    Args:
        df : pandas DataFrame
        variable : str : column name in df
        figsize : tuple : figure size
        bins : int : histogram bins
    """

    if variable not in df.columns:
        raise ValueError(f"{variable} not found in dataframe.")

    data = df[variable].dropna()

    if len(data) < 3:
        print(f"Not enough data to compute IQR for {variable}")
        return

    # Compute quartiles + IQR
    Q1 = data.quantile(0.25)
    Q2 = data.quantile(0.50)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    non_outliers = data[(data >= lower_bound) & (data <= upper_bound)]

    # --- Plot ---
    plt.figure(figsize=figsize)

    # Histogram
    sns.histplot(non_outliers, bins=bins, kde=True, color="steelblue", label="Non-outliers")
    if len(outliers) > 0:
        sns.histplot(outliers, bins=bins, kde=False, color="red", label="Outliers")

    # Quartile lines
    plt.axvline(Q1, color="orange", linestyle="--", label="Q1")
    plt.axvline(Q2, color="green", linestyle="-", label="Median")
    plt.axvline(Q3, color="orange", linestyle="--", label="Q3")

    # Outlier threshold lines
    plt.axvline(lower_bound, color="red", linestyle=":", label="Lower Outlier Threshold")
    plt.axvline(upper_bound, color="red", linestyle=":", label="Upper Outlier Threshold")

    plt.title(f"Distribution and Outliers: {variable}")
    plt.xlabel(variable)
    plt.ylabel("Count")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"----- {variable} -----")
    print(f"Q1: {Q1:.3f}")
    print(f"Median: {Q2:.3f}")
    print(f"Q3: {Q3:.3f}")
    print(f"IQR: {IQR:.3f}")
    print(f"Lower bound: {lower_bound:.3f}")
    print(f"Upper bound: {upper_bound:.3f}")
    print(f"Outliers detected: {len(outliers)}")

    return {
        "Q1": Q1,
        "Median": Q2,
        "Q3": Q3,
        "IQR": IQR,
        "Lower_bound": lower_bound,
        "Upper_bound": upper_bound,
        "Outliers": outliers
    }

# ==============================================
#       Remove non physiologic values
# ==============================================

def remove_nonphysiologic_values(df):
    """
    Removes clearly impossible values for specific clinical variables.
    Does NOT remove clinically plausible outliers.

    Returns a cleaned dataframe and prints what was removed.
    """

    df_clean = df.copy()

    rules = {
        "temp":              (25, 46),      # °C
        "bilirubin_lab":     (0, 35),       # mg/dL
        "pfratio":           (0, 600),      # mmHg
        "delta_spo2":        (-50, 100),   # %
        "delta_temp":        (-10, 10),     # °C change
    }

    for col, (lower, upper) in rules.items():
        if col not in df_clean.columns:
            print(f"Column '{col}' not in dataframe, skipping.")
            continue

        before = df_clean[col].notna().sum()

        # Replace out-of-range with NaN (do NOT drop rows)
        mask = (df_clean[col] < lower) | (df_clean[col] > upper)
        df_clean.loc[mask, col] = np.nan

        after = df_clean[col].notna().sum()
        removed = before - after

        print(f"In {col}: Removed {removed} non-physiologic values "
              f"({before} → {after})")

    return df_clean


# ==================================================================================
#       Imputation of missing variables and create new {variable}_missing column
# ==================================================================================

def mice_impute_splits(
    X_train, X_val, X_test,
    add_missing_indicators=True,
    random_state=42,
    max_iter=20
):
    """
    Applies MICE (Iterative Imputer) on training data only,
    and transforms validation and test sets consistently.

    Adds missing-indicator columns <col>_missing for every column
    that has at least one NaN in any of the splits.

    Parameters
    ----------
    X_train, X_val, X_test : pd.DataFrame
        Feature matrices from the temporal split.
    add_missing_indicators : bool
        Whether to add <col>_missing columns.
    random_state : int
        Reproducibility for IterativeImputer.
    max_iter : int
        Number of MICE iterations.

    Returns
    -------
    X_train_imp, X_val_imp, X_test_imp : pd.DataFrame
        Imputed and indicator-augmented feature matrices.
    imputer : IterativeImputer
        Fitted object that can transform new data.
    """

    # Copy to avoid mutation
    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    # Identify numeric columns for MICE
    num_cols = X_train.select_dtypes(include=["float", "int"]).columns.tolist()

    # Add missing indicators BEFORE imputation (same structure for all splits)
    if add_missing_indicators:
        for col in num_cols:
            if (
                X_train[col].isna().any()
                or X_val[col].isna().any()
                or X_test[col].isna().any()
            ):
                X_train[f"{col}_missing"] = X_train[col].isna().astype(int)
                X_val[f"{col}_missing"] = X_val[col].isna().astype(int)
                X_test[f"{col}_missing"] = X_test[col].isna().astype(int)

    # Fit MICE on *training numeric columns only*
    imputer = IterativeImputer(
        random_state=random_state,
        max_iter=max_iter,
        sample_posterior=True
    )

    # Fit on train
    X_train_num_imp = pd.DataFrame(
        imputer.fit_transform(X_train[num_cols]),
        columns=num_cols,
        index=X_train.index
    )

    # Transform val/test
    X_val_num_imp = pd.DataFrame(
        imputer.transform(X_val[num_cols]),
        columns=num_cols,
        index=X_val.index
    )
    X_test_num_imp = pd.DataFrame(
        imputer.transform(X_test[num_cols]),
        columns=num_cols,
        index=X_test.index
    )

    # Replace numeric columns with imputed versions
    X_train_imp = X_train.copy()
    X_val_imp = X_val.copy()
    X_test_imp = X_test.copy()

    X_train_imp[num_cols] = X_train_num_imp
    X_val_imp[num_cols] = X_val_num_imp
    X_test_imp[num_cols] = X_test_num_imp

    return X_train_imp, X_val_imp, X_test_imp, imputer



# ============================================================
#          LR model
# ============================================================
#%%
def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model assuming ALL categorical encoding
    has already been performed upstream.

    Only numeric features are scaled via RobustScaler.
    """

    # Identify numeric columns only
    numeric_cols = X_train.select_dtypes(include=["float", "int"]).columns

    # Preprocessor ONLY scales numeric features
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numeric_cols),
        ],
        remainder="passthrough"   # Keep already-encoded categorical columns
    )

    # Full pipeline
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("logreg", LogisticRegression(
                max_iter=2000,
                solver="lbfgs",
                n_jobs=-1
            ))
        ]
    )

    # Fit model
    model.fit(X_train, y_train)

    return model


# ============================================================
#          encoding categorical variables
# ============================================================

def encode_categorical_variables(X_train, X_val=None, X_test=None, drop_first=True):
    """
    One-hot encodes categorical variables and aligns columns across train/val/test.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_val   : pd.DataFrame or None
    X_test  : pd.DataFrame or None
    drop_first : bool
        Whether to drop first level to avoid multicollinearity.

    Returns
    -------
    X_train_enc, X_val_enc, X_test_enc
        Encoded and column-aligned DataFrames.
    """

    # Identify categorical columns
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    print(f"Categorical columns detected: {categorical_cols}")

    # Encode TRAIN
    X_train_enc = pd.get_dummies(X_train, columns=categorical_cols, drop_first=drop_first)

    # Encode VAL
    if X_val is not None:
        X_val_enc = pd.get_dummies(X_val, columns=categorical_cols, drop_first=drop_first)
        # Align colums
        X_val_enc = X_val_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    else:
        X_val_enc = None

    # Encode TEST
    if X_test is not None:
        X_test_enc = pd.get_dummies(X_test, columns=categorical_cols, drop_first=drop_first)
        # Align columns
        X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    else:
        X_test_enc = None

    return X_train_enc, X_val_enc, X_test_enc


#%%
# ============================================================
#       Helper functions for generating metrics and graphs
# ============================================================

def get_predicted_probabilities(model, X):
    """Return predicted probabilities for the positive class."""
    return model.predict_proba(X)[:, 1]


def find_best_threshold(y_true, y_scores):
    """Find threshold maximizing F1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    f1_scores = []
    for th in thresholds:
        y_pred = (y_scores >= th).astype(int)
        if len(np.unique(y_pred)) > 1:
            f1_scores.append(f1_score(y_true, y_pred))
        else:
            f1_scores.append(0)

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]


def evaluate_metrics_model(model, X, y):
    """Precision, recall, F1, accuracy using best threshold."""
    y_scores = get_predicted_probabilities(model, X)
    best_threshold = find_best_threshold(y, y_scores)
    y_pred = (y_scores >= best_threshold).astype(int)

    if len(np.unique(y_pred)) < 2:
        return 0, 0, 0, 0, best_threshold

    precision = precision_score(y, y_pred)
    recall    = recall_score(y, y_pred)
    f1        = f1_score(y, y_pred)
    accuracy  = accuracy_score(y, y_pred)

    return precision, recall, f1, accuracy, best_threshold


# ============================================================
# MAIN PLOTTING FUNCTION
# ============================================================

def plot_model_metrics(model, X, y_true, model_name="Model"):
    
    if len(np.unique(y_true)) < 2:
        raise ValueError("y_true contains only one class — cannot compute ROC/PR curves.")

    y_scores = get_predicted_probabilities(model, X)

    # --- Fixed threshold 0.5 ---
    y_pred_05 = (y_scores >= 0.5).astype(int)
    if y_pred_05.sum() > 0:
        precision_05 = precision_score(y_true, y_pred_05)
        recall_05    = recall_score(y_true, y_pred_05)
        f1_05        = f1_score(y_true, y_pred_05)
    else:
        precision_05 = recall_05 = f1_05 = 0.0
    acc_05 = accuracy_score(y_true, y_pred_05)

    # --- Best threshold ---
    precision_bt, recall_bt, f1_bt, acc_bt, best_th = evaluate_metrics_model(model, X, y_true)

    # --- ROC / PR ---
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_roc = auc(fpr, tpr)

    prec_curve, recall_curve, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall_curve, prec_curve)

    # Brier
    brier = brier_score_loss(y_true, y_scores)

    # --- Plotting ---
    fig, axes = plt.subplots(4, 1, figsize=(10, 20))
    fig.suptitle(f"Performance Metrics — {model_name}", fontsize=20)

    # 1. ROC
    axes[0].plot(fpr, tpr, label=f"AUC={auc_roc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    # 2. PR
    axes[1].plot(recall_curve, prec_curve, label=f"AUPRC={auprc:.3f}")
    axes[1].set_title("Precision–Recall Curve")
    axes[1].legend()

    # 3. Calibration
    axes[2].set_title("Calibration Curve")
    try:
        CalibrationDisplay.from_predictions(y_true, y_scores, n_bins=10, ax=axes[2])
    except ValueError:
        axes[2].text(0.3, 0.5, "Calibration not possible", fontsize=14)
    axes[2].plot([0, 1], [0, 1], "k--")

    # 4. Summary box
    axes[3].axis("off")
    txt = f"""
AUC ROC:                 {auc_roc:.3f}
AUPRC:                   {auprc:.3f}
Brier Score:             {brier:.3f}

--- Threshold = 0.5 ---
Accuracy:                {acc_05:.3f}
Precision:               {precision_05:.3f}
Recall:                  {recall_05:.3f}
F1-score:                {f1_05:.3f}

--- Optimal Threshold ---
Best Threshold:          {best_th:.3f}
Accuracy:                {acc_bt:.3f}
Precision:               {precision_bt:.3f}
Recall:                  {recall_bt:.3f}
F1-score:                {f1_bt:.3f}
"""
    axes[3].text(0, 0.5, txt, fontsize=13, fontfamily="monospace")

    plt.tight_layout()
    return fig


def generate_random_baseline(y_true):    
    np.random.seed(42)  # Pour reproductibilité
    y_scores_random = np.random.uniform(0, 1, size=len(y_true))
    return y_scores_random


# ============================================================
# Model tuning and hyperparameters
# ============================================================

def tune_model(
    model_class,
    X_train,
    y_train,
    param_grid=None,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    verbose=1,
    model_kwargs=None
):
    """
    Universal hyperparameter tuning function.
    - If model_class is LogisticRegression → uses your original LR search space (L1/L2/EN).
    - Otherwise → uses the provided param_grid (required).
    
    Parameters
    ----------
    model_class : any sklearn classifier class
        Example: LogisticRegression, RandomForestClassifier, XGBClassifier, etc.
    X_train, y_train : training data
    param_grid : dict or list of dicts
        Required for non-LR models.
    scoring : str
        Metric for GridSearchCV (default: "f1").
    cv : int
        Number of folds for cross-validation.
    n_jobs : int
        Parallel jobs.
    verbose : int
        Verbosity.
    model_kwargs : dict
        Additional arguments to pass when initializing model_class.

    Returns
    -------
    best_model : fitted model
    best_params : dict
    results_df : pd.DataFrame of all CV results
    """

    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    # -------------------------------
    # Construct base model
    # -------------------------------
    model_kwargs = model_kwargs or {}
    base_model = model_class(**model_kwargs)

    # -------------------------------
    # If LogisticRegression → use your original search space
    # -------------------------------
    if model_class is LogisticRegression:
        print("Using built-in Logistic Regression hyperparameter search space.")

        param_grid = [
            # L2 penalty
            {
                "penalty": ["l2"],
                "C": [0.001, 0.01, 0.1, 1, 10],
                "class_weight": [None, "balanced"],
                "solver": ["lbfgs"],
            },
            # L1 penalty
            {
                "penalty": ["l1"],
                "C": [0.001, 0.01, 0.1, 1, 10],
                "class_weight": [None, "balanced"],
                "solver": ["liblinear"],
            },
            # ElasticNet
            {
                "penalty": ["elasticnet"],
                "l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
                "C": [0.01, 0.1, 1, 10],
                "class_weight": [None, "balanced"],
                "solver": ["saga"],
            }
        ]

        # overwrite base_model with LR-specific defaults
        base_model = LogisticRegression(max_iter=5000, n_jobs=-1)

    # -------------------------------
    # If NOT LR → param_grid must be provided
    # -------------------------------
    else:
        if param_grid is None:
            raise ValueError(
                "param_grid must be provided for non-LogisticRegression models."
            )

    # -------------------------------
    # Grid Search
    # -------------------------------
    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )

    grid.fit(X_train, y_train)

    # Package CV results
    results_df = pd.DataFrame(grid.cv_results_)
    results_df = results_df.sort_values(by="mean_test_score", ascending=False)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    print("\n===============================")
    print(f"Best {model_class.__name__} Model")
    print("===============================")
    print(f"Best scoring metric ({scoring}): {grid.best_score_:.4f}\n")
    print("Best hyperparameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return best_model, best_params, results_df


# ============================================================
# Model tuning and hyperparameters
# ============================================================

def train_random_forest(
    X_train,
    y_train,
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
):
    """
    Train a RandomForest classifier assuming categorical encoding
    has already been performed upstream (i.e., no encoding here).

    Random Forest does NOT require scaling.
    """

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
    )

    model.fit(X_train, y_train)

    return model