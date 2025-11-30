import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    chi2,
    RFECV,
    SequentialFeatureSelector,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns
sns.set(style="whitegrid")


# ============================================================
# 1. Statistical Feature Selection (Mutual Information / Chi²)
# ============================================================

def perform_statistical_feature_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_features_to_select: int = 10,
    method: str = "mutual_info",
    output_path: Optional[str] = None,
    random_state: int = 42,
) -> Tuple[List[str], plt.Figure]:
    """
    Statistical feature selection using Mutual Information or Chi².

    Args
    ----
    X_train : DataFrame
    y_train : Series
    n_features_to_select : int
        Number of features to keep.
    method : {"mutual_info", "chi2"}
    output_path : optional str
        If given, saves the figure.
    random_state : int

    Returns
    -------
    selected_features : list[str]
    fig : matplotlib Figure
    """

    print(f"\n[Statistical FS] Method={method}, k={n_features_to_select}")

    if method == "mutual_info":
        score_func = lambda X, y: mutual_info_classif(
            X, y, random_state=random_state
        )
        method_name = "Mutual Information"
        X_used = X_train

    elif method == "chi2":
        # Chi² needs non-negative values → shift each column so min >= 0
        X_shifted = X_train.copy()
        for col in X_shifted.columns:
            col_min = X_shifted[col].min()
            if pd.notnull(col_min) and col_min < 0:
                X_shifted[col] = X_shifted[col] - col_min
        score_func = chi2
        method_name = "Chi-Squared"
        X_used = X_shifted

    else:
        raise ValueError(f"Unknown method: {method}")

    selector = SelectKBest(score_func=score_func, k=n_features_to_select)
    selector.fit(X_used, y_train)

    selected_features = X_train.columns[selector.get_support()].tolist()
    scores = selector.scores_

    print(f"  → Selected features ({len(selected_features)}): {selected_features}")

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(12, 6))

    sorted_idx = np.argsort(scores)[::-1]
    sorted_features = [X_train.columns[i] for i in sorted_idx]
    sorted_scores = scores[sorted_idx]

    colors = [
        "green" if feat in selected_features else "steelblue"
        for feat in sorted_features
    ]

    ax.barh(range(len(sorted_features)), sorted_scores, color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel(f"{method_name} score")
    ax.set_title(f"Feature Selection — {method_name}")
    # Threshold line at the k-th selected score
    if n_features_to_select <= len(sorted_scores):
        ax.axvline(
            sorted_scores[n_features_to_select - 1],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Top {n_features_to_select}",
        )
        ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return selected_features, fig

# ============================================================
# 2. Sequential Feature Selection (Forward / Backward)
# ============================================================

def perform_sequential_feature_selection(
    model_class: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_features_to_select: int = 10,
    direction: str = "forward",
    cv: int = 3,
    scale: bool = True,
    **model_kwargs,
) -> Tuple[List[str], plt.Figure]:
    """
    Sequential Feature Selection (SFS) using forward or backward selection.

    Args
    ----
    model_class : classifier class (e.g., LogisticRegression, RandomForestClassifier)
    X_train, y_train : training data
    X_val, y_val : validation data (for final AUC)
    n_features_to_select : int
    direction : {"forward", "backward"}
    cv : int
    scale : bool
        If True, wraps estimator in a StandardScaler pipeline (good for LR/NN).
    model_kwargs : passed to model_class

    Returns
    -------
    selected_features : list[str]
    fig : matplotlib Figure (AUC using selected features vs baseline)
    """
    from sklearn.feature_selection import SequentialFeatureSelector

    print(f"\n[SFS] Direction={direction}, k={n_features_to_select}")

    base_estimator = model_class(**model_kwargs)

    if scale:
        estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", base_estimator),
            ]
        )
    else:
        estimator = base_estimator

    sfs = SequentialFeatureSelector(
        estimator=estimator,
        n_features_to_select=n_features_to_select,
        direction=direction,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )

    sfs.fit(X_train, y_train)
    selected_mask = sfs.get_support()
    selected_features = X_train.columns[selected_mask].tolist()

    print(f"  → Selected features ({len(selected_features)}): {selected_features}")

    # Evaluate baseline (all features) vs SFS-selected features
    # (using the same model type)
    if scale:
        baseline_estimator = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", model_class(**model_kwargs)),
            ]
        )
    else:
        baseline_estimator = model_class(**model_kwargs)

    baseline_estimator.fit(X_train, y_train)
    y_val_proba_full = baseline_estimator.predict_proba(X_val)[:, 1]
    auc_full = roc_auc_score(y_val, y_val_proba_full)

    estimator.fit(X_train[selected_features], y_train)
    y_val_proba_sel = estimator.predict_proba(X_val[selected_features])[:, 1]
    auc_sel = roc_auc_score(y_val, y_val_proba_sel)

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        ["All features", f"SFS ({len(selected_features)})"],
        [auc_full, auc_sel],
        color=["steelblue", "green"],
        alpha=0.8,
    )
    for x, v in zip(
        ["All features", f"SFS ({len(selected_features)})"], [auc_full, auc_sel]
    ):
        ax.text(
            x,
            v + 0.005,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylabel("Validation AUC")
    ax.set_title("Sequential Feature Selection — AUC comparison")
    ax.set_ylim(0.0, max(auc_full, auc_sel) + 0.05)

    plt.tight_layout()

    return selected_features, fig

# ============================================================
# 3. Regularized Logistic Regression (L1 / L2 / Elastic Net)
# ============================================================

def perform_regularized_logistic_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    penalties: List[str] = ["l1", "l2", "elasticnet"],
    Cs: List[float] = None,
    l1_ratios: List[float] = None,
    max_iter: int = 2000,
    random_state: int = 42,
) -> Tuple[List[str], LogisticRegression, plt.Figure]:
    """
    Hyperparameter search for Logistic Regression with L1/L2/ElasticNet,
    and feature selection via sparsity for L1/EN.

    Args
    ----
    X_train, y_train, X_val, y_val
    penalties : list of penalties to try ["l1", "l2", "elasticnet"]
    Cs : list of C values (inverse regularization)
    l1_ratios : list of l1_ratio values (for elasticnet)
    max_iter : int
    random_state : int

    Returns
    -------
    selected_features : list[str]
    best_clf : fitted LogisticRegression estimator
    fig : matplotlib Figure
    """

    print("\n[Logistic Regularization] Searching over penalties + C (+ l1_ratio for EN)")

    if Cs is None:
        Cs = np.logspace(-3, 2, 10)
    if l1_ratios is None:
        l1_ratios = [0.2, 0.5, 0.8]

    param_grid = []

    for pen in penalties:
        if pen in ("l1", "l2"):
            param_grid.append(
                {
                    "clf__penalty": [pen],
                    "clf__C": Cs,
                    "clf__l1_ratio": [None],
                    "clf__solver": ["liblinear"],  # supports l1 & l2
                }
            )
        elif pen == "elasticnet":
            param_grid.append(
                {
                    "clf__penalty": ["elasticnet"],
                    "clf__C": Cs,
                    "clf__l1_ratio": l1_ratios,
                    "clf__solver": ["saga"],  # required for elasticnet
                }
            )

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=max_iter, random_state=random_state, n_jobs=-1
                ),
            ),
        ]
    )

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        verbose=0,
    )

    grid.fit(X_train, y_train)

    print(f"  → Best params: {grid.best_params_}")
    print(f"  → Best CV AUC: {grid.best_score_:.3f}")

    best_model: Pipeline = grid.best_estimator_
    # Evaluate on validation set
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    auc_val = roc_auc_score(y_val, y_val_proba)
    print(f"  → Validation AUC with best model: {auc_val:.3f}")

    # Extract underlying LogisticRegression
    best_clf: LogisticRegression = best_model.named_steps["clf"]

    # Determine selected features (non-zero coefficients for L1/EN)
    penalty_used = best_clf.penalty
    coef = best_clf.coef_.ravel()
    if penalty_used in ("l1", "elasticnet"):
        selected_mask = coef != 0
        selected_features = X_train.columns[selected_mask].tolist()
    else:
        selected_features = X_train.columns.tolist()

    print(f"  → Penalty used: {penalty_used}")
    print(f"  → Selected features ({len(selected_features)}): {selected_features}")

    # --- Visualization: sparsity & AUC evolution ---
    # We'll reconstruct a simple plot from the CV results
    results = pd.DataFrame(grid.cv_results_)
    # Use mean_test_score vs C (log-scale) for each penalty
    fig, ax = plt.subplots(figsize=(8, 6))

    for pen in penalties:
        mask = results["param_clf__penalty"] == pen
        if not mask.any():
            continue
        df_pen = results[mask].copy()
        # sort by C for readability
        df_pen["C"] = df_pen["param_clf__C"].astype(float)
        df_pen = df_pen.sort_values("C")
        ax.semilogx(
            df_pen["C"],
            df_pen["mean_test_score"],
            marker="o",
            label=pen,
        )

    ax.set_xlabel("C (inverse regularization)")
    ax.set_ylabel("Mean CV AUC")
    ax.set_title("Logistic Regression — Regularization search")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return selected_features, best_clf, fig

# ============================================================
# 4. Orchestrator: compare FS methods + models (Leaderboard)
# ============================================================

def train_model_for_features(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[float, Any]:
    """
    Helper: trains a given model on (X_train, y_train), evaluates AUC on val.

    model_name ∈ {"logistic", "rf", "mlp"}
    Returns (auc_val, fitted_model)
    """
    if model_name == "logistic":
        est = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        penalty="l2",
                        solver="liblinear",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        )
    elif model_name == "rf":
        est = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "mlp":
        est = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(32, 16),
                        activation="relu",
                        solver="adam",
                        max_iter=500,
                        random_state=42,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    est.fit(X_train, y_train)
    y_val_proba = est.predict_proba(X_val)[:, 1]
    auc_val = roc_auc_score(y_val, y_val_proba)

    return auc_val, est


def run_feature_selection_leaderboard(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_features: int = 10,
    output_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Run multiple feature selection methods, evaluate different models,
    build a leaderboard, and optionally export plots & tables.

    Methods:
      - Mutual Information
      - Chi-Squared
      - Sequential Feature Selection (SFS) with Logistic Regression
      - Regularized Logistic Regression (L1/L2/EN)

    Models:
      - Logistic Regression
      - Random Forest
      - MLP (Neural Network)

    Returns
    -------
    leaderboard : DataFrame
        One row per (FS method, model, feature subset)
    details : dict
        Nested results by method.
    """

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    methods_results: Dict[str, Dict[str, Any]] = {}
    leaderboard_records: List[Dict[str, Any]] = []

    # -------------------------
    # 1. Statistical — MI
    # -------------------------
    mi_feats, fig_mi = perform_statistical_feature_selection(
        X_train, y_train,
        n_features_to_select=n_features,
        method="mutual_info",
        output_path=str(output_dir / "fs_mutual_info.png") if output_dir else None,
    )
    methods_results["Mutual Information"] = {
        "features": mi_feats,
        "fig": fig_mi,
    }

    # -------------------------
    # 2. Statistical — Chi²
    # -------------------------
    chi2_feats, fig_chi2 = perform_statistical_feature_selection(
        X_train, y_train,
        n_features_to_select=n_features,
        method="chi2",
        output_path=str(output_dir / "fs_chi2.png") if output_dir else None,
    )
    methods_results["Chi2"] = {
        "features": chi2_feats,
        "fig": fig_chi2,
    }

    # -------------------------
    # 3. Sequential FS (SFS) with Logistic
    # -------------------------
    sfs_feats, fig_sfs = perform_sequential_feature_selection(
        model_class=LogisticRegression,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_features_to_select=n_features,
        direction="forward",
        cv=3,
        scale=True,
        max_iter=2000,
        solver="liblinear",
        random_state=42,
    )
    methods_results["SFS Logistic"] = {
        "features": sfs_feats,
        "fig": fig_sfs,
    }

    # -------------------------
    # 4. Logistic Regularization (L1/L2/EN)
    # -------------------------
    reg_feats, best_logreg, fig_reg = perform_regularized_logistic_selection(
        X_train, y_train,
        X_val, y_val,
        penalties=["l1", "l2", "elasticnet"],
        Cs=None,
        l1_ratios=None,
        max_iter=2000,
        random_state=42,
    )
    methods_results["LR Regularization"] = {
        "features": reg_feats,
        "clf": best_logreg,
        "fig": fig_reg,
    }

    # ======================================================
    # Evaluate each FS method with 3 models and build table
    # ======================================================

    model_names = ["logistic", "rf", "mlp"]

    for method_name, info in methods_results.items():
        feats = info["features"]
        Xtr = X_train[feats]
        Xv = X_val[feats]

        for mname in model_names:
            auc_val, est = train_model_for_features(
                mname, Xtr, y_train, Xv, y_val
            )
            leaderboard_records.append(
                {
                    "fs_method": method_name,
                    "model": mname,
                    "n_features": len(feats),
                    "features": feats,
                    "val_auc": auc_val,
                }
            )
            # store estimator as well
            info.setdefault("models", {})[mname] = {
                "estimator": est,
                "val_auc": auc_val,
            }

    leaderboard = pd.DataFrame(leaderboard_records)
    leaderboard = leaderboard.sort_values("val_auc", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print("FEATURE SELECTION + MODEL LEADERBOARD (sorted by validation AUC)")
    print("=" * 80)
    print(leaderboard[["fs_method", "model", "n_features", "val_auc"]])

    if output_dir:
        leaderboard.to_csv(output_dir / "feature_selection_LR.csv", index=False)

    # Quick barplot of top configurations
    fig_lb, ax = plt.subplots(figsize=(10, 6))
    top_k = min(20, len(leaderboard))
    sns.barplot(
        data=leaderboard.head(top_k),
        x="val_auc",
        y="fs_method",
        hue="model",
        ax=ax,
        orient="h",
    )
    ax.set_xlabel("Validation AUC")
    ax.set_ylabel("Feature Selection Method")
    ax.set_title("Top configurations by AUC")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    if output_dir:
        fig_lb.savefig(output_dir / "LR_barplot.png", dpi=150, bbox_inches="tight")

    methods_results["LR_leaderboard_fig"] = fig_lb

    return leaderboard, methods_results

