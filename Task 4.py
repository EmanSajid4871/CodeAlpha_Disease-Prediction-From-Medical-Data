import os, io, sys, warnings, time, urllib.request
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Tuple, Optional
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib

# Optional XGBoost
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


# ----------------------------- Utils ---------------------------------

def outdir_for(dsname: str) -> str:
    d = os.path.join("outputs", dsname)
    os.makedirs(d, exist_ok=True)
    return d

def download_text(url: str, timeout: int = 30, max_retries: int = 3) -> str:
    """
    Robust text downloader with UA header + retries.
    """
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", errors="ignore")
        except Exception as e:
            last_err = e
            time.sleep(0.8 * attempt)
    raise last_err


# -------------------------- Dataset loaders ---------------------------

def dataset_breast_cancer() -> Tuple[pd.DataFrame, pd.Series]:
    # UCI wdbc.data → 569 rows, 32 cols (ID, diagnosis, 30 features)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    raw = download_text(url)
    colnames = ["ID", "diagnosis"] + [f"f{i}" for i in range(1, 31)]
    df = pd.read_csv(io.StringIO(raw), header=None, names=colnames)
    y = df["diagnosis"].map({"M": 1, "B": 0}).astype(int)
    X = df.drop(columns=["ID", "diagnosis"])
    return X, y

def dataset_heart() -> Tuple[pd.DataFrame, pd.Series]:
    # UCI processed.cleveland.data (no header, '?' as missing)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    raw = download_text(url)
    cols = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal","num"]
    df = pd.read_csv(io.StringIO(raw), header=None, names=cols, na_values="?")
    y = (df["num"] > 0).astype(int)         # 0 = no disease, >0 = disease
    X = df.drop(columns=["num"])
    return X, y

def dataset_diabetes() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Pima Indians Diabetes (Classification)
    Primary: OpenML data_id=37 (labels: tested_negative/positive)
    Fallback: GitHub mirror (numeric 0/1 target).
    Also fix zero-as-missing for known fields and set to NaN for imputation.
    """
    # --- Primary: OpenML ---
    try:
        ds = fetch_openml(data_id=37, as_frame=True)   # Pima Indians Diabetes
        X = ds.data.copy()
        y = ds.target.copy()

        # Map string labels to 0/1 if present
        if y.dtype == "O" or str(y.dtype).startswith("category"):
            y = y.astype(str).str.lower().map({"tested_positive": 1, "tested_negative": 0}).astype(int)

        # Zero-as-missing: columns may use short aliases on OpenML
        alias_groups = [
            ["glucose", "plas"],
            ["blood_pressure", "pres"],
            ["skin_thickness", "skin"],
            ["insulin", "insu"],
            ["bmi", "mass"],
        ]
        for group in alias_groups:
            for col in group:
                if col in X.columns:
                    X.loc[pd.to_numeric(X[col], errors="coerce") == 0, col] = np.nan
                    break  # only one of the aliases will exist

        return X, y

    except Exception as e:
        print(f"[Diabetes] OpenML failed: {e}\nFalling back to mirror…")

    # --- Fallback: GitHub mirror (Jason Brownlee) ---
    try:
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        raw = download_text(url)
        cols = ["pregnancies","glucose","blood_pressure","skin_thickness",
                "insulin","bmi","diabetes_pedigree","age","outcome"]
        df = pd.read_csv(io.StringIO(raw), header=None, names=cols)
        y = df["outcome"].astype(int)
        X = df.drop(columns=["outcome"])
        for c in ["glucose","blood_pressure","skin_thickness","insulin","bmi"]:
            X.loc[X[c] == 0, c] = np.nan
        return X, y

    except Exception as e2:
        raise RuntimeError("Could not load Pima Indians Diabetes from OpenML or mirror. "
                           "If you are offline, download the CSV locally and add a manual loader.") from e2


# ------------------------ Modeling utilities --------------------------

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])
    return pre

def get_models(seed: int = 42) -> Dict[str, object]:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed),
        "SVM_RBF": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, n_jobs=-1, class_weight="balanced_subsample", random_state=seed
        )
    }
    if XGB_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss", random_state=seed
        )
    return models

def evaluate_all(X: pd.DataFrame, y: pd.Series, pre: ColumnTransformer,
                 models: Dict[str, object], outdir: str) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])

        # Cross-val probabilities for fair metrics
        y_proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)
        y_pred = (y_proba[:, 1] >= 0.5).astype(int)

        acc  = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec  = recall_score(y, y_pred, zero_division=0)
        f1   = f1_score(y, y_pred, zero_division=0)
        try:
            roc = roc_auc_score(y, y_proba[:, 1])
        except Exception:
            roc = np.nan

        rows.append({"model": name, "accuracy": acc, "precision": prec,
                     "recall": rec, "f1": f1, "roc_auc": roc})

        # Confusion matrix (save PNG)
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(4,4))
        im = ax.imshow(cm, interpolation="nearest")
        ax.set_title(f"Confusion Matrix — {name}")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha='center', va='center')
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"confusion_matrix_{name}.png"), dpi=150)
        plt.close(fig)

    df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    df.to_csv(os.path.join(outdir, "metrics_summary.csv"), index=False)

    # Combined ROC (fit-on-full for visualization only)
    plt.figure(figsize=(6,5))
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        try:
            pipe.fit(X, y)
            if hasattr(pipe.named_steps["clf"], "predict_proba"):
                ys = pipe.predict_proba(X)[:, 1]
            elif hasattr(pipe.named_steps["clf"], "decision_function"):
                ys = pipe.decision_function(X)
            else:
                continue
            RocCurveDisplay.from_predictions(y_true=y, y_pred=ys, name=name)
        except Exception:
            continue
    plt.title("ROC Curves (train fit for viz)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_curves.png"), dpi=150)
    plt.close()

    # Hold-out best model and save
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    best_name, best_score, best_pipe = None, -np.inf, None
    for name, clf in models.items():
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        pipe.fit(Xtr, ytr)
        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            ys = pipe.predict_proba(Xte)[:, 1]
        elif hasattr(pipe.named_steps["clf"], "decision_function"):
            ys = pipe.decision_function(Xte)
        else:
            ys = pipe.predict(Xte).astype(float)
        try:
            roc = roc_auc_score(yte, ys)
        except Exception:
            roc = np.nan
        if np.nan_to_num(roc) > best_score:
            best_name, best_score, best_pipe = name, roc, pipe

    joblib.dump(best_pipe, os.path.join(outdir, "best_model.joblib"))
    return df, best_name, best_score


# --------------------------- Runner -----------------------------------

def run_dataset(name: str, loader_func):
    print(f"\n=== {name} ===")
    X, y = loader_func()
    pre = build_preprocessor(X)
    out = outdir_for(name)
    models = get_models()

    metrics, best_name, best_score = evaluate_all(X, y, pre, models, out)
    print(metrics)
    print(f"Best model (hold-out ROC-AUC): {best_name} — {best_score:.4f}")
    print(f"Artifacts saved to: {out}")

def main():
    os.makedirs("outputs", exist_ok=True)
    run_dataset("breast_cancer_wdbc", dataset_breast_cancer)
    run_dataset("heart_cleveland",    dataset_heart)
    run_dataset("diabetes_pima",      dataset_diabetes)
    print("\nAll done. Check ./outputs/<dataset>/ for CSVs, PNGs, and best_model.joblib")

if __name__ == "__main__":
    main()