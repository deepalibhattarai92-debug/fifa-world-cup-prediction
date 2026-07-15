"""
Train Version 2 model — XGBoost with hyperparameter tuning on competitive matches only.

Improvements over V1:
  1. Trains on competitive matches only (friendlies excluded — less noise)
  2. New features: match_importance, h2h_win_rate, h2h_goal_diff
  3. RandomizedSearchCV hyperparameter tuning (time-series aware CV)

Input:
    data/processed/features_v2.csv

Outputs:
    data/processed/model_comparison_v2.csv
    models/best_model_v2.pkl
    models/label_encoder_v2.pkl
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")

INPUT_FEATURES    = PROCESSED_DIR / "features_v2.csv"
OUTPUT_COMPARISON = PROCESSED_DIR / "model_comparison_v2.csv"
OUTPUT_BEST_MODEL = MODELS_DIR / "best_model_v2.pkl"
OUTPUT_ENCODER    = MODELS_DIR / "label_encoder_v2.pkl"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEST_SPLIT_DATE = "2018-01-01"

# Competitive tournaments only — friendlies excluded from training
COMPETITIVE_KEYWORDS = [
    "FIFA World Cup",
    "UEFA Euro",
    "Copa América",
    "African Cup",
    "AFC Asian Cup",
    "Gold Cup",
    "Nations League",
    "qualification",
    "CONCACAF",
]

from src.models.feature_cols import FEATURE_COLS

TARGET_COL     = "result"
PRIMARY_METRIC = "log_loss"

# ---------------------------------------------------------------------------
# Step 1 — Load and filter
# ---------------------------------------------------------------------------

def load_and_split(path: Path) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
]:
    print("Loading features_v2.csv...")
    df = pd.read_csv(path, parse_dates=["date"])

    # Filter to competitive matches only
    comp_pattern = "|".join(COMPETITIVE_KEYWORDS)
    df_comp = df[df["tournament"].str.contains(comp_pattern, case=False, na=False)].copy()
    print(f"  Total rows:       {len(df):,}")
    print(f"  Competitive only: {len(df_comp):,}  ({len(df_comp)/len(df)*100:.1f}%)")

    train = df_comp[df_comp["date"] < TEST_SPLIT_DATE]
    test  = df_comp[df_comp["date"] >= TEST_SPLIT_DATE]

    print(f"  Train: {len(train):,} ({train['date'].min().date()} – {train['date'].max().date()})")
    print(f"  Test:  {len(test):,}  ({test['date'].min().date()} – {test['date'].max().date()})")

    X_train = train[FEATURE_COLS]
    X_test  = test[FEATURE_COLS]
    y_train = train[TARGET_COL]
    y_test  = test[TARGET_COL]

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step 2 — Encode labels
# ---------------------------------------------------------------------------

def encode_target(
    y_train: pd.Series, y_test: pd.Series
) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
    print(f"  Target classes: {list(le.classes_)}")
    return y_train_enc, y_test_enc, le


# ---------------------------------------------------------------------------
# Step 3 — Build pipelines (base models for comparison)
# ---------------------------------------------------------------------------

def build_base_pipelines() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=20,
                random_state=42, n_jobs=-1,
            )),
        ]),
        "XGBoost (base)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss", random_state=42, n_jobs=-1,
            )),
        ]),
    }


# ---------------------------------------------------------------------------
# Step 4 — Tune XGBoost
# ---------------------------------------------------------------------------

def tune_xgboost(
    X_train: pd.DataFrame,
    y_train_enc: np.ndarray,
) -> Pipeline:
    print("\nTuning XGBoost with RandomizedSearchCV (TimeSeriesSplit)...")

    param_dist = {
        "model__n_estimators":     [200, 300, 400, 500],
        "model__max_depth":        [3, 4, 5, 6, 7],
        "model__learning_rate":    [0.01, 0.03, 0.05, 0.1],
        "model__subsample":        [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5, 10],
        "model__gamma":            [0, 0.1, 0.2, 0.5],
        "model__reg_alpha":        [0, 0.01, 0.1, 1.0],
        "model__reg_lambda":       [0.5, 1.0, 2.0, 5.0],
    }

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   XGBClassifier(
            eval_metric="mlogloss", random_state=42, n_jobs=-1,
        )),
    ])

    tscv = TimeSeriesSplit(n_splits=5)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=40,
        scoring="neg_log_loss",
        cv=tscv,
        random_state=42,
        n_jobs=1,      # parallel workers blocked in sandbox
        verbose=1,
        refit=True,
    )

    search.fit(X_train, y_train_enc)

    best_params = {k: v for k, v in search.best_params_.items()}
    print(f"  Best log_loss (CV): {-search.best_score_:.4f}")
    print(f"  Best params: {best_params}")

    return search.best_estimator_


# ---------------------------------------------------------------------------
# Step 5 — Evaluate
# ---------------------------------------------------------------------------

def evaluate(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_enc: np.ndarray,
    y_test_enc: np.ndarray,
) -> dict:
    pipeline.fit(X_train, y_train_enc)
    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)

    avg = "weighted"
    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test_enc, y_pred), 4),
        "precision": round(precision_score(y_test_enc, y_pred, average=avg, zero_division=0), 4),
        "recall":    round(recall_score(y_test_enc, y_pred, average=avg), 4),
        "f1":        round(f1_score(y_test_enc, y_pred, average=avg), 4),
        "roc_auc":   round(roc_auc_score(y_test_enc, y_pred_prob, multi_class="ovr", average=avg), 4),
        "log_loss":  round(log_loss(y_test_enc, y_pred_prob), 4),
    }
    print(f"  {name:30} acc={metrics['accuracy']}  f1={metrics['f1']}  log_loss={metrics['log_loss']}  roc_auc={metrics['roc_auc']}")
    return metrics


# ---------------------------------------------------------------------------
# Step 6 — Save
# ---------------------------------------------------------------------------

def save_outputs(
    results: list[dict],
    tuned_pipeline: Pipeline,
    le: LabelEncoder,
) -> None:
    df = pd.DataFrame(results).sort_values("log_loss")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_COMPARISON, index=False)
    print(f"\nModel comparison:\n{df.to_string(index=False)}")
    print(f"\nSaved: {OUTPUT_COMPARISON}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name":   "XGBoost (tuned)",
        "pipeline":     tuned_pipeline,
        "feature_cols": FEATURE_COLS,
        "label_encoder": le,
        "metrics": next((r for r in results if "tuned" in r["model"].lower()), results[0]),
    }
    with open(OUTPUT_BEST_MODEL, "wb") as f:
        pickle.dump(payload, f)

    with open(OUTPUT_ENCODER, "wb") as f:
        pickle.dump(le, f)

    print(f"Best model saved: {OUTPUT_BEST_MODEL}")
    print(f"Label encoder:    {OUTPUT_ENCODER}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_models_v2() -> None:
    X_train, X_test, y_train, y_test = load_and_split(INPUT_FEATURES)

    print("\nEncoding target labels...")
    y_train_enc, y_test_enc, le = encode_target(y_train, y_test)

    # Base models
    pipelines = build_base_pipelines()
    print("\nEvaluating base models...")
    results = []
    for name, pipeline in pipelines.items():
        m = evaluate(name, pipeline, X_train, X_test, y_train_enc, y_test_enc)
        results.append(m)

    # Tuned XGBoost
    tuned = tune_xgboost(X_train, y_train_enc)
    m = evaluate("XGBoost (tuned)", tuned, X_train, X_test, y_train_enc, y_test_enc)
    results.append(m)

    save_outputs(results, tuned, le)


if __name__ == "__main__":
    train_models_v2()
