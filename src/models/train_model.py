"""
Train and compare Logistic Regression, Random Forest, and XGBoost classifiers
to predict international football match outcomes (home_win / draw / away_win).

Input:
    data/processed/features.csv

Outputs:
    data/processed/model_comparison.csv   — metric table for all three models
    models/best_model.pkl                 — serialised best model + pipeline
    models/label_encoder.pkl              — fitted LabelEncoder for the target
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")

INPUT_FEATURES   = PROCESSED_DIR / "features.csv"
OUTPUT_COMPARISON = PROCESSED_DIR / "model_comparison.csv"
OUTPUT_BEST_MODEL = MODELS_DIR / "best_model.pkl"
OUTPUT_ENCODER    = MODELS_DIR / "label_encoder.pkl"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Temporal train/test split — train on all matches before this date,
# test on matches from this date forward. Avoids data leakage from the future.
TEST_SPLIT_DATE = "2018-01-01"

# Features fed to each model
FEATURE_COLS = [
    # Rolling form (10-match window, shift-1 leakage protection)
    "home_form_win_pct",
    "home_form_goals_scored",
    "home_form_goals_conceded",
    "home_form_goal_diff",
    "home_form_clean_sheet_pct",
    "away_form_win_pct",
    "away_form_goals_scored",
    "away_form_goals_conceded",
    "away_form_goal_diff",
    "away_form_clean_sheet_pct",
    # FIFA rankings (current snapshot)
    "home_fifa_rank",
    "home_fifa_points",
    "away_fifa_rank",
    "away_fifa_points",
    # Elo ratings (current snapshot, high null rate for historical matches)
    "home_elo_rating",
    "away_elo_rating",
    # World Cup history
    "home_wc_appearances",
    "home_wc_titles",
    "home_wc_best_finish",
    "away_wc_appearances",
    "away_wc_titles",
    "away_wc_best_finish",
    # Derived match-level features
    "rank_diff",
    "points_diff",
    "elo_diff",
    "same_conf",
    # Match context
    "neutral",
]

TARGET_COL = "result"

# Evaluation metric used to pick the best model
PRIMARY_METRIC = "log_loss"  # lower is better


# ---------------------------------------------------------------------------
# Step 1 — Load and split
# ---------------------------------------------------------------------------

def load_and_split(path: Path) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
]:
    print("Loading features.csv...")
    df = pd.read_csv(path, parse_dates=["date"])

    # Temporal split
    train = df[df["date"] < TEST_SPLIT_DATE].copy()
    test  = df[df["date"] >= TEST_SPLIT_DATE].copy()

    print(f"  Train: {len(train):,} rows  ({train['date'].min().date()} – {train['date'].max().date()})")
    print(f"  Test:  {len(test):,}  rows  ({test['date'].min().date()} – {test['date'].max().date()})")

    X_train = train[FEATURE_COLS]
    X_test  = test[FEATURE_COLS]
    y_train = train[TARGET_COL]
    y_test  = test[TARGET_COL]

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Step 2 — Encode target labels
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
# Step 3 — Build model pipelines
# ---------------------------------------------------------------------------

def build_pipelines() -> dict[str, Pipeline]:
    """
    Each pipeline: median imputation → model.
    Logistic Regression also includes standard scaling.
    XGBoost handles nulls natively but is wrapped in the same imputer pipeline
    for consistency.
    """
    imputer = SimpleImputer(strategy="median")

    pipelines = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("model",   LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model",   XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
    }
    return pipelines


# ---------------------------------------------------------------------------
# Step 4 — Train and evaluate
# ---------------------------------------------------------------------------

def evaluate(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train_enc: np.ndarray,
    y_test_enc: np.ndarray,
    le: LabelEncoder,
) -> dict:
    print(f"  Training {name}...")
    pipeline.fit(X_train, y_train_enc)

    y_pred      = pipeline.predict(X_test)
    y_pred_prob = pipeline.predict_proba(X_test)

    n_classes = len(le.classes_)
    avg = "weighted"

    metrics = {
        "model":     name,
        "accuracy":  round(accuracy_score(y_test_enc, y_pred), 4),
        "precision": round(precision_score(y_test_enc, y_pred, average=avg, zero_division=0), 4),
        "recall":    round(recall_score(y_test_enc, y_pred, average=avg), 4),
        "f1":        round(f1_score(y_test_enc, y_pred, average=avg), 4),
        "roc_auc":   round(roc_auc_score(
            y_test_enc, y_pred_prob,
            multi_class="ovr", average=avg,
        ), 4),
        "log_loss":  round(log_loss(y_test_enc, y_pred_prob), 4),
    }

    print(
        f"    accuracy={metrics['accuracy']}  "
        f"f1={metrics['f1']}  "
        f"log_loss={metrics['log_loss']}  "
        f"roc_auc={metrics['roc_auc']}"
    )
    return metrics


# ---------------------------------------------------------------------------
# Step 5 — Save outputs
# ---------------------------------------------------------------------------

def save_comparison(results: list[dict]) -> None:
    df = pd.DataFrame(results)
    df = df.sort_values("log_loss")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_COMPARISON, index=False)
    print(f"\nModel comparison saved: {OUTPUT_COMPARISON}")
    print(df.to_string(index=False))


def save_best_model(
    best_name: str,
    pipelines: dict[str, Pipeline],
    le: LabelEncoder,
    results: list[dict],
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_name":    best_name,
        "pipeline":      pipelines[best_name],
        "feature_cols":  FEATURE_COLS,
        "label_encoder": le,
        "metrics":       next(r for r in results if r["model"] == best_name),
    }

    with open(OUTPUT_BEST_MODEL, "wb") as f:
        pickle.dump(payload, f)

    with open(OUTPUT_ENCODER, "wb") as f:
        pickle.dump(le, f)

    print(f"\nBest model ({best_name}) saved: {OUTPUT_BEST_MODEL}")
    print(f"Label encoder saved:           {OUTPUT_ENCODER}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_models() -> None:
    X_train, X_test, y_train, y_test = load_and_split(INPUT_FEATURES)

    print("\nEncoding target labels...")
    y_train_enc, y_test_enc, le = encode_target(y_train, y_test)

    pipelines = build_pipelines()

    print("\nTraining models...")
    results = []
    for name, pipeline in pipelines.items():
        metrics = evaluate(name, pipeline, X_train, X_test, y_train_enc, y_test_enc, le)
        results.append(metrics)

    save_comparison(results)

    # Best model = lowest log loss (probability calibration matters most for simulation)
    best = min(results, key=lambda r: r[PRIMARY_METRIC])
    best_name = best["model"]
    print(f"\nBest model by {PRIMARY_METRIC}: {best_name}")

    save_best_model(best_name, pipelines, le, results)


if __name__ == "__main__":
    train_models()
