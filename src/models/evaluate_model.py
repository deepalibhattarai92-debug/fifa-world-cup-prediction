"""
Pre-compute all model evaluation artifacts needed by the Streamlit dashboard.

Outputs written to data/processed/eval/:
    confusion_matrix.csv        - 3x3 confusion matrix counts
    calibration.csv             - calibration curve (predicted vs actual frequency)
    roc_curves.csv              - one-vs-rest ROC curve data for all 3 classes
    feature_importance.csv      - XGBoost feature importances
    test_predictions.csv        - test set with actual + predicted labels and probabilities
    temporal_accuracy.csv       - accuracy by year on the test set
    confederation_accuracy.csv  - accuracy by home confederation

Run from project root:
    python src/models/evaluate_model.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
EVAL_DIR      = PROCESSED_DIR / "eval"

INPUT_FEATURES = PROCESSED_DIR / "features_v2.csv"
INPUT_MODEL    = MODELS_DIR / "best_model_v2.pkl"
INPUT_FIFA     = PROCESSED_DIR / "fifa_rankings.csv"

EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRAIN_CUTOFF = 2018   # matches from 2018 onwards = test set
from src.models.feature_cols import FEATURE_COLS

FEATURE_DISPLAY_NAMES = {
    "elo_diff":                    "Elo Rating Difference",
    "rank_diff":                   "FIFA Ranking Difference",
    "home_form_goal_diff":         "Home Recent Goal Diff (10 matches)",
    "away_form_goal_diff":         "Away Recent Goal Diff (10 matches)",
    "points_diff":                 "FIFA Points Difference",
    "home_form_win_pct":           "Home Form Win Rate (10 matches)",
    "away_form_win_pct":           "Away Form Win Rate (10 matches)",
    "home_wc_appearances":         "Home World Cup Appearances",
    "away_wc_appearances":         "Away World Cup Appearances",
    "match_importance":            "Match Importance Score",
    "home_elo_rating":             "Home Elo Rating",
    "away_elo_rating":             "Away Elo Rating",
    "home_fifa_rank":              "Home FIFA Rank",
    "away_fifa_rank":              "Away FIFA Rank",
    "home_fifa_points":            "Home FIFA Points",
    "away_fifa_points":            "Away FIFA Points",
    "h2h_win_rate":                "Head-to-Head Win Rate",
    "h2h_goal_diff":               "Head-to-Head Goal Diff",
    "home_form_goals_scored":      "Home Form Goals Scored",
    "away_form_goals_scored":      "Away Form Goals Scored",
    "home_form_goals_conceded":    "Home Form Goals Conceded",
    "away_form_goals_conceded":    "Away Form Goals Conceded",
    "home_form_clean_sheet_pct":   "Home Clean Sheet Rate",
    "away_form_clean_sheet_pct":   "Away Clean Sheet Rate",
    "home_wc_titles":              "Home World Cup Titles",
    "away_wc_titles":              "Away World Cup Titles",
    "home_wc_best_finish":         "Home WC Best Finish",
    "away_wc_best_finish":         "Away WC Best Finish",
    "same_conf":                   "Same Confederation",
    "neutral":                     "Neutral Venue",
}


def main() -> None:
    print("Loading features and model...")
    features = pd.read_csv(INPUT_FEATURES, parse_dates=["date"])
    fifa     = pd.read_csv(INPUT_FIFA)

    with open(INPUT_MODEL, "rb") as f:
        model_payload = pickle.load(f)

    pipeline      = model_payload["pipeline"]
    label_encoder = model_payload["label_encoder"]
    print(f"  Model: {model_payload['model_name']}")

    # Filter to competitive matches only (same as training)
    competitive = features[features["match_importance"] > 1].copy()

    # Temporal test split
    test = competitive[competitive["date"].dt.year >= TRAIN_CUTOFF].copy()
    print(f"  Test set: {len(test):,} rows")

    X_test = test[FEATURE_COLS]
    y_true = label_encoder.transform(test["result"])
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    classes = list(label_encoder.classes_)
    print(f"  Classes: {classes}")

    # -----------------------------------------------------------------------
    # 1. Test predictions CSV
    # -----------------------------------------------------------------------
    pred_df = test[["date", "home_team", "away_team", "result", "home_confederation"]].copy()
    pred_df["predicted"] = label_encoder.inverse_transform(y_pred)
    pred_df["correct"]   = pred_df["result"] == pred_df["predicted"]
    for i, cls in enumerate(classes):
        pred_df[f"prob_{cls}"] = y_prob[:, i]
    pred_df.to_csv(EVAL_DIR / "test_predictions.csv", index=False)
    print(f"  Saved test_predictions.csv ({len(pred_df):,} rows)")

    # -----------------------------------------------------------------------
    # 2. Confusion matrix
    # -----------------------------------------------------------------------
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"actual_{c}" for c in classes],
        columns=[f"pred_{c}" for c in classes],
    )
    cm_df.to_csv(EVAL_DIR / "confusion_matrix.csv")
    print(f"  Saved confusion_matrix.csv")

    # -----------------------------------------------------------------------
    # 3. Calibration curves (one per class, one-vs-rest)
    # -----------------------------------------------------------------------
    cal_rows = []
    for i, cls in enumerate(classes):
        y_bin  = (y_true == i).astype(int)
        prob_c = y_prob[:, i]
        frac_pos, mean_pred = calibration_curve(y_bin, prob_c, n_bins=10, strategy="quantile")
        for fp, mp in zip(frac_pos, mean_pred):
            cal_rows.append({"class": cls, "mean_predicted_prob": mp, "fraction_positive": fp})
    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(EVAL_DIR / "calibration.csv", index=False)
    print(f"  Saved calibration.csv")

    # -----------------------------------------------------------------------
    # 4. ROC curves (one-vs-rest)
    # -----------------------------------------------------------------------
    roc_rows = []
    for i, cls in enumerate(classes):
        y_bin = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        for fp, tp in zip(fpr, tpr):
            roc_rows.append({"class": cls, "fpr": fp, "tpr": tp, "auc": round(roc_auc, 4)})
    roc_df = pd.DataFrame(roc_rows)
    roc_df.to_csv(EVAL_DIR / "roc_curves.csv", index=False)
    print(f"  Saved roc_curves.csv")

    # -----------------------------------------------------------------------
    # 5. Feature importance
    # -----------------------------------------------------------------------
    xgb_model = pipeline.named_steps["model"]
    importances = xgb_model.feature_importances_
    fi_df = pd.DataFrame({
        "feature":      FEATURE_COLS,
        "display_name": [FEATURE_DISPLAY_NAMES.get(f, f) for f in FEATURE_COLS],
        "importance":   importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    fi_df.to_csv(EVAL_DIR / "feature_importance.csv", index=False)
    print(f"  Saved feature_importance.csv")
    print(f"  Top 5 features:")
    for _, row in fi_df.head(5).iterrows():
        print(f"    {row['display_name']:45} {row['importance']:.4f}")

    # -----------------------------------------------------------------------
    # 6. Temporal accuracy (by year)
    # -----------------------------------------------------------------------
    pred_df["year"] = pred_df["date"].dt.year
    temporal = pred_df.groupby("year")["correct"].agg(
        accuracy="mean", n_matches="count"
    ).reset_index()
    temporal["accuracy"] = temporal["accuracy"].round(4)
    temporal.to_csv(EVAL_DIR / "temporal_accuracy.csv", index=False)
    print(f"  Saved temporal_accuracy.csv")

    # -----------------------------------------------------------------------
    # 7. Confederation accuracy
    # -----------------------------------------------------------------------
    conf_acc = pred_df.groupby("home_confederation")["correct"].agg(
        accuracy="mean", n_matches="count"
    ).reset_index().rename(columns={"home_confederation": "confederation"})
    conf_acc["accuracy"] = conf_acc["accuracy"].round(4)
    conf_acc = conf_acc.sort_values("accuracy", ascending=False)
    conf_acc.to_csv(EVAL_DIR / "confederation_accuracy.csv", index=False)
    print(f"  Saved confederation_accuracy.csv")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    overall_acc = pred_df["correct"].mean()
    print(f"\nOverall test accuracy: {overall_acc:.4f}")
    print(f"\nAll evaluation artifacts saved to {EVAL_DIR}/")


if __name__ == "__main__":
    main()
