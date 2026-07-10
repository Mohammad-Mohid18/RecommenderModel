"""Train the investor-startup recommender from Firestore data.

This script expects:
- serviceAccounts.json
- Firestore collections: startup_profiles and investor_profiles

If no curated recommended_investors.csv exists, it creates one with
rule-based positive labels from investor preferences to startup profiles.
"""

from __future__ import annotations

import json
from pathlib import Path

import firebase_admin
import joblib
import numpy as np
import pandas as pd
from firebase_admin import credentials, firestore
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from recommender import (
    PairFeatureBuilder,
    RAW_FEATURES,
    make_rule_based_matches,
    normalize_investors,
    normalize_startups,
    recommend,
)


DATA_DIR = Path(".")
MODEL_PATH = DATA_DIR / "startup_investor_pipeline.pkl"
METRICS_PATH = DATA_DIR / "model_metrics.json"
RECOMMENDED_PATH = DATA_DIR / "recommended_investors.csv"
STARTUP_SNAPSHOT_PATH = DATA_DIR / "firebase_startup_profiles.csv"
INVESTOR_SNAPSHOT_PATH = DATA_DIR / "firebase_investor_profiles.csv"


def init_firestore():
    if not firebase_admin._apps:
        firebase_admin.initialize_app(credentials.Certificate("serviceAccounts.json"))
    return firestore.client()


def collection_to_frame(db, collection_name: str, id_column: str) -> pd.DataFrame:
    rows = []
    for doc in db.collection(collection_name).stream(timeout=60):
        row = doc.to_dict() or {}
        row[id_column] = doc.id
        rows.append(row)
    if not rows:
        raise RuntimeError(f"Firestore collection '{collection_name}' is empty.")
    return pd.DataFrame(rows)


def build_pair_dataset(
    matches_df: pd.DataFrame,
    startups_df: pd.DataFrame,
    investors_df: pd.DataFrame,
    neg_ratio: float = 1.0,
    seed: int = 42,
    max_negatives_per_investor: int | None = 10,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    all_startups = startups_df["startup_id"].dropna().unique()
    negative_rows: list[dict] = []

    for investor_id, group in matches_df.groupby("investor_id"):
        positive_startups = set(group["startup_id"].dropna().tolist())
        candidates = np.setdiff1d(all_startups, list(positive_startups))
        n_pos = len(group)
        n_neg = int(np.ceil(n_pos * neg_ratio))
        if max_negatives_per_investor is not None:
            n_neg = min(n_neg, max_negatives_per_investor)
        n_neg = min(n_neg, len(candidates))
        if n_neg <= 0:
            continue

        for startup_id in rng.choice(candidates, size=n_neg, replace=False):
            negative_rows.append(
                {"startup_id": startup_id, "investor_id": investor_id, "label": 0}
            )

    negatives = pd.DataFrame(negative_rows)
    pairs = pd.concat(
        [matches_df[["startup_id", "investor_id", "label"]], negatives],
        ignore_index=True,
    )
    dataset = (
        pairs.merge(startups_df, on="startup_id", how="left")
        .merge(investors_df, on="investor_id", how="left")
    )
    return dataset.dropna(subset=["startup_id", "investor_id", "label"])


def build_pipeline() -> Pipeline:
    categorical_features = [
        "startup_category",
        "startup_budget_required",
        "startup_stage",
        "startup_location",
        "startup_risk_level",
        "startup_traction_level",
        "investor_category",
        "investor_budget_range",
        "investor_preferred_stage",
        "investor_location",
        "investor_risk_preference",
        "investor_traction_preference",
        "category_match_flag",
        "location_match_flag",
        "stage_match_flag",
        "budget_close_flag",
    ]
    numeric_features = ["budget_rank_diff"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("feature_builder", PairFeatureBuilder()),
            ("preprocess", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    db = init_firestore()
    startups_raw = collection_to_frame(db, "startup_profiles", "startup_id")
    investors_raw = collection_to_frame(db, "investor_profiles", "investor_id")

    startups_raw.to_csv(STARTUP_SNAPSHOT_PATH, index=False)
    investors_raw.to_csv(INVESTOR_SNAPSHOT_PATH, index=False)

    startups = normalize_startups(startups_raw)
    investors = normalize_investors(investors_raw)

    if RECOMMENDED_PATH.exists():
        matches = pd.read_csv(RECOMMENDED_PATH)
        if "label" not in matches.columns:
            matches["label"] = 1
        is_rule_based = (
            "source" in matches.columns
            and matches["source"].astype(str).str.contains("rule_based", case=False).any()
        )
        if is_rule_based:
            matches = make_rule_based_matches(startups_raw, investors_raw, top_k=5)
            matches["source"] = "rule_based_investor_preferences_from_firestore"
            matches.to_csv(RECOMMENDED_PATH, index=False)
            label_source = "rule_based_investor_preferences_from_firestore"
        else:
            label_source = "recommended_investors.csv"
    else:
        matches = make_rule_based_matches(startups_raw, investors_raw, top_k=5)
        matches["source"] = "rule_based_investor_preferences_from_firestore"
        matches.to_csv(RECOMMENDED_PATH, index=False)
        label_source = "rule_based_investor_preferences_from_firestore"

    pair_dataset = build_pair_dataset(
        matches,
        startups,
        investors,
        neg_ratio=1.0,
        max_negatives_per_investor=10,
    )

    X = pair_dataset[RAW_FEATURES].fillna("")
    y = pair_dataset["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    sample_investor = investors_raw.iloc[0].to_dict()
    sample_recommendations = recommend(sample_investor, startups_raw, pipeline, top_n=40)

    metrics = {
        "data_source": "firestore",
        "recommendation_direction": "investor_to_startups",
        "label_source": label_source,
        "startup_profiles": int(len(startups_raw)),
        "investor_profiles": int(len(investors_raw)),
        "positive_pairs": int((pair_dataset["label"] == 1).sum()),
        "negative_pairs": int((pair_dataset["label"] == 0).sum()),
        "total_pairs": int(len(pair_dataset)),
        "test_size": int(len(y_test)),
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "average_precision": float(average_precision_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(
            y_test,
            preds,
            output_dict=True,
            zero_division=0,
        ),
        "score_stats_test": {
            "min": float(np.min(proba)),
            "max": float(np.max(proba)),
            "mean": float(np.mean(proba)),
            "median": float(np.median(proba)),
        },
        "sample_investor_id": str(sample_investor.get("investor_id", "")),
        "sample_recommendation_scores": [
            {
                "startup_id": str(row.get("startup_id", "")),
                "match_score": float(row.get("match_score", 0.0)),
            }
            for row in sample_recommendations[["startup_id", "match_score"]].to_dict("records")
        ],
    }

    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved Firestore snapshots: {STARTUP_SNAPSHOT_PATH}, {INVESTOR_SNAPSHOT_PATH}")
    print(f"Recommended pairs file: {RECOMMENDED_PATH}")


if __name__ == "__main__":
    main()
