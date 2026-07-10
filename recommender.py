"""Shared startup-investor recommendation utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


BUDGET_ORDER = ["10k-100k", "50k-200k", "100k-500k", "500k-2M", "2M-10M"]
BUDGET_RANK = {label: idx for idx, label in enumerate(BUDGET_ORDER)}
RECOMMENDATION_TOP_N = 40

BUDGET_ALIASES = {
    "500k-1m": "500k-2M",
    "1m-2m": "500k-2M",
    "1m-5m": "2M-10M",
}

STAGE_ALIASES = {
    "pre-seed": "Idea",
    "pre seed": "Idea",
    "seed": "MVP",
    "series a": "Growth",
    "series b": "Scaling",
    "series c": "Scaling",
}

LOCATION_ALIASES = {
    "karachi": "Pakistan",
    "lahore": "Pakistan",
    "islamabad": "Pakistan",
    "riyadh": "Saudi Arabia",
    "jeddah": "Saudi Arabia",
    "dubai": "UAE",
    "abu dhabi": "UAE",
    "london": "UK",
    "new york": "USA",
    "san francisco": "USA",
}

TRACTION_ALIASES = {
    "high": "Revenue",
    "medium": "Users",
    "low": "",
    "none": "",
    "nan": "",
}

RAW_FEATURES = [
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
]

STARTUP_RENAME = {
    "category": "startup_category",
    "budget_required": "startup_budget_required",
    "budget_range": "startup_budget_required",
    "budget": "startup_budget_required",
    "status": "startup_stage",
    "stage": "startup_stage",
    "location": "startup_location",
    "risk_level": "startup_risk_level",
    "traction_level": "startup_traction_level",
}

INVESTOR_RENAME = {
    "category": "investor_category",
    "budget_range": "investor_budget_range",
    "budget": "investor_budget_range",
    "preferred_stage": "investor_preferred_stage",
    "stage": "investor_preferred_stage",
    "location": "investor_location",
    "risk_preference": "investor_risk_preference",
    "traction_preference": "investor_traction_preference",
}


def _split_stages(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {stage.strip() for stage in str(value).split(",") if stage.strip()}


def _budget_rank(value: object) -> float:
    return float(BUDGET_RANK.get(value, np.nan))


def _canonicalize(value: object, aliases: dict[str, str]) -> object:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return aliases.get(text.lower(), text)


def canonicalize_startup_features(startup: dict) -> dict:
    startup = startup.copy()
    startup["startup_budget_required"] = _canonicalize(
        startup.get("startup_budget_required", ""),
        BUDGET_ALIASES,
    )
    startup["startup_stage"] = _canonicalize(
        startup.get("startup_stage", ""),
        STAGE_ALIASES,
    )
    startup["startup_location"] = _canonicalize(
        startup.get("startup_location", ""),
        LOCATION_ALIASES,
    )
    startup["startup_traction_level"] = _canonicalize(
        startup.get("startup_traction_level", ""),
        TRACTION_ALIASES,
    )
    return startup


def canonicalize_investor_features(investor: dict) -> dict:
    investor = investor.copy()
    investor["investor_budget_range"] = _canonicalize(
        investor.get("investor_budget_range", ""),
        BUDGET_ALIASES,
    )
    investor["investor_location"] = _canonicalize(
        investor.get("investor_location", ""),
        LOCATION_ALIASES,
    )
    investor["investor_traction_preference"] = _canonicalize(
        investor.get("investor_traction_preference", ""),
        TRACTION_ALIASES,
    )
    return investor


class PairFeatureBuilder(BaseEstimator, TransformerMixin):
    """Add pairwise match features used by both training and inference."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["category_match_flag"] = (
            X["startup_category"].astype(str) == X["investor_category"].astype(str)
        ).astype(int)
        X["location_match_flag"] = (
            X["startup_location"].astype(str) == X["investor_location"].astype(str)
        ).astype(int)

        X["stage_match_flag"] = X.apply(
            lambda row: int(str(row["startup_stage"]) in _split_stages(row["investor_preferred_stage"])),
            axis=1,
        )

        startup_rank = X["startup_budget_required"].map(BUDGET_RANK)
        investor_rank = X["investor_budget_range"].map(BUDGET_RANK)
        rank_diff = (startup_rank - investor_rank).abs()
        X["budget_rank_diff"] = rank_diff.fillna(99).astype(float)
        X["budget_close_flag"] = (rank_diff <= 1).fillna(False).astype(int)
        return X


def normalize_startups(startups_df: pd.DataFrame) -> pd.DataFrame:
    startups = startups_df.rename(columns=STARTUP_RENAME)
    for feature in ["startup_risk_level", "startup_traction_level"]:
        if feature not in startups.columns:
            startups[feature] = ""
    return startups


def normalize_investors(investors_df: pd.DataFrame) -> pd.DataFrame:
    investors = investors_df.rename(columns=INVESTOR_RENAME)
    for feature in ["investor_risk_preference", "investor_traction_preference"]:
        if feature not in investors.columns:
            investors[feature] = ""
    return investors


def make_rule_based_matches(
    startups_df: pd.DataFrame,
    investors_df: pd.DataFrame,
    top_k: int = 3,
) -> pd.DataFrame:
    """Create positive training pairs: top matching startups per investor."""
    rows: list[dict[str, object]] = []
    startups_norm = normalize_startups(startups_df)

    for _, investor in normalize_investors(investors_df).iterrows():
        investor_budget_rank = _budget_rank(investor.get("investor_budget_range"))
        scored: list[tuple[float, object]] = []

        for _, startup in startups_norm.iterrows():
            startup_budget_rank = _budget_rank(startup.get("startup_budget_required"))
            budget_diff = abs(startup_budget_rank - investor_budget_rank)
            if np.isnan(budget_diff):
                budget_diff = 99

            score = 0.0
            score += 3.0 * (startup.get("startup_category") == investor.get("investor_category"))
            score += 2.0 * (
                startup.get("startup_stage") in _split_stages(investor.get("investor_preferred_stage"))
            )
            score += 1.5 * (budget_diff == 0)
            score += 0.75 * (budget_diff == 1)
            score += 1.0 * (startup.get("startup_location") == investor.get("investor_location"))
            score += 0.5 * (startup.get("startup_risk_level") == investor.get("investor_risk_preference"))
            score += 0.5 * (
                pd.notna(startup.get("startup_traction_level"))
                and startup.get("startup_traction_level") == investor.get("investor_traction_preference")
            )
            scored.append((score, startup["startup_id"]))

        scored.sort(key=lambda item: (-item[0], str(item[1])))
        for _, startup_id in scored[:top_k]:
            rows.append(
                {
                    "startup_id": startup_id,
                    "investor_id": investor["investor_id"],
                    "label": 1,
                }
            )

    return pd.DataFrame(rows)


def normalize_startup_input(startup: dict) -> dict:
    return canonicalize_startup_features({
        "startup_category": startup.get("startup_category", startup.get("category", "")),
        "startup_budget_required": startup.get(
            "startup_budget_required",
            startup.get("budget_required", startup.get("budget_range", startup.get("budget", ""))),
        ),
        "startup_stage": startup.get("startup_stage", startup.get("stage", "")),
        "startup_location": startup.get("startup_location", startup.get("location", "")),
        "startup_risk_level": startup.get("startup_risk_level", startup.get("risk_level", "")),
        "startup_traction_level": startup.get(
            "startup_traction_level",
            startup.get("traction_level", ""),
        ),
    })


def score_startup_against_investors(
    startup: dict,
    investors_df: pd.DataFrame,
    pipeline,
) -> pd.Series:
    startup_features = normalize_startup_input(startup)
    investors_norm = normalize_investors(investors_df.copy())

    for key, value in startup_features.items():
        investors_norm[key] = "" if value is None else value

    for feature in RAW_FEATURES:
        if feature not in investors_norm.columns:
            investors_norm[feature] = ""

    scoring_frame = investors_norm[RAW_FEATURES].fillna("")
    return pd.Series(pipeline.predict_proba(scoring_frame)[:, 1], index=investors_df.index)


def normalize_investor_input(investor: dict) -> dict:
    return canonicalize_investor_features({
        "investor_category": investor.get("investor_category", investor.get("category", "")),
        "investor_budget_range": investor.get(
            "investor_budget_range",
            investor.get("budget_range", investor.get("budget", "")),
        ),
        "investor_preferred_stage": investor.get(
            "investor_preferred_stage",
            investor.get("preferred_stage", investor.get("stage", "")),
        ),
        "investor_location": investor.get("investor_location", investor.get("location", "")),
        "investor_risk_preference": investor.get(
            "investor_risk_preference",
            investor.get("risk_preference", investor.get("risk_level", "")),
        ),
        "investor_traction_preference": investor.get(
            "investor_traction_preference",
            investor.get("traction_preference", investor.get("traction_level", "")),
        ),
    })


def score_investor_against_startups(
    investor: dict,
    startups_df: pd.DataFrame,
    pipeline,
) -> pd.Series:
    investor_features = normalize_investor_input(investor)
    startups_norm = normalize_startups(startups_df.copy())

    for key, value in investor_features.items():
        startups_norm[key] = "" if value is None else value

    for feature in RAW_FEATURES:
        if feature not in startups_norm.columns:
            startups_norm[feature] = ""

    scoring_frame = startups_norm[RAW_FEATURES].fillna("")
    return pd.Series(pipeline.predict_proba(scoring_frame)[:, 1], index=startups_df.index)


def recommend(
    investor: dict,
    startups_df: pd.DataFrame,
    pipeline,
    top_n: int = RECOMMENDATION_TOP_N,
    passthrough_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return top startups ranked by match score for one investor profile."""
    scores = score_investor_against_startups(investor, startups_df, pipeline)
    results = startups_df.copy()
    results["match_score"] = scores.round(4)

    if passthrough_columns is not None:
        columns = [column for column in passthrough_columns if column in results.columns]
        if "match_score" not in columns:
            columns.append("match_score")
        results = results[columns]

    return (
        results.sort_values("match_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def recommend_investors_for_startup(
    startup: dict,
    investors_df: pd.DataFrame,
    pipeline,
    top_n: int = RECOMMENDATION_TOP_N,
    passthrough_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Legacy helper: return top investors ranked for one startup profile."""
    scores = score_startup_against_investors(startup, investors_df, pipeline)
    results = investors_df.copy()
    results["match_score"] = scores.round(4)

    if passthrough_columns is not None:
        columns = [column for column in passthrough_columns if column in results.columns]
        if "match_score" not in columns:
            columns.append("match_score")
        results = results[columns]

    return (
        results.sort_values("match_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
