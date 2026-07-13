"""
main.py
FastAPI backend for the Startup-Investor ML Recommendation System.
Run with: uvicorn main:app --reload
"""

# ─── Standard library ────────────────────────────────────────────────────────
import asyncio
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Optional

# ─── Third-party ─────────────────────────────────────────────────────────────
import joblib
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from recommender import PairFeatureBuilder, RECOMMENDATION_TOP_N, recommend as rank_startups

# ─── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH           = "startup_investor_pipeline.pkl"
SERVICE_ACCOUNT_PATH = "serviceAccounts.json"
FIRESTORE_COLLECTION = "project"
LOCAL_STARTUPS_PATH  = Path(os.getenv("LOCAL_STARTUPS_PATH", "firebase_startup_profiles.csv"))
USE_FIRESTORE        = os.getenv("USE_FIRESTORE", "0").strip().lower() in {"1", "true", "yes"}

# Periodic Firestore → CSV sync. This runs in the background regardless of
# USE_FIRESTORE, so that even when the API serves requests from the local
# CSV (cheap, no per-request Firestore reads/quota risk), that CSV stays
# fresh on its own as new users are added in Firestore.
AUTO_REFRESH_CSV      = os.getenv("AUTO_REFRESH_CSV", "1").strip().lower() in {"1", "true", "yes"}
CSV_REFRESH_SECONDS   = int(os.getenv("CSV_REFRESH_SECONDS", "600"))  # 10 minutes


def resolve_service_account_path() -> str:
    """
    Return a filesystem path to the Firebase service-account JSON.

    - On Railway (or any host without a committed serviceAccounts.json),
      the JSON content is provided via the FIREBASE_CREDENTIALS_JSON
      env var. We write it out to a temp file once and return that path,
      since firebase_admin.credentials.Certificate() needs a file path.
    - Locally, if that env var isn't set, we fall back to the
      serviceAccounts.json file sitting next to this script.
    """
    creds_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
    if creds_json:
        creds_dict = json.loads(creds_json)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(creds_dict, tmp)
        tmp.close()
        return tmp.name
    return SERVICE_ACCOUNT_PATH


# ════════════════════════════════════════════════════════════════════════════
# GLOBALS  (loaded once at startup, reused across every request)
# ════════════════════════════════════════════════════════════════════════════
model    = None   # scikit-learn pipeline
db       = None   # Firestore client
_refresh_task = None   # background asyncio task handle


# ════════════════════════════════════════════════════════════════════════════
# BACKGROUND JOB — keep the local CSV in sync with Firestore
# ════════════════════════════════════════════════════════════════════════════
def map_project_to_startup_features(project_data: Optional[dict], startup_id: Optional[str] = None) -> dict:
    """Map only the project fields required by the existing recommendation model."""
    data = project_data or {}
    return {
        "startup_id": startup_id or data.get("startup_id", ""),
        "startup_category": data.get("category", ""),
        "startup_budget_required": data.get("budget_range", ""),
        "startup_stage": data.get("projectStage", ""),
        "startup_location": data.get("location", ""),
        "startup_risk_level": data.get("risk_level", ""),
        "startup_traction_level": data.get("traction_level", ""),
    }


def fetch_project_startup_rows() -> list[dict]:
    """Fetch documents from Firestore's projects collection and map them to model features."""
    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Firestore client is not initialised."
        )

    try:
        docs = db.collection(FIRESTORE_COLLECTION).stream()
        rows = []
        for doc in docs:
            data = doc.to_dict() or {}
            rows.append(map_project_to_startup_features(data, doc.id))
        return rows
    except Exception as exc:
        logger.error("Firestore read failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Could not fetch startups from Firestore: {exc}"
        )


def refresh_startups_csv_once() -> None:
    """
    Pull every document from Firestore's projects collection and overwrite the
    local CSV with the mapped startup feature rows. Blocking (uses the sync
    Firestore SDK), so it's always called via asyncio.to_thread from the async
    loop below.
    """
    if db is None:
        logger.warning("⏭️  Skipping CSV refresh — Firestore client not initialised.")
        return

    try:
        rows = fetch_project_startup_rows()
    except HTTPException as exc:
        logger.warning("⏭️  Skipping CSV refresh — %s", exc.detail)
        return

    if not rows:
        logger.warning("⏭️  Firestore returned 0 startup docs — leaving existing CSV untouched.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(LOCAL_STARTUPS_PATH, index=False)
    logger.info(
        "🔄 Refreshed '%s' from Firestore projects — %d startup rows.",
        LOCAL_STARTUPS_PATH, len(df),
    )


async def periodic_csv_refresh_loop():
    """
    Runs forever in the background: every CSV_REFRESH_SECONDS, re-pull
    Firestore and overwrite the local CSV. Any single failure (quota,
    network, etc.) is logged and skipped — it never crashes the server
    or blocks request handling, and just retries on the next tick.
    """
    while True:
        await asyncio.sleep(CSV_REFRESH_SECONDS)
        try:
            await asyncio.to_thread(refresh_startups_csv_once)
        except Exception as exc:
            logger.error("❌ Background CSV refresh failed (will retry next tick): %s", exc)


# ════════════════════════════════════════════════════════════════════════════
# LIFESPAN  (replaces deprecated @app.on_event)
# ════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs once when the server starts (before first request) and once
    when it shuts down.  Heavy initialisation lives here so it is never
    repeated per-request.
    """
    global model, db, _refresh_task

    # ── Load ML model ───────────────────────────────────────────────────────
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("✅ Model loaded from '%s'", MODEL_PATH)
    except FileNotFoundError:
        logger.error("❌ Model file '%s' not found. Exiting.", MODEL_PATH)
        raise RuntimeError(f"Model file '{MODEL_PATH}' not found.")
    except Exception as exc:
        logger.error("❌ Failed to load model: %s", exc)
        raise RuntimeError(f"Model load error: {exc}") from exc

    # Firebase needs to be initialised if EITHER the API serves requests
    # straight from Firestore (USE_FIRESTORE=1) OR the background CSV
    # refresh job needs it (AUTO_REFRESH_CSV=1), even while requests are
    # served from the cheaper local CSV.
    if USE_FIRESTORE or AUTO_REFRESH_CSV:
        try:
            if not firebase_admin._apps:
                resolved_path = resolve_service_account_path()
                cred = credentials.Certificate(resolved_path)
                firebase_admin.initialize_app(cred)
                logger.info("✅ Firebase initialised.")
            else:
                logger.info("ℹ️  Firebase already initialised — skipping.")

            db = firestore.client()
            logger.info("✅ Firestore client ready.")
        except FileNotFoundError:
            logger.error(
                "❌ No Firebase credentials found. Set FIREBASE_CREDENTIALS_JSON "
                "(deployed) or add serviceAccounts.json (local)."
            )
            if USE_FIRESTORE:
                # Serving directly from Firestore with no credentials is fatal.
                raise RuntimeError(
                    "Firebase credentials not found: set FIREBASE_CREDENTIALS_JSON "
                    "env var or provide serviceAccounts.json locally."
                )
            # Otherwise just disable the background refresh and keep serving
            # from whatever CSV is already on disk.
            db = None
            logger.warning("⚠️  Auto-refresh disabled for this run — no Firebase credentials.")
        except Exception as exc:
            logger.error("❌ Firebase init failed: %s", exc)
            if USE_FIRESTORE:
                raise RuntimeError(f"Firebase init error: {exc}") from exc
            db = None
    else:
        db = None
        logger.info(
            "Using local startups file '%s'. Set USE_FIRESTORE=1 to serve from "
            "Firestore directly, or AUTO_REFRESH_CSV=1 to keep this CSV synced.",
            LOCAL_STARTUPS_PATH,
        )

    # ── Start the background CSV-refresh loop ───────────────────────────────
    if AUTO_REFRESH_CSV and db is not None:
        _refresh_task = asyncio.create_task(periodic_csv_refresh_loop())
        logger.info(
            "🔁 Background CSV refresh enabled — every %d seconds.",
            CSV_REFRESH_SECONDS,
        )

    yield  # ← server is live and handling requests here

    # ── Shutdown cleanup ─────────────────────────────────────────────────────
    if _refresh_task is not None:
        _refresh_task.cancel()
    logger.info("🔒 Shutting down — releasing resources.")


# ════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ════════════════════════════════════════════════════════════════════════════
app = FastAPI(
    title="Investor–Startup Recommendation API",
    description=(
        "Scores every startup in Firestore against an incoming investor profile "
        f"and returns up to {RECOMMENDATION_TOP_N} ranked startup matches."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/", summary="API index")
async def root():
    """Return basic API navigation for browser checks."""
    return {
        "status": "ok",
        "message": "Investor-Startup Recommendation API is running.",
        "api_version": "2.0.0",
        "recommendation_direction": "investor_to_startups",
        "recommendation_limit": RECOMMENDATION_TOP_N,
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "recommend": "/recommend",
        },
    }

# ════════════════════════════════════════════════════════════════════════════
# PYDANTIC REQUEST SCHEMA
# ════════════════════════════════════════════════════════════════════════════
class InvestorProfile(BaseModel):
    """
    Validated request body for the /recommend endpoint.
    Optional fields fall back to empty string so the model pipeline
    can apply its own imputation/encoding logic.
    """
    investor_category:        str   = Field(..., example="FinTech")
    investor_budget_range:    str   = Field(..., example="500k-2M")
    investor_preferred_stage: str   = Field(..., example="Growth")
    investor_location:        str   = Field(..., example="Pakistan")
    investor_risk_preference:     Optional[str] = Field(None, example="Medium")
    investor_traction_preference: Optional[str] = Field(None, example="Revenue")

    @model_validator(mode="before")
    @classmethod
    def accept_raw_firestore_fields(cls, data):
        if not isinstance(data, dict):
            return data
        data = data.copy()
        aliases = {
            "investor_category": ["category"],
            "investor_budget_range": ["budget_range", "budget"],
            "investor_preferred_stage": ["preferred_stage", "stage"],
            "investor_location": ["location"],
            "investor_risk_preference": ["risk_preference", "risk_level"],
            "investor_traction_preference": ["traction_preference", "traction_level"],
        }
        for canonical, fallback_keys in aliases.items():
            if data.get(canonical) in (None, ""):
                for key in fallback_keys:
                    if data.get(key) not in (None, ""):
                        data[canonical] = data[key]
                        break
        return data

    class Config:
        # Allow extra fields to pass through without raising a validation error
        extra = "ignore"


# ════════════════════════════════════════════════════════════════════════════
# HELPER — FETCH STARTUPS FROM FIRESTORE
# ════════════════════════════════════════════════════════════════════════════
def fetch_startups() -> pd.DataFrame:
    """
    Pull every project document from Firestore and return a tidy DataFrame
    containing only the fields expected by the existing recommendation model.

    Each Firestore document ID is stored in a 'startup_id' column so
    downstream code can reference it without needing the raw doc reference.

    Raises
    ------
    HTTPException 503  if Firestore is unreachable or returns no data.
    """
    if not USE_FIRESTORE:
        try:
            local_path = LOCAL_STARTUPS_PATH
            if not local_path.exists() and local_path.name != "startup2.csv":
                local_path = Path("startup2.csv")
            startups_df = pd.read_csv(local_path)
            logger.info("Loaded %d startup profiles from '%s'.", len(startups_df), local_path)
            return startups_df
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail=f"Local startups file '{LOCAL_STARTUPS_PATH}' was not found."
            )
        except Exception as exc:
            logger.error("Local startup CSV read failed: %s", exc)
            raise HTTPException(
                status_code=503,
                detail=f"Could not read local startups file: {exc}"
            )

    try:
        rows = fetch_project_startup_rows()
    except HTTPException:
        raise

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{FIRESTORE_COLLECTION}' is empty or does not exist."
        )

    startups_df = pd.DataFrame(rows)
    logger.info("📥 Fetched %d startup rows from Firestore projects.", len(startups_df))
    return startups_df


# ════════════════════════════════════════════════════════════════════════════
# CORE RECOMMENDATION LOGIC
# ════════════════════════════════════════════════════════════════════════════
def make_json_safe(value):
    """
    Recursively convert values that json.dumps can't handle on its own —
    pandas.Timestamp, Firestore's DatetimeWithNanoseconds, datetime.date,
    numpy scalars, etc. — into plain JSON-friendly types.
    """
    if isinstance(value, dict):
        return {key: make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    # Covers pandas.Timestamp, datetime.datetime, Firestore's
    # DatetimeWithNanoseconds (all are datetime.datetime subclasses),
    # and datetime.date.
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if pd.isna(value) if not isinstance(value, (list, dict)) else False:
        return None
    return value


def recommend(
    investor: dict,
    startups: pd.DataFrame,
    pipeline,
    top_n: int = 50,
) -> pd.DataFrame:
    try:
        return rank_startups(investor, startups, pipeline, top_n=top_n)
    except Exception as exc:
        logger.error("Model prediction failed: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Model prediction error: {exc}"
        )


# ════════════════════════════════════════════════════════════════════════════
# ENDPOINT — POST /recommend
# ════════════════════════════════════════════════════════════════════════════
@app.post(
    "/recommend",
    summary=f"Get top {RECOMMENDATION_TOP_N} startup matches for an investor",
    response_description=f"List of up to {RECOMMENDATION_TOP_N} startups ranked by match score",
)
async def recommend_endpoint(investor_profile: InvestorProfile):
    """
    Accepts an investor profile and returns up to **40** matching startups
    from Firestore, ranked by the ML model's predicted match probability.

    **Example request body:**
```json
    {
        "investor_category":        "FinTech",
        "investor_budget_range":    "500k-2M",
        "investor_preferred_stage": "Growth,Scaling",
        "investor_location":        "Pakistan",
        "investor_risk_preference":     "Medium",
        "investor_traction_preference": "Revenue"
    }
```
    """
    # ── Guard: model must be loaded ─────────────────────────────────────────
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="ML model is not loaded. Check server startup logs."
        )

    # ── Convert Pydantic model → plain dict ─────────────────────────────────
    investor_dict = investor_profile.model_dump()

    # Replace None optionals with empty string
    for key, value in investor_dict.items():
        if value is None:
            investor_dict[key] = ""

    logger.info("🔍 Recommendation request: %s", investor_dict)

    # ── Fetch fresh startup data from Firestore ──────────────────────────────
    # NOTE: For high-traffic APIs, cache this DataFrame and refresh periodically
    #       (e.g. every 5 minutes) instead of fetching on every request.
    startups_df = fetch_startups()

    # ── Run recommendation logic ─────────────────────────────────────────────
    top_startups = recommend(
        investor   = investor_dict,
        startups   = startups_df,
        pipeline   = model,
        top_n      = RECOMMENDATION_TOP_N,
    )

    # ── Serialise and return ─────────────────────────────────────────────────
    output = (
        top_startups.astype(object)
        .where(pd.notna(top_startups), None)
        .to_dict(orient="records")
    )
    output = make_json_safe(output)
    logger.info("✅ Returning %d recommendations.", len(output))

    return JSONResponse(content={"recommendations": output})


# ════════════════════════════════════════════════════════════════════════════
# HEALTH-CHECK ENDPOINT
# ════════════════════════════════════════════════════════════════════════════
@app.get("/health", summary="Health check")
async def health():
    """Returns the live status of the model, data source, and API configuration."""
    return {
        "status":                   "ok",
        "api_version":              "2.0.0",
        "recommendation_direction": "startups_to_investor",
        "recommendation_limit":     RECOMMENDATION_TOP_N,
        "model_loaded":             model is not None,
        "data_source":              "firestore" if USE_FIRESTORE else "local_csv",
        "firestore_ready":          db is not None,
        "auto_refresh_enabled":     AUTO_REFRESH_CSV and db is not None,
        "auto_refresh_interval_seconds": CSV_REFRESH_SECONDS if AUTO_REFRESH_CSV else None,
    }