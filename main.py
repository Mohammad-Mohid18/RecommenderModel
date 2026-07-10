"""
main.py
FastAPI backend for the Startup-Investor ML Recommendation System.
Run with: uvicorn main:app --reload
"""

# ─── Standard library ────────────────────────────────────────────────────────
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# ─── Third-party ─────────────────────────────────────────────────────────────
import joblib
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
import os

SERVICE_ACCOUNT_PATH = os.environ.get("FIREBASE_CRED_PATH", "serviceAccounts.json")
FIRESTORE_COLLECTION = "startup_profiles"
LOCAL_STARTUPS_PATH  = Path(os.getenv("LOCAL_STARTUPS_PATH", "firebase_startup_profiles.csv"))
USE_FIRESTORE        = os.getenv("USE_FIRESTORE", "0").strip().lower() in {"1", "true", "yes"}


# ════════════════════════════════════════════════════════════════════════════
# GLOBALS  (loaded once at startup, reused across every request)
# ════════════════════════════════════════════════════════════════════════════
model    = None   # scikit-learn pipeline
db       = None   # Firestore client


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
    global model, db

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

    if not USE_FIRESTORE:
        db = None
        logger.info(
            "Using local startups file '%s'. Set USE_FIRESTORE=1 to use Firestore.",
            LOCAL_STARTUPS_PATH,
        )
        yield
        logger.info("Shutting down - releasing resources.")
        return

    # ── Initialise Firebase (idempotent guard) ──────────────────────────────
    try:
        if not firebase_admin._apps:
            cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
            firebase_admin.initialize_app(cred)
            logger.info("✅ Firebase initialised.")
        else:
            logger.info("ℹ️  Firebase already initialised — skipping.")

        db = firestore.client()
        logger.info("✅ Firestore client ready.")
    except FileNotFoundError:
        logger.error("❌ Service account file '%s' not found.", SERVICE_ACCOUNT_PATH)
        raise RuntimeError(f"Service account file '{SERVICE_ACCOUNT_PATH}' not found.")
    except Exception as exc:
        logger.error("❌ Firebase init failed: %s", exc)
        raise RuntimeError(f"Firebase init error: {exc}") from exc

    yield  # ← server is live and handling requests here

    # ── Shutdown cleanup (optional) ─────────────────────────────────────────
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
    Pull every document from the 'startup_profiles' Firestore collection
    and return a tidy DataFrame.

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

    if db is None:
        raise HTTPException(
            status_code=503,
            detail="Firestore client is not initialised."
        )

    try:
        docs = db.collection(FIRESTORE_COLLECTION).stream()
        rows = []
        for doc in docs:
            data = doc.to_dict()
            data["startup_id"] = doc.id   # preserve Firestore document ID
            rows.append(data)
    except Exception as exc:
        logger.error("Firestore read failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail=f"Could not fetch startups from Firestore: {exc}"
        )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{FIRESTORE_COLLECTION}' is empty or does not exist."
        )

    startups_df = pd.DataFrame(rows)
    logger.info("📥 Fetched %d startup profiles from Firestore.", len(startups_df))
    return startups_df


# ════════════════════════════════════════════════════════════════════════════
# CORE RECOMMENDATION LOGIC
# ════════════════════════════════════════════════════════════════════════════
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
        "recommendation_direction": "investor_to_startups",
        "recommendation_limit":     RECOMMENDATION_TOP_N,
        "model_loaded":             model is not None,
        "data_source":              "firestore" if USE_FIRESTORE else "local_csv",
        "firestore_ready":          db is not None,
    }
