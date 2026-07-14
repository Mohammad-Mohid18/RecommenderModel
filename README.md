# Investor-Startup Recommendation API

A machine-learning recommendation system that ranks startup projects for an
investor profile based on compatibility. Built with scikit-learn, Firebase
Firestore, and FastAPI.

---

## Current Status

### ✅ Working

- `POST /recommend` returns up to **40** ranked startups with realistic,
  non-zero `match_score` values for any investor profile.
- All project details (name, description, photo, images, location, etc.) are
  preserved in the API response and local CSV.
- Data is sourced directly from the Firestore **`project`** collection — no
  manually maintained startup CSV required.
- A local snapshot `firebase_project_profiles.csv` is written on startup and
  refreshed automatically every 10 minutes in the background.
- The model **auto-reloads** when the pipeline file is updated, so retraining
  immediately affects live recommendations without a server restart.
- `GET /health` reports full server state: model status, data source, CSV row
  count, Firestore connection, and refresh interval.
- `GET /` returns a small API index for browser checks.
- The model trains from Firestore `project` and `investor_profiles` collections.
- Evaluation metrics are saved in `model_metrics.json`.
- **Pakistan city locations** are fully supported and normalised — see
  [Supported Values](#supported-values).

### Recent Fixes

| Fix | Detail |
|-----|--------|
| Zero match score | Fixed `_coalesce_columns` bug in `recommender.py` that overwrote pre-mapped canonical feature columns with empty strings |
| CSV data loss | Local CSV now written with `QUOTE_ALL` quoting; read with matching `quoting` parameter |
| Auto model reload | API detects pipeline file mtime on every request and reloads without restart |
| Collection name | Aligned to `project` (singular) to match Firestore |
| Pakistan cities | `peshawar`, `islamabad`, `lahore`, `karachi`, `faisalabad` (and more) now map correctly in `LOCATION_ALIASES` |
| Immediate startup sync | CSV is refreshed from Firestore before the first request, not just after the first 10-minute tick |

### Important Modelling Note

No human-curated recommendation labels exist in Firestore. `train_model.py`
generates `recommended_investors.csv` using transparent rule-based
compatibility logic (investor preferences → startup profiles). The model
pipeline and API are fully functional, but business-grade accuracy should later
be improved with real historical or curated match labels.

---

## Project Files

### Core

| File | Purpose |
|------|---------|
| `main.py` | FastAPI backend (API v3.0.0, investor → startups) |
| `recommender.py` | Shared utilities: feature engineering, schema mapping, canonicalisation, recommendation logic |
| `train_model.py` | Trains model from Firestore data and saves metrics |
| `3-cosine-based-system-6f-v2.1.ipynb` | Notebook version of the training flow |
| `MOBILE_INTEGRATION.md` | React Native integration guide |
| `requirements.txt` | Python dependencies |
| `serviceAccounts.json` | Firebase service account credentials (**never commit publicly**) |

### Generated / Model Files

| File | Purpose |
|------|---------|
| `startup_investor_pipeline.pkl` | Trained scikit-learn pipeline used by the API |
| `model_metrics.json` | Latest evaluation metrics |
| `firebase_project_profiles.csv` | Live snapshot of Firestore `project` collection — updated every 10 minutes |
| `firebase_investor_profiles.csv` | Snapshot of Firestore `investor_profiles` |
| `recommended_investors.csv` | Generated positive training pairs (rule-based) |

### Legacy / Fallback CSVs

| File | Purpose |
|------|---------|
| `startup2.csv` | Legacy startup data — used as last-resort fallback only |
| `investor2.csv` | Legacy investor data — used as last-resort fallback only |

The preferred training path is Firestore snapshots. These files are kept for
local testing without Firebase access.

---

## Architecture

The system works as a pairwise binary classifier.

For each investor–startup pair the model predicts the probability of a good
match. The API scores one investor against all projects in Firestore and returns
the top 40 results.

```
Firestore 'project' collection
        │
        ▼ (on startup + every 10 min)
firebase_project_profiles.csv  ◄──── background refresh loop
        │
        ▼ (per /recommend request)
normalize_startups()
        │
        ▼
score_investor_against_startups()
        │   PairFeatureBuilder → OneHotEncoder → StandardScaler
        │   LogisticRegression.predict_proba()
        ▼
top 40 results sorted by match_score DESC
```

---

## Data Sources

### Firestore Collections

| Collection | Role |
|------------|------|
| `project` | Live startup/project listings — primary source |
| `investor_profiles` | Investor preference profiles used for training |

### Checked but Empty / Missing

- `recommended_investors`
- `recommendedinvestor`
- `recommendedInvestor`
- `recommendations`

Because no real labels exist, `train_model.py` generates rule-based positive
pairs and saves them as `recommended_investors.csv`.

---

## Firestore Schemas

### `project` collection fields used by the model

| Firestore field | Maps to model feature |
|-----------------|----------------------|
| `category` | `startup_category` |
| `budget_range` | `startup_budget_required` |
| `projectStage` | `startup_stage` (falls back to `status` if empty) |
| `location` | `startup_location` |
| `risk_level` | `startup_risk_level` |
| `traction_level` | `startup_traction_level` |

All other fields (`name`, `owner`, `description`, `photo`, `images`,
`equityOffered`, `likes`, `createdAt`, `updatedAt`, etc.) are passed through
to the API response unchanged.

### `investor_profiles` collection fields used by the model

| Firestore field | Maps to model feature |
|-----------------|----------------------|
| `category` | `investor_category` |
| `budget_range` | `investor_budget_range` |
| `preferred_stage` | `investor_preferred_stage` |
| `location` | `investor_location` |
| `risk_preference` | `investor_risk_preference` |
| `traction_preference` | `investor_traction_preference` |

---

## Supported Values

### Categories

`AI` · `AgriTech` · `CleanTech` · `E-commerce` · `EdTech` · `FinTech` ·
`HealthTech` · `LogisticsTech` · `PropTech` · `SaaS`

### Budget Ranges

`10k-100k` · `50k-200k` · `100k-500k` · `500k-2M` · `2M-10M` ·
`5M-15M` · `15M-50M` · `50M-250M`

### Startup Stages

`Idea` · `MVP` · `Early Traction` · `Growth` · `Scaling`

### Locations

**Pakistan cities (primary — set these in Firestore):**

| City | Normalises to |
|------|--------------|
| Karachi | Pakistan |
| Lahore | Pakistan |
| Islamabad | Pakistan |
| Peshawar | Pakistan |
| Faisalabad | Pakistan |
| Rawalpindi | Pakistan |
| Quetta | Pakistan |
| Multan | Pakistan |
| Sialkot | Pakistan |
| Hyderabad | Pakistan |

**Other supported locations:**

`Bangladesh` · `Egypt` · `India` · `Kenya` · `Nigeria` ·
`Saudi Arabia` · `Turkey` · `UAE` · `UK` · `USA`

City aliases are also handled: `Dubai` → `UAE`, `Riyadh` → `Saudi Arabia`,
`London` → `UK`, `Mumbai` → `India`, `Cairo` → `Egypt`, etc.

### Risk Values

`Low` · `Medium` · `High`

### Traction Values

`Users` · `Revenue`

### Compatibility Aliases

| Input | Resolves to |
|-------|------------|
| `500k-1M` | `500k-2M` |
| `1M-5M` | `2M-10M` |
| `Series A` | `Growth` |
| `Series B` / `Series C` | `Scaling` |
| `Pre-Seed` | `Idea` |
| `Seed` | `MVP` |
| City names (lowercase) | Country name |

---

## Model Features

### Raw Input Features

```
startup_category          investor_category
startup_budget_required   investor_budget_range
startup_stage             investor_preferred_stage
startup_location          investor_location
startup_risk_level        investor_risk_preference
startup_traction_level    investor_traction_preference
```

### Engineered Features (added inside pipeline)

| Feature | Description |
|---------|-------------|
| `category_match_flag` | 1 if startup and investor category match exactly |
| `location_match_flag` | 1 if locations match exactly |
| `stage_match_flag` | 1 if startup stage is in investor's preferred stages |
| `budget_rank_diff` | Absolute difference in budget tier ranks |
| `budget_close_flag` | 1 if budget tiers differ by ≤ 1 step |

### Model Stack

```
PairFeatureBuilder
    └── ColumnTransformer
            ├── OneHotEncoder(handle_unknown="ignore")  [categorical]
            └── StandardScaler                          [numeric]
                    └── LogisticRegression(class_weight="balanced")
```

---

## Latest Evaluation

Trained from Firestore data with rule-based investor-preference labels.

| Metric | Value |
|--------|-------|
| Startup profiles | 500 |
| Investor profiles | 500 |
| Positive pairs | 2,500 |
| Negative pairs | 2,500 |
| Total pairs | 5,000 |
| Test size | 1,000 |
| Accuracy | 0.988 |
| Precision | 0.9803 |
| Recall | 0.9960 |
| F1 | 0.9881 |
| ROC-AUC | 0.9981 |
| Average Precision | 0.9979 |

Score distribution on test set: min `0.00`, max `1.00`, mean `0.50`, median `0.68`.

> These scores are against generated rule-based labels. They verify the pipeline
> and API are working correctly but should not be treated as final business
> accuracy.

---

## Setup

```powershell
cd C:\Users\moham\OneDrive\Documents\model
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m py_compile main.py recommender.py train_model.py
```

---

## Training

```powershell
.\.venv\Scripts\python.exe train_model.py
```

This command:

1. Reads Firestore `project` collection
2. Reads Firestore `investor_profiles` collection
3. Saves local snapshots with all fields preserved (`QUOTE_ALL`)
4. Creates `recommended_investors.csv` if curated labels are missing
5. Trains the scikit-learn pipeline
6. Saves `startup_investor_pipeline.pkl`
7. Saves `model_metrics.json`

**No server restart needed.** The API detects the updated pipeline file on the
next request and reloads it automatically.

---

## Running the API

### Local CSV mode (default — fastest)

```powershell
Remove-Item Env:USE_FIRESTORE -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Reads startups from `firebase_project_profiles.csv`. If that file does not
exist, triggers an immediate Firestore sync to create it.

### Firestore mode (reads live on every request)

```powershell
$env:USE_FIRESTORE = "1"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

### LAN mode (phones on same Wi-Fi)

```powershell
$env:USE_FIRESTORE = "1"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

LAN IP used during testing: `192.168.18.207`

> Do not use `--reload` unless you exclude `.venv`. On Windows the reloader
> watches `.venv` and restarts repeatedly due to notebook packages.

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `USE_FIRESTORE` | `0` | Set to `1` to read startups from Firestore on every request |
| `AUTO_REFRESH_CSV` | `1` | Set to `0` to disable background CSV refresh |
| `CSV_REFRESH_SECONDS` | `600` | How often (seconds) to refresh the local CSV |
| `LOCAL_PROJECTS_PATH` | `firebase_project_profiles.csv` | Path to the local projects CSV cache |
| `FIREBASE_CREDENTIALS_JSON` | *(unset)* | Full service account JSON for Railway/cloud deployment |

---

## API Endpoints

### GET /

```json
{
  "status": "ok",
  "message": "Investor-Startup Recommendation API is running.",
  "api_version": "3.0.0",
  "endpoints": { "health": "/health", "docs": "/docs", "recommend": "/recommend" }
}
```

### GET /health

```json
{
  "status": "ok",
  "api_version": "3.0.0",
  "model_loaded": true,
  "firestore_ready": true,
  "data_source": "local_csv",
  "local_csv": "firebase_project_profiles.csv",
  "local_csv_exists": true,
  "local_csv_rows": 12,
  "auto_refresh_enabled": true,
  "auto_refresh_interval_seconds": 600,
  "recommendation_limit": 40,
  "firestore_collection": "project"
}
```

### POST /recommend

Canonical request body:

```json
{
  "investor_category":        "EdTech",
  "investor_budget_range":    "2M-10M",
  "investor_preferred_stage": "Growth,Scaling",
  "investor_location":        "Pakistan",
  "investor_risk_preference":     "High",
  "investor_traction_preference": "Users"
}
```

Firestore-style fields are also accepted (`category`, `budget_range`,
`preferred_stage`, `location`, `risk_preference`, `traction_preference`).

Example response item:

```json
{
  "startup_id": "La7XOi9oflVspQb6uRUG",
  "name": "E-test",
  "category": "EdTech",
  "description": "Online tests platform",
  "budget_range": "500k-2M",
  "projectStage": "Scaling",
  "location": "Peshawar",
  "traction_level": "Users",
  "risk_level": "High",
  "startup_category": "EdTech",
  "startup_budget_required": "500k-2M",
  "startup_stage": "Scaling",
  "startup_location": "Peshawar",
  "startup_risk_level": "High",
  "startup_traction_level": "Users",
  "match_score": 0.9933
}
```

---

## Testing

### PowerShell

```powershell
# Health check
Invoke-RestMethod http://127.0.0.1:8000/health

# Recommendation
$body = @{
  investor_category        = "EdTech"
  investor_budget_range    = "2M-10M"
  investor_preferred_stage = "Growth,Scaling"
  investor_location        = "Pakistan"
  investor_risk_preference     = "High"
  investor_traction_preference = "Users"
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8000/recommend -Method Post -Body $body -ContentType "application/json"
```

### React Native

```js
const response = await fetch("http://192.168.18.207:8000/recommend", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    investor_category:        "EdTech",
    investor_budget_range:    "2M-10M",
    investor_preferred_stage: "Growth,Scaling",
    investor_location:        "Pakistan",
    investor_risk_preference:     "High",
    investor_traction_preference: "Users",
  }),
});
const data = await response.json();
console.log(data.recommendations);
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `404 Not Found` on base URL | Use `http://host:8000/` or `/docs` |
| `405 Method Not Allowed` on `/recommend` | Use POST, not GET |
| All scores are 0.0 | Retrain with `train_model.py`; check field names match schema |
| `firebase_project_profiles.csv` is stale | Wait for 10-min refresh, or restart API (immediate sync on startup) |
| Phone cannot connect | Use `--host 0.0.0.0`; confirm same Wi-Fi; disable VPN |
| Port 8000 already in use | `netstat -ano \| Select-String ':8000'` then `Stop-Process -Id <PID> -Force` |
| `/recommend` returns old results | An old Uvicorn process is still running — kill all PIDs on port 8000 |
| `local_csv_rows: 0` in `/health` | Firestore `project` collection is empty or credentials are wrong |

---

## Stopping the API

```powershell
# Current terminal
Ctrl+C

# Background process
Stop-Process -Id (Get-Content .uvicorn.pid) -Force
```

---

## Security Notes

- **Never commit `serviceAccounts.json`** — add it to `.gitignore`.
- The API has no authentication layer. Before production, add auth, rate
  limiting, CORS policy, HTTPS, and proper secret management.
- On Railway, supply credentials via `FIREBASE_CREDENTIALS_JSON` env var.

---

## Recommended Next Improvements

- Replace generated rule-based labels with real curated or historical match data.
- Add API authentication (JWT or API key) for the mobile app.
- Add CORS configuration for deployed frontend/mobile environments.
- Add a `/retrain` endpoint to trigger retraining from Railway without SSH.
- Add automated tests for training, scoring, and API endpoints.
- Add a Dockerfile for portable deployment.