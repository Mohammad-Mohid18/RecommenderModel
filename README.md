# Investor-Startup Recommendation API

This project contains a machine-learning recommendation system that ranks
startups for an investor profile based on the investor's preferences. It includes:

- a reusable scikit-learn model pipeline
- Firebase/Firestore data loading
- local CSV fallback data
- a FastAPI backend
- a notebook for exploration/retraining
- React Native integration-ready API endpoints

The current API has been tested end to end in both local CSV mode and Firestore
mode.

## Current Status

Working pieces:

- `POST /recommend` returns up to **40** ranked startups with nonzero `match_score` values for an investor profile.
- `GET /health` reports API version, recommendation direction, recommendation limit, model status, and data source.
- `GET /` returns a small API index for browser checks.
- The model trains from Firestore `startup_profiles` and `investor_profiles`.
- Evaluation metrics are saved in `model_metrics.json`.
- Firestore snapshots are saved locally for reproducible development.
- LAN testing works when Uvicorn binds to `0.0.0.0` on port `8000`.

Important modeling note:

The project does not currently have human-curated recommendation labels.
No populated recommendation-label collection was found in Firestore, and the
original local `recommended_investors.csv` was missing. Because of that,
`train_model.py` generates `recommended_investors.csv` using transparent
rule-based compatibility logic from investor preferences to startup profiles.
The model is technically working, but true business accuracy should later be
improved with real historical or curated match labels.

## Project Files

Core files:

- `main.py` - FastAPI backend (API v2.0.0, investor → startups).
- `recommender.py` - shared model utilities, feature engineering, schema mapping,
canonicalization, and recommendation logic (`RECOMMENDATION_TOP_N = 40`).
- `train_model.py` - trains the model from Firestore data and saves metrics.
- `3-cosine-based-system-6f-v2.1.ipynb` - notebook version of the training flow.
- `MOBILE_INTEGRATION.md` - React Native integration guide for the investor-centric API.
- `requirements.txt` - Python dependencies.
- `serviceAccounts.json` - Firebase service account credentials.

Generated/model files:



- `startup_investor_pipeline.pkl` - trained scikit-learn pipeline used by the API.
- `model_metrics.json` - latest evaluation metrics.
- `firebase_startup_profiles.csv` - snapshot of Firestore `startup_profiles`.
- `firebase_investor_profiles.csv` - snapshot of Firestore `investor_profiles`.
- `recommended_investors.csv` - generated positive training pairs when no curated
labels exist.

Legacy/source CSVs:

- `startup2.csv`
- `investor2.csv`

These are still useful as fallback data, but the current preferred training path
uses Firestore snapshots.

## Architecture

The recommendation system works as a pairwise classifier.

For each investor-startup pair, the model predicts the probability that the pair
is a good match. The API scores one investor against all startup profiles and
returns the top ranked startups (up to 40 per request) that fit the investor's preferences.

Flow:

1. Load startup profiles and investor profiles.
2. Normalize schemas into model feature names.
3. Build positive and negative investor-startup pairs (top startups per investor).
4. Add engineered match features inside the scikit-learn pipeline.
5. Encode categorical fields and scale numeric fields.
6. Train a logistic regression classifier.
7. Save one pipeline artifact.
8. Use the saved pipeline inside FastAPI for inference.

## Data Sources

Firestore collections used:

- `startup_profiles`
- `investor_profiles`

Checked but empty/missing recommendation-label collections:

- `recommended_investors`
- `recommendedinvestor`
- `recommendedInvestor`
- `recommendations`

Because no real recommendation label data was available, the training script
creates rule-based positive pairs (top matching startups per investor) and saves
them as `recommended_investors.csv`.

## Firestore Schemas

Startup profile fields from Firestore:

- `category`
- `budget_range`
- `status`
- `location`
- `risk_level`
- `traction_level`

Investor profile fields from Firestore:

- `category`
- `budget_range`
- `preferred_stage`
- `location`
- `risk_preference`
- `traction_preference`

The model internally uses canonical names:

Startup-side:

- `startup_category`
- `startup_budget_required`
- `startup_stage`
- `startup_location`
- `startup_risk_level`
- `startup_traction_level`

Investor-side:

- `investor_category`
- `investor_budget_range`
- `investor_preferred_stage`
- `investor_location`
- `investor_risk_preference`
- `investor_traction_preference`

`recommender.py` maps Firestore fields into these canonical fields.

## Supported Values

Known startup/investor categories:

- `AI`
- `AgriTech`
- `CleanTech`
- `E-commerce`
- `EdTech`
- `FinTech`
- `HealthTech`
- `LogisticsTech`
- `PropTech`
- `SaaS`

Known budget ranges:

- `10k-100k`
- `50k-200k`
- `100k-500k`
- `500k-2M`
- `2M-10M`

Known startup stages:

- `Idea`
- `MVP`
- `Early Traction`
- `Growth`
- `Scaling`

Known locations:

- `Bangladesh`
- `Egypt`
- `India`
- `Nigeria`
- `Pakistan`
- `Saudi Arabia`
- `Turkey`
- `UAE`
- `UK`
- `USA`

Known risk values:

- `Low`
- `Medium`
- `High`

Known traction values:

- `Users`
- `Revenue`

Compatibility aliases were added so older API examples still work:

- `500k-1M` maps to `500k-2M`
- `Series A` maps to `Growth`
- `Series B` / `Series C` map to `Scaling`
- `Karachi`, `Lahore`, `Islamabad` map to `Pakistan`
- `Dubai`, `Abu Dhabi` map to `UAE`
- `Riyadh`, `Jeddah` map to `Saudi Arabia`

## Model Features

Raw features:

- startup category
- startup budget
- startup stage
- startup location
- startup risk level
- startup traction level
- investor category
- investor budget range
- investor preferred stage
- investor location
- investor risk preference
- investor traction preference

Engineered features added inside the pipeline:

- `category_match_flag`
- `location_match_flag`
- `stage_match_flag`
- `budget_rank_diff`
- `budget_close_flag`

Model stack:

- `PairFeatureBuilder`
- `ColumnTransformer`
- `OneHotEncoder(handle_unknown="ignore")`
- `StandardScaler`
- `LogisticRegression(class_weight="balanced")`

## Latest Evaluation

The latest model was trained from Firestore data with investor-preference labels:

- Recommendation direction: `investor_to_startups`
- Startup profiles: `500`
- Investor profiles: `500`
- Positive pairs: `2500`
- Negative pairs: `2500`
- Total pairs: `5000`
- Test size: `1000`

Latest metrics saved in `model_metrics.json`:

- Accuracy: `0.988`
- Precision: `0.9803`
- Recall: `0.996`
- F1: `0.9881`
- ROC-AUC: `0.9981`
- Average precision: `0.9979`

Score distribution on the test set:

- Minimum score: `0.0000000101`
- Maximum score: `0.99998`
- Mean score: `0.5018`
- Median score: `0.6811`

These scores are against generated rule-based labels. They are useful for
verifying the pipeline and API behavior, but they should not be treated as final
business-grade accuracy.

## Setup

Use PowerShell from the project directory:

```powershell
cd C:\Users\moham\OneDrive\Documents\model
```

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Check dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

Compile-check Python files:

```powershell
.\.venv\Scripts\python.exe -m py_compile main.py recommender.py train_model.py
```

## Training

Train from Firebase:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

This command:

- reads Firestore `startup_profiles`
- reads Firestore `investor_profiles`
- saves local Firebase snapshots
- creates `recommended_investors.csv` if curated labels are missing
- trains the scikit-learn pipeline
- saves `startup_investor_pipeline.pkl`
- saves `model_metrics.json`

After retraining, restart the API so the new model is loaded.

## Running the API Locally

Local CSV/snapshot mode:

```powershell
Remove-Item Env:USE_FIRESTORE -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

In this mode, the API reads startups from `firebase_startup_profiles.csv`.
If that file does not exist, it falls back to `startup2.csv`.

Firestore mode:

```powershell
$env:USE_FIRESTORE = "1"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

In this mode, the API loads startups directly from Firestore
`startup_profiles`.

Do not use `--reload` for this project unless you exclude `.venv`. On Windows,
the reloader can watch `.venv` and repeatedly restart after notebook packages
such as `ipykernel` are installed.

## Running as a LAN Server

To allow phones or other devices on the same Wi-Fi to access the API, bind to
all interfaces:

```powershell
cd C:\Users\moham\OneDrive\Documents\model
$env:USE_FIRESTORE = "1"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

For this machine, the LAN IP used during testing was:

```text
192.168.18.207
```

Frontend/base URL for devices on the same Wi-Fi:

```text
http://192.168.18.207:8000
```

Health URL:

```text
http://192.168.18.207:8000/health
```

Recommendation URL:

```text
http://192.168.18.207:8000/recommend
```

If a phone cannot reach the API:

- confirm the API is running
- confirm the phone and laptop are on the same Wi-Fi
- temporarily disable mobile data on the phone
- turn off VPN on both devices
- allow Python/Uvicorn through Windows Firewall on private networks
- check `http://192.168.18.207:8000/health` first

## API Endpoints

### GET /

Browser-friendly index route.

Example:

```text
GET http://127.0.0.1:8000/
```

Response:

```json
{
  "status": "ok",
  "message": "Investor-Startup Recommendation API is running.",
  "api_version": "2.0.0",
  "recommendation_direction": "investor_to_startups",
  "recommendation_limit": 40,
  "endpoints": {
    "health": "/health",
    "docs": "/docs",
    "recommend": "/recommend"
  }
}
```

### GET /health

Checks whether the API and model are ready.

Example:

```text
GET http://127.0.0.1:8000/health
```

Local mode response:

```json
{
  "status": "ok",
  "api_version": "2.0.0",
  "recommendation_direction": "investor_to_startups",
  "recommendation_limit": 40,
  "model_loaded": true,
  "data_source": "local_csv",
  "firestore_ready": false
}
```

Firestore mode response:

```json
{
  "status": "ok",
  "api_version": "2.0.0",
  "recommendation_direction": "investor_to_startups",
  "recommendation_limit": 40,
  "model_loaded": true,
  "data_source": "firestore",
  "firestore_ready": true
}
```

### POST /recommend

Returns up to 40 top startup recommendations for one investor profile.

Important: `/recommend` only accepts `POST`. Opening `/recommend` in a browser
sends `GET`, so FastAPI returns:

```json
{"detail":"Method Not Allowed"}
```

Use `/docs`, PowerShell, Postman, or React Native `fetch` to send a POST request.

Canonical request body:

```json
{
  "investor_category": "FinTech",
  "investor_budget_range": "500k-2M",
  "investor_preferred_stage": "Growth,Scaling",
  "investor_location": "Pakistan",
  "investor_risk_preference": "Medium",
  "investor_traction_preference": "Revenue"
}
```

Firestore-style request body is also accepted:

```json
{
  "category": "FinTech",
  "budget_range": "500k-2M",
  "preferred_stage": "Growth,Scaling",
  "location": "Pakistan",
  "risk_preference": "Medium",
  "traction_preference": "Revenue"
}
```

Example response:

```json
{
  "recommendations": [
    {
      "startup_id": "S140",
      "category": "FinTech",
      "budget_range": "2M-10M",
      "status": "Scaling",
      "location": "Pakistan",
      "traction_level": "Revenue",
      "risk_level": "Medium",
      "match_score": 0.9991
    }
  ]
}
```

## Testing with PowerShell

Health:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Recommendation:

```powershell
$body = @{
  category = "FinTech"
  budget_range = "500k-2M"
  preferred_stage = "Growth,Scaling"
  location = "Pakistan"
  risk_preference = "Medium"
  traction_preference = "Revenue"
} | ConvertTo-Json

Invoke-RestMethod http://127.0.0.1:8000/recommend -Method Post -Body $body -ContentType "application/json"
```

## React Native Integration

Use this endpoint when the API is running on the same computer:

```text
http://127.0.0.1:8000/recommend
```

Use this endpoint for Android emulator:

```text
http://10.0.2.2:8000/recommend
```

Use this endpoint for a physical phone on the same Wi-Fi:

```text
http://192.168.18.207:8000/recommend
```

Example React Native request:

```js
const response = await fetch("http://192.168.18.207:8000/recommend", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    category: "FinTech",
    budget_range: "500k-2M",
    preferred_stage: "Growth,Scaling",
    location: "Pakistan",
    risk_preference: "Medium",
    traction_preference: "Revenue",
  }),
});

const data = await response.json();
console.log(data.recommendations);
```

## Notebook

The notebook `3-cosine-based-system-6f-v2.1.ipynb` mirrors the model training
workflow.

It now prefers:

- `firebase_startup_profiles.csv`
- `firebase_investor_profiles.csv`
- `recommended_investors.csv`

If Firebase snapshots do not exist, it falls back to:

- `startup2.csv`
- `investor2.csv`

Recommended workflow:

1. Run `train_model.py` first to refresh Firebase snapshots.
2. Open the notebook.
3. Run all cells from top to bottom.

## Troubleshooting

`{"detail":"Not Found"}` on the base URL:

- Fixed by adding `GET /`.
- Use `http://host:8000/` or `http://host:8000/docs`.

`{"detail":"Method Not Allowed"}` on `/recommend`:

- `/recommend` is a POST endpoint.
- Use `/docs`, PowerShell, Postman, or React Native `fetch`.

Scores are all zero or too low:

- Make sure the latest `startup_investor_pipeline.pkl` is loaded.
- Restart Uvicorn after retraining.
- Make sure request fields match accepted schema.
- Prefer Firestore-style fields: `category`, `budget_range`, `preferred_stage`,
`location`, `risk_preference`, `traction_preference`.
- Check `model_metrics.json` and test `/recommend` with the sample payload.

Phone cannot connect to LAN server:

- Start Uvicorn with `--host 0.0.0.0`.
- Use the laptop LAN IP, for example `http://192.168.18.207:8000`.
- Ensure both devices are on the same Wi-Fi.
- Allow Python through Windows Firewall.

Port 8000 already in use:

```powershell
netstat -ano | Select-String ':8000'
```

Then stop **all** processes using port 8000 (old servers can keep running and serve outdated code):

```powershell
Stop-Process -Id <PID> -Force
```

Restart with a single server:

```powershell
cd C:\Users\moham\OneDrive\Documents\model
$env:USE_FIRESTORE = "1"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Confirm the correct version via `/health` — you should see `"recommendation_limit": 40` and `"recommendation_direction": "investor_to_startups"`.

`/recommend` still returns only 10 results or investors instead of startups:

- An old Uvicorn process is probably still bound to port `8000`.
- Run `netstat -ano | Select-String ':8000'` and stop every PID listed.
- Start only one server with the latest code (see commands above).
- Re-check `/health` before testing `/docs`.

## Stopping the API

If the server is running in the current terminal, press:

```text
Ctrl+C
```

If it was started in the background and `.uvicorn.pid` exists:

```powershell
Stop-Process -Id (Get-Content .uvicorn.pid) -Force
```

## Security Notes

- Do not commit or share `serviceAccounts.json` publicly.
- The current API has no authentication layer.
- Before production deployment, add authentication, rate limiting, CORS policy,
HTTPS, and proper secret management.

## Recommended Next Improvements

- Replace generated rule-based labels with real curated or historical match data.
- Add API authentication for the mobile app.
- Add CORS configuration for deployed frontend/mobile environments.
- Cache Firestore startup profiles for faster repeated recommendation calls.
- Add a deployment script or Dockerfile.
- Add automated tests for training, scoring, and API endpoints.

