# Mobile App Integration Guide (React Native CLI)

This guide explains how to integrate the **investor–startup recommendation pipeline** into a React Native (CLI, JS) mobile app. The API accepts an **investor profile** and returns up to **40 ranked startups** that match the investor's preferences.

---

## What you already have

From this repo, you now have:

- A trained, production-ready pipeline saved as `startup_investor_pipeline.pkl`
- The inference schema (raw inputs only)
- A shared `recommend(...)` function in `recommender.py`
- A FastAPI backend in `main.py`
- A notebook (`3-cosine-based-system-6f-v2.1.ipynb`) that trains and saves the pipeline

---

## Recommended architecture for mobile

Because React Native cannot run Python/scikit-learn directly, expose the pipeline via a **backend API**.

### Architecture overview

1. **Backend (Python)**
   - Loads `startup_investor_pipeline.pkl`
   - Accepts an investor profile
   - Scores all startups and returns up to 40 ranked matches

2. **Mobile (React Native)**
   - Collects investor preferences (form)
   - Calls backend API
   - Displays ranked startups

---

## Input schema (raw JSON)

The mobile app should send a JSON payload containing the investor profile:

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

These map into:

- `investor_category`
- `investor_budget_range`
- `investor_preferred_stage`
- `investor_location`
- `investor_risk_preference` (optional)
- `investor_traction_preference` (optional)

---

## Backend flow (Python API)

Core logic in `main.py`:

1. Load the `.pkl` pipeline once at startup
2. Load or fetch startup data (Firestore or local CSV)
3. Call `recommend(investor, startups_df, pipeline, top_n=40)`
4. Return ranked startups

### Minimal API flow (pseudo-code)

```python
from recommender import RECOMMENDATION_TOP_N, recommend

model = joblib.load("startup_investor_pipeline.pkl")
startups = pd.read_csv("firebase_startup_profiles.csv")

@app.post("/recommend")
def recommend_api(investor: dict):
    results = recommend(investor, startups, model, top_n=RECOMMENDATION_TOP_N)
    return {"recommendations": results.to_dict(orient="records")}
```

---

## React Native flow (JS)

### 1) Collect investor preferences

Use a form and map values to the schema.

### 2) Call API

```js
const response = await fetch("https://your-api.com/recommend", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(investorData),
});
const data = await response.json();
const results = data.recommendations;
```

### 3) Render results

Display up to 40 startups in a list or cards.

---

## Suggested screens in mobile app

1. **Investor Preferences Screen**
   - Category (dropdown)
   - Budget range (dropdown)
   - Preferred stage (multi-select or comma-separated)
   - Location (text)
   - Risk / traction preference (optional)

2. **Results Screen**
   - Ranked startup list (up to 40)
   - Match score
   - Startup metadata (category, stage, location, budget)

---

## Deployment options for backend

- **Render / Railway / Fly.io** (fastest to deploy)
- **AWS / GCP / Azure** (scalable)
- **Local dev** with tunneling (e.g., `ngrok`)

### Local / LAN testing

```powershell
cd C:\Users\moham\OneDrive\Documents\model
$env:USE_FIRESTORE = "1"
.\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

- This PC: `http://127.0.0.1:8000/docs`
- Other device on same Wi-Fi: `http://<your-lan-ip>:8000/docs`

---

## Recommendation for production

- Store startups in a DB (Postgres / MongoDB) or Firestore
- Load pipeline once at startup
- Cache startup profiles if repeated queries
- Add validation on input schema

---

## Summary

You already have the ML pipeline. The API:

- Accepts an **investor profile**
- Returns up to **40 ranked startups**
- Is ready to call from React Native via `POST /recommend`
