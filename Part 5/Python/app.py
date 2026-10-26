"""
Part 5 - AQI Risk Advisor (Streamlit app)
"""
import json
import os
import re
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st
from jsonschema import ValidationError, validate

from aqi_features import enso_category, make_features

try:  # local runs: pick up OPENROUTER_API_KEY from the repo-root .env
    from dotenv import find_dotenv, load_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

PART_DIR = Path(__file__).resolve().parent.parent          # .../Project/Part 5
MODELS_DIR = PART_DIR / "models"

LLM_MODEL = "openai/gpt-4.1-mini"
LLM_URL = "https://openrouter.ai/api/v1/chat/completions"

ADVISORY_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "risk_level": {"type": "string", "enum": ["High", "Low"]},
        "probability": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
        "precautions": {"type": "array", "items": {"type": "string"},
                        "minItems": 1, "maxItems": 3},
    },
    "required": ["city", "risk_level", "probability", "summary", "precautions"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You write short public-health air quality advisories.
Return ONLY valid JSON (no markdown, no explanation) with exactly these keys:
city, risk_level ("High" or "Low"), probability (number 0-1),
summary (one plain-language sentence), precautions (list of 1-3 short strings).
Base the advice only on the prediction and weather values provided."""

st.set_page_config(page_title="AQI Risk Advisor", page_icon="🌫️", layout="centered")


# ---------------- helpers ----------------
@st.cache_resource
def load_artifacts():
    risk_model = joblib.load(MODELS_DIR / "aqi_risk_rf.joblib")
    aqi_model = joblib.load(MODELS_DIR / "aqi_ridge.joblib")
    meta = json.loads((MODELS_DIR / "metadata.json").read_text())
    return risk_model, aqi_model, meta


def get_api_key():
    try:
        return st.secrets["OPENROUTER_API_KEY"]   # Streamlit Cloud
    except Exception:
        return os.getenv("OPENROUTER_API_KEY")    # local .env


def has_pii(text: str) -> bool:
    """Same guardrail as Part 4: block emails and phone numbers."""
    email_pattern = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"
    phone_pattern = r"\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b"
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))


def rule_based_advisory(city: str, risk: str, prob: float) -> dict:
    if risk == "High":
        return {"city": city, "risk_level": "High", "probability": round(prob, 3),
                "summary": "Air quality is likely unhealthy; reduce outdoor exposure.",
                "precautions": ["Limit outdoor exercise",
                                "Wear an N95 mask outdoors",
                                "Keep windows closed during peak hours"]}
    return {"city": city, "risk_level": "Low", "probability": round(prob, 3),
            "summary": "Air quality is likely acceptable for most people.",
            "precautions": ["Sensitive groups should still monitor local AQI"]}


def llm_advisory(payload: dict, api_key: str):
    """Returns (advisory_dict or None, status_message)."""
    user_prompt = json.dumps(payload)
    if has_pii(user_prompt):
        return None, "Blocked by guardrail: PII detected in input."
    try:
        resp = requests.post(
            LLM_URL,
            headers={"Authorization": f"Bearer {api_key}",
                     "Content-Type": "application/json"},
            json={"model": LLM_MODEL, "temperature": 0, "max_tokens": 300,
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": user_prompt}]},
            timeout=30,
        )
    except requests.RequestException as e:
        return None, f"LLM request error: {e.__class__.__name__}"
    if resp.status_code != 200:
        return None, f"LLM request failed (HTTP {resp.status_code})."

    text = resp.json()["choices"][0]["message"]["content"].strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(text)
        validate(instance=data, schema=ADVISORY_SCHEMA)
        return data, f"Schema-valid output from {LLM_MODEL}."
    except (json.JSONDecodeError, ValidationError) as e:
        return None, f"LLM output failed validation ({str(e)[:100]}); used fallback."


# ---------------- UI ----------------
risk_model, aqi_model, meta = load_artifacts()
m = meta["metrics"]

st.title("🌫️ AQI Risk Advisor")
st.caption(
    f"Tuned Random Forest · test ROC-AUC {m['test_roc_auc']} · recall {m['test_recall']}  |  "
    f"Ridge AQI estimate · R² {m['ridge_r2']}  |  Advisory: {LLM_MODEL}"
)

with st.sidebar:
    st.header("Inputs")
    city = st.selectbox("City", meta["cities"])
    season = st.selectbox("Season", meta["seasons"])
    min_t = st.number_input("Min temperature (°C)", -5.0, 45.0, 18.0, 0.5)
    max_t = st.number_input("Max temperature (°C)", 0.0, 50.0, 30.0, 0.5)
    avg_t = st.number_input("Avg temperature (°C)", -5.0, 50.0, 24.0, 0.5)
    wind = st.number_input("Wind speed (km/h)", 0.0, 40.0, 8.0, 0.5)
    pressure = st.number_input("Air pressure (hPa)", 950.0, 1030.0, 1010.0, 0.5)
    oni = st.slider("ONI (ENSO index)", -2.0, 2.0, 0.0, 0.05)
    threshold = st.slider(
        "High-risk decision threshold", 0.3, 0.7, 0.5, 0.05,
        help="Lower = catch more high-risk days (higher recall, lower precision).")
    use_llm = st.toggle("Generate LLM advisory", value=True)

if not (min_t <= avg_t <= max_t):
    st.warning("Check temperatures: expected min ≤ avg ≤ max.")

row = pd.DataFrame([{"City": city, "season": season, "avg_temp": avg_t,
                     "min_temp": min_t, "max_temp": max_t, "wind_speed": wind,
                     "air_pressure": pressure, "ONI": oni}])
X = make_features(row, columns=meta["feature_columns"])

if st.button("Predict risk", type="primary"):
    prob = float(risk_model.predict_proba(X)[0, 1])
    aqi_est = max(float(aqi_model.predict(X)[0]), 0.0)
    risk = "High" if prob >= threshold else "Low"

    c1, c2, c3 = st.columns(3)
    c1.metric("High-risk probability", f"{prob:.0%}")
    c2.metric("Risk level", risk)
    c3.metric("Estimated AQI", f"{aqi_est:.0f}")
    st.caption(f"ENSO phase: {enso_category(oni)} · "
               f"AQI estimate is a rough guide (R² ≈ {m['ridge_r2']}).")

    st.subheader("Plain-language advisory")
    payload = {"city": city, "season": season, "risk_level": risk,
               "probability": round(prob, 3), "estimated_aqi": round(aqi_est),
               "avg_temp": avg_t, "wind_speed": wind, "air_pressure": pressure}

    advisory, status = None, "LLM advisory turned off: using rule-based advisory."
    if use_llm:
        api_key = get_api_key()
        if api_key:
            with st.spinner("Generating advisory..."):
                advisory, status = llm_advisory(payload, api_key)
        else:
            status = "No OPENROUTER_API_KEY found: using rule-based advisory."
    if advisory is None:
        advisory = rule_based_advisory(city, risk, prob)

    (st.error if risk == "High" else st.success)(advisory["summary"])
    for p in advisory["precautions"]:
        st.write(f"• {p}")
    st.caption(status)
    with st.expander("Structured JSON output"):
        st.json(advisory)

st.subheader("What drives the model")
imp = pd.Series(meta["top_importances"]).sort_values()
st.bar_chart(imp, horizontal=True)
st.caption("Random Forest feature importance (top 10).")
