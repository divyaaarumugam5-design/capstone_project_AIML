"""

Samples 20 rows from cleaned_data.csv, predicts risk with the saved model, asks the LLM
for an advisory, and checks each response against ADVISORY_SCHEMA.


"""
import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv
from jsonschema import ValidationError, validate

from aqi_features import make_features

load_dotenv(find_dotenv(usecwd=True))
PART_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PART_DIR / "models"
N_RECORDS = 20

LLM_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
LLM_URL = "https://openrouter.ai/api/v1/chat/completions"
SCHEMA = {
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


def main():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not found in .env")

    model = joblib.load(MODELS_DIR / "aqi_risk_rf.joblib")
    meta = json.loads((MODELS_DIR / "metadata.json").read_text())
    df = pd.read_csv(PART_DIR / "Data" / "cleaned_data.csv")
    sample = df.sample(N_RECORDS, random_state=7).reset_index(drop=True)
    probs = model.predict_proba(make_features(sample, meta["feature_columns"]))[:, 1]

    rows = []
    for i, r in sample.iterrows():
        prob = float(probs[i])
        payload = {"city": r["City"], "season": r["season"],
                   "risk_level": "High" if prob >= 0.5 else "Low",
                   "probability": round(prob, 3), "avg_temp": r["avg_temp"],
                   "wind_speed": r["wind_speed"], "air_pressure": r["air_pressure"]}
        valid, error, api_ok = False, "", False
        try:
            for attempt in range(3):                     
                resp = requests.post(
                    LLM_URL, timeout=60,
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": LLM_MODEL, "temperature": 0, "max_tokens": 300,
                          "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                       {"role": "user", "content": json.dumps(payload)}]})
                if resp.status_code == 429:
                    time.sleep(20)
                    continue
                break
            if resp.status_code != 200:
                raise requests.RequestException(f"HTTP {resp.status_code}: {resp.text[:80]}")
            api_ok = True
            text = resp.json()["choices"][0]["message"]["content"].strip()
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(text)
            validate(instance=data, schema=SCHEMA)
            valid = data["risk_level"] == payload["risk_level"]
            if not valid:
                error = "risk_level disagrees with model"
        except (json.JSONDecodeError, ValidationError, KeyError, requests.RequestException) as e:
            error = f"{e.__class__.__name__}: {str(e)[:80]}"
        rows.append({**payload, "api_ok": api_ok, "schema_valid": valid, "error": error})
        print(f"{i + 1:2d}/{N_RECORDS} {r['City']:<15} valid={valid} {error}")
        time.sleep(4)                                      # stay under 20 requests/minute

    out = pd.DataFrame(rows)
    out.to_csv(PART_DIR / "outputs" / "llm_eval_results.csv", index=False)
    print(f"\nSchema-valid: {out['schema_valid'].sum()}/{N_RECORDS}  -> put this in the resume")


if __name__ == "__main__":
    main()
