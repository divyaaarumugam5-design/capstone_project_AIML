# Part 5: Deployment (AQI Risk Advisor)

This part takes the tuned model from Part 3 and the LLM advisory from Part 4 and serves them as a Streamlit web app.
 http://192.168.29.105:8501

Steps to run app:

1.  enter a city, season, weather values and the ONI (ENSO) index.
2. tuned Random Forest pipeline (imputer ---> scaler-->  RF; 5-fold CV ROC-AUC 0.96) returns the probability of a high-risk AQI day (Poor, Very Poor or Severe).
3. A threshold slider lets you trade recall against precision.
4. A Ridge model gives a rough AQI estimate (R² ≈ 0.61).
5. GPT-4.1-mini (via OpenRouter, temperature 0) writes a plain-language advisory as JSON. The JSON is validated against a JSON Schema. A regex guardrail blocks emails and phone numbers before any API call. If the LLM fails, the app uses a rule-based advisory instead.
6. A feature-importance chart shows what drives the model.

## Run locally (from the `CAPSTONE PROJECT` folder)




## Results


Metrics: {
  "test_roc_auc": 0.9703,
  "test_accuracy": 0.9263,
  "test_precision": 0.8574,
  "test_recall": 0.7604,
  "ridge_r2": 0.6112
}
aqi_risk_rf.joblib: 6.9 MB
aqi_ridge.joblib: 0.0 MB