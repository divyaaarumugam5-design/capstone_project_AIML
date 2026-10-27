"""
Part 5 - Task 1: Serialize the best model 


"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aqi_features import RISK_MAPPING, make_features

PART_DIR = Path(__file__).resolve().parent.parent          
DATA_PATH = PART_DIR / "Data" / "cleaned_data.csv"
MODELS_DIR = PART_DIR / "models"
OUTPUTS_DIR = PART_DIR / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
print(f"Loaded {DATA_PATH.name}: {df.shape}")

X = make_features(df)
y_clf = df["AQI_Bucket"].map(RISK_MAPPING)
y_reg = df["AQI"]


X_train, X_test, y_train_clf, y_test_clf = train_test_split(
    X, y_clf, test_size=0.2, random_state=42)
_, _, y_train_reg, y_test_reg = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)


risk_model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(n_estimators=200, max_depth=None,
                           min_samples_leaf=1, random_state=42, n_jobs=-1),
)
risk_model.fit(X_train, y_train_clf)

proba = risk_model.predict_proba(X_test)[:, 1]
pred = (proba >= 0.5).astype(int)

aqi_model = make_pipeline(SimpleImputer(strategy="median"),
                          StandardScaler(), Ridge(alpha=1.0))
aqi_model.fit(X_train, y_train_reg)

metrics = {
    "test_roc_auc": round(roc_auc_score(y_test_clf, proba), 4),
    "test_accuracy": round(accuracy_score(y_test_clf, pred), 4),
    "test_precision": round(precision_score(y_test_clf, pred), 4),
    "test_recall": round(recall_score(y_test_clf, pred), 4),
    "ridge_r2": round(r2_score(y_test_reg, aqi_model.predict(X_test)), 4),
}

rf = risk_model.named_steps["randomforestclassifier"]
importances = (pd.Series(rf.feature_importances_, index=X.columns)
               .sort_values(ascending=False))

metadata = {
    "feature_columns": list(X.columns),
    "cities": sorted(df["City"].unique().tolist()),
    "seasons": sorted(df["season"].unique().tolist()),
    "metrics": metrics,
    "top_importances": importances.head(10).round(4).to_dict(),
}

joblib.dump(risk_model, MODELS_DIR / "aqi_risk_rf.joblib", compress=3)
joblib.dump(aqi_model, MODELS_DIR / "aqi_ridge.joblib", compress=3)
(MODELS_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))

# Feature importance chart for README / interviews
top = importances.head(10).sort_values()
plt.figure(figsize=(7, 4.5))
plt.barh(top.index, top.values)
plt.title("Random Forest - Top 10 Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=150)

print("Metrics:", json.dumps(metrics, indent=2))
for f in ["aqi_risk_rf.joblib", "aqi_ridge.joblib"]:
    size_mb = (MODELS_DIR / f).stat().st_size / 1e6
    print(f"{f}: {size_mb:.1f} MB")   # must stay under 100 MB for GitHub
