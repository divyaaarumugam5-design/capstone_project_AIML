

import pandas as pd

BASE_COLS = ["City", "season", "avg_temp", "min_temp", "max_temp",
             "wind_speed", "air_pressure", "ONI"]
CAT_COLS = ["City", "season", "ENSO_Category"]

RISK_MAPPING = {
    "Good": 0,
    "Satisfactory": 0,
    "Moderate": 0,
    "Poor": 1,
    "Very Poor": 1,
    "Severe": 1,
}


def enso_category(oni: float) -> str:
    """ENSO phase from the NOAA Oceanic Nino Index."""
    if oni >= 0.5:
        return "EL Nino"
    if oni <= -0.5:
        return "La Nina"
    return "Neutral"


def make_features(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """
    Training   (columns=None): one-hot with drop_first=True, same as the notebooks.
    Inference  (columns=list): one-hot every category, then align to the saved
                               training columns so a single input row is encoded
                               correctly (the base category simply gets dropped).
    """
    X = df[BASE_COLS].copy()
    X["ENSO_Category"] = X["ONI"].apply(enso_category)
    X = pd.get_dummies(X, columns=CAT_COLS, drop_first=columns is None)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X.astype(float)
