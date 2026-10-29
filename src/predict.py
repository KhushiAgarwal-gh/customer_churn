"""Run: python -m src.predict path/to/customers.csv  -> reports/predictions.csv"""
import sys
import joblib
import pandas as pd
from .config import CATEGORICAL_FEATURES, MODEL_PATH, NUMERIC_FEATURES, REPORTS_DIR


def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    model = joblib.load(MODEL_PATH)
    proba = model.predict_proba(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]
    out = df.copy()
    out["churn_probability"] = proba.round(4)
    out["risk"] = pd.cut(proba, [-1, 0.33, 0.66, 2], labels=["Low", "Medium", "High"])
    return out


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "data/churn.csv"
    res = predict_df(pd.read_csv(src))
    out_path = REPORTS_DIR / "predictions.csv"
    res.to_csv(out_path, index=False)
    print(res["risk"].value_counts()); print(f"Saved -> {out_path}")
