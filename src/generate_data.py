"""Synthetic telecom customer data banata hai (agar aapke paas real dataset nahi hai).
Real data use karna ho to data/churn.csv replace kar do (same columns ke saath)."""
import numpy as np
import pandas as pd
from .config import DATA_PATH, RANDOM_STATE


def generate(n: int = 7000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customerID": [f"CUST-{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice(["Yes", "No"], n, p=[0.16, 0.84]),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure": rng.integers(1, 73, n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2]),
        "OnlineSecurity": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "TechSupport": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2]),
        "PaperlessBilling": rng.choice(["Yes", "No"], n, p=[0.6, 0.4]),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
        "SupportCalls": rng.poisson(1.5, n),
    })
    base = np.where(df.InternetService == "Fiber optic", 80,
                    np.where(df.InternetService == "DSL", 55, 25))
    df["MonthlyCharges"] = np.round(base + rng.normal(0, 8, n), 2).clip(18, 120)
    df["TotalCharges"] = np.round(df.MonthlyCharges * df.tenure + rng.normal(0, 30, n), 2).clip(18)

    z = (-1.2
         + 1.3 * (df.Contract == "Month-to-month")
         - 1.0 * (df.Contract == "Two year")
         + 0.6 * (df.InternetService == "Fiber optic")
         + 0.5 * (df.PaymentMethod == "Electronic check")
         + 0.35 * df.SupportCalls
         - 0.04 * df.tenure
         + 0.015 * (df.MonthlyCharges - 65)
         - 0.5 * (df.OnlineSecurity == "Yes")
         - 0.4 * (df.TechSupport == "Yes")
         + 0.3 * (df.SeniorCitizen == "Yes"))
    p = 1 / (1 + np.exp(-z))
    df["Churn"] = np.where(rng.random(n) < p, "Yes", "No")

    df.loc[rng.choice(n, 20, replace=False), "TotalCharges"] = np.nan
    return df


if __name__ == "__main__":
    df = generate()
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df)} rows -> {DATA_PATH} | churn rate: {(df.Churn=='Yes').mean():.1%}")
