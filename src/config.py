from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
REPORTS_DIR = ROOT / "reports"

TARGET = "Churn"
ID_COL = "customerID"
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges", "SupportCalls"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "InternetService", "OnlineSecurity", "TechSupport",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
