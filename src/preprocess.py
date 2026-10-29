import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET, ID_COL, DATA_PATH


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    if ID_COL in df:
        df = df.drop_duplicates(subset=ID_COL)
    y = (df[TARGET] == "Yes").astype(int)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return X, y


def build_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIC_FEATURES),
                              ("cat", cat, CATEGORICAL_FEATURES)])
