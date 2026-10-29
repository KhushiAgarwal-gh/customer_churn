"""Run: python -m src.train"""
import json
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from .config import MODEL_PATH, RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from .preprocess import build_preprocessor, load_data


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                               class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    cv_scores, pipes = {}, {}
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        score = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="roc_auc").mean()
        cv_scores[name], pipes[name] = score, pipe
        print(f"{name:20s} CV ROC-AUC: {score:.4f}")

    best = max(cv_scores, key=cv_scores.get)
    model = pipes[best].fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "best_model": best,
        "cv_roc_auc": cv_scores,
        "test_roc_auc": roc_auc_score(y_te, proba),
        "test_precision": precision_score(y_te, pred),
        "test_recall": recall_score(y_te, pred),
        "test_f1": f1_score(y_te, pred),
    }
    print(f"\nBest: {best}\n", classification_report(y_te, pred, target_names=["Stay", "Churn"]))
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=["Stay", "Churn"])
    plt.title(f"Confusion Matrix - {best}"); plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=120); plt.close()
    RocCurveDisplay.from_predictions(y_te, proba)
    plt.title("ROC Curve"); plt.savefig(REPORTS_DIR / "roc_curve.png", dpi=120); plt.close()

    names = model.named_steps["prep"].get_feature_names_out()
    clf = model.named_steps["clf"]
    imp = clf.coef_[0] if hasattr(clf, "coef_") else clf.feature_importances_
    top = np.argsort(np.abs(imp))[-15:]
    plt.figure(figsize=(8, 6)); plt.barh(names[top], imp[top])
    plt.title("Top 15 Features"); plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=120); plt.close()

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
