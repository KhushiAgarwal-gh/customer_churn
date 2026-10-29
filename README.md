# 📉 Customer Churn Prediction

End-to-end ML project: data → preprocessing → model comparison → evaluation → Streamlit app.

## Folder structure
```
customer-churn-project/
├── data/            # churn.csv (generated ya aapka apna)
├── src/
│   ├── config.py        # paths, feature lists
│   ├── generate_data.py # synthetic data generator
│   ├── preprocess.py    # imputation, scaling, encoding
│   ├── train.py         # 3 models compare, best save
│   └── predict.py       # batch predictions
├── models/          # saved model (.joblib)
├── reports/         # metrics.json + plots
├── notebooks/       # EDA notebook
├── app.py           # Streamlit app
└── requirements.txt
```

## Quick start
```bash
pip install -r requirements.txt
python -m src.generate_data     # data/churn.csv banata hai
python -m src.train             # model train + reports/
python -m src.predict           # predictions -> reports/predictions.csv
streamlit run app.py            # web app
```

## Apna real dataset use karna hai?
`data/churn.csv` replace karo (jaise Kaggle Telco Churn). Columns `src/config.py` ke
hisaab se match karo, aur `Churn` column me `Yes/No` hona chahiye.

## Models
Logistic Regression, Random Forest, Gradient Boosting - 5-fold CV ROC-AUC se best select hota hai.

## Next steps
- Hyperparameter tuning (GridSearchCV / Optuna)
- SHAP explainability
- Threshold tuning based on retention cost vs. customer lifetime value
