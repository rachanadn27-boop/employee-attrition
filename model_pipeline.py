import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import VarianceThreshold
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

def train_model():
    df = pd.read_csv(DATA_PATH)

    y = df["Attrition"].map({"Yes": 1, "No": 0})
    X = df.drop("Attrition", axis=1)

    X = pd.get_dummies(X, drop_first=True)
    columns = X.columns.tolist()

    selector = VarianceThreshold(threshold=0.01)
    X = selector.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 🔥 Balanced learning without SMOTE
    scale = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    # 🔥 Threshold tuning (KEY for higher accuracy)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.6).astype(int)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    joblib.dump(selector, os.path.join(MODEL_DIR, "selector.pkl"))
    joblib.dump(columns, os.path.join(MODEL_DIR, "columns.pkl"))

    print("Model saved!")

def load_model():
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    selector = joblib.load(os.path.join(MODEL_DIR, "selector.pkl"))
    columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))
    return model, selector, columns

def predict(data):
    model, selector, columns = load_model()

    df = pd.DataFrame([data])
    df["OverTime"] = df["OverTime"].apply(lambda x: "Yes" if x == 1 else "No")

    df = pd.get_dummies(df)

    for col in columns:
        if col not in df.columns:
            df[col] = 0

    df = df[columns]
    df = selector.transform(df)

    prob = model.predict_proba(df)[0][1]

    return prob
if __name__ == "__main__":
    train_model()