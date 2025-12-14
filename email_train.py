import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib

# === Load Dataset ===
if not os.path.exists("dataset/emails_dataset.csv"):
    raise FileNotFoundError(" emails_dataset.csv not found. Please run generate_email_dataset.py first.")

data = pd.read_csv("dataset/emails_dataset.csv")
data = data.dropna()
data['label'] = data['label'].map({'normal': 0, 'spam': 1})

# === TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), max_features=10000)
X = vectorizer.fit_transform(data['email'])
y = data['label']

# === Split Train/Test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

os.makedirs("models", exist_ok=True)

# === Define Models ===
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        min_samples_split=3,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        eval_metric='logloss',
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
}

best_model = None
best_acc = 0
best_name = ""

# === Train & Evaluate ===
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)
    print(report)
    print(f"{name} Accuracy: {acc:.2%}")

    # === Extra Metrics ===
    if y_prob is not None:
        logloss = log_loss(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        print(f"🔹 Log Loss: {logloss:.4f}")
        print(f"🔹 ROC-AUC: {auc:.4f}")
    else:
        logloss = np.nan
        auc = np.nan

    # === Save individual model ===
    joblib.dump(model, f"models/email_{name}_model.joblib")

    # === Track best model ===
    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

# === Save Vectorizer & Best Model ===
joblib.dump(vectorizer, "models/email_vectorizer.joblib")
joblib.dump(best_model, "models/email_best_model.joblib")

print(f"\n Best Model: {best_name} ({best_acc:.2%}) saved successfully!")
