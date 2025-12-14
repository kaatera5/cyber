import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, log_loss
import joblib

# === Load dataset ===
data = pd.read_csv("dataset/message.csv", encoding='latin-1')
data = data[['v1', 'v2']]  # columns: v1 = label, v2 = message
data.columns = ['label', 'message']
data = data.dropna()

# Encode labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# === TF-IDF Vectorizer ===
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=10000,
    ngram_range=(1, 2)  # word + bigram
)
X = vectorizer.fit_transform(data['message'])
y = data['label']

# === Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Models ===
models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=30, n_jobs=-1, random_state=42
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

os.makedirs("models", exist_ok=True)
best_acc = 0
best_model = None
best_name = ""

# === Train and Evaluate ===
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    loss = log_loss(y_test, y_prob)

    print(f"{name} Accuracy: {acc:.2%}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Log Loss: {loss:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, f"models/sms_{name}_model.joblib")

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

# Save best model and vectorizer
joblib.dump(vectorizer, "models/sms_vectorizer.joblib")
joblib.dump(best_model, "models/sms_best_model.joblib")

print(f"\n Best Model: {best_name} ({best_acc:.2%}) saved successfully!")
