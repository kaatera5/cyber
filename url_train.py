import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os

# === Load Data ===
data = pd.read_csv("dataset/urls.csv")

# === Encode Labels ===
data['label'] = data['label'].map({'normal': 0, 'spam': 1})

# === Vectorize URLs (character-level TF-IDF) ===
vectorizer = TfidfVectorizer(
    analyzer='char_wb',      
    ngram_range=(3, 6),      # slightly longer n-grams
    max_features=10000       # richer feature set
)
X = vectorizer.fit_transform(data['url'])
y = data['label']

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Prepare models folder ===
os.makedirs("models", exist_ok=True)

# === Models ===
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
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {acc:.2%}")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, f"models/url_{name}_model.joblib")

    if acc > best_acc:
        best_acc = acc
        best_model = model
        best_name = name

# === Save Best Model & Vectorizer ===
joblib.dump(vectorizer, "models/url_vectorizer.joblib")
joblib.dump(best_model, "models/url_best_model.joblib")

print(f"\n Best Model: {best_name} ({best_acc:.2%}) saved as models/url_best_model.joblib")
