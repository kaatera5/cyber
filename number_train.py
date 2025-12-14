import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# === Load dataset ===
data = pd.read_csv("dataset/phone_numbers_dataset.csv")

# Clean data
data = data.dropna()
data['number'] = data['number'].astype(str).str.strip()
data['label'] = data['label'].map({'normal': 0, 'spam': 1})



# === Enhanced Feature Extraction ===
def extract_features(number):
    number = str(number)
    digits = [int(d) for d in number if d.isdigit()]
    if not digits:
        digits = [0]
    
    # Digit statistics
    mean_digit = np.mean(digits)
    std_digit = np.std(digits)
    unique_digits = len(set(digits))
    
    # Patterns
    repeated_pairs = sum(1 for i in range(1, len(digits)) if digits[i] == digits[i-1])
    increasing_seq = sum(1 for i in range(1, len(digits)) if digits[i] > digits[i-1])
    decreasing_seq = sum(1 for i in range(1, len(digits)) if digits[i] < digits[i-1])
    digit_sum = sum(digits)
    zeros = number.count('0')
    nines = number.count('9')
    
    # Frequency-based
    counts = [digits.count(i) for i in range(10)]
    max_freq = max(counts)
    min_freq = min(counts)
    entropy = -sum((c/len(digits)) * np.log2(c/len(digits)) for c in counts if c > 0)
    
    return [
        len(number),               # total length
        unique_digits,             # number of unique digits
        mean_digit, std_digit,     # mean and std of digits
        repeated_pairs,            # consecutive repeats
        increasing_seq,            # increasing pattern
        decreasing_seq,            # decreasing pattern
        zeros, nines,              # specific digits
        digit_sum,                 # total digit sum
        max_freq, min_freq,        # digit frequency extremes
        entropy,                   # randomness measure
        int(number[0]),            # first digit
        int(number[-1]),           # last digit
    ]

# === Feature Matrix ===
X = np.array([extract_features(num) for num in data['number']])
y = data['label']

# === Scale Features ===
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Model Setup ===
os.makedirs("models", exist_ok=True)

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=3,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    "XGBoost": XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42
    )
}

best_model, best_acc, best_name = None, 0, ""

# === Train & Evaluate ===
for name, model in models.items():
    print(f"\n--- Training {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2%}")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(model, f"models/number_{name}_model.joblib")

    if acc > best_acc:
        best_acc, best_model, best_name = acc, model, name

# === Save Scaler & Best Model ===
joblib.dump(scaler, "models/number_scaler.joblib")
joblib.dump(best_model, "models/number_best_model.joblib")

print(f"\n Best Model: {best_name} ({best_acc:.2%}) saved successfully!")
