import sys
import joblib
import numpy as np

# === Load Model and Scaler ===
model = joblib.load("models/number_best_model.joblib")
scaler = joblib.load("models/number_scaler.joblib")

# === Feature Extraction  ===
def extract_features(number):
    number = str(number)
    digits = [int(d) for d in number if d.isdigit()]
    if not digits:
        digits = [0]
    
    mean_digit = np.mean(digits)
    std_digit = np.std(digits)
    unique_digits = len(set(digits))
    repeated_pairs = sum(1 for i in range(1, len(digits)) if digits[i] == digits[i-1])
    increasing_seq = sum(1 for i in range(1, len(digits)) if digits[i] > digits[i-1])
    decreasing_seq = sum(1 for i in range(1, len(digits)) if digits[i] < digits[i-1])
    digit_sum = sum(digits)
    zeros = number.count('0')
    nines = number.count('9')
    counts = [digits.count(i) for i in range(10)]
    max_freq = max(counts)
    min_freq = min(counts)
    entropy = -sum((c/len(digits)) * np.log2(c/len(digits)) for c in counts if c > 0)
    
    return np.array([[  # must be 2D for scaler
        len(number),
        unique_digits,
        mean_digit, std_digit,
        repeated_pairs,
        increasing_seq,
        decreasing_seq,
        zeros, nines,
        digit_sum,
        max_freq, min_freq,
        entropy,
        int(number[0]),
        int(number[-1])
    ]])

# === Prediction Function ===
def predict_number(phone_number):
    X = extract_features(phone_number)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    label = "SPAM" if pred == 1 else "NORMAL"
    print(f"\n Number: {phone_number}")
    print(f" Prediction: {label}")

# === Run Mode ===
if len(sys.argv) > 1:
    # Command-line mode
    phone_number = sys.argv[1]
    predict_number(phone_number)
else:
    # Interactive mode
    print(" Phone Number Spam Detector")
    print("Type 'exit' to quit.\n")
    while True:
        phone_number = input("Enter phone number: ").strip()
        if phone_number.lower() == "exit":
            print(" Goodbye!")
            break
        if not phone_number.isdigit() or len(phone_number) < 5:
            print(" Please enter a valid numeric phone number.")
            continue
        predict_number(phone_number)
