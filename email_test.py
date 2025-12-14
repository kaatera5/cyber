import sys
import joblib


vectorizer = joblib.load("models/email_vectorizer.joblib")
model = joblib.load("models/email_best_model.joblib")

def predict_email(email):
    # Vectorize the input email
    X = vectorizer.transform([email])
    # Get probability (if supported)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0][1]
    else:
        prob = 0.0
    pred = model.predict(X)[0]

    label = "SPAM" if pred == 1 else "NORMAL"
    print(f"\n Email: {email}")
    print(f" Prediction: {label}  ")

# === Command-line mode ===
if len(sys.argv) > 1:
    email = sys.argv[1]
    predict_email(email)

# === Interactive mode ===
else:
    print(" Email Spam Detector — type 'exit' to quit\n")
    while True:
        email = input("Enter an email address: ").strip()
        if email.lower() == "exit":
            print(" Goodbye!")
            break
        if "@" not in email:
            print(" Invalid email format. Try again.")
            continue
        predict_email(email)
