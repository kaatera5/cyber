import joblib


model = joblib.load("models/sms_best_model.joblib")
vectorizer = joblib.load("models/sms_vectorizer.joblib")

def predict_message(text):
    X = vectorizer.transform([text])
    prob = model.predict_proba(X)[0][1]
    label = "SPAM" if prob > 0.5 else "HAM"
    print(f"\n Message: {text}")
    print(f"Prediction: {label} ")

print(" SMS / Message Spam Detector — type 'exit' to quit\n")
while True:
    msg = input("Enter message: ").strip()
    if msg.lower() == "exit":
        break
    predict_message(msg)
