import sys
import joblib

# === Load vectorizer & model ===
vectorizer = joblib.load("models/url_vectorizer.joblib")
model = joblib.load("models/url_best_model.joblib")

def predict_url(url):
    X = vectorizer.transform([url])
    pred = model.predict(X)[0]
    label = "SPAM" if pred == 1 else "NORMAL"
    print(f"\n URL: {url}")
    print(f" Prediction: {label}")

if len(sys.argv) > 1:
    # Command-line mode
    url = sys.argv[1]
    predict_url(url)
else:
    # Interactive mode
    print(" URL Spam Detector — type 'exit' to quit\n")
    while True:
        url = input("Enter a URL: ").strip()
        if url.lower() == "exit":
            print(" Goodbye!")
            break
        if not url.startswith("http"):
            print(" Please enter a valid URL (starting with http/https)")
            continue
        predict_url(url)
