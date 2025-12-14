
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash
import os
import requests, time
import json
from datetime import datetime
import numpy as np
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


app = Flask(__name__)
app.secret_key = 'wverihdfuvuwi2482'



def get_db_connection():
    conn = sqlite3.connect('cyber_db') 
    conn.row_factory = sqlite3.Row  
    return conn


def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()
    
   
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            number TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    cursor.close()
    conn.close()

create_tables()




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        number = request.form['number']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                'INSERT INTO users (name, email, number, password) VALUES (?, ?, ?, ?)',
                (name, email, number, hashed_password)
            )
            conn.commit()
            flash('Registration successful. Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already exists.', 'danger')
        finally:
            cursor.close()
            conn.close()

    return render_template('register.html')





@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()

        if user and check_password_hash(user['password'], password):
            session['email'] = user['email']
            session['name'] = user['name']
            flash('Login successful!', 'success')

            return redirect(url_for('index'))
        else:
            flash('Invalid email or password', 'danger')

    return render_template('login.html')

import joblib

@app.route("/email", methods=["GET", "POST"])
def email():
    if 'email' not in session:
        flash("Please log in to access this page.", "warning")
        return redirect(url_for('login'))
    result = None
    email_input = ""

    if request.method == "POST":
        email_input = request.form.get("email_content", "")

        if email_input.strip():
          
            email_vectorizer = joblib.load("models/email_vectorizer.joblib")
            email_model = joblib.load("models/email_best_model.joblib")

           
            X = email_vectorizer.transform([email_input])
            pred = email_model.predict(X)[0]
            prob = email_model.predict_proba(X)[0][1] if hasattr(email_model, "predict_proba") else 0.0

            label = "SPAM" if pred == 1 else "NORMAL"
            result = {"label": label, "prob": round(prob * 100, 2)}

    return render_template("email.html", result=result, email_content=email_input)


@app.route("/message", methods=["GET", "POST"])
def message():
  
    if 'email' not in session:
        flash("Please log in to access this page.", "warning")
        return redirect(url_for('login'))

    result = None
    message_text = ""

    if request.method == "POST":
        message_text = request.form.get("message_text", "")

        if message_text.strip():
           
            sms_vectorizer = joblib.load("models/sms_vectorizer.joblib")
            sms_model = joblib.load("models/sms_best_model.joblib")

        
            X = sms_vectorizer.transform([message_text])
            prob = sms_model.predict_proba(X)[0][1]
            label = "SPAM" if prob > 0.5 else "HAM"

            result = {"label": label, "prob": round(prob * 100, 2)}

    return render_template("message.html", result=result, message_text=message_text)



@app.route("/number", methods=["GET", "POST"])
def number():
  
    if 'email' not in session:
        flash("Please log in to access this page.", "warning")
        return redirect(url_for('login'))
    
    result = None
    number_input = ""

    if request.method == "POST":
        number_input = request.form.get("phone_number", "").strip()

        if number_input and number_input.isdigit():
            # === Load model and scaler INSIDE the route ===
            number_model = joblib.load("models/number_best_model.joblib")
            number_scaler = joblib.load("models/number_scaler.joblib")

            # === Feature extraction (inline) ===
            digits = [int(d) for d in number_input if d.isdigit()]
            mean_digit = np.mean(digits)
            std_digit = np.std(digits)
            unique_digits = len(set(digits))
            repeated_pairs = sum(1 for i in range(1, len(digits)) if digits[i] == digits[i-1])
            increasing_seq = sum(1 for i in range(1, len(digits)) if digits[i] > digits[i-1])
            decreasing_seq = sum(1 for i in range(1, len(digits)) if digits[i] < digits[i-1])
            digit_sum = sum(digits)
            zeros = number_input.count('0')
            nines = number_input.count('9')
            counts = [digits.count(i) for i in range(10)]
            max_freq = max(counts)
            min_freq = min(counts)
            entropy = -sum((c/len(digits)) * np.log2(c/len(digits)) for c in counts if c > 0)
            
            # === Combine features ===
            X = np.array([[  
                len(number_input),
                unique_digits,
                mean_digit, std_digit,
                repeated_pairs,
                increasing_seq,
                decreasing_seq,
                zeros, nines,
                digit_sum,
                max_freq, min_freq,
                entropy,
                int(number_input[0]),
                int(number_input[-1])
            ]])

            # === Scale & Predict ===
            X_scaled = number_scaler.transform(X)
            pred = number_model.predict(X_scaled)[0]
            label = "SPAM" if pred == 1 else "NORMAL"

            result = {"label": label}

        elif number_input:
            result = {"error": "Please enter a valid numeric phone number."}

    return render_template("number.html", result=result, number_input=number_input)


@app.route("/url", methods=["GET", "POST"])
def url():
    # === Require login ===
    if 'email' not in session:
        flash("Please log in to access this page.", "warning")
        return redirect(url_for('login'))

    result = None
    url_input = ""

    if request.method == "POST":
        url_input = request.form.get("url_text", "").strip()

        if url_input:
           
            url_vectorizer = joblib.load("models/url_vectorizer.joblib")
            url_model = joblib.load("models/url_best_model.joblib")

            # === Predict ===
            X = url_vectorizer.transform([url_input])
            pred = url_model.predict(X)[0]
            label = "SPAM" if pred == 1 else "NORMAL"

            result = {"label": label}

        elif url_input == "":
            result = {"error": "Please enter a URL."}

    return render_template("url.html", result=result, url_input=url_input)


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)