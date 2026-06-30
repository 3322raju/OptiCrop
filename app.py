from flask import Flask, render_template, request
import pickle
import numpy as np
import sqlite3

from crop_info import crop_data
# This creates the database and history table if they don't exist
import database.database


app = Flask(__name__)

# Load trained model
model = pickle.load(open("models/model.pkl", "rb"))


# ---------------- HOME ----------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------- ABOUT ----------------

@app.route("/about")
def about():
    return render_template("about.html")


# ---------------- DASHBOARD ----------------

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ---------------- PREDICT PAGE ----------------

@app.route("/predict")
def predict():
    return render_template("predict.html")


# ---------------- PREDICTION ----------------

@app.route("/predict", methods=["POST"])
def prediction():

    N = float(request.form["N"])
    P = float(request.form["P"])
    K = float(request.form["K"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    ph = float(request.form["ph"])
    rainfall = float(request.form["rainfall"])

    values = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    crop = model.predict(values)[0]

    # Crop Information
    info = crop_data.get(crop.lower(), {})

    # Save Prediction History
    conn = sqlite3.connect("database/history.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO history
        (
            crop,
            nitrogen,
            phosphorus,
            potassium,
            temperature,
            humidity,
            ph,
            rainfall
        )
        VALUES (?,?,?,?,?,?,?,?)
    """, (
        crop,
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall
    ))

    conn.commit()
    conn.close()

    return render_template(
        "result.html",
        prediction=crop,
        info=info
    )


# ---------------- HISTORY ----------------

@app.route("/history")
def history():

    conn = sqlite3.connect("database/history.db")

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
        crop,
        nitrogen,
        phosphorus,
        potassium,
        temperature,
        humidity,
        ph,
        rainfall,
        prediction_date
        FROM history
        ORDER BY id DESC
    """)

    rows = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        history=rows
    )


# ---------------- CONTACT ----------------

@app.route("/contact")
def contact():
    return render_template("contact.html")


# ---------------- MAIN ----------------

if __name__ == "__main__":
    app.run(debug=True)