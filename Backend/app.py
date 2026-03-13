from flask import Flask, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
import os
import datetime

app = Flask(__name__)

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = "distress_cnn_model.h5"

SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
EXPECTED_WIDTH = 130

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)

labels = {
    0: "distress",
    1: "non_distress"
}

# =========================
# HOME ROUTE
# =========================
@app.route("/")
def home():
    return {"message": "ResQ API is running"}

# =========================
# MFCC FEATURE EXTRACTION
# =========================
def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

    # Normalize
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    # Padding / Truncation
    if mfcc.shape[1] < EXPECTED_WIDTH:
        pad_width = EXPECTED_WIDTH - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :EXPECTED_WIDTH]

    mfcc = mfcc[np.newaxis, :, :, np.newaxis]

    return mfcc


# =========================
# INCIDENT LOGGER
# =========================
def log_incident(label, confidence, risk):

    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_entry = f"{time} | {label} | Confidence: {confidence:.2f} | Risk: {risk}\n"

    with open("incident_log.txt", "a") as file:
        file.write(log_entry)


# =========================
# PREDICTION API
# =========================
@app.route("/predict", methods=["POST"])
def predict():

    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio = request.files["audio"]

    temp_path = "temp.wav"
    audio.save(temp_path)

    try:

        features = extract_features(temp_path)

        prediction = model.predict(features)

        pred_class = np.argmax(prediction)

        label = labels[pred_class]

        confidence = float(prediction[0][pred_class])

        # ALERT SYSTEM
        alert_status = "ALERT_TRIGGERED" if label == "distress" else "SAFE"

        # RISK LEVEL
        if confidence > 0.85:
            risk_level = "HIGH"
        elif confidence > 0.65:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # LOG INCIDENT
        if label == "distress":
            log_incident(label, confidence, risk_level)

        result = {
            "prediction": label,
            "confidence": round(confidence, 3),
            "alert_status": alert_status,
            "risk_level": risk_level
        }

    except Exception as e:

        result = {"error": str(e)}

    finally:

        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify(result)


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)