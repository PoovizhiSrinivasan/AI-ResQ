from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import librosa
import os

app = Flask(__name__)

# =========================
# CONFIG
# =========================
MODEL_PATH = "distress_cnn_model.h5"
SAMPLE_RATE = 22050
DURATION = 3          # seconds
N_MFCC = 40
THRESHOLD = 0.5       # keep 0.5 since model outputs inverted labels

model = tf.keras.models.load_model(MODEL_PATH)

# =========================
# FEATURE EXTRACTION
# =========================
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    # normalize (VERY important)
    y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC
    )

    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc.reshape(1, -1)

# =========================
# API ENDPOINT
# =========================
@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["file"]
    file_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(file_path)

    try:
        features = extract_mfcc(file_path)

        # raw model output
        prediction = model.predict(features)[0][0]
        confidence = float(prediction)

        # 🔥 FIXED LABEL INTERPRETATION
        # Based on your observed behavior:
        #   0.0001  → DISTRESS
        #   ~1.0    → NON_DISTRESS
        if confidence < THRESHOLD:
            label = "DISTRESS"
        else:
            label = "NON_DISTRESS"

        return jsonify({
            "confidence": round(confidence, 4),
            "label": label
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
                                