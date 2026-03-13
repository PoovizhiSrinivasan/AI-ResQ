import os
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from scipy.io.wavfile import write
import time

# =============================
# CONFIG
# =============================

SAMPLE_RATE = 22050
RECORD_SECONDS = 3
TEMP_FILE = "temp_audio.wav"
THRESHOLD = 0.90

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ai_model", "distress_cnn_model.h5")

# =============================
# LOAD MODEL
# =============================

print("\n🔹 Loading AI ResQ Model...")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully\n")
except Exception as e:
    print("❌ Model loading failed:", e)
    exit()


# =============================
# FEATURE EXTRACTION
# =============================

def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    # Remove silence
    audio, _ = librosa.effects.trim(audio)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    # Normalize MFCC
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

    # Ensure fixed width
    if mfcc.shape[1] < 130:
        pad = 130 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode="constant")
    else:
        mfcc = mfcc[:, :130]

    # CNN input shape
    mfcc = mfcc.reshape(1, 40, 130, 1)

    return mfcc


# =============================
# AUDIO RECORDING
# =============================

def record_audio():

    print("🎤 Listening...")

    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    write(TEMP_FILE, SAMPLE_RATE, audio)

    return TEMP_FILE


# =============================
# PREDICTION
# =============================

def predict_audio(file_path):

    features = extract_features(file_path)

    prediction = model.predict(features, verbose=0)

    confidence = float(prediction[0][0])

    print(f"Raw Prediction: {confidence:.3f}")

    if confidence >= THRESHOLD:
        print(f"🚨 DISTRESS DETECTED | Confidence: {confidence:.2f}\n")
    else:
        print(f"✅ Normal Sound | Confidence: {confidence:.2f}\n")


# =============================
# MAIN LOOP
# =============================

print("=================================")
print("  AI ResQ Live Monitoring Started")
print("=================================\n")

while True:

    try:

        audio_file = record_audio()

        predict_audio(audio_file)

        time.sleep(1)

    except KeyboardInterrupt:
        print("\n🛑 Monitoring Stopped")
        break

    except Exception as e:
        print("⚠ Error:", e)