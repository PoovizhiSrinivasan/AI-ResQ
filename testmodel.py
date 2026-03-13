import tensorflow as tf
import numpy as np
import librosa
import sys
import os

# --------------------------
# CONFIG
# --------------------------
MODEL_PATH = "ai_model/distress_cnn_model.h5"  # adjust path if needed
SAMPLE_RATE = 22050
DURATION = 3       # seconds
N_MFCC = 40
EXPECTED_WIDTH = 130  # time frames expected by model

# --------------------------
# UTILITY: extract 2D MFCC
# --------------------------
def extract_mfcc_2d(file_path, n_mfcc=N_MFCC, duration=DURATION, sr=SAMPLE_RATE, expected_width=EXPECTED_WIDTH):
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Pad or truncate to match expected width
    if mfcc.shape[1] < expected_width:
        pad_width = expected_width - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    elif mfcc.shape[1] > expected_width:
        mfcc = mfcc[:, :expected_width]
    
    # Add batch and channel dimensions
    mfcc = mfcc[np.newaxis, :, :, np.newaxis]  # shape (1, 40, 130, 1)
    return mfcc

# --------------------------
# LOAD MODEL
# --------------------------
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    sys.exit(1)

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# --------------------------
# PREDICTION FUNCTION
# --------------------------
def predict_audio(file_path):
    mfcc = extract_mfcc_2d(file_path)
    prediction = model.predict(mfcc)
    
    # Determine if softmax or sigmoid
    if prediction.shape[1] == 1:
        # Sigmoid output
        score = float(prediction[0][0])
        label = "distress" if score > 0.5 else "non_distress"
    elif prediction.shape[1] == 2:
        # Softmax output
        predicted_class = np.argmax(prediction[0])
        score = float(prediction[0][predicted_class])
        # Adjust mapping depending on training label order
        label = "distress" if predicted_class == 0 else "non_distress"
    else:
        print("❌ Unexpected model output shape:", prediction.shape)
        return
    
    print(f"File: {os.path.basename(file_path)}")
    print(f"Prediction: {label}")
    print(f"Confidence score: {score:.4f}\n")

# --------------------------
# MAIN: test audio file(s)
# --------------------------
if len(sys.argv) < 2:
    print("Usage: python testmodel.py <audio_file1> [<audio_file2> ...]")
    sys.exit(1)

for file_path in sys.argv[1:]:
    if os.path.exists(file_path):
        predict_audio(file_path)
    else:
        print(f"❌ File not found: {file_path}")