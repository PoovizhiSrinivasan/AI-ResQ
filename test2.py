import numpy as np
import librosa
from tensorflow.keras.models import load_model

MODEL_PATH = "ai_model/distress_cnn_model.h5"
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
EXPECTED_WIDTH = 130

# Load model
model = load_model(MODEL_PATH)

def predict_audio(file_path):
    # Load and preprocess audio
    y, _ = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC)
    
    # Pad/truncate
    if mfcc.shape[1] < EXPECTED_WIDTH:
        pad_width = EXPECTED_WIDTH - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :EXPECTED_WIDTH]
    
    mfcc = mfcc[np.newaxis, :, :, np.newaxis]  # shape (1, 40, 130, 1)
    
    # Predict
    probs = model.predict(mfcc)
    pred_class = np.argmax(probs)
    label = "distress" if pred_class == 0 else "non-distress"
    confidence = probs[0][pred_class]
    
    return label, confidence

# Example usage
file_path = "dataset_organized/distress/distress_010.wav"
label, confidence = predict_audio(file_path)
print(f"Prediction: {label}, Confidence: {confidence:.4f}")