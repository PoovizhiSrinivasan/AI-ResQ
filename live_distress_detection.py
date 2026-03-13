import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf

MODEL_PATH = "distress_cnn_model.h5"

SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
EXPECTED_WIDTH = 130

model = tf.keras.models.load_model(MODEL_PATH)

labels = {
0: "distress",
1: "non_distress"
}

def extract_features(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

    if mfcc.shape[1] < EXPECTED_WIDTH:
        pad_width = EXPECTED_WIDTH - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :EXPECTED_WIDTH]

    mfcc = mfcc[np.newaxis, :, :, np.newaxis]

    return mfcc


print("🎤 Listening for distress sounds...")

while True:

    audio = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1
    )

    sd.wait()

    audio = audio.flatten()

    features = extract_features(audio)

    prediction = model.predict(features)

    pred_class = np.argmax(prediction)

    label = labels[pred_class]

    confidence = prediction[0][pred_class]

    print("Prediction:", label, "| Confidence:", round(float(confidence),3))

    if label == "distress":
        print("🚨 ALERT: Distress detected!")