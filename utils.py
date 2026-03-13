# utils.py
import librosa
import numpy as np

def extract_mfcc(file_path, n_mfcc=40, duration=3, sr=22050):
    y, _ = librosa.load(file_path, sr=sr, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc = np.mean(mfcc.T, axis=0)  # flatten to 1D
    return mfcc