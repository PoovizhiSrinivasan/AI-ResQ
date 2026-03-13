import os
import numpy as np
import librosa

# --------------------------
# CONFIG
# --------------------------
DATASET_PATH = r"C:\Users\poovi\AI ResQ\dataset_organized"  # updated dataset path
N_MFCC = 40
DURATION = 3          # seconds
SAMPLE_RATE = 22050
EXPECTED_WIDTH = 130  # time frames for CNN
DISTRESS_FOLDER = "distress"
NON_DISTRESS_FOLDER = "non_distress"

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
    
    # Add channel dimension
    mfcc = mfcc[:, :, np.newaxis]  # shape (n_mfcc, width, 1)
    return mfcc

# --------------------------
# LOOP THROUGH DATASET
# --------------------------
X = []
y = []

for label, folder in enumerate([DISTRESS_FOLDER, NON_DISTRESS_FOLDER]):
    folder_path = os.path.join(DATASET_PATH, folder)
    if not os.path.exists(folder_path):
        print(f"⚠️ Folder not found: {folder_path}")
        continue
    
    for file in os.listdir(folder_path):
        if file.lower().endswith((".wav", ".mp3", ".ogg")):
            file_path = os.path.join(folder_path, file)
            try:
                mfcc = extract_mfcc_2d(file_path)
                X.append(mfcc)
                y.append(label)  # distress=0, non_distress=1
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"✅ Extracted MFCCs: X.shape={X.shape}, y.shape={y.shape}")

# --------------------------
# SAVE FEATURES AND LABELS
# --------------------------
np.save("X.npy", X)
np.save("y.npy", y)
print("✅ Saved X.npy and y.npy for training!")