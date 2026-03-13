import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG
# =========================

DATASET_PATH = "dataset_organized"
SAMPLE_RATE = 22050
DURATION = 3
N_MFCC = 40
EXPECTED_WIDTH = 130

# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(file_path):
    
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)

        # normalize
        mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)

        # pad or truncate
        if mfcc.shape[1] < EXPECTED_WIDTH:
            pad_width = EXPECTED_WIDTH - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :EXPECTED_WIDTH]

        return mfcc

    except:
        return None


# =========================
# LOAD DATASET
# =========================

X = []
y = []

labels = {
    "distress":0,
    "non_distress":1
}

for label in labels:

    folder = os.path.join(DATASET_PATH, label)

    for file in os.listdir(folder):

        if file.endswith(".wav"):

            file_path = os.path.join(folder, file)

            features = extract_features(file_path)

            if features is not None:

                X.append(features)
                y.append(labels[label])


X = np.array(X)
y = np.array(y)

# reshape for CNN
X = X[..., np.newaxis]

print("Dataset shape:", X.shape)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# CLASS WEIGHTS (Fix imbalance)
# =========================

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))

print("Class weights:", class_weights)

# =========================
# CNN MODEL
# =========================

model = Sequential([

    Input(shape=(40,130,1)),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# TRAIN MODEL
# =========================

history = model.fit(

    X_train,
    y_train,

    epochs=20,
    batch_size=16,

    validation_split=0.2,

    class_weight=class_weights
)

# =========================
# SAVE MODEL
# =========================

model.save("distress_cnn_model.h5")

print("\n✅ Model saved as distress_cnn_model.h5")

# =========================
# EVALUATE MODEL
# =========================

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

print("\nConfusion Matrix:\n", cm)

# =========================
# HEATMAP
# =========================

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Distress","Non Distress"],
    yticklabels=["Distress","Non Distress"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()