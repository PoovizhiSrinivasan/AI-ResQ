import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

DATASET_PATH = "dataset"

X = []
y = []

labels = {}
label_index = 0


# =============================
# RENAME AUDIO FILES
# =============================

print("Renaming audio files...\n")

for main_folder in os.listdir(DATASET_PATH):

    main_path = os.path.join(DATASET_PATH, main_folder)

    for subfolder in os.listdir(main_path):

        sub_path = os.path.join(main_path, subfolder)

        count = 1

        for file in os.listdir(sub_path):

            if file.endswith(".wav"):

                old_path = os.path.join(sub_path, file)

                new_name = f"{subfolder}_{count}.wav"

                new_path = os.path.join(sub_path, new_name)

                os.rename(old_path, new_path)

                count += 1

print("Renaming complete\n")


# =============================
# FEATURE EXTRACTION
# =============================

def extract_features(file):

    audio, sr = librosa.load(file, sr=22050)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)

    if mfcc.shape[1] < 130:
        pad = 130 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0),(0,pad)), mode='constant')
    else:
        mfcc = mfcc[:, :130]

    return mfcc


# =============================
# LOAD DATASET
# =============================

print("Loading dataset...\n")

for main_folder in os.listdir(DATASET_PATH):

    main_path = os.path.join(DATASET_PATH, main_folder)

    for subfolder in os.listdir(main_path):

        sub_path = os.path.join(main_path, subfolder)

        if subfolder not in labels:
            labels[subfolder] = label_index
            label_index += 1

        for file in os.listdir(sub_path):

            if file.endswith(".wav"):

                file_path = os.path.join(sub_path, file)

                features = extract_features(file_path)

                X.append(features)

                y.append(labels[subfolder])


X = np.array(X)
y = np.array(y)

X = X.reshape(X.shape[0], 40, 130, 1)

print("Dataset Loaded")
print("Total Samples:", len(X))
print("Classes:", labels)


# =============================
# TRAIN TEST SPLIT
# =============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =============================
# CNN MODEL
# =============================

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(40,130,1)))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(len(labels), activation='softmax'))


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


print("\nTraining model...\n")

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test)
)


# =============================
# SAVE MODEL
# =============================

model.save("distress_sound_classifier.h5")

print("\nModel Saved Successfully!")