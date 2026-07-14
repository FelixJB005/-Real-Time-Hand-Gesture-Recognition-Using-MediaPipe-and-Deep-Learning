import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# ======================================================
# Load Dataset
# ======================================================

df = pd.read_csv("outputs/dataset_landmarks_clean.csv")

print("Original Dataset Shape:", df.shape)

# ======================================================
# Features and Labels
# ======================================================

X = df.drop("label", axis=1).values
y = df["label"].values

# ======================================================
# Encode Labels
# (Fixes missing class numbers like label 22)
# ======================================================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

num_classes = len(np.unique(y))

print("\nNumber of classes:", num_classes)

# ======================================================
# Scale Features
# ======================================================

scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler and encoder
os.makedirs("model", exist_ok=True)

joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")

# ======================================================
# One-hot Encoding
# ======================================================

y = to_categorical(y, num_classes=num_classes)

# ======================================================
# Train Test Split
# ======================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=np.argmax(y, axis=1)
)

# ======================================================
# Neural Network
# ======================================================

model = Sequential([
    Input(shape=(63,)),

    Dense(256, activation="relu"),
    Dropout(0.30),

    Dense(128, activation="relu"),
    Dropout(0.30),

    Dense(64, activation="relu"),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ======================================================
# Training
# ======================================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    verbose=1
)

# ======================================================
# Evaluation
# ======================================================

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\n==============================")
print(f"Test Accuracy : {accuracy*100:.2f}%")
print("==============================")

# ======================================================
# Save Model
# ======================================================

model.save("model/sign_model.keras")

print("\nModel Saved!")
print("model/sign_model.keras")
print("model/scaler.pkl")
print("model/label_encoder.pkl")