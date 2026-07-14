import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# ============================
# Load Model
# ============================

model = tf.keras.models.load_model("model/sign_model.keras")

# Load labels
with open("model/labels.txt", "r") as f:
    labels = [line.strip().split()[-1] for line in f]

# ============================
# MediaPipe Setup
# ============================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================
# Webcam
# ============================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press Q to quit.")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    prediction = ""
    confidence = 0

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = []

            for lm in hand_landmarks.landmark:

                landmarks.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            landmarks = np.array(landmarks).reshape(1, 63)

            pred = model.predict(landmarks, verbose=0)

            class_id = np.argmax(pred)

            confidence = np.max(pred)

            prediction = labels[class_id]

    cv2.putText(
        frame,
        f"Prediction : {prediction}",
        (10,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence : {confidence*100:.2f}%",
        (10,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,0,0),
        2
    )

    cv2.imshow("Hand Gesture Recognition", frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()