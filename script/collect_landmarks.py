import os
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# =====================================================
# PATHS
# =====================================================

DATASET_PATH = r"archive/Gesture Image Data"

OUTPUT_CSV = "outputs/dataset_landmarks.csv"
LABEL_FILE = "model/labels.txt"

os.makedirs("outputs", exist_ok=True)
os.makedirs("model", exist_ok=True)

# =====================================================
# MEDIAPIPE
# =====================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# =====================================================
# FIND CLASSES
# =====================================================

classes = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

print("=" * 60)
print("Classes Found")
print("=" * 60)

for i, c in enumerate(classes):
    print(f"{i:2d} -> {c}")

with open(LABEL_FILE, "w") as f:
    for i, c in enumerate(classes):
        f.write(f"{i} {c}\n")

print("=" * 60)

# =====================================================
# EXTRACTION
# =====================================================

rows = []

total = 0
success = 0
failed = 0

for label_index, folder in enumerate(classes):

    folder_path = os.path.join(DATASET_PATH, folder)

    print(f"\nProcessing [{folder}]...")

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            failed += 1
            continue

        total += 1

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if not result.multi_hand_landmarks:
            failed += 1
            continue

        hand = result.multi_hand_landmarks[0]

        landmarks = []

        for lm in hand.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks)

        # ================================
        # NORMALIZATION
        # ================================

        landmarks = landmarks - landmarks[0]

        scale = np.max(np.abs(landmarks))

        if scale != 0:
            landmarks = landmarks / scale

        row = [label_index]

        for point in landmarks:
            row.extend(point)

        rows.append(row)

        success += 1

        if success % 500 == 0:
            print(f"Processed {success} images...")

# =====================================================
# SAVE CSV
# =====================================================

columns = ["label"]

for i in range(21):
    columns.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

df = pd.DataFrame(rows, columns=columns)

print("\nRemoving duplicate rows...")

before = len(df)

df = df.drop_duplicates()

after = len(df)

print(f"Removed {before-after} duplicates.")

df.to_csv(OUTPUT_CSV, index=False)

print("\n" + "=" * 60)
print("DATASET EXTRACTION COMPLETE")
print("=" * 60)

print(f"Total Images   : {total}")
print(f"Successful     : {success}")
print(f"Failed         : {failed}")

print(f"Detection Rate : {(success/total)*100:.2f}%")

print("\nDataset saved to")

print(OUTPUT_CSV)

print("\nLabels saved to")

print(LABEL_FILE)