import os
import cv2
import json
import numpy as np
import mediapipe as mp
from tqdm import tqdm

# --- Configuration ---
DATASET_PATH = "../dataset"  # path to your dataset folder
OUTPUT_PATH = "../data"      # where X.npy, y.npy will be saved
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                       min_detection_confidence=0.5)

# Prepare lists
X = []
y = []
labels = []

# Get gesture folders
gesture_folders = sorted([f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))])
labels = gesture_folders  # store class names

print("📂 Processing gestures:", labels)

for idx, gesture in enumerate(gesture_folders):
    folder_path = os.path.join(DATASET_PATH, gesture)
    images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    print(f"\n👉 Processing gesture '{gesture}' ({len(images)} images)")

    for img_name in tqdm(images):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if not results.multi_hand_landmarks:
            # Skip images where hand is not detected
            continue

        hand_landmarks = results.multi_hand_landmarks[0]
        coords = []
        for lm in hand_landmarks.landmark:
            coords.extend([lm.x, lm.y, lm.z])  # 21 points x 3 = 63
        X.append(coords)
        y.append(idx)

# Convert to NumPy arrays
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# Save features and labels
np.save(os.path.join(OUTPUT_PATH, "X.npy"), X)
np.save(os.path.join(OUTPUT_PATH, "y.npy"), y)

# Save label map
with open(os.path.join(OUTPUT_PATH, "label_map.json"), "w") as f:
    json.dump(labels, f)

print(f"\n✅ Landmarks extracted and saved!")
print(f"X.npy shape: {X.shape}")
print(f"y.npy shape: {y.shape}")
print(f"Label map saved: {labels}")
