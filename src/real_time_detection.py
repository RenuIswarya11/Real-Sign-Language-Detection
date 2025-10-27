import cv2
import numpy as np
import os
import json
import mediapipe as mp
from tensorflow.keras.models import load_model

# --- Paths ---
MODEL_PATH = "../models"
DATA_PATH = "../data"

# --- Load model and scaler ---
model = load_model(os.path.join(MODEL_PATH, "final_model.h5"))
scaler_mean = np.load(os.path.join(MODEL_PATH, "scaler_mean.npy"))
scaler_scale = np.load(os.path.join(MODEL_PATH, "scaler_scale.npy"))

# --- Load label map ---
with open(os.path.join(DATA_PATH, "label_map.json")) as f:
    labels = json.load(f)

# --- Initialize MediaPipe Hands ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# --- Start webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

print("🎥 Webcam opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    frame = cv2.flip(frame, 1)  # mirror
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmarks
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            # Normalize and predict
            X_input = (np.array(coords).reshape(1, -1) - scaler_mean) / scaler_scale
            prediction = model.predict(X_input, verbose=0)
            class_id = np.argmax(prediction)
            class_name = labels[class_id]
            confidence = prediction[0][class_id]

            # Display
            cv2.putText(frame, f"{class_name} ({confidence*100:.1f}%)",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Real-Time Sign Language Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
