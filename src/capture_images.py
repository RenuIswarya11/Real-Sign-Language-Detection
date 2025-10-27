import cv2
import os

# --- Configuration ---
GESTURES = ["hello", "thank_you", "yes", "no", "i_love_you"]
DATASET_PATH = "../dataset"
IMAGES_PER_GESTURE = 5  # You can increase later if needed

# Create folders if not exist
for gesture in GESTURES:
    os.makedirs(os.path.join(DATASET_PATH, gesture), exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot open webcam.")
    exit()

print("\n✅ Webcam opened successfully.")
print("Press 's' to save an image, 'n' to go to next gesture, and 'q' to quit.\n")

for gesture in GESTURES:
    print(f"👉 Capturing images for gesture: {gesture.upper()}")
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # Flip image for mirror effect
        frame = cv2.flip(frame, 1)

        # Draw rectangle to help align your hand
        cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {gesture} | Images: {count}/{IMAGES_PER_GESTURE}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Capture Hand Gestures", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # Save image
            img_name = os.path.join(DATASET_PATH, gesture, f"{count}.jpg")
            cropped = frame[100:400, 100:400]
            cv2.imwrite(img_name, cropped)
            print(f"📸 Saved: {img_name}")
            count += 1

            if count >= IMAGES_PER_GESTURE:
                print(f"✅ Completed {gesture} ({IMAGES_PER_GESTURE} images)")
                break

        elif key == ord('n'):
            print("➡️ Skipping to next gesture.")
            break

        elif key == ord('q'):
            print("❌ Quitting capture.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()
print("\n🎉 Image capture completed for all gestures!")
