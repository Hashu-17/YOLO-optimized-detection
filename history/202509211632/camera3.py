import cv2
import os

# Open Pi camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access Pi Camera")
    exit()

# Capture one frame
ret, frame = cap.read()
cap.release()

if ret:
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Path to Desktop
    desktop_path = os.path.expanduser("~/Desktop/pi_camera_gray.jpg")

    # Save image
    cv2.imwrite(desktop_path, gray)
    print(f"Saved grayscale image at: {desktop_path}")
else:
    print("Failed to capture image")

