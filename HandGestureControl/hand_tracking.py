import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pycaw.pycaw import AudioUtilities

# ---------------- AUDIO SETUP ----------------
devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume

min_vol, max_vol = volume.GetVolumeRange()[:2]

# ---------------- MEDIAPIPE SETUP ----------------
model_path = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# Distance calibration (VERY IMPORTANT)
MIN_DIST = 30    # fingers almost touching
MAX_DIST = 170   # fingers fully apart

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        h, w, _ = frame.shape

        # Thumb tip (4) and Index tip (8)
        x1, y1 = int(hand[4].x * w), int(hand[4].y * h)
        x2, y2 = int(hand[8].x * w), int(hand[8].y * h)

        # Draw points & line
        cv2.circle(frame, (x1, y1), 8, (0, 255, 0), -1)
        cv2.circle(frame, (x2, y2), 8, (0, 255, 0), -1)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Distance calculation
        distance = np.hypot(x2 - x1, y2 - y1)

        # Clamp distance so we ALWAYS hit min & max volume
        distance = np.clip(distance, MIN_DIST, MAX_DIST)

        # Map distance to full volume range
        vol = np.interp(distance, [MIN_DIST, MAX_DIST], [min_vol, max_vol])
        volume.SetMasterVolumeLevel(vol, None)

        # Visual volume bar
        vol_bar = np.interp(distance, [MIN_DIST, MAX_DIST], [400, 150])
        cv2.rectangle(frame, (50, 150), (85, 400), (0, 0, 255), 3)
        cv2.rectangle(frame, (50, int(vol_bar)), (85, 400), (0, 0, 255), -1)

        # Convert volume to percentage
        vol_percent = np.interp(vol, [min_vol, max_vol], [0, 100])

        cv2.putText(
            frame,
            f"Volume: {int(vol_percent)} %",
            (30, 430),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.imshow("Hand Gesture Volume Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
