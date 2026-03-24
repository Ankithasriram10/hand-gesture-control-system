import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from pycaw.pycaw import AudioUtilities
import pyautogui
import time

# ---------------- SETTINGS ----------------
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ---------------- AUDIO ----------------
devices = AudioUtilities.GetSpeakers()
volume = devices.EndpointVolume
min_vol, max_vol = volume.GetVolumeRange()[:2]

# ---------------- MEDIAPIPE ----------------
model_path = "hand_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# ---------------- CONFIG ----------------
MIN_DIST = 30
MAX_DIST = 170

volume_locked = False
locked_volume = None

last_action_time = 0
ACTION_DELAY = 1.2   # seconds

# Gesture confidence
gesture_frames = 0
CONFIDENCE_FRAMES = 15   # ~0.5 sec

# ---------------- MAIN LOOP ----------------
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

        # -------- FINGER STATES --------
        fingers = []

        # Thumb
        fingers.append(1 if hand[4].x > hand[3].x else 0)

        # Other fingers
        finger_tips = [8, 12, 16, 20]
        finger_joints = [6, 10, 14, 18]

        for tip, joint in zip(finger_tips, finger_joints):
            fingers.append(1 if hand[tip].y < hand[joint].y else 0)

        # -------- LOCK / UNLOCK VOLUME --------
        if fingers == [0, 0, 0, 0, 0]:   # FIST
            volume_locked = True

        if fingers == [0, 1, 0, 0, 0]:   # INDEX ONLY
            volume_locked = False

        # -------- VOLUME CONTROL --------
        if not volume_locked:
            x1, y1 = int(hand[4].x * w), int(hand[4].y * h)
            x2, y2 = int(hand[8].x * w), int(hand[8].y * h)

            distance = np.hypot(x2 - x1, y2 - y1)
            distance = np.clip(distance, MIN_DIST, MAX_DIST)

            vol = np.interp(distance, [MIN_DIST, MAX_DIST], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)

            locked_volume = vol
            vol_percent = int(np.interp(vol, [min_vol, max_vol], [0, 100]))

            cv2.putText(
                frame, f"Volume: {vol_percent} %",
                (30, 430), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255,255,255), 2
            )

        # -------- KEEP LOCKED VOLUME --------
        if volume_locked and locked_volume is not None:
            volume.SetMasterVolumeLevel(locked_volume, None)
            cv2.putText(
                frame, "VOLUME LOCKED",
                (300, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,0,255), 3
            )

        current_time = time.time()

        # -------- PLAY / PAUSE (WITH CONFIDENCE BAR) --------
        if fingers == [1, 1, 1, 1, 1]:
            gesture_frames += 1
        else:
            gesture_frames = 0

        # Draw confidence bar
        bar_width = int((gesture_frames / CONFIDENCE_FRAMES) * 200)
        bar_width = min(bar_width, 200)

        cv2.rectangle(frame, (30, 460), (230, 480), (255,255,255), 2)
        cv2.rectangle(frame, (30, 460), (30 + bar_width, 480), (0,255,0), -1)

        cv2.putText(
            frame, "Play/Pause Confidence",
            (30, 450), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (255,255,255), 2
        )

        if gesture_frames >= CONFIDENCE_FRAMES:
            if current_time - last_action_time > ACTION_DELAY:
                pyautogui.press('space')
                last_action_time = current_time
            gesture_frames = 0

        # -------- NEXT VIDEO (YouTube: Shift + N) --------
        if volume_locked and fingers == [0, 1, 1, 0, 0]:
            if current_time - last_action_time > ACTION_DELAY:
                pyautogui.hotkey('shift', 'n')
                last_action_time = current_time
                cv2.putText(
                    frame, "NEXT VIDEO",
                    (300, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,255,0), 3
                )

        # -------- PREVIOUS VIDEO (YouTube: Shift + P) --------
        if volume_locked and fingers == [0, 1, 0, 0, 1]:
            if current_time - last_action_time > ACTION_DELAY:
                pyautogui.hotkey('shift', 'p')
                last_action_time = current_time
                cv2.putText(
                    frame, "PREVIOUS VIDEO",
                    (300, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,255,0), 3
                )

        # -------- DRAW LANDMARKS --------
        for lm in hand:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0,255,0), -1)

    cv2.imshow("Hand Gesture Media Control (Confidence Mode)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()