Hand Gesture Controlled Media Player

A real-time **hand gesture–based media control system** that allows users to control **system volume and media playback** (Play/Pause, Next, Previous) using simple hand gestures captured via a webcam.

Built using **Python, OpenCV, MediaPipe, and Pycaw**, this project focuses on **usability, gesture stability, and real-world compatibility (YouTube, Spotify, VLC)**.

---

Features

- **Real-time Volume Control**
  - Adjust system volume using thumb–index finger distance
  - Supports full range (0% – 100%)

- **Volume Lock**
  - Lock the volume at the desired level
  - Prevents accidental volume changes (no unintended mute)

- **Play / Pause Control**
  - Open-palm gesture with a **gesture confidence bar**
  - Prevents false triggers

- **Next / Previous Video (YouTube Compatible)**
  - Gesture-based navigation using platform-specific shortcuts
  - Works reliably on YouTube using `Shift + N` / `Shift + P`

- **Gesture Confidence Mechanism**
  - Actions trigger only after consistent detection across multiple frames
  - Improves stability and user experience

---

## System Design Overview

1. Webcam captures real-time video  
2. MediaPipe detects hand landmarks  
3. Finger states and distances are analyzed  
4. Gesture confidence is calculated  
5. Corresponding media action is triggered  

---

##Gesture Mapping

| Gesture | Action |
|------|------|
| 🤏 Thumb + Index movement | Adjust Volume |
| ✊ Fist | Lock Volume |
| ☝️ Index finger only | Unlock Volume |
| ✋ Open Palm (hold) | Play / Pause |
| ✌️ Index + Middle | Next Video |
| 🤘 Index + Pinky | Previous Video |

---

##Technologies Used

- Python  
- OpenCV  
- MediaPipe (Hand Landmarker)  
- PyCaw  
- PyAutoGUI  
- NumPy  

---
## Installation & Setup

```bash
git clone https://github.com/your-username/HandGestureMediaControl.git
cd HandGestureMediaControl
pip install opencv-python mediapipe pycaw pyautogui numpy
python hand_gesture_media_control_confidence.py
```

---

## Usage Instructions

1. Open YouTube / Spotify / VLC  
2. Click once inside the media player  
3. Run the program  
4. Perform gestures in front of the webcam  
5. Press **Q** to exit  

---

## Applications

- Touchless media control  
- Smart TVs and kiosks  
- Assistive technology  
- Human–Computer Interaction systems  

---

## Future Enhancements

- Multi-hand support  
- Gesture customization  
- Application auto-detection  
- ML-based gesture classification  

---

## Author
**Ankitha**  
B.Tech – Information Technology  

---

## Acknowledgements

- MediaPipe by Google  
- OpenCV Community  
