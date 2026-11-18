import math
import time
import cv2
import mediapipe as mp
import numpy as np
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# 1. Initialize Webcam
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# 2. Initialize MediaPipe Hand Detector
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
mpDraw = mp.solutions.drawing_utils

# 3. Initialize Pycaw (Windows Audio Control) - Updated for latest pycaw
device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume

# Get Volume Range (typically min: -63.5, max: 0.0)
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]

volBar = 400
volPer = 0
pTime = 0

while True:
  success, img = cap.read()
  if not success:
    break

  # Convert BGR image to RGB for MediaPipe
  imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  results = hands.process(imgRGB)

  lmList = []
  if results.multi_hand_landmarks:
    for handLms in results.multi_hand_landmarks:
      for id, lm in enumerate(handLms.landmark):
        # Get pixel coordinates
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])

      mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

  if len(lmList) != 0:
    # Thumb tip is landmark 4, Index finger tip is landmark 8
    x1, y1 = lmList[4][1], lmList[4][2]
    x2, y2 = lmList[8][1], lmList[8][2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Draw circles on fingertips and a line between them
    cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
    cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
    cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

    # Calculate distance between thumb and index finger
    length = math.hypot(x2 - x1, y2 - y1)

    # Hand range typically varies from 50 (closed) to 300 (fully open)
    vol = np.interp(length, [50, 300], [minVol, maxVol])
    volBar = np.interp(length, [50, 300], [400, 150])
    volPer = np.interp(length, [50, 300], [0, 100])

    # Set system volume
    volume.SetMasterVolumeLevel(vol, None)

    # Visual feedback when fingers are pinched very close
    if length < 50:
      cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

  # Draw Volume Bar and Percentage on Screen
  cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
  cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
  cv2.putText(
      img,
      f"{int(volPer)} %",
      (40, 440),
      cv2.FONT_HERSHEY_COMPLEX,
      1,
      (255, 0, 0),
      3,
  )

  # Calculate and display FPS
  cTime = time.time()
  fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
  pTime = cTime
  cv2.putText(
      img,
      f"FPS: {int(fps)}",
      (40, 50),
      cv2.FONT_HERSHEY_COMPLEX,
      1,
      (255, 0, 0),
      3,
  )

  # Show Image Window
  cv2.imshow("Volume Controller", img)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()