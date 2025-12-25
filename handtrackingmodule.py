import os
import time
import math
import cv2
import numpy as np
import mediapipe as mp


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # NOTE: mode/trackCon not used by Tasks API the same way,
        # but we keep arguments so your main code doesn't change.
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # --- Model path (absolute) ---
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, "models", "hand_landmarker.task")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Put hand_landmarker.task inside: {os.path.join(base_dir,'models')}"
            )

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=self.maxHands,
            min_hand_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )

        self.landmarker = HandLandmarker.create_from_options(options)

        # last results
        self._result = None
        self.lmlist = []
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        # OpenCV gives BGR; MediaPipe expects SRGB (RGB) :contentReference[oaicite:1]{index=1}
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

        timestamp_ms = int(time.time() * 1000)
        self._result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        # Optional simple drawing: draw landmark dots
        if draw and self._result and self._result.hand_landmarks:
            h, w, _ = img.shape
            for hand_landmarks in self._result.hand_landmarks:
                for lm in hand_landmarks:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmlist = []
        bbox = []

        if not self._result or not self._result.hand_landmarks:
            return self.lmlist, bbox

        if handNo >= len(self._result.hand_landmarks):
            return self.lmlist, bbox

        h, w, _ = img.shape
        hand_landmarks = self._result.hand_landmarks[handNo]

        xList, yList = [], []
        for idx, lm in enumerate(hand_landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            xList.append(cx)
            yList.append(cy)
            self.lmlist.append([idx, cx, cy])

        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        bbox = [xmin, ymin, xmax, ymax]

        if draw:
            cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmlist, bbox

    def fingersUp(self):
        fingers = []
        if not self.lmlist:
            return [0, 0, 0, 0, 0]

        # Thumb (this heuristic depends on your mirroring; adjust if needed)
        if self.lmlist[self.tipIds[0]][1] > self.lmlist[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for i in range(1, 5):
            if self.lmlist[self.tipIds[i]][2] < self.lmlist[self.tipIds[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmlist[p1][1], self.lmlist[p1][2]
        x2, y2 = self.lmlist[p2][1], self.lmlist[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]
