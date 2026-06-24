import cv2
import numpy as np
from picamera2 import Picamera2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Résolution écran Waveshare
OEIL_W, OEIL_H = 800, 480
OEIL_CAPTURE_W = 300
OEIL_CAPTURE_H = 180
DECALAGE_SOURCIL = 20
OEIL_CENTRE_IDX = [133, 33, 159, 145]

def generer_vignette(h, w):
    cx, cy = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    dist = dist / dist.max()
    vignette = 1 - dist * 1.2
    vignette = np.clip(vignette, 0, 1)
    return np.stack([vignette]*3, axis=-1).astype(np.float32)

VIGNETTE = generer_vignette(OEIL_H, OEIL_W)

def effet_naturel(zone):
    b, g, r = cv2.split(zone)
    b = cv2.convertScaleAbs(b, alpha=1.1, beta=5)
    r = cv2.convertScaleAbs(r, alpha=0.88, beta=0)
    zone = cv2.merge([b, g, r])
    zone = (zone.astype(np.float32) * VIGNETTE).astype(np.uint8)
    return zone

def extraire_zone_orbitale(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [landmarks[i] for i in OEIL_CENTRE_IDX]
    cx = int(sum(p.x for p in pts) / len(pts) * w)
    cy = int(sum(p.y for p in pts) / len(pts) * h)
    cy -= DECALAGE_SOURCIL

    x_min = max(0, cx - OEIL_CAPTURE_W // 2)
    x_max = x_min + OEIL_CAPTURE_W
    y_min = max(0, cy - OEIL_CAPTURE_H // 2)
    y_max = y_min + OEIL_CAPTURE_H

    if x_max > w:
        x_max = w
        x_min = w - OEIL_CAPTURE_W
    if y_max > h:
        y_max = h
        y_min = h - OEIL_CAPTURE_H

    zone = frame[y_min:y_max, x_min:x_max]
    zone = cv2.resize(zone, (OEIL_W, OEIL_H))
    return zone

# Initialisation caméra Pi
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1920, 1080), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

# Chargement modèle MediaPipe
model_path = "face_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        results = landmarker.detect(mp_image)

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            zone = extraire_zone_orbitale(frame_bgr, landmarks)
            rendu = effet_naturel(zone)
            cv2.imshow('Ecran oeil — 800x480', rendu)

        cv2.imshow('Yet The Faces — Pi', frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

picam2.stop()
cv2.destroyAllWindows()
