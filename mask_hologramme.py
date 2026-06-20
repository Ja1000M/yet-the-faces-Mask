import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Résolution écran Waveshare
OEIL_W, OEIL_H = 800, 480
OEIL_CAPTURE_W = 300
OEIL_CAPTURE_H = 180
DECALAGE_SOURCIL = 20
OEIL_CENTRE_IDX = [133, 33, 159, 145]

# Masque scanlines plus marquées — généré une seule fois
def generer_scanlines(h, w):
    masque = np.ones((h, w, 3), dtype=np.float32)
    masque[::2] *= 0.15
    return masque

SCANLINES = generer_scanlines(OEIL_H, OEIL_W)

def effet_hologramme(zone):
    # 1 — Niveaux de gris
    gris = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)

    # 2 — Cyan froid — plus de bleu, moins de vert
    hologramme = np.zeros((OEIL_H, OEIL_W, 3), dtype=np.uint8)
    hologramme[:, :, 0] = gris
    hologramme[:, :, 1] = (gris * 0.55).astype(np.uint8)
    hologramme[:, :, 2] = (gris * 0.05).astype(np.uint8)

    # 3 — Aberration chromatique forte
    b, g, r = cv2.split(hologramme)
    decalage = 7
    M_plus = np.float32([[1, 0, decalage], [0, 1, 0]])
    M_moins = np.float32([[1, 0, -decalage], [0, 1, 0]])
    b = cv2.warpAffine(b, M_plus, (OEIL_W, OEIL_H))
    r = cv2.warpAffine(r, M_moins, (OEIL_W, OEIL_H))
    hologramme = cv2.merge([b, g, r])

    # 4 — Scanlines
    hologramme = (hologramme.astype(np.float32) * SCANLINES).astype(np.uint8)

    return hologramme

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

# Chargement du modèle
model_path = "face_landmarker.task"
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

with vision.FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = landmarker.detect(mp_image)

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            zone = extraire_zone_orbitale(frame, landmarks)
            hologramme = effet_hologramme(zone)
            cv2.imshow('Hologramme oeil — 800x480', hologramme)

        cv2.imshow('Yet The Faces — Phase A', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()