import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Indices MediaPipe pour les zones
OEIL_GAUCHE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
OEIL_DROIT  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
BOUCHE      = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

def extraire_zone(frame, landmarks, indices, marge=20):
    h, w = frame.shape[:2]
    points = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    x_min = max(0, min(p[0] for p in points) - marge)
    x_max = min(w, max(p[0] for p in points) + marge)
    y_min = max(0, min(p[1] for p in points) - marge)
    y_max = min(h, max(p[1] for p in points) + marge)
    return frame[y_min:y_max, x_min:x_max]

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

            zone_oeil = extraire_zone(frame, landmarks, OEIL_GAUCHE)
            zone_bouche = extraire_zone(frame, landmarks, BOUCHE)

            if zone_oeil.size > 0:
                cv2.imshow('Zone oeil', zone_oeil)
            if zone_bouche.size > 0:
                cv2.imshow('Zone bouche', zone_bouche)

        cv2.imshow('Yet The Faces — Phase A', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()