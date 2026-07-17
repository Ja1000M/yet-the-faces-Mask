import cv2
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2

MODEL_PATH = "/home/jamil/yet-the-faces-mask/face_landmarker.task"

# === ROTATION CAMERA ===
# La camera est montee tournee sur le masque : on redresse la frame des la capture
# ROTATE_90_COUNTERCLOCKWISE ou ROTATE_90_CLOCKWISE selon le sens de montage
CAMERA_ROTATION = cv2.ROTATE_90_COUNTERCLOCKWISE

# === CONFIG ECRANS (validee par le test couleurs) ===
BOUCHE_ECRAN_W = 480
BOUCHE_ECRAN_H = 800
BOUCHE_ECRAN_X = 0

OEIL_ECRAN_W = 800
OEIL_ECRAN_H = 480
OEIL_ECRAN_X = 480

# === ZONES DE CAPTURE ===
ORBITE = [133, 33, 159, 145]
OEIL_CAPTURE_W = 400
OEIL_CAPTURE_H = 240
DECALAGE_SOURCIL = 20

BOUCHE_LM = [61, 291, 0, 17]
BOUCHE_CAPTURE_W = 200
BOUCHE_CAPTURE_H = 333

# === CAMERA ===
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

# === MEDIAPIPE ===
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)

# === FENETRES (moveWindow PUIS fullscreen) ===
cv2.namedWindow("BOUCHE", cv2.WINDOW_NORMAL)
cv2.moveWindow("BOUCHE", BOUCHE_ECRAN_X, 0)
cv2.setWindowProperty("BOUCHE", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.namedWindow("OEIL", cv2.WINDOW_NORMAL)
cv2.moveWindow("OEIL", OEIL_ECRAN_X, 0)
cv2.setWindowProperty("OEIL", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def traiter_crop(crop_rgb, dest_w, dest_h):
    if crop_rgb.size == 0:
        return np.zeros((dest_h, dest_w, 3), dtype=np.uint8)
    resized = cv2.resize(crop_rgb, (dest_w, dest_h))
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.6
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    bgr[:, :, 0] = np.clip(bgr[:, :, 0].astype(np.int16) + 10, 0, 255)
    return bgr

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        frame_brute = picam2.capture_array()
        # Redressement de la camera montee a 90 degres
        frame_rgb = cv2.rotate(frame_brute, CAMERA_ROTATION)
        h, w = frame_rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)

        out_oeil = np.zeros((OEIL_ECRAN_H, OEIL_ECRAN_W, 3), dtype=np.uint8)
        out_bouche = np.zeros((BOUCHE_ECRAN_H, BOUCHE_ECRAN_W, 3), dtype=np.uint8)

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            # --- OEIL ---
            xs = [lm[i].x * w for i in ORBITE]
            ys = [lm[i].y * h for i in ORBITE]
            cx = int(np.mean(xs))
            cy = int(np.mean(ys)) - DECALAGE_SOURCIL
            x1 = max(0, cx - OEIL_CAPTURE_W // 2)
            y1 = max(0, cy - OEIL_CAPTURE_H // 2)
            x2 = min(w, x1 + OEIL_CAPTURE_W)
            y2 = min(h, y1 + OEIL_CAPTURE_H)
            crop_oeil = frame_rgb[y1:y2, x1:x2]
            out_oeil = traiter_crop(crop_oeil, OEIL_ECRAN_W, OEIL_ECRAN_H)

            # --- BOUCHE ---
            bxs = [lm[i].x * w for i in BOUCHE_LM]
            bys = [lm[i].y * h for i in BOUCHE_LM]
            bcx = int(np.mean(bxs))
            bcy = int(np.mean(bys))
            bx1 = max(0, bcx - BOUCHE_CAPTURE_W)
            by1 = max(0, bcy - 20)
            bx2 = min(w, bcx)
            by2 = min(h, by1 + BOUCHE_CAPTURE_H)
            crop_bouche = frame_rgb[by1:by2, bx1:bx2]
            bouche_traitee = traiter_crop(crop_bouche, BOUCHE_ECRAN_W, BOUCHE_ECRAN_H)
            out_bouche = bouche_traitee

        cv2.imshow("OEIL", out_oeil)
        cv2.imshow("BOUCHE", out_bouche)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

picam2.stop()
cv2.destroyAllWindows()
