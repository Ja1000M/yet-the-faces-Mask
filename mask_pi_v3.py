import cv2
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2

MODEL_PATH = "/home/jamil/yet-the-faces-mask/face_landmarker.task"

# === ROTATION CAMERA ===
CAMERA_ROTATION = cv2.ROTATE_90_COUNTERCLOCKWISE

# === CONFIG ECRANS ===
BOUCHE_ECRAN_W = 480
BOUCHE_ECRAN_H = 800
BOUCHE_ECRAN_X = 0

OEIL_ECRAN_W = 800
OEIL_ECRAN_H = 480
OEIL_ECRAN_X = 480

# === CADRAGE (tailles cibles a l'ecran, ajustables) ===
BOUCHE_LARGEUR_ECRAN = 300   # largeur de la demi-bouche affichee (px ecran)
BOUCHE_POSITION_HAUT = 200   # position verticale de la bouche (px depuis le haut)
OEIL_LARGEUR_ECRAN = 500     # largeur de l'oeil affiche (px ecran)

# === LANDMARKS ===
ORBITE = [362, 263, 386, 374]
OEIL_COIN_EXT = 263
OEIL_COIN_INT = 362
DECALAGE_SOURCIL = 0

BOUCHE_LM = [61, 291, 0, 17]
BOUCHE_COIN_G = 61
BOUCHE_COIN_D = 291

# === CAMERA ===
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
from libcamera import controls
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous, "Sharpness": 2.0})

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

# === FENETRES ===
cv2.namedWindow("BOUCHE", cv2.WINDOW_NORMAL)
cv2.moveWindow("BOUCHE", BOUCHE_ECRAN_X, 0)
cv2.setWindowProperty("BOUCHE", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.namedWindow("OEIL", cv2.WINDOW_NORMAL)
cv2.moveWindow("OEIL", OEIL_ECRAN_X, 0)
cv2.setWindowProperty("OEIL", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def crop_sur(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    return frame[y1:y2, x1:x2]

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
        frame_rgb = cv2.rotate(frame_brute, CAMERA_ROTATION)
        h, w = frame_rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)

        out_oeil = np.zeros((OEIL_ECRAN_H, OEIL_ECRAN_W, 3), dtype=np.uint8)
        out_bouche = np.zeros((BOUCHE_ECRAN_H, BOUCHE_ECRAN_W, 3), dtype=np.uint8)

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            # ================= OEIL =================
            # Largeur reelle de l'oeil dans l'image source
            oeil_l = abs(lm[OEIL_COIN_EXT].x - lm[OEIL_COIN_INT].x) * w
            # Zone a capturer pour que l'oeil fasse OEIL_LARGEUR_ECRAN a l'ecran
            zone_w = int(oeil_l * OEIL_ECRAN_W / OEIL_LARGEUR_ECRAN)
            zone_h = int(zone_w * OEIL_ECRAN_H / OEIL_ECRAN_W)  # meme ratio que l'ecran
            xs = [lm[i].x * w for i in ORBITE]
            ys = [lm[i].y * h for i in ORBITE]
            cx = int(np.mean(xs))
            cy = int(np.mean(ys)) - DECALAGE_SOURCIL
            x1 = cx - zone_w // 2
            y1 = cy - zone_h // 2
            crop_oeil = crop_sur(frame_rgb, x1, y1, x1 + zone_w, y1 + zone_h)
            out_oeil = traiter_crop(crop_oeil, OEIL_ECRAN_W, OEIL_ECRAN_H)

            # ================= BOUCHE =================
            # Largeur reelle de la demi-bouche (coin -> centre)
            centre_x = (lm[BOUCHE_COIN_G].x + lm[BOUCHE_COIN_D].x) / 2 * w
            demi_l = abs(lm[BOUCHE_COIN_G].x * w - centre_x)
            # Zone a capturer pour que la demi-bouche fasse BOUCHE_LARGEUR_ECRAN
            zone_w = int(demi_l * BOUCHE_ECRAN_W / BOUCHE_LARGEUR_ECRAN)
            zone_h = int(zone_w * BOUCHE_ECRAN_H / BOUCHE_ECRAN_W)  # ratio ecran
            bys = [lm[i].y * h for i in BOUCHE_LM]
            bcy = int(np.mean(bys))
            bcx = int(centre_x)
            # Cote gauche du centre (cote valide precedemment)
            bx1 = bcx - zone_w
            # La bouche positionnee a BOUCHE_POSITION_HAUT px ecran du haut
            decal_haut = int(zone_h * BOUCHE_POSITION_HAUT / BOUCHE_ECRAN_H)
            by1 = bcy - decal_haut
            crop_bouche = crop_sur(frame_rgb, bx1, by1, bx1 + zone_w, by1 + zone_h)
            out_bouche = traiter_crop(crop_bouche, BOUCHE_ECRAN_W, BOUCHE_ECRAN_H)

        cv2.imshow("OEIL", out_oeil)
        cv2.imshow("BOUCHE", out_bouche)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

picam2.stop()
cv2.destroyAllWindows()
