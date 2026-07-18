import cv2
import numpy as np
import time
import mediapipe as mp
from picamera2 import Picamera2
from libcamera import controls

MODEL_PATH = "/home/jamil/yet-the-faces-mask/face_landmarker.task"
OEIL_MILENA_PATH = "/home/jamil/yet-the-faces-mask/oeil_milena.png"
BOUCHE_MILENA_PATH = "/home/jamil/yet-the-faces-mask/bouche_milena.png"

# === ROTATION CAMERA ===
CAMERA_ROTATION = cv2.ROTATE_90_COUNTERCLOCKWISE

# === CONFIG ECRANS ===
BOUCHE_ECRAN_W = 480
BOUCHE_ECRAN_H = 800
BOUCHE_ECRAN_X = 0

OEIL_ECRAN_W = 800
OEIL_ECRAN_H = 480
OEIL_ECRAN_X = 480

# === CADRAGE ===
BOUCHE_LARGEUR_ECRAN = 300
BOUCHE_POSITION_HAUT = 200
OEIL_LARGEUR_ECRAN = 500

# === SUPERPOSITION MILENA ===
POIDS_FLUX = 0.75
POIDS_MILENA = 0.25
MILENA_DESATURATION = 0.4   # 0 = noir et blanc, 1 = couleurs d'origine
MILENA_FLOU = 7             # impair : 5, 7, 9, 11...

# === FONDU (v5) ===
# Duree du fondu enchaine entre "Milena seule" et la capture live.
FADE_IN_DUREE = 0.8    # secondes, apparition du visage capte
FADE_OUT_DUREE = 1.2   # secondes, effacement quand le visage disparait

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
picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous, "Sharpness": 2.0})

def charger_image_milena(chemin, largeur, hauteur):
    img = cv2.imread(chemin)
    if img is None:
        print(f"ATTENTION : image introuvable a {chemin}")
        return None
    img = cv2.resize(img, (largeur, hauteur))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= MILENA_DESATURATION
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    img = cv2.GaussianBlur(img, (MILENA_FLOU, MILENA_FLOU), 0)
    return img

oeil_milena = charger_image_milena(OEIL_MILENA_PATH, OEIL_ECRAN_W, OEIL_ECRAN_H)
bouche_milena = charger_image_milena(BOUCHE_MILENA_PATH, BOUCHE_ECRAN_W, BOUCHE_ECRAN_H)

# Fonds de reference : Milena si disponible, sinon noir
fond_oeil = oeil_milena if oeil_milena is not None else \
    np.zeros((OEIL_ECRAN_H, OEIL_ECRAN_W, 3), dtype=np.uint8)
fond_bouche = bouche_milena if bouche_milena is not None else \
    np.zeros((BOUCHE_ECRAN_H, BOUCHE_ECRAN_W, 3), dtype=np.uint8)

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

def lissage(t):
    """Adoucit le fondu (smoothstep) : demarrage et fin en douceur."""
    return t * t * (3.0 - 2.0 * t)

# === ETAT DU FONDU (v5) ===
fondu = 0.0             # 0 = Milena seule, 1 = capture live pleine
derniere_oeil = None    # derniere image live connue (pour le fade out)
derniere_bouche = None
t_precedent = time.monotonic()

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        frame_brute = picam2.capture_array()
        frame_rgb = cv2.rotate(frame_brute, CAMERA_ROTATION)
        h, w = frame_rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)

        # --- horloge du fondu ---
        maintenant = time.monotonic()
        dt = maintenant - t_precedent
        t_precedent = maintenant

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            # ================= OEIL =================
            oeil_l = abs(lm[OEIL_COIN_EXT].x - lm[OEIL_COIN_INT].x) * w
            zone_w = int(oeil_l * OEIL_ECRAN_W / OEIL_LARGEUR_ECRAN)
            zone_h = int(zone_w * OEIL_ECRAN_H / OEIL_ECRAN_W)
            xs = [lm[i].x * w for i in ORBITE]
            ys = [lm[i].y * h for i in ORBITE]
            cx = int(np.mean(xs))
            cy = int(np.mean(ys)) - DECALAGE_SOURCIL
            x1 = cx - zone_w // 2
            y1 = cy - zone_h // 2
            crop_oeil = crop_sur(frame_rgb, x1, y1, x1 + zone_w, y1 + zone_h)
            live_oeil = traiter_crop(crop_oeil, OEIL_ECRAN_W, OEIL_ECRAN_H)
            if oeil_milena is not None:
                live_oeil = cv2.addWeighted(
                    live_oeil, POIDS_FLUX, oeil_milena, POIDS_MILENA, 0)

            # ================= BOUCHE =================
            centre_x = (lm[BOUCHE_COIN_G].x + lm[BOUCHE_COIN_D].x) / 2 * w
            demi_l = abs(lm[BOUCHE_COIN_G].x * w - centre_x)
            zone_w = int(demi_l * BOUCHE_ECRAN_W / BOUCHE_LARGEUR_ECRAN)
            zone_h = int(zone_w * BOUCHE_ECRAN_H / BOUCHE_ECRAN_W)
            bys = [lm[i].y * h for i in BOUCHE_LM]
            bcy = int(np.mean(bys))
            bcx = int(centre_x)
            bx1 = bcx - zone_w
            decal_haut = int(zone_h * BOUCHE_POSITION_HAUT / BOUCHE_ECRAN_H)
            by1 = bcy - decal_haut
            crop_bouche = crop_sur(frame_rgb, bx1, by1, bx1 + zone_w, by1 + zone_h)
            live_bouche = traiter_crop(crop_bouche, BOUCHE_ECRAN_W, BOUCHE_ECRAN_H)
            if bouche_milena is not None:
                live_bouche = cv2.addWeighted(
                    live_bouche, POIDS_FLUX, bouche_milena, POIDS_MILENA, 0)

            # memorise la derniere image live (support du fade out)
            derniere_oeil = live_oeil
            derniere_bouche = live_bouche

            # fondu monte vers 1 (fade in)
            fondu = min(1.0, fondu + dt / FADE_IN_DUREE)
        else:
            # fondu descend vers 0 (fade out)
            fondu = max(0.0, fondu - dt / FADE_OUT_DUREE)

        # --- composition finale : fondu enchaine Milena <-> live ---
        if fondu <= 0.0 or derniere_oeil is None:
            out_oeil = fond_oeil
            out_bouche = fond_bouche
        elif fondu >= 1.0:
            out_oeil = derniere_oeil
            out_bouche = derniere_bouche
        else:
            a = lissage(fondu)
            out_oeil = cv2.addWeighted(
                derniere_oeil, a, fond_oeil, 1.0 - a, 0)
            out_bouche = cv2.addWeighted(
                derniere_bouche, a, fond_bouche, 1.0 - a, 0)

        cv2.imshow("OEIL", out_oeil)
        cv2.imshow("BOUCHE", out_bouche)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

picam2.stop()
cv2.destroyAllWindows()
