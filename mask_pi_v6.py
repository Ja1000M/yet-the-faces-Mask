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

# === RENDU DU FLUX LIVE (v6) ===
# C'est ici que se joue la "temperature" de l'image.
# Anciennes valeurs (rendu froid/clinique) : saturation 0.6, temperature -10
FLUX_SATURATION = 1.05    # 0 = noir et blanc, 1 = couleurs d'origine, >1 = renforcees
FLUX_TEMPERATURE = 5      # >0 = plus chaud (rouge), <0 = plus froid (bleu), 0 = neutre
TEMPERATURE_SEUIL_BLANC = 200  # luminosite au-dela de laquelle un pixel n'est plus teinte
FLUX_LUMINOSITE = 1.05    # 1 = inchangee, 1.1 = +10%, 0.9 = -10%
FLUX_CONTRASTE = 1.18     # 1 = inchange, 1.2 = +20%, 0.9 = plus doux

# === SUPERPOSITION MILENA ===
# --- Mode de melange live/Milena pendant la capture ---
# "alpha"  : moyenne classique (l'ancien rendu, tendance grisatre)
# "screen" : superposition lumineuse (double exposition, les deux regards
#            se traversent sans s'eteindre — plus percutant)
# "aucun"  : flux live pur, Milena invisible pendant la capture
MODE_MELANGE = "screen"

# --- Proximite : la personne captee se revele en s'approchant ---
# La distance est estimee par la largeur de l'oeil capte (en pixels).
# Loin  -> flux flou + Milena tres presente
# Proche -> flux net + Milena effacee
OEIL_PX_LOIN = 50        # largeur d'oeil (px) consideree "loin"
OEIL_PX_PROCHE = 180     # largeur d'oeil (px) consideree "tout proche"
FLOU_LOIN = 31           # flou du flux quand loin (impair ; 0 = pas de flou)
PRESENCE_MILENA_LOIN = 0.95    # presence de Milena quand la personne est loin
PRESENCE_MILENA_PROCHE = 0.0   # presence de Milena au plus proche
PROXIMITE_LISSAGE = 0.12  # 0.05 = tres amorti, 0.3 = tres reactif

# --- Zone d'intimite : trop pres, Milena reemerge et defend sa zone ---
OEIL_PX_INTIME = 240       # largeur d'oeil (px) ou commence la zone d'intimite
OEIL_PX_INTIME_MAX = 320   # largeur d'oeil (px) ou Milena est pleinement revenue
PRESENCE_MILENA_INTIME = 0.9   # presence de Milena au coeur de la zone d'intimite
MILENA_DESATURATION = 0.55  # 0 = noir et blanc, 1 = couleurs d'origine (avant : 0.4)
MILENA_FLOU = 7             # impair : 5, 7, 9, 11...

# === FONDU (v5) ===
FADE_IN_DUREE = 1.2    # secondes, apparition du visage capte
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
# Sequence robuste (validee avec test_ecrans v2) : creer -> afficher ->
# positionner -> fullscreen -> re-positionner, avec pauses pour le WM.
def ouvrir_fenetre(nom, x, w, h):
    noir = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.namedWindow(nom, cv2.WINDOW_NORMAL)
    cv2.imshow(nom, noir)
    cv2.waitKey(200)
    cv2.moveWindow(nom, x, 0)
    cv2.resizeWindow(nom, w, h)
    cv2.waitKey(200)
    cv2.setWindowProperty(nom, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(200)
    cv2.moveWindow(nom, x, 0)   # re-force la position apres fullscreen
    cv2.waitKey(200)

ouvrir_fenetre("BOUCHE", BOUCHE_ECRAN_X, BOUCHE_ECRAN_W, BOUCHE_ECRAN_H)
ouvrir_fenetre("OEIL", OEIL_ECRAN_X, OEIL_ECRAN_W, OEIL_ECRAN_H)

def crop_sur(frame, x1, y1, x2, y2):
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))
    return frame[y1:y2, x1:x2]

def traiter_crop(crop_rgb, dest_w, dest_h):
    """Rendu du flux live : saturation, luminosite, temperature (v6)."""
    if crop_rgb.size == 0:
        return np.zeros((dest_h, dest_w, 3), dtype=np.uint8)
    resized = cv2.resize(crop_rgb, (dest_w, dest_h))
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

    # saturation + luminosite
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= FLUX_SATURATION
    hsv[:, :, 2] *= FLUX_LUMINOSITE
    hsv[:, :, 1:] = np.clip(hsv[:, :, 1:], 0, 255)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # temperature : rechauffe (rouge+/bleu-) ou refroidit (l'inverse)
    # avec protection des blancs : plus un pixel est clair, moins il est
    # teinte (evite les blancs jaunis, ex. le blanc de l'oeil)
    if FLUX_TEMPERATURE != 0:
        t = float(FLUX_TEMPERATURE)
        lum = bgr.astype(np.float32).mean(axis=2)
        # 1 dans les ombres, 0 des que lum >= TEMPERATURE_SEUIL_BLANC
        protection = np.clip(
            (TEMPERATURE_SEUIL_BLANC - lum) / TEMPERATURE_SEUIL_BLANC,
            0.0, 1.0) ** 1.5
        shift = t * protection
        b = bgr[:, :, 0].astype(np.float32) - shift
        r = bgr[:, :, 2].astype(np.float32) + shift
        bgr[:, :, 0] = np.clip(b, 0, 255).astype(np.uint8)
        bgr[:, :, 2] = np.clip(r, 0, 255).astype(np.uint8)

    # contraste : ecarte les valeurs autour du gris moyen
    if FLUX_CONTRASTE != 1.0:
        bgr = cv2.convertScaleAbs(
            bgr, alpha=FLUX_CONTRASTE,
            beta=128.0 * (1.0 - FLUX_CONTRASTE))
    return bgr

def lissage(t):
    """Adoucit le fondu (smoothstep) : demarrage et fin en douceur."""
    return t * t * (3.0 - 2.0 * t)

def melanger(live, milena, presence):
    """Fusionne le flux live avec l'image de Milena selon MODE_MELANGE.
    presence : 0 = Milena invisible, 1 = pleine presence."""
    if milena is None or MODE_MELANGE == "aucun" or presence <= 0.0:
        return live
    if MODE_MELANGE == "screen":
        # superposition lumineuse : 255 - (255-a)(255-b)/255
        a = live.astype(np.float32)
        b = milena.astype(np.float32)
        fusion = 255.0 - (255.0 - a) * (255.0 - b) / 255.0
        fusion = fusion.astype(np.uint8)
    else:  # "alpha" : moyenne ponderee classique
        fusion = cv2.addWeighted(live, 0.5, milena, 0.5, 0)
    return cv2.addWeighted(live, 1.0 - presence, fusion, presence, 0)

def flou_selon_proximite(img, proximite):
    """Floute d'autant plus que la personne est loin (proximite 0 -> 1)."""
    if FLOU_LOIN <= 0 or proximite >= 1.0:
        return img
    noyau = int(FLOU_LOIN * (1.0 - proximite))
    if noyau < 1:
        return img
    if noyau % 2 == 0:
        noyau += 1
    return cv2.GaussianBlur(img, (noyau, noyau), 0)

# === ETAT DU FONDU ===
fondu = 0.0
derniere_oeil = None
derniere_bouche = None
t_precedent = time.monotonic()

# === ETAT DE LA PROXIMITE ===
proximite = 0.0          # 0 = loin, 1 = tout proche (valeur lissee)
intimite = 0.0           # 0 = hors zone d'intimite, 1 = en plein dedans (lissee)
t_dernier_debug = 0.0

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        frame_brute = picam2.capture_array()
        frame_rgb = cv2.rotate(frame_brute, CAMERA_ROTATION)
        h, w = frame_rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)

        maintenant = time.monotonic()
        dt = maintenant - t_precedent
        t_precedent = maintenant

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            # ================= PROXIMITE =================
            oeil_l = abs(lm[OEIL_COIN_EXT].x - lm[OEIL_COIN_INT].x) * w
            cible = (oeil_l - OEIL_PX_LOIN) / (OEIL_PX_PROCHE - OEIL_PX_LOIN)
            cible = max(0.0, min(1.0, cible))
            proximite += (cible - proximite) * PROXIMITE_LISSAGE

            # zone d'intimite : au-dela de OEIL_PX_INTIME, Milena revient
            cible_intime = (oeil_l - OEIL_PX_INTIME) / \
                (OEIL_PX_INTIME_MAX - OEIL_PX_INTIME)
            cible_intime = max(0.0, min(1.0, cible_intime))
            intimite += (cible_intime - intimite) * PROXIMITE_LISSAGE

            presence_milena = PRESENCE_MILENA_LOIN + \
                (PRESENCE_MILENA_PROCHE - PRESENCE_MILENA_LOIN) * lissage(proximite)
            li = lissage(intimite)
            presence_milena = presence_milena * (1.0 - li) + \
                PRESENCE_MILENA_INTIME * li

            if maintenant - t_dernier_debug > 1.0:
                t_dernier_debug = maintenant
                fps = 1.0 / dt if dt > 0 else 0.0
                print(f"oeil {oeil_l:5.0f}px  proximite {proximite:4.2f}  "
                      f"intimite {intimite:4.2f}  "
                      f"presence Milena {presence_milena:4.2f}  {fps:4.1f} fps")

            # ================= OEIL =================
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
            live_oeil = flou_selon_proximite(live_oeil, proximite)
            live_oeil = melanger(live_oeil, oeil_milena, presence_milena)

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
            live_bouche = flou_selon_proximite(live_bouche, proximite)
            live_bouche = melanger(live_bouche, bouche_milena, presence_milena)

            derniere_oeil = live_oeil
            derniere_bouche = live_bouche
            fondu = min(1.0, fondu + dt / FADE_IN_DUREE)
        else:
            fondu = max(0.0, fondu - dt / FADE_OUT_DUREE)

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
