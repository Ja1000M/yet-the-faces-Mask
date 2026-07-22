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

# === PERFORMANCE (v7) ===
# MediaPipe analyse une image reduite (les landmarks sont normalises,
# la precision reste largement suffisante) ; les crops restent en pleine
# resolution pour la qualite d'image.
DETECTION_ECHELLE = 0.5   # 0.5 = detection sur image moitie (4x moins de pixels)

# === CADRAGE ===
# La bouche affichee = 2 x BOUCHE_LARGEUR_ECRAN. 300 -> 270 : bouche -10%.
BOUCHE_LARGEUR_ECRAN = 270
# Position verticale du centre de la bouche sur l'ecran (sur 800 px de haut).
# 200 -> 250 : la bouche descend de 50 px.
BOUCHE_POSITION_HAUT = 250
OEIL_LARGEUR_ECRAN = 500

# === STABILISATION DU CADRE (anti-tressautement) ===
# Lisse dans le temps la position et le zoom du crop pour absorber le bruit
# des landmarks (le cadre suit le visage en douceur au lieu de vibrer).
# 1.0 = aucun lissage (brut, tressaute) ; 0.15 = tres stable mais un peu mou ;
# 0.3 = bon compromis reactivite/stabilite.
STAB_LISSAGE = 0.3

# === RENDU DU FLUX LIVE ===
FLUX_SATURATION = 1.05    # 0 = noir et blanc, 1 = couleurs d'origine, >1 = renforcees
FLUX_TEMPERATURE = 7      # >0 = plus chaud (rouge), <0 = plus froid (bleu), 0 = neutre
TEMPERATURE_SEUIL_BLANC = 235  # luminosite au-dela de laquelle un pixel n'est plus teinte
FLUX_LUMINOSITE = 1.15    # 1 = inchangee, 1.1 = +10%, 0.9 = -10%
FLUX_CONTRASTE = 1.10     # 1 = inchange, 1.2 = +20%, 0.9 = plus doux

# --- Nettete progressive : l'image se precise a l'approche ---
# Contre le "flou non assume" (mollesse d'agrandissement) : un masque de
# nettete (unsharp) monte avec la proximite. Au point optimum l'image
# "claque", loin elle reste franchement floue (flou assume).
NETTETE_MAX = 0.7   # accentuation max au point optimum (0 = aucune ; 0.4 doux, 1.0 fort)

# === MELANGE LIVE/MILENA ===
# "alpha"  : moyenne classique (tendance grisatre)
# "screen" : superposition lumineuse (double exposition, plus percutant)
# "aucun"  : flux live pur
MODE_MELANGE = "screen"

# --- Recolorisation vers Milena (mode fusion "Couleur") ---
# Garde la LUMINANCE du flux live (details, mouvement) mais emprunte la
# TEINTE + SATURATION de l'image de Milena. Corrige le rendu violet de la
# camera NoIR en unifiant vers les tons chauds de Milena.
# 0 = couleur d'origine (violet NoIR) ; 1 = couleur de Milena pleine.
RECOLOR_MILENA = 0.5

# --- Proximite : la personne captee se revele en s'approchant ---
OEIL_PX_LOIN = 33        # largeur d'oeil (px) consideree "loin" (mesure reelle NoIR grand angle)
OEIL_PX_PROCHE = 55      # largeur d'oeil (px) : revelation pleine atteinte plus tot (plus loin)
FLOU_LOIN = 55           # flou du flux quand loin (impair ; 0 = pas de flou) - defocus assume
PRESENCE_MILENA_LOIN = 1.0     # presence de Milena quand la personne est loin
PRESENCE_MILENA_PROCHE = 0.2   # au point optimum : on sent encore l'oeil de Milena, en plus doux
PROXIMITE_LISSAGE = 0.12  # 0.05 = tres amorti, 0.3 = tres reactif
REVELATION_COURBE = 0.8   # <1 = revelation precoce (net tot) ; 1 = lineaire ; 2,3 = tardive/spectaculaire

# --- Zone d'intimite : trop pres, Milena reemerge et defend sa zone ---
OEIL_PX_INTIME = 72        # largeur d'oeil (px) ou commence la zone d'intimite (Milena revient plus tot)
OEIL_PX_INTIME_MAX = 92    # largeur d'oeil (px) ou Milena est pleinement revenue
PRESENCE_MILENA_INTIME = 0.9   # presence de Milena au coeur de la zone d'intimite

# === IMAGES MILENA ===
MILENA_DESATURATION = 0.55  # 0 = noir et blanc, 1 = couleurs d'origine
MILENA_FLOU = 7             # impair : 5, 7, 9, 11...

# === FONDU ===
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
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_c, s_c, v_c = cv2.split(hsv)
    s_c = cv2.convertScaleAbs(s_c, alpha=MILENA_DESATURATION)
    img = cv2.cvtColor(cv2.merge([h_c, s_c, v_c]), cv2.COLOR_HSV2BGR)
    img = cv2.GaussianBlur(img, (MILENA_FLOU, MILENA_FLOU), 0)
    return img

oeil_milena = charger_image_milena(OEIL_MILENA_PATH, OEIL_ECRAN_W, OEIL_ECRAN_H)
bouche_milena = charger_image_milena(BOUCHE_MILENA_PATH, BOUCHE_ECRAN_W, BOUCHE_ECRAN_H)

fond_oeil = oeil_milena if oeil_milena is not None else \
    np.zeros((OEIL_ECRAN_H, OEIL_ECRAN_W, 3), dtype=np.uint8)
fond_bouche = bouche_milena if bouche_milena is not None else \
    np.zeros((BOUCHE_ECRAN_H, BOUCHE_ECRAN_W, 3), dtype=np.uint8)

# === LUT TEMPERATURE (precalculee : decalage par niveau de luminosite) ===
# Protection des blancs : plus un pixel est clair, moins il est teinte.
_lut_temp = np.zeros(256, dtype=np.uint8)
if FLUX_TEMPERATURE != 0:
    for l in range(256):
        prot = max(0.0, (TEMPERATURE_SEUIL_BLANC - l) / TEMPERATURE_SEUIL_BLANC)
        _lut_temp[l] = int(round(abs(FLUX_TEMPERATURE) * (prot ** 1.5)))

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
# Sequence robuste : creer -> afficher -> positionner -> fullscreen ->
# re-positionner, avec pauses pour le WM.
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
    cv2.moveWindow(nom, x, 0)
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

def flouter_crop(crop, proximite, dest_w):
    """Floute le petit crop AVANT agrandissement (quasi gratuit).
    Le noyau est mis a l'echelle du crop pour un rendu equivalent."""
    if FLOU_LOIN <= 0 or proximite >= 1.0 or crop.size == 0:
        return crop
    k = FLOU_LOIN * (1.0 - proximite) * crop.shape[1] / float(dest_w)
    k = int(k)
    if k < 1:
        return crop
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(crop, (k, k), 0)

def traiter_crop(crop_rgb, dest_w, dest_h):
    """Rendu du flux live : saturation, luminosite, temperature, contraste.
    v7 : operations OpenCV natives uint8 (SIMD), plus de flottants numpy.
    v8 : agrandissement en INTER_CUBIC (bords plus francs, moins mou)."""
    if crop_rgb.size == 0:
        return np.zeros((dest_h, dest_w, 3), dtype=np.uint8)
    resized = cv2.resize(crop_rgb, (dest_w, dest_h), interpolation=cv2.INTER_CUBIC)
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

    # saturation + luminosite (canaux HSV en uint8)
    if FLUX_SATURATION != 1.0 or FLUX_LUMINOSITE != 1.0:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h_c, s_c, v_c = cv2.split(hsv)
        if FLUX_SATURATION != 1.0:
            s_c = cv2.convertScaleAbs(s_c, alpha=FLUX_SATURATION)
        if FLUX_LUMINOSITE != 1.0:
            v_c = cv2.convertScaleAbs(v_c, alpha=FLUX_LUMINOSITE)
        bgr = cv2.cvtColor(cv2.merge([h_c, s_c, v_c]), cv2.COLOR_HSV2BGR)

    # temperature avec protection des blancs (LUT precalculee)
    if FLUX_TEMPERATURE != 0:
        lum = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        shift = cv2.LUT(lum, _lut_temp)
        b_c, g_c, r_c = cv2.split(bgr)
        if FLUX_TEMPERATURE > 0:
            r_c = cv2.add(r_c, shift)
            b_c = cv2.subtract(b_c, shift)
        else:
            r_c = cv2.subtract(r_c, shift)
            b_c = cv2.add(b_c, shift)
        bgr = cv2.merge([b_c, g_c, r_c])

    # contraste autour du gris moyen
    if FLUX_CONTRASTE != 1.0:
        bgr = cv2.convertScaleAbs(
            bgr, alpha=FLUX_CONTRASTE,
            beta=128.0 * (1.0 - FLUX_CONTRASTE))
    return bgr

def recolorer(live, milena, intensite):
    """Mode fusion 'Couleur' : luminance du live + teinte/saturation de Milena.
    Neutralise le violet de la NoIR en unifiant vers les tons de Milena."""
    if milena is None or intensite <= 0.0:
        return live
    live_hsv = cv2.cvtColor(live, cv2.COLOR_BGR2HSV)
    mil_hsv = cv2.cvtColor(milena, cv2.COLOR_BGR2HSV)
    _, _, lv = cv2.split(live_hsv)      # luminance (V) du live
    mh, ms, _ = cv2.split(mil_hsv)      # teinte (H) + saturation (S) de Milena
    recolored = cv2.cvtColor(cv2.merge([mh, ms, lv]), cv2.COLOR_HSV2BGR)
    return cv2.addWeighted(live, 1.0 - intensite, recolored, intensite, 0)

def accentuer(img, amount):
    """Masque de nettete (unsharp mask). amount 0 = image inchangee.
    Reintroduit une nettete franche a l'approche pour defaire la mollesse
    d'agrandissement : le flou devient un choix, plus un accident."""
    if amount <= 0.0 or img.size == 0:
        return img
    flou = cv2.GaussianBlur(img, (0, 0), 1.2)
    return cv2.addWeighted(img, 1.0 + amount, flou, -amount, 0)

def lissage(t):
    """Adoucit le fondu (smoothstep) : demarrage et fin en douceur."""
    return t * t * (3.0 - 2.0 * t)

def melanger(live, milena, presence):
    """Fusionne le flux live avec l'image de Milena selon MODE_MELANGE.
    presence : 0 = Milena invisible, 1 = pleine presence.
    v7 : screen calcule en uint8 natif (SIMD)."""
    if milena is None or MODE_MELANGE == "aucun" or presence <= 0.0:
        return live
    if MODE_MELANGE == "screen":
        # screen : 255 - (255-a)(255-b)/255, en uint8 natif
        inv = cv2.multiply(cv2.bitwise_not(live), cv2.bitwise_not(milena),
                           scale=1.0 / 255.0)
        fusion = cv2.bitwise_not(inv)
    else:  # "alpha"
        fusion = cv2.addWeighted(live, 0.5, milena, 0.5, 0)
    return cv2.addWeighted(live, 1.0 - presence, fusion, presence, 0)

# === ETATS ===
fondu = 0.0
derniere_oeil = None
derniere_bouche = None
t_precedent = time.monotonic()

proximite = 0.0          # 0 = loin, 1 = tout proche (valeur lissee)
intimite = 0.0           # 0 = hors zone d'intimite, 1 = en plein dedans (lissee)
t_dernier_debug = 0.0

# stabilisation du cadre (valeurs lissees, None = non initialise)
oeil_cx_s = oeil_cy_s = oeil_zw_s = None
bou_cx_s = bou_cy_s = bou_zw_s = None

def lisser_cadre(actuel, cible, a):
    """Filtre passe-bas 1er ordre. actuel None -> initialise a cible."""
    if actuel is None:
        return float(cible)
    return actuel + (cible - actuel) * a

with FaceLandmarker.create_from_options(options) as landmarker:
    while True:
        frame_brute = picam2.capture_array()
        frame_rgb = cv2.rotate(frame_brute, CAMERA_ROTATION)
        h, w = frame_rgb.shape[:2]

        # v7 : detection sur image reduite, crops sur image pleine
        if DETECTION_ECHELLE < 1.0:
            petite = cv2.resize(frame_rgb, None,
                                fx=DETECTION_ECHELLE, fy=DETECTION_ECHELLE)
            petite = np.ascontiguousarray(petite)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=petite)
        else:
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

            # courbe de revelation : la nettete/effacement de Milena se
            # concentre sur la fin de l'approche (plus spectaculaire)
            p_eff = proximite ** REVELATION_COURBE
            presence_milena = PRESENCE_MILENA_LOIN + \
                (PRESENCE_MILENA_PROCHE - PRESENCE_MILENA_LOIN) * lissage(p_eff)
            li = lissage(intimite)
            presence_milena = presence_milena * (1.0 - li) + \
                PRESENCE_MILENA_INTIME * li

            # nettete progressive : nulle au loin, maximale au point optimum
            nettete = NETTETE_MAX * p_eff

            if maintenant - t_dernier_debug > 1.0:
                t_dernier_debug = maintenant
                fps = 1.0 / dt if dt > 0 else 0.0
                print(f"oeil {oeil_l:5.0f}px  proximite {proximite:4.2f}  "
                      f"intimite {intimite:4.2f}  "
                      f"presence Milena {presence_milena:4.2f}  {fps:4.1f} fps")

            # ================= OEIL =================
            zone_w = oeil_l * OEIL_ECRAN_W / OEIL_LARGEUR_ECRAN
            xs = [lm[i].x * w for i in ORBITE]
            ys = [lm[i].y * h for i in ORBITE]
            cx = np.mean(xs)
            cy = np.mean(ys) - DECALAGE_SOURCIL
            # stabilisation : lisse centre + zoom du cadre
            oeil_cx_s = lisser_cadre(oeil_cx_s, cx, STAB_LISSAGE)
            oeil_cy_s = lisser_cadre(oeil_cy_s, cy, STAB_LISSAGE)
            oeil_zw_s = lisser_cadre(oeil_zw_s, zone_w, STAB_LISSAGE)
            zone_w = int(oeil_zw_s)
            zone_h = int(zone_w * OEIL_ECRAN_H / OEIL_ECRAN_W)
            x1 = int(oeil_cx_s) - zone_w // 2
            y1 = int(oeil_cy_s) - zone_h // 2
            crop_oeil = crop_sur(frame_rgb, x1, y1, x1 + zone_w, y1 + zone_h)
            crop_oeil = flouter_crop(crop_oeil, p_eff, OEIL_ECRAN_W)
            live_oeil = traiter_crop(crop_oeil, OEIL_ECRAN_W, OEIL_ECRAN_H)
            live_oeil = recolorer(live_oeil, oeil_milena, RECOLOR_MILENA)
            live_oeil = accentuer(live_oeil, nettete)
            live_oeil = melanger(live_oeil, oeil_milena, presence_milena)

            # ================= BOUCHE =================
            centre_x = (lm[BOUCHE_COIN_G].x + lm[BOUCHE_COIN_D].x) / 2 * w
            demi_l = abs(lm[BOUCHE_COIN_G].x * w - centre_x)
            zone_w = demi_l * BOUCHE_ECRAN_W / BOUCHE_LARGEUR_ECRAN
            bys = [lm[i].y * h for i in BOUCHE_LM]
            bcy = np.mean(bys)
            bcx = centre_x
            # stabilisation : lisse centre + zoom du cadre
            bou_cx_s = lisser_cadre(bou_cx_s, bcx, STAB_LISSAGE)
            bou_cy_s = lisser_cadre(bou_cy_s, bcy, STAB_LISSAGE)
            bou_zw_s = lisser_cadre(bou_zw_s, zone_w, STAB_LISSAGE)
            zone_w = int(bou_zw_s)
            zone_h = int(zone_w * BOUCHE_ECRAN_H / BOUCHE_ECRAN_W)
            bx1 = int(bou_cx_s) - zone_w
            decal_haut = int(zone_h * BOUCHE_POSITION_HAUT / BOUCHE_ECRAN_H)
            by1 = int(bou_cy_s) - decal_haut
            crop_bouche = crop_sur(frame_rgb, bx1, by1, bx1 + zone_w, by1 + zone_h)
            crop_bouche = flouter_crop(crop_bouche, p_eff, BOUCHE_ECRAN_W)
            live_bouche = traiter_crop(crop_bouche, BOUCHE_ECRAN_W, BOUCHE_ECRAN_H)
            live_bouche = recolorer(live_bouche, bouche_milena, RECOLOR_MILENA)
            live_bouche = accentuer(live_bouche, nettete)
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
