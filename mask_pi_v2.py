import cv2
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2
import pygame

MODEL_PATH = "/home/jamil/yet-the-faces-mask/face_landmarker.task"

ORBITE = [133, 33, 159, 145]
OEIL_CAPTURE_W = 400
OEIL_CAPTURE_H = 200
DECALAGE_SOURCIL = 20

BOUCHE = [61, 291, 0, 17]
BOUCHE_CAPTURE_W = 200
BOUCHE_CAPTURE_H = 350

picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)
picam2.configure(config)
picam2.start()

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
)

pygame.init()
nb_ecrans = pygame.display.get_num_displays()
print(f"Nombre d'écrans détectés : {nb_ecrans}")

# HDMI-1 portrait bouche : écran index 0
# HDMI-2 paysage oeil : écran index 1
ecran_bouche = pygame.display.set_mode((480, 800), pygame.FULLSCREEN | pygame.NOFRAME, display=1)
ecran_oeil = pygame.display.set_mode((800, 480), pygame.FULLSCREEN | pygame.NOFRAME, display=0)

def traiter_crop(crop_rgb, dest_w, dest_h):
    if crop_rgb.size == 0:
        return np.zeros((dest_h, dest_w, 3), dtype=np.uint8)
    resized = cv2.resize(crop_rgb, (dest_w, dest_h))
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= 0.6
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    result[:, :, 2] = np.clip(result[:, :, 2].astype(np.int16) + 10, 0, 255)
    return result

def np_to_surface(arr):
    return pygame.surfarray.make_surface(np.transpose(arr, (1, 0, 2)))

with FaceLandmarker.create_from_options(options) as landmarker:
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                running = False

        frame_rgb = picam2.capture_array()
        h, w = frame_rgb.shape[:2]

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)

        out_oeil = np.zeros((480, 800, 3), dtype=np.uint8)
        out_bouche = np.zeros((800, 480, 3), dtype=np.uint8)

        if result.face_landmarks:
            lm = result.face_landmarks[0]

            # Zone oeil
            xs = [lm[i].x * w for i in ORBITE]
            ys = [lm[i].y * h for i in ORBITE]
            cx = int(np.mean(xs))
            cy = int(np.mean(ys)) - DECALAGE_SOURCIL
            x1 = max(0, cx - OEIL_CAPTURE_W // 2)
            y1 = max(0, cy - OEIL_CAPTURE_H // 2)
            x2 = min(w, x1 + OEIL_CAPTURE_W)
            y2 = min(h, y1 + OEIL_CAPTURE_H)
            crop_oeil = frame_rgb[y1:y2, x1:x2]
            out_oeil = traiter_crop(crop_oeil, 800, 480)

            # Zone bouche
            bxs = [lm[i].x * w for i in BOUCHE]
            bys = [lm[i].y * h for i in BOUCHE]
            bcx = int(np.mean(bxs))
            bcy = int(np.mean(bys))
            bx1 = max(0, bcx)
            by1 = max(0, bcy - 20)
            bx2 = min(w, bcx + BOUCHE_CAPTURE_W)
            by2 = min(h, by1 + BOUCHE_CAPTURE_H)
            crop_bouche = frame_rgb[by1:by2, bx1:bx2]
            bouche = traiter_crop(crop_bouche, 480, 800)
            out_bouche = cv2.flip(bouche, 0)

        ecran_oeil.blit(np_to_surface(out_oeil), (0, 0))
        ecran_bouche.blit(np_to_surface(out_bouche), (0, 0))
        pygame.display.flip()

picam2.stop()
pygame.quit()
