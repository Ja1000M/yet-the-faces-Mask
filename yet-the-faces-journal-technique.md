# Yet The Faces — Journal technique complet
**État au 19 juillet 2026**

## Contexte du projet

Masque en fil de cuivre pour Milena Trivier (prosopagnosique), basé sur les 478 points de reconnaissance faciale MediaPipe comme nœuds soudés. Le masque intègre deux écrans qui diffusent en temps réel des fragments du visage de la personne en interaction avec Milena (zone œil et zone bouche) — pas le visage de Milena elle-même. Collectif G_1000&6 lots (Jamil Mehdaoui, Cécile Kretschmar). Performeuse : Milena. Production : Maxime, Jorge, Anna.

---

## Phase A — Simulation sur Mac (terminée)

### Environnement
- macOS, Xcode Command Line Tools installés
- Python 3.9.6, venv dans `~/yet-the-faces-mask/venv`
- OpenCV 4.13.0 + MediaPipe 0.10.35 (API moderne `mediapipe.tasks`, pas l'ancienne `mp.solutions`)
- VS Code avec interpréteur venv sélectionné
- Modèle `face_landmarker.task` téléchargé dans le dossier projet

### Scripts développés
- `mask_phase_a_033.py` — capture webcam + 478 landmarks affichés (1 point sur 3, rayon 2px, vert)
- `mask_phase_a_ecrans.py` — extraction zones œil (paysage 800×480) et bouche (portrait 480×800) avec letterbox noir pour respecter les proportions
- `mask_hologramme3.py` — version finale du rendu : extraction zone orbitale à l'échelle 1 (zone capturée 300×180px recentrée vers le sourcil via `DECALAGE_SOURCIL`), désaturation froide légère, vignette douce aux bords noirs. **C'est la direction retenue après que Milena a partagé une photo du masque réel** — rendu photographique/clinique plutôt qu'hologramme cyan futuriste (testé et abandonné : scanlines, aberration chromatique RGB, teinte cyan — trop "déjà-vu numérique")

### Paramètres clés à retenir
- Indices landmarks orbite : `[133, 33, 159, 145]`
- `OEIL_CAPTURE_W=300, OEIL_CAPTURE_H=180` (zone source avant resize)
- Webcam Mac capture en 1920×1080
- Idée non implémentée mais validée conceptuellement : superposer en transparence une boucle vidéo de l'œil de Milena avec le flux temps réel de la personne en face (pas encore de vidéo source disponible)

---

## Phase B — Migration Raspberry Pi (en cours)

### Matériel
- Raspberry Pi 5 8GB (occasion, 171€, ventilateur Active Cooler + alim 27W officiels)
- Camera Module 3 Wide Noir (autofocus, 12MP, grand angle 120°) + câble CSI FPC 22Pin→15Pin — reçus et branchés
- 2× écrans Waveshare 4" 480×800 HDMI — fonctionnels
- Carte SanDisk Extreme Pro 64GB
- Batterie Anker 20000mAh
- 2× câbles micro HDMI vers HDMI (Thsucords, 30cm)
- Aimants néodyme (lot commandé sur Amazon)
- Oreillette Bluetooth — **hors périmètre**, projet de Jorge/Anna (reconnaissance faciale + retour audio à Milena)

### Configuration système — point critique
- **Première tentative ratée** : Raspberry Pi OS par défaut = Trixie (Debian 13, Python 3.13) → MediaPipe totalement incompatible, aucune wheel ARM64 pour Python 3.13 (`ModuleNotFoundError: No module named 'mediapipe.python._framework_bindings'`)
- **Solution** : reflasher avec **Raspberry Pi OS (Legacy, 64-bit)** = Debian 12 Bookworm, Python 3.11 natif
- Hostname : `yetthefaces.local`, user : `jamil`, SSH activé, WiFi via Raspberry Pi Imager
- Connexion : `ssh jamil@yetthefaces.local`

### Installation logicielle réussie (sur Bookworm)
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv python3-opencv
mkdir ~/yet-the-faces-mask && cd ~/yet-the-faces-mask
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install mediapipe --break-system-packages
```
**Ne pas utiliser** `mediapipe-rpi4` ou `mediapipe-rpi5` (packages piwheels cassés). Le simple `pip install mediapipe` résout la bonne wheel ARM64.

Stack validée : OpenCV 4.11.0, MediaPipe 0.10.18. Modèle `face_landmarker.task` dans `~/yet-the-faces-mask/`.

---

## Phase B — Session du 17 juillet 2026 (dispositif double écran)

### Config matérielle
- **Pi 5 sous Bookworm, passé de Wayland à X11** (via `raspi-config`) — décision clé : `cv2.moveWindow` ne fonctionne que sous X11
- 2 écrans Waveshare 4" 480×800 : **HDMI-1 = portrait bouche** (x=0), **HDMI-2 = paysage œil**, rotation `right` (x=480)
- Caméra montée physiquement à 90° → redressée dans le script (`cv2.ROTATE_90_COUNTERCLOCKWISE`)

### Scripts (`~/yet-the-faces-mask/`, GitHub à jour)
- **`mask_pi_v4.py` = version fonctionnelle complète** :
  - Cadrage dynamique par landmarks (œil droit `[362, 263, 386, 374]`, demi-bouche)
  - Superposition images `oeil_milena.png` + `bouche_milena.png` (`addWeighted` 75/25, désaturation 0.4, flou gaussien 7)
  - Autofocus continu + `Sharpness 2.0`
  - Milena seule affichée quand aucun visage détecté
- Lancement :
  ```bash
  source ~/yet-the-faces-mask/venv/bin/activate
  DISPLAY=:0 python3 ~/yet-the-faces-mask/mask_pi_v4.py
  ```

### Problème rencontré (résolu le 18/07)
- Config xrandr non persistante au reboot ; xrandr **gèle systématiquement sur tout modeset à chaud** (lecture OK)
- Modeline inventée dans xorg.conf.d → écrans brouillés (fichier supprimé, à ne jamais refaire)
- Après reboot, perte d'EDID : mode natif `480×800 62.29` absent de xrandr (`0mm×0mm`)

---

## Phase B — Session du 18 juillet 2026 : CONFIG ÉCRANS RÉSOLUE ✔

### Diagnostic final
1. **Le shutdown complet + débranchage total** (Pi ET USB écrans) a restauré l'EDID — confirmé au boot : `480x800 62.29 +` sur les deux sorties, dimensions physiques réelles (150mm×100mm)
2. **Cause racine identifiée** : les écrans Waveshare (alimentés par l'USB du Pi) ne fournissent pas leur EDID assez tôt dans la séquence de boot → X démarre en fallback 1024×768 miroir (« deux bureaux »). Un fichier xorg.conf avec `PreferredMode` seul ne suffit donc pas (rotation/position appliquées, mode ignoré)
3. **Confirmé** : tout modeset xrandr à chaud gèle X (reproductible, driver vc4). Règle : **ne plus jamais changer de mode à chaud** — tout se joue au boot

### Solution définitive (deux fichiers)
**a) EDID capturé sur disque** — le vrai EDID de l'écran, lu une fois et sauvegardé :
```bash
sudo mkdir -p /lib/firmware/edid
cat /sys/class/drm/card*-HDMI-A-1/edid | sudo tee /lib/firmware/edid/waveshare-480x800.bin > /dev/null
# vérification : wc -c → 128 octets (bloc EDID valide)
```
Paramètre kernel ajouté en fin de ligne dans `/boot/firmware/cmdline.txt` :
```
drm.edid_firmware=HDMI-A-1:edid/waveshare-480x800.bin,HDMI-A-2:edid/waveshare-480x800.bin
```

**b) `/etc/X11/xorg.conf.d/10-ytf-ecrans.conf`** — AUCUNE Modeline :
```
Section "Monitor"
    Identifier "HDMI-1"
    Option "PreferredMode" "480x800"
    Option "Rotate" "normal"
    Option "Position" "0 0"
    Option "Primary" "true"
EndSection

Section "Monitor"
    Identifier "HDMI-2"
    Option "PreferredMode" "480x800"
    Option "Rotate" "right"
    Option "Position" "480 0"
EndSection
```

### Résultat validé après cold boot
```
Monitors: 2
 0: +*HDMI-1 480/150x800/100+0+0   HDMI-1
 1: +HDMI-2 800/150x480/100+480+0  HDMI-2
```
Bureau étendu 1280×800, définitions natives, rotation et position correctes, **au boot, sans aucune commande xrandr**.

### Validation visuelle : `test_ecrans.py` (v2, dans le repo)
Script de calibration : aplat rouge « BOUCHE / HDMI-1 / 480x800 » et vert « OEIL / HDMI-2 / 800x480 », flèche HAUT, lecture de la géométrie via `xrandr --listmonitors` (rien en dur). Les deux écrans affichent simultanément, à l'endroit, en définition native. ✔

**Leçon fenêtres OpenCV** : deux fullscreen ouverts coup sur coup s'empilent sur le même écran. Séquence robuste : `namedWindow` → `imshow` → `waitKey(200)` → `moveWindow` → `resizeWindow` → `waitKey(200)` → fullscreen → `waitKey(200)` → **re-`moveWindow`**. À reporter dans `mask_pi_v4.py` si le problème s'y présente.

### Divers
- Reboot après gel de X : le warm reboot peut accrocher (LED verte fixe, hors réseau) → coupure franche de l'alim, sans danger, et bénéfique pour l'EDID
- Alimentation écrans : décision = **rester sur l'USB du Pi** pour la portabilité (ou tout sur la batterie Anker, à traiter plus tard). Attention Pi 5 : sur alim non-5A, limite USB à 600mA → si overcurrent sur batterie, ajouter `usb_max_current_enable=1` dans `/boot/firmware/config.txt` ou brancher les écrans directement sur les ports de l'Anker
- Barre de tâches encore visible en haut de l'écran bouche pendant le test — à masquer à l'étape autostart/kiosque

### `mask_pi_v5.py` — fondu enchaîné (18/07, validé)
- v4 + fade in/out entre « Milena seule » et la capture live :
  - `FADE_IN_DUREE = 0.8` s (apparition du visage), `FADE_OUT_DUREE = 1.2` s (effacement)
  - Fondu basé sur l'horloge réelle (`time.monotonic`), indépendant du framerate
  - Courbe smoothstep (`t*t*(3-2t)`) pour un départ/arrivée en douceur
  - Dernière image live mémorisée → le fade out part du visage capté et se dissout vers Milena
  - Effet secondaire bienvenu : les micro-pertes de détection ondulent au lieu de faire clignoter
- `mask_pi_v4.py` conservé comme version de référence sans fondu

---

## Phase B — Session du 19 juillet 2026 : rendu de l'image (v6 → v7)

### Problème d'allumage constaté (à résoudre avec l'alim)
Au boot, les écrans restent noirs (clignotement vague) : alimentés par l'USB du Pi,
ils s'allument trop tard et ratent l'accrochage du signal HDMI (que notre config
EDID émet pourtant très tôt). **Un débranchement/rebranchement des USB écrans suffit**
à accrocher l'image — inacceptable une fois le dispositif monté sur le masque.
À traiter avec le chantier alimentation (piste : USB écrans sur la batterie Anker,
allumés avant le Pi ; et/ou `usb_max_current_enable=1`).

Autre point : la barre de tâches (lxpanel) reste devant la fenêtre BOUCHE.
Contournement immédiat : `pkill lxpanel` (retour : `DISPLAY=:0 lxpanel -p LXDE-pi &`).
Solution définitive au chantier autostart/kiosque.

### `mask_pi_v6.py` — travail du rendu (validé par étapes successives)
Le rendu « froid/clinique » hérité de la Phase A (désaturation 0.6 + décalage bleu)
était jugé triste et grisâtre. Paramètres exposés et ajustés à l'œil sur les écrans :
- `FLUX_SATURATION 1.05`, `FLUX_LUMINOSITE 1.05`, `FLUX_CONTRASTE 1.18`
- `FLUX_TEMPERATURE 5` avec **protection des blancs** (`TEMPERATURE_SEUIL_BLANC 200`,
  courbe progressive) — corrige le blanc de l'œil qui jaunissait
- Séquence fenêtres robuste appliquée (la fenêtre BOUCHE s'empilait derrière OEIL)
- **Mode de mélange « screen »** (superposition lumineuse, double exposition) à la
  place de la moyenne addWeighted : les deux regards (capté + Milena) coexistent
  sans se noyer dans le gris — nettement plus percutant

### Dramaturgie de la proximité (introduite, à affiner)
La distance est estimée par la largeur de l'œil capté en pixels. Trois actes :
1. **Loin** : flux flouté (45) + Milena pleine présence (1.0)
2. **Approche** : révélation progressive — le visage capté devient net, Milena
   s'efface (courbe `REVELATION_COURBE 2.0` : dévoilement concentré sur la fin)
3. **Zone d'intimité** (`oeil ≥ 240px`) : Milena réémerge (0.9) et « défend sa
   zone » — trouvaille née d'un comportement accidentel, conservée volontairement
Paramètres de calibration : `OEIL_PX_LOIN 50 / PROCHE 180 / INTIME 240-320`,
lissage temporel 0.12. Debug console 1×/s : largeur œil px, proximité, intimité,
présence Milena, fps.
**État : l'effet fonctionne mais n'est pas encore concluant — calibration des
seuils et force de l'effet à reprendre à la prochaine session, en conditions réelles.**

### `mask_pi_v7.py` — optimisation : 7 → 18 fps
- Détection MediaPipe sur image réduite de moitié (`DETECTION_ECHELLE 0.5`) —
  les landmarks étant normalisés, la précision reste suffisante ; les crops
  restent en pleine résolution
- Flou appliqué au petit crop **avant** agrandissement (noyau mis à l'échelle)
- Calculs couleur en opérations OpenCV natives uint8/SIMD (LUT température
  précalculée, `convertScaleAbs`, screen en `multiply`/`bitwise_not`) —
  plus de flottants numpy par image
- Mesuré : **18 fps** (contre 7 avec v6) — fluide pour le dispositif

---

## Phase B — Session du 22 juillet 2026 : effet proximité VALIDÉ (v8)

### `mask_pi_v8.py` = version de référence
Apports depuis v7 : masque de netteté progressif (unsharp, `NETTETE_MAX`),
agrandissement `INTER_CUBIC`, cadrage bouche affiné (`BOUCHE_LARGEUR_ECRAN 270`,
`BOUCHE_POSITION_HAUT 250`).

### Calibration de la dramaturgie (mesures réelles console)
Découverte clé : avec la caméra **NoIR grand angle (imx708_wide_noir)**, la largeur
d'œil captée plafonne à ~90-105 px (jamais 180 comme supposé). Les seuils d'origine
rendaient l'effet inopérant (proximité bloquée < 0.15, Milena jamais effacée).
**Seuils recalés sur la plage réelle** : `OEIL_PX_LOIN 33`, `OEIL_PX_PROCHE 68`,
zone d'intimité `OEIL_PX_INTIME 85 → INTIME_MAX 105`.
Arc validé en conditions réelles : loin (~33px) presence 1.00 → approche → révélation
pleine → très près (~105px) Milena réémerge (« défend sa zone »). fps stables 15-16.

**VALEURS DE CALIBRATION VALIDÉES (22/07) — à conserver, ne changer qu'avec mesure console :**

| Paramètre | Valeur | Rôle |
|---|---|---|
| `OEIL_PX_LOIN` | 33 | œil (px) = « loin », Milena pleine |
| `OEIL_PX_PROCHE` | 55 | œil (px) = révélation pleine (net + seul). ↓ = net plus tôt/loin, ↑ = plus tard/près |
| `OEIL_PX_INTIME` | 85 | œil (px) = début zone d'intimité (Milena revient) |
| `OEIL_PX_INTIME_MAX` | 105 | œil (px) = Milena pleinement revenue (≈ max atteignable) |
| `PRESENCE_MILENA_LOIN` | 1.0 | présence Milena au loin |
| `PRESENCE_MILENA_PROCHE` | 0.2 | présence Milena au point de révélation (0.2 = on la sent encore un peu) |
| `PRESENCE_MILENA_INTIME` | 0.9 | présence Milena au cœur de l'intimité |
| `REVELATION_COURBE` | 2.0 | 1 = linéaire, 2-3 = révélation tardive/spectaculaire |
| `PROXIMITE_LISSAGE` | 0.12 | amortissement temporel (0.05 lent, 0.3 réactif) |
| `FLOU_LOIN` | 55 | flou assumé quand loin |
| `NETTETE_MAX` | 0.7 | accentuation (unsharp) au point net |

Repères px ↔ distance (caméra NoIR grand angle) : ~33px = loin / ~55px = point de
révélation validé / ~105px = collé au masque (max).

### Rendu couleur ajusté (retour Milena) — VALEURS VALIDÉES 22/07
Le live était jugé trop sombre/contrasté/bleu vs la teinte chaude des images de
Milena. Réglages validés : `FLUX_SATURATION 1.05`, `FLUX_LUMINOSITE 1.15`,
`FLUX_CONTRASTE 1.10`, `FLUX_TEMPERATURE 12`, `TEMPERATURE_SEUIL_BLANC 235`.
Piste si encore trop froid : forcer la balance des blancs caméra (`AwbMode`
tungsten/indoor) plutôt que corriger après coup — la NoIR tire vers le bleu-violet.

### Reste à faire (dans l'ordre)
1. ~~Config écrans pérenne au boot~~ ✔ **FAIT (18/07)**
2. ~~Affiner l'effet proximité/révélation~~ ✔ **FAIT (22/07, v8 calibrée)**
3. Alimentation : écrans allumés avant/indépendamment du Pi (batterie Anker,
   `usb_max_current_enable=1`) — règle aussi le problème d'allumage au boot
4. Systemd autostart de `mask_pi_v7.py` (+ masquer lxpanel / mode kiosque)
5. Streaming pour le module de Jorge/Anna (reco faciale — **hors périmètre**).
   **Piste privilégiée (19/07) : liaison Ethernet filaire directe entre les deux
   Pi 5** (câble RJ45 seul, IP fixes) — le Gigabit permet le flux 720p quasi sans
   compression (~400 Mbit/s brut) avec latence minimale, là où le WiFi imposerait
   compression et aléas. Notes qualité : la capture reste en 1280×720 pleine
   résolution (la réduction DETECTION_ECHELLE ne concerne que la copie analysée
   par MediaPipe en interne) ; le stream partirait de `frame_rgb` intact ; la
   Camera Module 3 peut monter au-delà de 720p si leur reco l'exige. Contrainte
   à discuter avec l'équipe : le câble physique masque → second Pi (scénographie).
6. Boucle vidéo œil/bouche de Milena à la place des images fixes

---

## Phase C — Intégration physique (pas commencée)

Câbles FPC dorés pour les écrans (esthétique cuivre/cohérence visuelle avec le masque) — commande AliExpress envisagée mais pas faite, les câbles HDMI noirs Thsucords servent de remplacement temporaire pour les tests.

---

## Échéance

Prototype shoot fin juillet avec Milena, Maxime, Cécile, Jorge, Anna.

---

## Substack "Yet The Faces"

Diffusion privée : Milena, Maxime, Cécile, Jorge, Anna. Trois premiers posts rédigés (français + traduction anglaise à faire), non publiés :
- Post 1 — présentation générale du dispositif (basé sur le document `dispositif_prosopagnosique.pdf`)
- Post 2 — session du 4 juin (installation environnement Mac, premiers landmarks)
- Post 3 — session du 5 juin (extraction zones, matériel) — *à vérifier, possible confusion de dates entre posts 2 et 3 à reclarifier*
- Post 4 (à rédiger) — session du 6 juin (recherche du rendu visuel, pivot hologramme→naturel)

Signature : "Jamil, pour G_1000&6 lots". Ton factuel pour ces premiers posts, registre libre pour la suite.
