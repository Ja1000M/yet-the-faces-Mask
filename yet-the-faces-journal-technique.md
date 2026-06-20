# Yet The Faces — Journal technique complet
**État au 19 juin 2026**

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

### Matériel Pi 5 reçu
- Raspberry Pi 5 8GB acheté d'occasion (171€, avec ventilateur Active Cooler + alim 27W officiels inclus)
- Camera Module 3 Wide Noir (autofocus, 12MP, grand angle 120°)
- Carte SanDisk Extreme Pro 64GB (achetée par sécurité après une carte 1TB suspecte de contrefaçon, marque HUNYEIZ, écartée)
- Batterie Anker 20000mAh
- 2× câbles micro HDMI vers HDMI (marque Thsucords, 30cm)

### Matériel en attente
- Câble CSI FPC 22Pin→15Pin (Pi 5 vers Camera Module 3) — livraison prévue 19/06, **bloquant actuel**
- Écrans (2×, format 4 pouces 800×480 HDMI) — Waveshare écarté car délai Amazon trop long (livraison annoncée 2 août, après le tournage) ; alternative Kubii (34€, réf WV12030, compatibilité Pi 5 confirmée en fiche technique) envisagée mais pas commandée, pas de marque fabricant claire
- Aimants néodyme (lot commandé sur Amazon)
- Oreillette Bluetooth — **hors périmètre**, c'est le projet de Jorge/Anna (reconnaissance faciale + retour audio à Milena), pas celui de Jamil

### Configuration système — point critique
- **Première tentative ratée** : Raspberry Pi OS par défaut = Trixie (Debian 13, Python 3.13) → MediaPipe totalement incompatible, aucune wheel ARM64 disponible pour Python 3.13, même en compilant Python 3.11 depuis les sources en parallèle (essayé, fonctionnel mais MediaPipe restait cassé : `ModuleNotFoundError: No module named 'mediapipe.python._framework_bindings'`)
- **Solution** : reflasher la carte microSD avec **Raspberry Pi OS (Legacy, 64-bit)** = Debian 12 Bookworm, qui inclut nativement Python 3.11
- Hostname : `yetthefaces.local`, user : `jamil`, SSH activé, WiFi configuré via Raspberry Pi Imager
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
**Ne pas utiliser** `mediapipe-rpi4` ou `mediapipe-rpi5` (packages piwheels cassés/obsolètes, échouent avec la même erreur `_framework_bindings`). Le simple `pip install mediapipe` fonctionne et résout automatiquement la bonne wheel ARM64.

### Résultat validé
```
OpenCV : 4.11.0
MediaPipe : 0.10.18
Environnement OK
```
Modèle `face_landmarker.task` déjà téléchargé dans `~/yet-the-faces-mask/` sur le Pi.

---

## Phase C — Intégration physique (pas commencée)

Câbles FPC dorés pour les écrans (esthétique cuivre/cohérence visuelle avec le masque) — commande AliExpress envisagée mais pas faite, les câbles HDMI noirs Thsucords servent de remplacement temporaire pour les tests.

---

## Prochaines étapes immédiates

1. Recevoir et brancher le câble CSI FPC
2. Tester la Camera Module 3 sur le Pi (remplacer la webcam Mac par `picamera2` dans le pipeline)
3. Migrer/adapter les scripts Phase A (`mask_hologramme3.py` notamment) pour tourner sur le Pi avec la caméra native
4. Trancher sur les écrans (Kubii vs autre solution) en fonction du délai contraint par le tournage fin juillet
5. Reprendre la commande Phase C (câbles dorés, aimants déjà en cours)

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
