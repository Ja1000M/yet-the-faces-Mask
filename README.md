# Yet The Faces — Et pourtant les visages

🇫🇷 [Français](#français) | 🇬🇧 [English](#english)

---

## Français

Masque connecté pour la performeuse Milena Trivier, conçu par le collectif **G_1000&6 lots** (Jamil Mehdaoui & Cécile Kretschmar).

### Le projet

Milena est prosopagnosique. Le masque, réalisé en fil de cuivre soudé selon les 478 points de reconnaissance faciale MediaPipe (calqués sur son propre visage), capte en temps réel le visage de la personne en face d'elle et en diffuse des fragments — œil et bouche — sur deux écrans intégrés au masque. Ces écrans sont visibles par la personne en interaction, pas par Milena.

### Structure du dépôt

- `mask_phase_a_*.py` — scripts de simulation développés sur Mac (Phase A) : détection de landmarks, extraction de zones, rendus visuels
- `mask_hologramme*.py` — itérations successives du rendu visuel de l'écran œil (versions explorées : effet holographique cyan, puis rendu naturel/photographique retenu dans `mask_hologramme3.py`)
- `yet-the-faces-journal-technique.md` — journal technique complet : état d'avancement, configuration matérielle et logicielle, problèmes résolus

### Phases de développement

1. **Phase A — Simulation sur Mac** (terminée) : pipeline complet validé via webcam
2. **Phase B — Migration Raspberry Pi 5** (en cours) : environnement Python/MediaPipe opérationnel sur Raspberry Pi OS Bookworm
3. **Phase C — Intégration physique** (à venir) : montage dans le masque, câblage, alimentation

### Démarrage rapide (Mac)

```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python mediapipe
python3 mask_hologramme3.py
```

### Équipe

Performeuse : Milena Trivier
Création : Cécile Kretschmar, Jamil Mehdaoui (G_1000&6 lots)
Production : Maxime Coton
Module reconnaissance faciale (projet distinct, même dispositif physique) : Jorge, Anna

---

## English

Connected mask for performer Milena Trivier, designed by the **G_1000&6 lots** collective (Jamil Mehdaoui & Cécile Kretschmar).

### The project

Milena has prosopagnosia. The mask, made of soldered copper wire following MediaPipe's 478 facial recognition points (mapped from her own face), captures in real time the face of the person interacting with her and broadcasts fragments of it — eye and mouth — onto two screens embedded in the mask. These screens are visible to the person interacting with her, not to Milena herself.

### Repository structure

- `mask_phase_a_*.py` — simulation scripts developed on Mac (Phase A): landmark detection, zone extraction, visual rendering
- `mask_hologramme*.py` — successive iterations of the eye screen's visual rendering (explored: cyan holographic effect, then the natural/photographic render retained in `mask_hologramme3.py`)
- `yet-the-faces-journal-technique.md` — full technical journal: progress, hardware/software setup, issues resolved

### Development phases

1. **Phase A — Mac simulation** (complete): full pipeline validated via webcam
2. **Phase B — Raspberry Pi 5 migration** (in progress): Python/MediaPipe environment operational on Raspberry Pi OS Bookworm
3. **Phase C — Physical integration** (upcoming): mounting into the mask, wiring, power

### Quick start (Mac)

```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python mediapipe
python3 mask_hologramme3.py
```

### Team

Performer: Milena Trivier
Creation: Cécile Kretschmar, Jamil Mehdaoui (G_1000&6 lots)
Production: Maxime Coton
Facial recognition module (separate project, same physical device): Jorge, Anna
