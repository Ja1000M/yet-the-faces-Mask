# Yet The Faces — Et pourtant les visages

Masque connecté pour la performeuse Milena Trivier, conçu par le collectif **G_1000&6 lots** (Jamil Mehdaoui & Cécile Kretschmar).

## Le projet

Milena est prosopagnosique. Le masque, réalisé en fil de cuivre soudé selon les 478 points de reconnaissance faciale MediaPipe (calqués sur son propre visage), capte en temps réel le visage de la personne en face d'elle et en diffuse des fragments — œil et bouche — sur deux écrans intégrés au masque. Ces écrans sont visibles par la personne en interaction, pas par Milena.

## Structure du dépôt

- `mask_phase_a_*.py` — scripts de simulation développés sur Mac (Phase A) : détection de landmarks, extraction de zones, rendus visuels
- `mask_hologramme*.py` — itérations successives du rendu visuel de l'écran œil (versions explorées : effet holographique cyan, puis rendu naturel/photographique retenu dans `mask_hologramme3.py`)
- `yet-the-faces-journal-technique.md` — journal technique complet : état d'avancement, configuration matérielle et logicielle, problèmes résolus

## Phases de développement

1. **Phase A — Simulation sur Mac** (terminée) : pipeline complet validé via webcam
2. **Phase B — Migration Raspberry Pi 5** (en cours) : environnement Python/MediaPipe opérationnel sur Raspberry Pi OS Bookworm
3. **Phase C — Intégration physique** (à venir) : montage dans le masque, câblage, alimentation

## Démarrage rapide (Mac)

```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python mediapipe
python3 mask_hologramme3.py
```

## Équipe

Performeuse : Milena Trivier
Création : Cécile Kretschmar, Jamil Mehdaoui (G_1000&6 lots)
Production : Maxime Coton
Module reconnaissance faciale (projet distinct, même dispositif physique) : Jorge, Anna
