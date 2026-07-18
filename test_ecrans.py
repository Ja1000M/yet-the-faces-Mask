#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_ecrans.py (v2) — Calibration écrans « Yet The Faces »
----------------------------------------------------------
Affiche un aplat de couleur plein écran sur CHAQUE sortie détectée :
  - BOUCHE = ROUGE   (sortie HDMI-1)
  - OEIL   = VERT    (sortie HDMI-2)

But : valider écran par écran, sans la logique du masque,
      1. l'IDENTITE     (quelle sortie = quel écran physique)
      2. l'ORIENTATION  (le texte et la flèche HAUT doivent être à l'endroit)
      3. la DEFINITION  (la résolution affichée doit être la native)

v2 : séquence fenêtres renforcée — créer -> afficher -> positionner ->
     fullscreen -> re-positionner, avec pauses pour laisser le gestionnaire
     de fenêtres suivre (évite l'empilement des deux fenêtres sur un écran).

Lancement :
    source ~/yet-the-faces-mask/venv/bin/activate
    DISPLAY=:0 python3 ~/yet-the-faces-mask/test_ecrans.py

Quitter : touche Échap ou 'q' (fenêtre active).
"""

import subprocess
import re
import sys
import cv2
import numpy as np

# --- Association sortie -> (étiquette, couleur BGR) --------------------------
ASSOCIATION = {
    "HDMI-1": ("BOUCHE", (0, 0, 255)),   # rouge
    "HDMI-2": ("OEIL",   (0, 255, 0)),   # vert
}
SECOURS = [("ECRAN A", (0, 0, 255)), ("ECRAN B", (0, 255, 0)),
           ("ECRAN C", (255, 0, 0)), ("ECRAN D", (0, 255, 255))]


def lire_moniteurs():
    """Retourne [(nom, x, y, w, h), ...] depuis `xrandr --listmonitors`."""
    sortie = subprocess.check_output(
        ["xrandr", "--listmonitors"], text=True)
    moniteurs = []
    # ex. ' 0: +*HDMI-1 480/150x800/100+0+0  HDMI-1'
    motif = re.compile(r"(\d+)/\d+x(\d+)/\d+\+(\d+)\+(\d+)\s+(\S+)")
    for ligne in sortie.splitlines():
        m = motif.search(ligne)
        if m:
            w, h, x, y, nom = m.groups()
            moniteurs.append((nom, int(x), int(y), int(w), int(h)))
    return moniteurs


def dessiner(nom, w, h, label, couleur):
    """Construit l'image de test pour un écran donné."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = couleur

    # Cadre blanc épais (repère les bords / le débordement)
    cv2.rectangle(img, (4, 4), (w - 5, h - 5), (255, 255, 255), 4)

    # Carré blanc en HAUT-GAUCHE (repère miroir / rotation)
    cv2.rectangle(img, (12, 12), (52, 52), (255, 255, 255), -1)

    blanc = (255, 255, 255)
    noir = (0, 0, 0)

    def texte_centre(txt, cy, echelle, epais):
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX,
                                      echelle, epais)
        x = (w - tw) // 2
        cv2.putText(img, txt, (x, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    echelle, noir, epais + 3, cv2.LINE_AA)
        cv2.putText(img, txt, (x, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    echelle, blanc, epais, cv2.LINE_AA)

    cy = int(h * 0.42)
    texte_centre(label, cy, 1.6, 4)
    texte_centre(nom, cy + 55, 1.0, 2)
    texte_centre(f"{w}x{h}  +", cy + 100, 0.9, 2)
    texte_centre("^ HAUT", int(h * 0.14), 1.0, 2)
    return img


def main():
    moniteurs = lire_moniteurs()
    if not moniteurs:
        print("Aucun moniteur detecte via xrandr --listmonitors.")
        print("Verifie : DISPLAY=:0 xrandr --listmonitors")
        sys.exit(1)

    print("Ecrans detectes :")
    for nom, x, y, w, h in moniteurs:
        print(f"  {nom}: {w}x{h} a la position +{x}+{y}")

    fenetres = []
    for i, (nom, x, y, w, h) in enumerate(moniteurs):
        label, couleur = ASSOCIATION.get(
            nom, SECOURS[i % len(SECOURS)])
        img = dessiner(nom, w, h, label, couleur)

        # Sequence robuste : creer -> afficher -> positionner -> fullscreen
        # -> re-positionner, avec pauses pour laisser le WM suivre.
        cv2.namedWindow(nom, cv2.WINDOW_NORMAL)
        cv2.imshow(nom, img)
        cv2.waitKey(200)
        cv2.moveWindow(nom, x, y)
        cv2.resizeWindow(nom, w, h)
        cv2.waitKey(200)
        cv2.setWindowProperty(nom, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(200)
        cv2.moveWindow(nom, x, y)   # re-force la position apres fullscreen
        cv2.waitKey(200)
        fenetres.append(nom)

    print("\nEchap ou 'q' pour quitter.")
    while True:
        k = cv2.waitKey(50) & 0xFF
        if k in (27, ord('q')):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
