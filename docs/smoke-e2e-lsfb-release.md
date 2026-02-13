# Smoke E2E - Release LSFB V1

Date: 2026-02-13

## Objectif

Valider le flux complet:

`unknown sign -> train few-shot -> deploy -> translate live`

## Préparation

1. `cp .env.example .env`
2. `docker compose up --build`
3. Vérifier:
   - `http://localhost:8000/healthz`
   - `http://localhost:3000`

## Scénario

1. Ouvrir `Translate`.
2. Provoquer un signe inconnu jusqu’à la popup.
3. Choisir `Ajouter un nouveau signe`.
4. Vérifier que `Train` reçoit le clip pre-roll.
5. Compléter à 5 clips valides minimum.
6. Démarrer entraînement.
7. Attendre status `completed`.
8. Ouvrir validation step 4:
   - vérifier prédiction live WS
   - vérifier badge deployment
9. Cliquer `Deploy model`.
10. Vérifier redirection vers `Translate`.
11. Effectuer le signe entraîné et confirmer la détection.

## Vérifications complémentaires

1. `Dictionary`:
   - ouvrir le signe
   - vérifier backlinks
   - exporter puis réimporter (format markdown ou obsidian)
2. `Dashboard`:
   - courbe accuracy affichée
   - signes par catégorie affichés

## Critère de succès

Le signe inconnu est ajouté, entraîné, validé, déployé, puis détecté en traduction live sans intervention DB manuelle.
