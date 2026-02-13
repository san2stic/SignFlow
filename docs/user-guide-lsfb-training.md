# Guide Utilisateur - Entraînement LSFB V1

## Objectif

Ajouter un nouveau signe LSFB, l’entraîner, le valider en live, puis le déployer.

## Prérequis

- Caméra autorisée dans le navigateur.
- Éclairage suffisant (idéalement frontal).
- 5 clips valides minimum par signe.

## Étape 1 - Préparer le signe

1. Ouvrir `Train`.
2. Nommer le signe (préfixe `lsfb_` appliqué automatiquement).
3. Vérifier catégorie/tags/description.
4. Cliquer `Next`.

## Étape 2 - Enregistrer les clips

1. Se positionner dans le guide visuel.
2. Vérifier l’indicateur de qualité mains:
   - Vert: deux mains visibles
   - Orange: une main
   - Rouge: aucune main
3. Enregistrer des clips 2-5 secondes.
4. Les clips sont validés avec règles:
   - mains détectées sur >= 80% des frames
   - geste suffisamment centré
   - luminosité suffisante
5. Atteindre `5/5` clips valides puis cliquer `Start Training`.

## Étape 3 - Suivre l’entraînement

1. Observer progression + métriques (`loss`, `acc`, `val_accuracy`).
2. Lire la recommandation:
   - `deploy`
   - `collect_more_examples`
3. Cliquer `Validate` une fois status `completed`.

## Étape 4 - Validation live + déploiement

1. Effectuer le signe devant la caméra dans l’écran de validation.
2. Vérifier la prédiction live (WS `/translate/stream`).
3. Si prêt, cliquer `Deploy model`.
4. Après succès, redirection vers `Translate`.

## Unknown sign flow

Depuis `Translate`, si la confiance reste faible:

1. Popup `Signe inconnu`.
2. Choisir:
   - créer un nouveau signe
   - assigner à un signe existant
3. Le clip pre-roll est transmis automatiquement vers `Train`.

## Dictionnaire

Dans `Dictionary`:

- vue graphe interactive (zoom/pan/drag)
- édition des notes markdown
- backlinks
- import/export zip (`json`, `markdown`, `obsidian-vault`)

## Dépannage rapide

- `Hands visibility too low`: rapprocher les mains et stabiliser le geste.
- `Gesture too often outside center frame`: recentrer le corps/les mains.
- `Lighting is too low`: augmenter l’éclairage.
- `Deployment failed`: vérifier backend/worker et relancer.
