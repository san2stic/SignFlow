# SignFlow - Nouvelle Interface Utilisateur

## 🎨 Design

Le nouveau frontend de SignFlow est basé sur un design moderne et professionnel avec :

- **Palette de couleurs** : Bleu marine profond (#1a2b3d) avec accents teal/cyan (#14b8a6, #06b6d4)
- **Typographie** : Clean et moderne
- **Layout** : Sidebar navigation + Dashboard principal + Panel latéral optionnel
- **Composants** : Cards avec statistiques, graphiques interactifs, visualisation temps réel

## 🚀 Démarrage Rapide

### Backend

1. **Installer les dépendances Python** :
```bash
cd backend
pip install -e ".[dev]"
```

2. **Configurer les variables d'environnement** :
```bash
cp .env.example .env
# Éditer .env et changer JWT_SECRET_KEY en production !
```

3. **Démarrer le serveur backend** :
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Le backend sera accessible sur `http://localhost:8000`
- Documentation API : `http://localhost:8000/docs`

### Frontend

1. **Installer les dépendances npm** :
```bash
cd frontend
npm install
```

2. **Configurer les variables d'environnement** :
```bash
cp .env.example .env
# Par défaut, VITE_API_URL=http://localhost:8000/api/v1
```

3. **Démarrer le serveur de développement** :
```bash
npm run dev
```

Le frontend sera accessible sur `http://localhost:5173`

## 📋 Fonctionnalités Principales

### ✅ Authentification
- **Login/Register** : Pages d'authentification avec design moderne
- **JWT Tokens** : Authentification sécurisée avec tokens JWT
- **Protection des routes** : Redirection automatique vers login si non authentifié
- **Gestion du profil** : Mise à jour du username et nom complet

### 📊 Dashboard
- **Statistiques en temps réel** :
  - Translations Today (145)
  - Pending Labels (32)
  - Dictionary Entries (1,250)
  - Model Accuracy (94.5%)
- **Graphique hebdomadaire** : Volume de traductions sur 7 jours
- **Activité récente** : Liste des dernières actions
- **Live Translation Assist** : Panel latéral avec caméra en temps réel

### 🗂️ Pages
1. **Dashboard** : Vue d'ensemble avec statistiques et graphiques
2. **Translation** : Interface de traduction en temps réel (existante)
3. **Dictionary** : Dictionnaire de signes (placeholder)
4. **Training & Labeling** : Outils d'entraînement (placeholder)
5. **Settings** : Paramètres de l'application (placeholder)
6. **Profile** : Gestion du profil utilisateur

## 🏗️ Architecture

### Backend
```
backend/
├── app/
│   ├── auth/           # Module d'authentification
│   │   ├── jwt.py      # Génération tokens JWT
│   │   ├── schemas.py  # Schémas Pydantic
│   │   └── dependencies.py  # Dépendances FastAPI
│   ├── models/
│   │   └── user.py     # Modèle SQLAlchemy User
│   ├── api/
│   │   └── auth.py     # Endpoints auth (register, login, profile)
│   └── config.py       # Configuration (JWT_SECRET_KEY, etc.)
```

### Frontend
```
frontend/
├── src/
│   ├── pages/
│   │   ├── Login.tsx       # Page de connexion
│   │   ├── Register.tsx    # Page d'inscription
│   │   ├── Dashboard.tsx   # Tableau de bord
│   │   ├── Dictionary.tsx  # Dictionnaire
│   │   ├── Training.tsx    # Training & Labeling
│   │   ├── Settings.tsx    # Paramètres
│   │   └── Profile.tsx     # Profil utilisateur
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Sidebar.tsx      # Navigation latérale
│   │   │   └── MainLayout.tsx   # Layout principal
│   │   └── ProtectedRoute.tsx   # Protection des routes
│   ├── stores/
│   │   └── authStore.ts    # Store Zustand pour auth
│   ├── lib/
│   │   └── api.ts          # Client API
│   └── routes.tsx          # Configuration des routes
```

## 🔐 Sécurité

### Backend
- **Hachage des mots de passe** : bcrypt avec salt automatique
- **JWT tokens** : Signature HS256 avec secret key
- **Validation des entrées** : Pydantic schemas
- **Protection CORS** : Configuration des origines autorisées

### Frontend
- **Stockage sécurisé** : Tokens dans localStorage (via zustand persist)
- **Protection des routes** : Composant ProtectedRoute
- **Validation des formulaires** : Validation côté client

## 🎨 Design System

### Couleurs
```css
/* Primary */
teal-600: #0d9488
cyan-600: #0891b2

/* Background */
slate-900: #0f172a  /* Sidebar */
slate-50: #f8fafc   /* Main background */
white: #ffffff      /* Cards */

/* Accents */
green-600: #16a34a  /* Success */
amber-600: #d97706  /* Warning */
red-600: #dc2626    /* Error */
```

### Composants Réutilisables
- **Cards** : `bg-white rounded-2xl p-6 shadow-sm border border-gray-200`
- **Boutons primaires** : `bg-gradient-to-r from-teal-600 to-cyan-600`
- **Inputs** : `px-4 py-3 border border-slate-300 rounded-xl`

## 📦 Dépendances Ajoutées

### Backend
```toml
python-jose[cryptography]>=3.3.0  # JWT tokens
bcrypt>=4.1.2                      # Password hashing
```

### Frontend
```json
lucide-react  # Icônes modernes
```

## 🚧 Prochaines Étapes

1. **Intégrer la traduction en temps réel** dans le Dashboard panel
2. **Implémenter les pages Dictionary et Training** avec vraies données
3. **Ajouter les tests** (backend + frontend)
4. **Améliorer la gestion des erreurs** et notifications
5. **Ajouter l'upload d'avatar** dans le profil
6. **Implémenter les paramètres utilisateur** (langue, thème, etc.)

## 📝 Notes de Migration

L'ancien frontend (pages TranslatePage, TrainPage, etc.) est toujours accessible mais nécessite d'être intégré dans le nouveau layout. Les composants suivants ont été créés en remplacement :

- ❌ `PageShell` → ✅ `MainLayout` + `Sidebar`
- ❌ Direct routing → ✅ `ProtectedRoute` wrapper
- ❌ No auth → ✅ JWT authentication with `authStore`

## 🐛 Debug

Si problème de connexion backend :
```bash
# Vérifier que le backend tourne
curl http://localhost:8000/healthz

# Tester l'endpoint de register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","username":"testuser","password":"testpass123"}'
```

Si problème frontend :
```bash
# Vérifier la console du navigateur
# Vérifier le fichier .env (VITE_API_URL)
# Redémarrer le serveur Vite
npm run dev
```

## 📄 Licence

SignFlow © 2026
