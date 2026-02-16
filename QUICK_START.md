# 🚀 SignFlow - Guide de Démarrage Rapide

## Nouveautés

✨ **Nouveau Design Complet** basé sur l'interface SignFlow moderne  
🔐 **Authentification JWT** avec système de profil utilisateur  
📊 **Dashboard** avec statistiques et graphiques en temps réel  
🎨 **UI Moderne** : Bleu marine + Teal/Cyan, navigation sidebar

## Démarrage en 3 Étapes

### 1️⃣ Backend
```bash
cd backend
python3 -m pip install -e .
cp .env.example .env
# ⚠️ IMPORTANT: Éditez .env et changez JWT_SECRET_KEY !
python3 -m uvicorn app.main:app --reload --port 8000
```
→ Backend : http://localhost:8000  
→ API Docs : http://localhost:8000/docs

### 2️⃣ Frontend  
```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```
→ Frontend : http://localhost:5173

### 3️⃣ Utilisation
1. Ouvrir http://localhost:5173
2. Cliquer sur "Create one" pour créer un compte
3. S'inscrire avec email + username + password
4. Vous êtes redirigé vers le Dashboard ! 🎉

## 📱 Pages Disponibles

- **Dashboard** (`/dashboard`) : Vue d'ensemble avec stats
- **Translation** (`/translate`) : Traduction en temps réel
- **Dictionary** (`/dictionary`) : Dictionnaire de signes
- **Training** (`/training`) : Entraînement et labeling
- **Settings** (`/settings`) : Paramètres
- **Profile** (`/profile`) : Gestion du profil

## 🔑 Endpoints API

### Authentication
```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","username":"testuser","password":"testpass123"}'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@test.com","password":"testpass123"}'

# Profile (avec token)
curl http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

## 🛠️ Dépendances Installées

### Backend
- `python-jose[cryptography]` : JWT tokens
- `bcrypt` : Password hashing
- `email-validator` : Email validation

### Frontend
- `lucide-react` : Icônes modernes

## 🎨 Design System

**Couleurs principales** :
- Teal : `#0d9488` (teal-600)
- Cyan : `#0891b2` (cyan-600)
- Slate : `#0f172a` (slate-900)

**Composants** :
- Cards : `rounded-2xl shadow-sm`
- Buttons : `rounded-xl gradient`
- Sidebar : Bleu marine avec navigation

## ⚠️ Important

1. **JWT_SECRET_KEY** : Changez-le dans `.env` ! (32+ caractères)
2. **Database** : SQLite par défaut, PostgreSQL recommandé en production
3. **CORS** : Configurez `CORS_ORIGINS` correctement

## 🐛 Problèmes Courants

**Backend ne démarre pas** :
```bash
python3 -m pip install "python-jose[cryptography]>=3.3.0" "bcrypt>=4.1.2" "email-validator>=2.0.0"
```

**Frontend erreurs de build** :
```bash
npm install
npm run build
```

**Erreur de connexion** :
- Vérifier que backend tourne sur :8000
- Vérifier VITE_API_URL dans frontend/.env

## 📚 Documentation Complète

- `NEW_FRONTEND_README.md` : Documentation détaillée frontend
- `IMPLEMENTATION_SUMMARY.md` : Résumé complet de l'implémentation
- `docs/plans/quizzical-jumping-karp.md` : Plan master du projet

## 🎯 Prochaines Étapes

1. Tester la création de compte et login
2. Explorer le Dashboard
3. Intégrer les vraies données dans Dictionary
4. Connecter l'API de traduction au Dashboard panel

**Bon développement ! 🚀**
