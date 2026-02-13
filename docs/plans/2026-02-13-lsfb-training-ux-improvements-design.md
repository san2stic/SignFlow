# Design Document : Amélioration UX Entraînement LSFB

**Date** : 2026-02-13
**Auteur** : Claude Sonnet 4.5
**Statut** : Approuvé
**Version** : 1.0

---

## 1. Contexte et Objectif

### Problème
L'utilisateur souhaite enrichir le vocabulaire SignFlow avec ses propres signes LSFB (Langue des Signes de Belgique Francophone) via l'enregistrement en temps réel. Le système d'entraînement existe déjà mais nécessite des améliorations UX pour optimiser l'expérience utilisateur et garantir la qualité des vidéos enregistrées.

### Objectif
Améliorer l'interface d'entraînement existante (`TrainingWizard`) pour :
- Faciliter l'enregistrement de signes LSFB de haute qualité
- Fournir un feedback visuel temps réel sur la qualité des clips
- Automatiser le déploiement intelligent des modèles
- Intégrer un flow de validation immédiat

### Scope
- **Frontend uniquement** : Composants React/TypeScript
- **Backend déjà fonctionnel** : API deployment, WebSocket metrics, few-shot training
- **Mobile-first** : Toutes les améliorations doivent fonctionner sur smartphone

---

## 2. Architecture Globale

### Flow Utilisateur Optimisé

```
┌─────────────────────────────────────────────────┐
│ Step 1 : Nommer le Signe LSFB                   │
│  - Auto-complétion signes existants             │
│  - Préfixe "lsfb_" automatique                  │
│  - Catégorie/Tags pré-remplis (lsfb-v1)        │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│ Step 2 : Enregistrement Clips (5-10 clips)     │
│  - Guide visuel overlay (silhouette)            │
│  - Feedback qualité temps réel (🟢🟡🔴)        │
│  - Compteur visuel chips colorés                │
│  - Validation durée recommandée (3-4 sec)       │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│ Upload + Extraction Landmarks (Backend)         │
│  - MediaPipe extraction automatique             │
│  - < 10 sec pour 5 clips                        │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│ Step 3 : Few-Shot Training (2-3 min)           │
│  - Progress bar animée                          │
│  - Chart loss/accuracy temps réel               │
│  - Badge deployment readiness                   │
│  - Actions recommandées affichées              │
└─────────────────┬───────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────┐
│ Step 4 : Validation & Deployment               │
│  - Test live intégré (caméra active)            │
│  - Deployment auto si accuracy ≥ 85%           │
│  - Sinon : Suggestion "Record More Clips"       │
│  - Success animation → Redirect /translate      │
└─────────────────────────────────────────────────┘
```

---

## 3. Composants Frontend à Créer/Modifier

### 3.1 Step 1 : Nommer le Signe

**Composant** : `TrainingWizard.tsx` (Step 1)

**Améliorations** :
- ✅ **Auto-complétion** : Recherche dans signes existants via API `GET /signs?search={query}`
- ✅ **Préfixe "lsfb_"** : Ajouté automatiquement si absent (ex: "bonjour" → "lsfb_bonjour")
- ✅ **Catégorie par défaut** : `"lsfb-v1"` pré-remplie dans l'input
- ✅ **Tags suggérés** : `["lsfb", "v1"]` pré-remplis via `TagInput` component
- ✅ **Description** : Textarea optionnelle pour notes (markdown support)

**State Management** :
```typescript
const [name, setName] = useState("");
const [category, setCategory] = useState("lsfb-v1");
const [tags, setTags] = useState<string[]>(["lsfb", "v1"]);
const [description, setDescription] = useState("");
```

**UI Layout** :
```
┌──────────────────────────────────┐
│ Sign Name                        │
│ [lsfb_bonjour_________]          │
│                                  │
│ Category                         │
│ [lsfb-v1_______________]         │
│                                  │
│ Description (optional)           │
│ [Textarea_______________]        │
│                                  │
│ Tags                             │
│ [lsfb] [v1] [+ Add]              │
│                                  │
│ [Next →]                         │
└──────────────────────────────────┘
```

---

### 3.2 Step 2 : Enregistrement Clips

**Composant** : `ClipRecorder.tsx` (amélioré)

**Nouvelles Features** :

#### A. Guide Visuel Overlay
- **Composant** : `SignGuideOverlay.tsx` (nouveau)
- **Implémentation** : SVG overlay statique avec silhouette mains/corps
- **Affichage** : Transparent 30% opacity, centré sur caméra
- **Design** : Silhouette simple (pas de 3D, juste contours)

```typescript
// SignGuideOverlay.tsx
export function SignGuideOverlay(): JSX.Element {
  return (
    <svg className="absolute inset-0 pointer-events-none opacity-30">
      {/* Silhouette corps */}
      <ellipse cx="50%" cy="40%" rx="60" ry="100" stroke="white" fill="none" />
      {/* Silhouette mains gauche/droite */}
      <circle cx="30%" cy="50%" r="40" stroke="cyan" fill="none" />
      <circle cx="70%" cy="50%" r="40" stroke="yellow" fill="none" />
    </svg>
  );
}
```

#### B. Feedback Qualité Temps Réel
- **Composant** : `QualityIndicator.tsx` (nouveau)
- **Input** : `visibleHands` (déjà calculé par `countVisibleHands(frame)`)
- **Logique** :
  - `visibleHands === 2` → 🟢 Vert "Perfect - Both hands detected"
  - `visibleHands === 1` → 🟡 Orange "Good - One hand detected"
  - `visibleHands === 0` → 🔴 Rouge "No hands detected"

```typescript
// QualityIndicator.tsx
interface QualityIndicatorProps {
  visibleHands: number;
}

export function QualityIndicator({ visibleHands }: QualityIndicatorProps): JSX.Element {
  const status = visibleHands === 2 ? "perfect" : visibleHands === 1 ? "good" : "poor";
  const color = visibleHands === 2 ? "bg-green-500" : visibleHands === 1 ? "bg-amber-500" : "bg-red-500";
  const text = visibleHands === 2 ? "Perfect - Both hands" : visibleHands === 1 ? "Good - One hand" : "No hands detected";

  return (
    <div className={`flex items-center gap-2 rounded-btn px-3 py-2 ${color}/20`}>
      <div className={`h-3 w-3 rounded-full ${color}`} />
      <span className="text-sm">{text}</span>
    </div>
  );
}
```

#### C. Compteur Visuel Clips
- **Composant** : `ClipCounter.tsx` (nouveau)
- **Affichage** : Chips colorés pour chaque clip (5 minimum)
- **States** : `pending` (gris), `valid` (vert), `invalid` (rouge)

```typescript
// ClipCounter.tsx
interface ClipCounterProps {
  clips: RecordedClip[];
  minClips: number;
}

export function ClipCounter({ clips, minClips }: ClipCounterProps): JSX.Element {
  const validCount = clips.filter(c => c.quality === "valid").length;
  const slots = Array.from({ length: Math.max(minClips, clips.length) });

  return (
    <div className="flex gap-2">
      {slots.map((_, idx) => {
        const clip = clips[idx];
        const color = !clip ? "bg-slate-700" : clip.quality === "valid" ? "bg-green-500" : "bg-red-500";
        return <div key={idx} className={`h-10 w-10 rounded-full ${color}`} />;
      })}
      <span className="self-center text-sm text-slate-400">{validCount}/{minClips} minimum</span>
    </div>
  );
}
```

---

### 3.3 Step 3 : Training Progress

**Composant** : `TrainingProgress.tsx` (amélioré)

**Nouvelles Features** :

#### A. Chart Metrics Temps Réel
- **Library** : Recharts (déjà installé ?)
- **Affichage** : Mini line chart loss/accuracy
- **Update** : Chaque epoch via WebSocket

```typescript
// TrainingProgress.tsx (ajout)
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer } from "recharts";

const chartData = progressState.metrics_history || [];

<ResponsiveContainer width="100%" height={150}>
  <LineChart data={chartData}>
    <XAxis dataKey="epoch" />
    <YAxis domain={[0, 1]} />
    <Line type="monotone" dataKey="loss" stroke="#F59E0B" name="Loss" />
    <Line type="monotone" dataKey="val_accuracy" stroke="#10B981" name="Accuracy" />
  </LineChart>
</ResponsiveContainer>
```

#### B. Deployment Readiness Badge
- **Input** : `deployment_ready` (WebSocket)
- **Affichage** :
  - `true` → Badge vert "✓ Ready to Deploy (92%)"
  - `false` → Badge orange "⚠ Below Threshold (78% < 85%)"

```typescript
// DeploymentReadinessBadge.tsx
interface DeploymentReadinessBadgeProps {
  ready: boolean;
  accuracy: number | null;
  threshold: number;
}

export function DeploymentReadinessBadge({ ready, accuracy, threshold }: DeploymentReadinessBadgeProps): JSX.Element {
  const bgColor = ready ? "bg-green-500/20" : "bg-amber-500/20";
  const textColor = ready ? "text-green-400" : "text-amber-400";
  const icon = ready ? "✓" : "⚠";
  const label = ready ? "Ready to Deploy" : "Below Threshold";
  const detail = accuracy !== null ? `(${(accuracy * 100).toFixed(1)}% ${ready ? "" : `< ${(threshold * 100).toFixed(0)}%`})` : "";

  return (
    <div className={`flex items-center gap-2 rounded-btn px-3 py-2 ${bgColor}`}>
      <span className={textColor}>{icon}</span>
      <span className={`text-sm ${textColor}`}>{label} {detail}</span>
    </div>
  );
}
```

#### C. Recommended Action Display
- **Input** : `recommended_next_action` (WebSocket)
- **Affichage** :
  - `"deploy"` → "✓ Model ready - Click Validate to deploy"
  - `"collect_more_examples"` → "⚠ Add 3-5 more clips to improve accuracy"
  - `"wait"` → "⏳ Training in progress..."
  - `"review_error"` → "❌ Training failed - Check logs"

```typescript
// RecommendedActionMessage.tsx
interface RecommendedActionMessageProps {
  action: "deploy" | "collect_more_examples" | "wait" | "review_error";
}

export function RecommendedActionMessage({ action }: RecommendedActionMessageProps): JSX.Element {
  const messages = {
    deploy: { icon: "✓", text: "Model ready - Click Validate to deploy", color: "text-green-400" },
    collect_more_examples: { icon: "⚠", text: "Add 3-5 more clips to improve accuracy", color: "text-amber-400" },
    wait: { icon: "⏳", text: "Training in progress...", color: "text-slate-400" },
    review_error: { icon: "❌", text: "Training failed - Check logs", color: "text-red-400" }
  };

  const msg = messages[action];
  return (
    <p className={`text-sm ${msg.color}`}>
      {msg.icon} {msg.text}
    </p>
  );
}
```

---

### 3.4 Step 4 : Validation & Deployment

**Composant** : `ValidationTest.tsx` (amélioré)

**Nouvelles Features** :

#### A. Deployment Automatique Intelligent
- **Logique** :
  - Si `deployment_ready === true` → Appel API automatique `POST /training/sessions/{id}/deploy`
  - Sinon → Afficher bouton "Record More Clips" (retour Step 2)

```typescript
// ValidationTest.tsx (ajout)
interface ValidationTestProps {
  prediction: string;
  confidence: number;
  deploymentReady: boolean;
  recommendedAction: string;
  onDeploy: () => void;
  onCollectMore: () => void;
  isDeploying: boolean;
  deployError: string | null;
}

export function ValidationTest({
  prediction,
  confidence,
  deploymentReady,
  recommendedAction,
  onDeploy,
  onCollectMore,
  isDeploying,
  deployError
}: ValidationTestProps): JSX.Element {
  return (
    <div className="card space-y-4 p-4">
      <h2 className="text-xl font-heading">Validation Complete</h2>

      {/* Résultat prédiction */}
      <div className="rounded-btn bg-slate-800 p-4">
        <p className="text-sm text-slate-400">Predicted Sign</p>
        <p className="text-2xl font-bold">{prediction}</p>
        <p className="text-sm text-slate-400">Confidence: {(confidence * 100).toFixed(1)}%</p>
      </div>

      {/* Deployment readiness */}
      <DeploymentReadinessBadge
        ready={deploymentReady}
        accuracy={confidence}
        threshold={0.85}
      />

      {/* Actions basées sur recommendation */}
      {recommendedAction === "deploy" && (
        <button
          className="touch-btn bg-green-500 text-white disabled:bg-slate-700"
          onClick={onDeploy}
          disabled={isDeploying}
        >
          {isDeploying ? "Deploying..." : "✓ Deploy Model"}
        </button>
      )}

      {recommendedAction === "collect_more_examples" && (
        <button
          className="touch-btn bg-amber-500 text-slate-950"
          onClick={onCollectMore}
        >
          ⚠ Record More Clips
        </button>
      )}

      {deployError && (
        <p className="text-sm text-red-400">{deployError}</p>
      )}

      {/* Test live caméra */}
      <div className="space-y-2">
        <p className="text-sm text-slate-400">Test your sign:</p>
        {/* Caméra + overlay landmarks (réutiliser TranslatePage logic) */}
      </div>
    </div>
  );
}
```

#### B. Success Flow
- **Animation** : Confetti ou checkmark animé (Framer Motion)
- **Message** : "✓ Model Deployed! Redirecting to Translate..."
- **Redirect** : Automatique vers `/translate` après 2 secondes

```typescript
// ValidationTest.tsx (ajout après deployment success)
const [deploySuccess, setDeploySuccess] = useState(false);

useEffect(() => {
  if (deploySuccess) {
    setTimeout(() => {
      window.location.href = "/translate";
    }, 2000);
  }
}, [deploySuccess]);

// Dans le render après deployment success
{deploySuccess && (
  <motion.div
    initial={{ scale: 0 }}
    animate={{ scale: 1 }}
    className="rounded-btn bg-green-500/20 p-6 text-center"
  >
    <p className="text-3xl">✓</p>
    <p className="text-lg text-green-400">Model Deployed!</p>
    <p className="text-sm text-slate-400">Redirecting to Translate...</p>
  </motion.div>
)}
```

---

## 4. API Backend (Déjà Implémenté)

### 4.1 Endpoint Deployment
```http
POST /api/v1/training/sessions/{session_id}/deploy
```

**Validation** :
- ✅ Session status === "completed"
- ✅ deployment_ready === true
- ✅ Activation modèle (is_active = true)
- ✅ Reload pipeline automatique

**Response** :
```json
{
  "status": "deployed",
  "session_id": "uuid",
  "active_model_id": "uuid",
  "version": "v13"
}
```

### 4.2 WebSocket Live Training
```
WS /api/v1/training/sessions/{session_id}/live
```

**Payload** (toutes les 500ms) :
```json
{
  "status": "training",
  "progress": 72.5,
  "metrics": {
    "loss": 0.234,
    "accuracy": 0.89,
    "val_accuracy": 0.86,
    "current_epoch": 36
  },
  "deployment_ready": false,
  "deploy_threshold": 0.85,
  "final_val_accuracy": null,
  "recommended_next_action": "wait",
  "estimated_remaining": "45s"
}
```

---

## 5. Décisions Techniques & Trade-offs

### 5.1 Few-Shot Learning
**Choix** : Garder le système actuel (déjà optimal)
- ✅ 5-10 clips suffisent
- ✅ Training rapide (2-3 min CPU)
- ✅ Prototypical Networks en fallback si < 5 exemples

### 5.2 Guide Visuel
**Choix** : SVG overlay statique (pas de 3D)
- ✅ Pro : Léger, performant mobile
- ❌ Con : Moins réaliste qu'un modèle 3D animé
- **Justification** : Balance simplicité/performance vs réalisme

### 5.3 Feedback Qualité
**Choix** : Basé sur `visibleHands` uniquement (simple)
- ✅ Pro : Déjà calculé, pas de processing additionnel
- ❌ Con : Ne détecte pas éclairage/contraste/centrage avancé
- **Justification** : Suffisant pour validation basique, évite over-engineering

### 5.4 Deployment
**Choix** : Semi-automatique (auto si ≥85%, sinon manuel)
- ✅ Pro : Évite modèles pourris, garde contrôle utilisateur
- ❌ Con : Une étape supplémentaire si < 85%
- **Justification** : Balance qualité vs friction UX

---

## 6. Performance Targets

| Phase | Métrique | Target |
|-------|----------|--------|
| Enregistrement clips | FPS | 30 FPS stable |
| Upload + extraction | Temps | < 10 sec pour 5 clips |
| Training | Durée | 2-3 minutes (50 epochs, CPU) |
| Deployment | Latency | < 2 sec (activation + reload) |
| UI Feedback | Refresh | 500ms (WebSocket updates) |

---

## 7. Structure Fichiers

### Nouveaux Composants
```
frontend/src/
  components/
    training/
      SignGuideOverlay.tsx       # Nouveau : Guide visuel overlay
      QualityIndicator.tsx        # Nouveau : Feedback qualité temps réel
      ClipCounter.tsx             # Nouveau : Compteur visuel chips
      DeploymentReadinessBadge.tsx # Nouveau : Badge deployment ready
      RecommendedActionMessage.tsx # Nouveau : Messages actions recommandées
    common/
      TagInput.tsx                # Existant (à utiliser dans Step 1)
```

### Composants Modifiés
```
frontend/src/
  components/
    training/
      TrainingWizard.tsx          # Step 1 : Auto-complétion, préfixes
      ClipRecorder.tsx            # Step 2 : Intégration nouveaux composants
      TrainingProgress.tsx        # Step 3 : Chart metrics, badges
      ValidationTest.tsx          # Step 4 : Deployment auto, success flow
```

---

## 8. Dépendances

### Packages à Vérifier
- ✅ `recharts` : Chart metrics (à installer si absent)
- ✅ `framer-motion` : Animations success flow (déjà installé ?)
- ✅ `@mediapipe/holistic` : Déjà installé (Phase 4)

### Installation
```bash
cd frontend
npm install recharts framer-motion  # Si pas déjà installés
```

---

## 9. Tests de Validation

### Test End-to-End
1. **Step 1** : Nommer signe "lsfb_test_bonjour"
   - ✅ Vérifier préfixe auto-ajouté
   - ✅ Vérifier catégorie/tags pré-remplis

2. **Step 2** : Enregistrer 5 clips
   - ✅ Vérifier guide overlay visible
   - ✅ Vérifier feedback qualité (🟢 avec 2 mains)
   - ✅ Vérifier compteur chips colorés

3. **Step 3** : Observer training
   - ✅ Vérifier chart metrics mis à jour
   - ✅ Vérifier badge deployment readiness
   - ✅ Vérifier message action recommandée

4. **Step 4** : Déploiement
   - ✅ Si accuracy ≥ 85% → Bouton "Deploy" affiché
   - ✅ Clic deploy → Success animation → Redirect /translate
   - ✅ Si accuracy < 85% → Bouton "Record More"

---

## 10. Prochaines Étapes

### Implémentation (Frontend uniquement)
1. Créer les 5 nouveaux composants (SignGuideOverlay, QualityIndicator, etc.)
2. Modifier TrainingWizard.tsx (Step 1 : auto-complétion)
3. Intégrer composants dans ClipRecorder.tsx (Step 2)
4. Améliorer TrainingProgress.tsx avec chart (Step 3)
5. Refactoriser ValidationTest.tsx avec deployment auto (Step 4)
6. Tests E2E avec vraies vidéos LSFB

### Post-Implémentation
- Tests utilisateur avec signes LSFB réels
- Ajustement thresholds si nécessaire (85% deployment)
- Documentation utilisateur (guide vidéo ?)

---

**Auteur** : Claude Sonnet 4.5
**Date de création** : 2026-02-13
**Dernière modification** : 2026-02-13
**Statut** : ✅ Approuvé pour implémentation
