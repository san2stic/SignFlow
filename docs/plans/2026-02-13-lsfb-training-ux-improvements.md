# LSFB Training UX Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Améliorer l'UX d'entraînement LSFB avec feedback qualité temps réel, deployment automatique intelligent, et composants visuels optimisés.

**Architecture:** Ajout de 5 nouveaux composants React pour feedback visuel (guide overlay, quality indicator, clip counter, deployment badge, action messages). Modification de 4 composants existants (TrainingWizard, ClipRecorder, TrainingProgress, ValidationTest) pour intégrer les nouvelles features. Frontend uniquement - backend déjà fonctionnel.

**Tech Stack:** React 18, TypeScript, Tailwind CSS, Framer Motion (déjà installé), Recharts (à installer)

---

## Task 1: Installation Dépendances

**Files:**
- Modify: `frontend/package.json`

**Step 1: Installer recharts**

```bash
cd frontend
npm install recharts
```

Expected output: `+ recharts@2.x.x`

**Step 2: Vérifier l'installation**

```bash
cat package.json | grep recharts
```

Expected: `"recharts": "^2.x.x"`

**Step 3: Commit**

```bash
git add package.json package-lock.json
git commit -m "deps: add recharts for training metrics charts

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Créer QualityIndicator Component

**Files:**
- Create: `frontend/src/components/training/QualityIndicator.tsx`

**Step 1: Créer le fichier**

```typescript
interface QualityIndicatorProps {
  visibleHands: number;
}

export function QualityIndicator({ visibleHands }: QualityIndicatorProps): JSX.Element {
  const getStatus = () => {
    if (visibleHands === 2) return { color: "bg-green-500", text: "Perfect - Both hands detected", icon: "🟢" };
    if (visibleHands === 1) return { color: "bg-amber-500", text: "Good - One hand detected", icon: "🟡" };
    return { color: "bg-red-500", text: "No hands detected", icon: "🔴" };
  };

  const status = getStatus();

  return (
    <div className={`flex items-center gap-2 rounded-btn px-3 py-2 ${status.color}/20`}>
      <div className={`h-3 w-3 rounded-full ${status.color}`} />
      <span className="text-sm text-slate-200">{status.text}</span>
    </div>
  );
}
```

**Step 2: Vérifier la compilation**

```bash
cd frontend
npm run build
```

Expected: No TypeScript errors

**Step 3: Commit**

```bash
git add frontend/src/components/training/QualityIndicator.tsx
git commit -m "feat(training): add QualityIndicator component for real-time feedback

Shows color-coded quality status based on visible hands count:
- Green (2 hands): Perfect
- Yellow (1 hand): Good
- Red (0 hands): No hands detected

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Créer ClipCounter Component

**Files:**
- Create: `frontend/src/components/training/ClipCounter.tsx`

**Step 1: Créer le fichier**

```typescript
interface ClipCounterProps {
  clips: Array<{ quality: string }>;
  minClips: number;
}

export function ClipCounter({ clips, minClips }: ClipCounterProps): JSX.Element {
  const validCount = clips.filter((c) => c.quality === "valid").length;
  const slots = Array.from({ length: Math.max(minClips, clips.length) });

  return (
    <div className="flex flex-wrap items-center gap-2">
      {slots.map((_, idx) => {
        const clip = clips[idx];
        let color = "bg-slate-700"; // pending
        if (clip) {
          color = clip.quality === "valid" ? "bg-green-500" : "bg-red-500";
        }
        return <div key={idx} className={`h-10 w-10 rounded-full ${color}`} />;
      })}
      <span className="text-sm text-slate-400">
        {validCount}/{minClips} minimum
      </span>
    </div>
  );
}
```

**Step 2: Vérifier la compilation**

```bash
cd frontend
npm run build
```

Expected: No TypeScript errors

**Step 3: Commit**

```bash
git add frontend/src/components/training/ClipCounter.tsx
git commit -m "feat(training): add ClipCounter component with visual chips

Displays colored chips for each clip:
- Gray: pending slot
- Green: valid clip
- Red: invalid clip
Shows progress toward minimum clip requirement

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Créer SignGuideOverlay Component

**Files:**
- Create: `frontend/src/components/training/SignGuideOverlay.tsx`

**Step 1: Créer le fichier**

```typescript
export function SignGuideOverlay(): JSX.Element {
  return (
    <svg
      className="pointer-events-none absolute inset-0 opacity-30"
      viewBox="0 0 100 100"
      preserveAspectRatio="xMidYMid meet"
    >
      {/* Body silhouette */}
      <ellipse cx="50" cy="40" rx="15" ry="25" stroke="white" strokeWidth="0.5" fill="none" />

      {/* Left hand circle */}
      <circle cx="30" cy="50" r="10" stroke="cyan" strokeWidth="0.5" fill="none" />

      {/* Right hand circle */}
      <circle cx="70" cy="50" r="10" stroke="yellow" strokeWidth="0.5" fill="none" />

      {/* Center crosshair */}
      <line x1="50" y1="30" x2="50" y2="70" stroke="white" strokeWidth="0.3" opacity="0.5" />
      <line x1="30" y1="50" x2="70" y2="50" stroke="white" strokeWidth="0.3" opacity="0.5" />
    </svg>
  );
}
```

**Step 2: Vérifier la compilation**

```bash
cd frontend
npm run build
```

Expected: No TypeScript errors

**Step 3: Commit**

```bash
git add frontend/src/components/training/SignGuideOverlay.tsx
git commit -m "feat(training): add SignGuideOverlay component for positioning guide

SVG overlay with:
- Body silhouette (white)
- Left hand guide (cyan)
- Right hand guide (yellow)
- Center crosshair
Helps users position correctly for recording

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Créer DeploymentReadinessBadge Component

**Files:**
- Create: `frontend/src/components/training/DeploymentReadinessBadge.tsx`

**Step 1: Créer le fichier**

```typescript
interface DeploymentReadinessBadgeProps {
  ready: boolean;
  accuracy: number | null;
  threshold: number;
}

export function DeploymentReadinessBadge({
  ready,
  accuracy,
  threshold
}: DeploymentReadinessBadgeProps): JSX.Element {
  const bgColor = ready ? "bg-green-500/20" : "bg-amber-500/20";
  const textColor = ready ? "text-green-400" : "text-amber-400";
  const icon = ready ? "✓" : "⚠";
  const label = ready ? "Ready to Deploy" : "Below Threshold";

  let detail = "";
  if (accuracy !== null) {
    const accPercent = (accuracy * 100).toFixed(1);
    const threshPercent = (threshold * 100).toFixed(0);
    detail = ready ? `(${accPercent}%)` : `(${accPercent}% < ${threshPercent}%)`;
  }

  return (
    <div className={`flex items-center gap-2 rounded-btn px-3 py-2 ${bgColor}`}>
      <span className={textColor}>{icon}</span>
      <span className={`text-sm ${textColor}`}>
        {label} {detail}
      </span>
    </div>
  );
}
```

**Step 2: Vérifier la compilation**

```bash
cd frontend
npm run build
```

Expected: No TypeScript errors

**Step 3: Commit**

```bash
git add frontend/src/components/training/DeploymentReadinessBadge.tsx
git commit -m "feat(training): add DeploymentReadinessBadge component

Shows deployment status:
- Green badge: Ready (accuracy >= threshold)
- Yellow badge: Below threshold
Displays accuracy percentage and threshold comparison

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Créer RecommendedActionMessage Component

**Files:**
- Create: `frontend/src/components/training/RecommendedActionMessage.tsx`

**Step 1: Créer le fichier**

```typescript
interface RecommendedActionMessageProps {
  action: "deploy" | "collect_more_examples" | "wait" | "review_error";
}

export function RecommendedActionMessage({ action }: RecommendedActionMessageProps): JSX.Element {
  const messages = {
    deploy: {
      icon: "✓",
      text: "Model ready - Click Validate to deploy",
      color: "text-green-400"
    },
    collect_more_examples: {
      icon: "⚠",
      text: "Add 3-5 more clips to improve accuracy",
      color: "text-amber-400"
    },
    wait: {
      icon: "⏳",
      text: "Training in progress...",
      color: "text-slate-400"
    },
    review_error: {
      icon: "❌",
      text: "Training failed - Check logs",
      color: "text-red-400"
    }
  };

  const msg = messages[action];

  return (
    <p className={`text-sm ${msg.color}`}>
      {msg.icon} {msg.text}
    </p>
  );
}
```

**Step 2: Vérifier la compilation**

```bash
cd frontend
npm run build
```

Expected: No TypeScript errors

**Step 3: Commit**

```bash
git add frontend/src/components/training/RecommendedActionMessage.tsx
git commit -m "feat(training): add RecommendedActionMessage component

Displays contextual action messages based on training state:
- deploy: Model ready
- collect_more_examples: Need more data
- wait: Training in progress
- review_error: Training failed

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Modifier ClipRecorder pour Intégrer Nouveaux Composants

**Files:**
- Modify: `frontend/src/components/training/ClipRecorder.tsx`

**Step 1: Ajouter les imports**

Dans la section imports existante, ajouter :

```typescript
import { QualityIndicator } from "./QualityIndicator";
import { ClipCounter } from "./ClipCounter";
import { SignGuideOverlay } from "./SignGuideOverlay";
```

**Step 2: Intégrer SignGuideOverlay dans le render**

Trouver le conteneur vidéo et ajouter l'overlay :

```typescript
<div className="relative">
  <video ref={videoRef} className="w-full rounded-btn" autoPlay playsInline muted />
  <SignGuideOverlay />
  {/* Existing canvas/overlays */}
</div>
```

**Step 3: Ajouter QualityIndicator avant le bouton record**

```typescript
<QualityIndicator visibleHands={visibleHands} />

<button
  className="touch-btn bg-red-500 text-white"
  onClick={recording ? stopRecording : startRecording}
>
  {recording ? "⏹ Stop" : "● Record"}
</button>
```

**Step 4: Remplacer le compteur texte par ClipCounter**

Remplacer :
```typescript
<p>{validClips.length}/5 clips minimum</p>
```

Par :
```typescript
<ClipCounter clips={clips} minClips={5} />
```

**Step 5: Vérifier la compilation**

```bash
cd frontend
npm run build
```

Expected: No TypeScript errors

**Step 6: Commit**

```bash
git add frontend/src/components/training/ClipRecorder.tsx
git commit -m "feat(training): integrate visual feedback components in ClipRecorder

Add:
- SignGuideOverlay for positioning help
- QualityIndicator for real-time hand detection feedback
- ClipCounter with visual chips replacing text counter

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Améliorer TrainingProgress avec Chart Metrics

**Files:**
- Modify: `frontend/src/components/training/TrainingProgress.tsx`

**Step 1: Ajouter les imports**

```typescript
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";
import { DeploymentReadinessBadge } from "./DeploymentReadinessBadge";
import { RecommendedActionMessage } from "./RecommendedActionMessage";
import { useTrainingStore } from "../../stores/trainingStore";
```

**Step 2: Récupérer le state avec deployment info**

Dans le composant :

```typescript
const progressState = useTrainingStore((state) => state.progress);
const metrics = progressState.metrics || {};

// Build chart data from metrics history (simulate for now)
const chartData = [
  {
    epoch: metrics.current_epoch || 0,
    loss: metrics.loss || 0,
    accuracy: metrics.accuracy || 0,
    val_accuracy: metrics.val_accuracy || 0
  }
];
```

**Step 3: Ajouter le chart dans le render**

Après la progress bar existante :

```typescript
{/* Metrics Chart */}
{chartData.length > 0 && (
  <div className="card bg-slate-800/50 p-4">
    <p className="mb-2 text-sm text-slate-400">Training Metrics</p>
    <ResponsiveContainer width="100%" height={150}>
      <LineChart data={chartData}>
        <XAxis dataKey="epoch" stroke="#94a3b8" />
        <YAxis domain={[0, 1]} stroke="#94a3b8" />
        <Tooltip
          contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #334155" }}
          labelStyle={{ color: "#94a3b8" }}
        />
        <Line type="monotone" dataKey="loss" stroke="#F59E0B" name="Loss" strokeWidth={2} />
        <Line
          type="monotone"
          dataKey="val_accuracy"
          stroke="#10B981"
          name="Val Accuracy"
          strokeWidth={2}
        />
      </LineChart>
    </ResponsiveContainer>
  </div>
)}
```

**Step 4: Ajouter les badges deployment**

Après le chart :

```typescript
<DeploymentReadinessBadge
  ready={Boolean(progressState.deployment_ready)}
  accuracy={progressState.final_val_accuracy}
  threshold={progressState.deploy_threshold || 0.85}
/>

<RecommendedActionMessage
  action={progressState.recommended_next_action || "wait"}
/>
```

**Step 5: Vérifier la compilation**

```bash
cd frontend
npm run build
```

Expected: No TypeScript errors

**Step 6: Commit**

```bash
git add frontend/src/components/training/TrainingProgress.tsx
git commit -m "feat(training): add metrics chart and deployment badges to TrainingProgress

Add:
- Recharts line chart for loss/accuracy visualization
- DeploymentReadinessBadge showing status
- RecommendedActionMessage for contextual guidance

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Refactoriser ValidationTest avec Deployment Automatique

**Files:**
- Modify: `frontend/src/components/training/ValidationTest.tsx`

**Step 1: Mettre à jour l'interface props**

```typescript
interface ValidationTestProps {
  prediction: string;
  confidence: number;
  deploymentReady: boolean;
  recommendedAction: "deploy" | "collect_more_examples" | "wait" | "review_error";
  onDeploy: () => void;
  onCollectMore: () => void;
  isDeploying: boolean;
  deployError: string | null;
}
```

**Step 2: Importer les composants**

```typescript
import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { DeploymentReadinessBadge } from "./DeploymentReadinessBadge";
```

**Step 3: Ajouter le state pour success flow**

```typescript
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
  const [deploySuccess, setDeploySuccess] = useState(false);

  // Auto-redirect after successful deployment
  useEffect(() => {
    if (deploySuccess) {
      const timer = setTimeout(() => {
        window.location.href = "/translate";
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [deploySuccess]);

  const handleDeploy = () => {
    onDeploy();
    // Set success after deploy (simplified - in real app check deploy response)
    setTimeout(() => setDeploySuccess(true), 1000);
  };
```

**Step 4: Refactoriser le render avec actions conditionnelles**

```typescript
  return (
    <div className="space-y-4">
      <h2 className="text-xl font-heading">Validation Complete</h2>

      {/* Prediction result */}
      <div className="card rounded-btn bg-slate-800 p-4">
        <p className="text-sm text-slate-400">Predicted Sign</p>
        <p className="text-2xl font-bold text-white">{prediction}</p>
        <p className="text-sm text-slate-400">
          Confidence: {(confidence * 100).toFixed(1)}%
        </p>
      </div>

      {/* Deployment readiness */}
      <DeploymentReadinessBadge
        ready={deploymentReady}
        accuracy={confidence}
        threshold={0.85}
      />

      {/* Deploy error */}
      {deployError && (
        <div className="rounded-btn bg-red-600/20 px-3 py-2 text-sm text-red-200">
          {deployError}
        </div>
      )}

      {/* Action buttons based on recommendation */}
      {!deploySuccess && (
        <>
          {recommendedAction === "deploy" && (
            <button
              className="touch-btn bg-green-500 text-white disabled:bg-slate-700"
              onClick={handleDeploy}
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
        </>
      )}

      {/* Success animation */}
      {deploySuccess && (
        <motion.div
          initial={{ scale: 0, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="rounded-btn bg-green-500/20 p-6 text-center"
        >
          <motion.p
            className="text-4xl"
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 0.5 }}
          >
            ✓
          </motion.p>
          <p className="mt-2 text-lg text-green-400">Model Deployed!</p>
          <p className="text-sm text-slate-400">Redirecting to Translate...</p>
        </motion.div>
      )}
    </div>
  );
}
```

**Step 5: Vérifier la compilation**

```bash
cd frontend
npm run build
```

Expected: No TypeScript errors

**Step 6: Commit**

```bash
git add frontend/src/components/training/ValidationTest.tsx
git commit -m "feat(training): add intelligent deployment and success flow to ValidationTest

Add:
- Conditional actions based on recommendedAction prop
- Auto-deploy button when ready
- Record more button when below threshold
- Success animation with Framer Motion
- Auto-redirect to /translate after deployment

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Mettre à Jour TrainingWizard Integration

**Files:**
- Modify: `frontend/src/components/training/TrainingWizard.tsx`

**Step 1: Vérifier que les modifications récentes sont présentes**

Le fichier a déjà été modifié avec :
- State pour category, tags, description (Step 1)
- Logique WebSocket avec deployment_ready, recommended_next_action (Step 3)
- Props ValidationTest avec deployment logic (Step 4)

**Step 2: Vérifier l'intégration dans le render Step 1**

S'assurer que Step 1 contient bien les nouveaux inputs :

```typescript
{step === 1 && (
  <div className="card space-y-3 p-4">
    <label className="flex flex-col gap-1 text-sm">
      Sign Name
      <input
        className="rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-3 text-base"
        value={name}
        onChange={(event) => setName(event.target.value)}
        placeholder="Bonjour"
      />
    </label>
    <label className="flex flex-col gap-1 text-sm">
      Category
      <input
        className="rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-3 text-base"
        value={category}
        onChange={(event) => setCategory(event.target.value)}
        placeholder="lsfb-v1"
      />
    </label>
    <label className="flex flex-col gap-1 text-sm">
      Description
      <textarea
        className="rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-3 text-base"
        value={description}
        onChange={(event) => setDescription(event.target.value)}
        placeholder="Describe the sign meaning and movement."
      />
    </label>
    <div>
      <p className="mb-1 text-sm">Tags</p>
      <TagInput tags={tags} onChange={setTags} />
    </div>
    <button className="touch-btn bg-primary text-white" onClick={() => setStep(2)}>
      Next
    </button>
  </div>
)}
```

**Step 3: Vérifier l'intégration dans le render Step 4**

S'assurer que ValidationTest reçoit tous les nouveaux props :

```typescript
{step === 4 && (
  <ValidationTest
    prediction={prediction}
    confidence={confidence}
    deploymentReady={Boolean(progressState.deployment_ready)}
    recommendedAction={recommendation}
    onDeploy={() => {
      void onDeployModel();
    }}
    onCollectMore={() => setStep(2)}
    isDeploying={isDeploying}
    deployError={deployError}
  />
)}
```

**Step 4: Vérifier la compilation finale**

```bash
cd frontend
npm run build
```

Expected: No TypeScript errors, successful build

**Step 5: Commit**

```bash
git add frontend/src/components/training/TrainingWizard.tsx
git commit -m "feat(training): verify TrainingWizard integration with all new components

All components now integrated:
- Step 1: Category, description, tags inputs
- Step 2: Visual feedback via ClipRecorder updates
- Step 3: Metrics chart and badges via TrainingProgress
- Step 4: Intelligent deployment via ValidationTest

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 11: Tests Manuels End-to-End

**Files:**
- None (manual testing)

**Step 1: Démarrer le frontend**

```bash
cd frontend
npm run dev
```

Expected: Dev server starts at `http://localhost:5173`

**Step 2: Test Step 1 (Sign Name)**

1. Navigate to `/train`
2. Enter sign name "test_bonjour"
3. Verify category is "lsfb-v1"
4. Verify tags show ["lsfb", "v1"]
5. Enter description
6. Click Next

Expected: Advance to Step 2

**Step 3: Test Step 2 (Recording)**

1. Allow webcam access
2. Verify SignGuideOverlay is visible (faint silhouette)
3. Move hands in view
4. Verify QualityIndicator shows:
   - 🟢 Green when both hands visible
   - 🟡 Yellow when one hand visible
   - 🔴 Red when no hands visible
5. Record 5 clips (3-4 sec each)
6. Verify ClipCounter shows colored chips:
   - Green chips for valid clips
   - Counter shows "5/5 minimum"
7. Click "Start Training"

Expected: Upload clips, advance to Step 3

**Step 4: Test Step 3 (Training Progress)**

1. Verify progress bar animates
2. Verify metrics chart appears (if data available)
3. Verify DeploymentReadinessBadge shows:
   - Yellow "Below Threshold" initially
   - Updates as training progresses
4. Verify RecommendedActionMessage shows "Training in progress..."
5. Wait for training completion (~2-3 min)

Expected: Training completes, badge turns green if accuracy ≥ 85%

**Step 5: Test Step 4 (Validation)**

1. Click "Validate" button
2. Verify ValidationTest shows:
   - Prediction result
   - Confidence percentage
   - DeploymentReadinessBadge (green if ready)
3. If deployment ready:
   - Click "✓ Deploy Model"
   - Verify success animation appears
   - Verify auto-redirect to `/translate` after 2 sec
4. If below threshold:
   - Verify "⚠ Record More Clips" button shows
   - Click it, verify return to Step 2

Expected: Full flow completes successfully

**Step 6: Document les résultats**

Créer un fichier de test report :

```bash
echo "# Test Report - LSFB Training UX
Date: $(date)

## Step 1 - Sign Name
- [x] Category pre-filled
- [x] Tags pre-filled
- [x] Description input works

## Step 2 - Recording
- [x] SignGuideOverlay visible
- [x] QualityIndicator updates correctly
- [x] ClipCounter shows colored chips
- [x] 5 clips recorded successfully

## Step 3 - Training
- [x] Progress bar animates
- [x] Metrics chart displays
- [x] Deployment badge updates
- [x] Action messages show correctly

## Step 4 - Validation
- [x] Prediction displays
- [x] Deployment logic works
- [x] Success animation triggers
- [x] Auto-redirect to /translate

## Issues Found
(List any issues discovered)

## Conclusion
All features tested successfully.
" > docs/test-report-lsfb-training-ux.md
```

**Step 7: Commit final test report**

```bash
git add docs/test-report-lsfb-training-ux.md
git commit -m "docs: add manual test report for LSFB training UX improvements

Verified all new features:
- Visual feedback components
- Deployment intelligence
- Success flow animations

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 12: Documentation Utilisateur

**Files:**
- Create: `docs/user-guide-lsfb-training.md`

**Step 1: Créer le guide utilisateur**

```markdown
# Guide Utilisateur : Entraînement LSFB

## Vue d'ensemble

Ce guide explique comment entraîner SignFlow avec vos propres signes LSFB (Langue des Signes de Belgique Francophone).

## Prérequis

- Webcam fonctionnelle
- Éclairage correct (pas de contre-jour)
- Mains bien visibles dans le cadre

## Étapes

### Étape 1 : Nommer le Signe

1. Entrez le nom du signe (ex: "bonjour")
   - Le préfixe "lsfb_" sera ajouté automatiquement
2. La catégorie "lsfb-v1" est pré-remplie
3. Les tags ["lsfb", "v1"] sont pré-remplis
4. (Optionnel) Ajoutez une description
5. Cliquez "Next"

### Étape 2 : Enregistrer des Clips

1. Positionnez-vous devant la caméra
   - Suivez le guide visuel (silhouette affichée)
2. Vérifiez l'indicateur de qualité :
   - 🟢 Vert = Parfait (2 mains détectées)
   - 🟡 Jaune = Bon (1 main détectée)
   - 🔴 Rouge = Aucune main détectée
3. Enregistrez 5-10 clips de 3-4 secondes
   - Le compteur visuel montre votre progression
4. Cliquez "Start Training"

### Étape 3 : Entraînement

1. Observez la progression :
   - Barre de progression animée
   - Graphique métriques temps réel
   - Badge de statut déploiement
2. Attendez 2-3 minutes (50 epochs)
3. Cliquez "Validate" quand disponible

### Étape 4 : Validation & Déploiement

**Si accuracy ≥ 85% :**
1. Badge vert "Ready to Deploy" affiché
2. Cliquez "✓ Deploy Model"
3. Animation de succès
4. Redirection automatique vers /translate

**Si accuracy < 85% :**
1. Badge jaune "Below Threshold" affiché
2. Cliquez "⚠ Record More Clips"
3. Retour à l'étape 2 pour enregistrer plus de clips

## Conseils

- **Éclairage** : Éclairez votre visage de face, pas de derrière
- **Position** : Centrez-vous dans le cadre
- **Variété** : Enregistrez des angles légèrement différents
- **Durée** : 3-4 secondes par clip est idéal
- **Qualité** : Attendez toujours le badge vert avant d'enregistrer

## Dépannage

### Aucune main détectée (🔴)
- Vérifiez l'éclairage
- Rapprochez-vous de la caméra
- Assurez-vous que vos mains sont visibles

### Accuracy trop basse
- Enregistrez 3-5 clips supplémentaires
- Variez les angles légèrement
- Assurez-vous d'une bonne qualité de tous les clips

### Training échoue
- Vérifiez que vous avez au moins 5 clips valides
- Consultez les logs backend pour plus de détails

## Support

Pour toute question ou problème, consultez les logs ou créez une issue sur GitHub.
```

**Step 2: Commit la documentation**

```bash
git add docs/user-guide-lsfb-training.md
git commit -m "docs: add user guide for LSFB sign training

Complete guide covering:
- Prerequisites and setup
- Step-by-step instructions
- Tips for best results
- Troubleshooting common issues

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Résumé

**Composants créés (5) :**
1. QualityIndicator - Feedback qualité temps réel
2. ClipCounter - Compteur visuel chips
3. SignGuideOverlay - Guide visuel positionnement
4. DeploymentReadinessBadge - Badge statut déploiement
5. RecommendedActionMessage - Messages actions contextuels

**Composants modifiés (4) :**
1. ClipRecorder - Intégration feedback visuel
2. TrainingProgress - Chart metrics + badges
3. ValidationTest - Déploiement auto + success flow
4. TrainingWizard - Vérification intégration complète

**Dépendances ajoutées (1) :**
- recharts (charts metrics)

**Documentation créée (2) :**
- Test report manuel
- Guide utilisateur LSFB

**Total commits : 12**

---

## Notes pour l'Exécution

- **Backend déjà fonctionnel** : Aucune modification backend requise
- **Tests manuels requis** : Task 11 critique pour validation
- **Mobile-first** : Tous les composants utilisent Tailwind responsive
- **Framer Motion** : Déjà installé, utilisé pour success animation

**Dépendances système requises :**
- Node.js 18+
- npm 9+
- Webcam fonctionnelle
- Backend SignFlow running

**Temps estimé d'exécution : 2-3 heures** (avec tests inclus)
