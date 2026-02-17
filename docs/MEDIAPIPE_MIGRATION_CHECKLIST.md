# Checklist de Migration vers MediaPipe Optimisé

## ✅ Pré-requis

- [ ] Node.js >= 18.0.0
- [ ] Vite >= 5.0.0
- [ ] TypeScript >= 5.0.0
- [ ] React >= 18.0.0
- [ ] @mediapipe/holistic >= 0.5.0

## 📦 Étape 1: Installation des Fichiers

Copier les nouveaux fichiers dans le projet:

```bash
# Librairie optimisée
frontend/src/lib/mediapipe-optimized.ts

# Hook optimisé
frontend/src/hooks/useMediaPipeOptimized.ts

# Composant overlay optimisé
frontend/src/components/camera/LandmarkOverlayOptimized.tsx

# Web Worker
frontend/src/workers/mediapipe.worker.ts
```

- [ ] Fichiers copiés
- [ ] Import path aliases configurés dans vite.config.ts
- [ ] Worker configuration ajoutée à vite.config.ts

## 🔧 Étape 2: Configuration Vite

Mettre à jour `vite.config.ts`:

```typescript
export default defineConfig({
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src")
    }
  },
  worker: {
    format: "es",
    plugins: () => [react()],
  },
  optimizeDeps: {
    exclude: ["@mediapipe/holistic"],
  },
});
```

- [ ] `worker.format` configuré
- [ ] Alias `@` configuré
- [ ] `@mediapipe/holistic` exclu des optimizeDeps

## 🔄 Étape 3: Migration Composants

### Option A: Migration Complète

Remplacer dans vos composants:

```typescript
// AVANT
import { useMediaPipe } from "@/hooks/useMediaPipe";
import { LandmarkOverlay } from "@/components/camera/LandmarkOverlay";

const { frame, ready } = useMediaPipe({ videoRef, enabled: true });

<LandmarkOverlay frame={frame} videoRef={videoRef} />
```

```typescript
// APRÈS
import { useMediaPipeOptimized } from "@/hooks/useMediaPipeOptimized";
import { LandmarkOverlayOptimized } from "@/components/camera/LandmarkOverlayOptimized";

const { frame, ready, metrics } = useMediaPipeOptimized({
  videoRef,
  enabled: true,
  adaptiveQuality: true,
  adaptiveFps: true
});

<LandmarkOverlayOptimized
  frame={frame}
  metrics={metrics}
  videoRef={videoRef}
  showPerformanceStats={true}
/>
```

- [ ] Imports mis à jour
- [ ] Hook signature mise à jour
- [ ] Props overlay mises à jour

### Option B: Migration Progressive (A/B Testing)

Utiliser feature flag:

```typescript
const USE_OPTIMIZED = import.meta.env.VITE_USE_OPTIMIZED_MEDIAPIPE === "true";

const { frame, ready, metrics } = USE_OPTIMIZED
  ? useMediaPipeOptimized({ ... })
  : useMediaPipe({ ... });
```

- [ ] Feature flag configuré
- [ ] A/B testing activé
- [ ] Métriques comparées

## 🧪 Étape 4: Tests

### Tests Visuels
- [ ] Landmarks affichés correctement
- [ ] Connexions dessinées
- [ ] Couleurs (cyan=left, yellow=right, green=pose)
- [ ] Performance overlay visible (si activé)

### Tests Performance
- [ ] FPS >= 25fps (scène normale)
- [ ] Latence < 60ms (moyenne)
- [ ] Cache hit rate > 30% (scène statique)
- [ ] Dropped frames < 5% du total

### Tests Robustesse
- [ ] Mouvement rapide des mains → détection maintenue
- [ ] Occlusion partielle → récupération rapide
- [ ] Faible luminosité → fallback qualité actif
- [ ] Scène statique → FPS throttling actif

## 📊 Étape 5: Monitoring

Activer les statistiques de debug:

```typescript
<LandmarkOverlayOptimized
  showPerformanceStats={true}
  metrics={metrics}
/>
```

Vérifier dans la console:
- [ ] Cache stats logués
- [ ] FPS controller status affiché
- [ ] Multi-stage detector events tracés
- [ ] Aucune erreur worker

## 🎯 Étape 6: Configuration par Environnement

### Développement
```typescript
{
  targetFps: 30,
  modelComplexity: 1,        // Balanced
  adaptiveQuality: true,
  adaptiveFps: false,        // Predictable debugging
  showPerformanceStats: true
}
```

### Production
```typescript
{
  targetFps: 30,
  modelComplexity: 2,        // Max quality
  adaptiveQuality: true,     // Battery saving
  adaptiveFps: true,
  showPerformanceStats: false
}
```

### Training
```typescript
{
  targetFps: 15,             // Slower, more accurate
  modelComplexity: 2,
  adaptiveQuality: false,    // Consistent data
  adaptiveFps: false,
  minDetectionConfidence: 0.5 // Permissive
}
```

- [ ] Config dev créée
- [ ] Config prod créée
- [ ] Config training créée
- [ ] Variables d'environnement configurées

## 🐛 Étape 7: Debugging

### Worker ne démarre pas
```typescript
// Vérifier support browser
if (typeof Worker === "undefined") {
  console.error("Web Workers not supported");
}

// Vérifier console errors
workerRef.current.onerror = (error) => {
  console.error("Worker error:", error);
};
```

### FPS bas persistant
```typescript
// Vérifier métriques
console.log("Processing time:", metrics.processingTime);
console.log("Dropped frames:", metrics.droppedFrames);

// Réduire qualité manuellement
const { frame } = useMediaPipeOptimized({
  modelComplexity: 1, // ou 0
  adaptiveQuality: false
});
```

### Cache inefficace
```typescript
// Vérifier cache stats
const stats = predictionCache.getStats();
console.log("Hit rate:", stats.hitRate);

// Si < 10% → scene très dynamique (normal)
// Si 0% → vérifier calculateMovementDelta
```

## ✅ Checklist Finale

### Fonctionnalités
- [ ] Détection landmarks fonctionne
- [ ] Overlay rendering fonctionne
- [ ] Web Worker actif (vérifier DevTools → Sources → Threads)
- [ ] Métriques affichées

### Performance
- [ ] FPS >= 25fps
- [ ] Latence < 60ms
- [ ] Cache hit rate > 20%
- [ ] Pas de memory leaks (DevTools → Memory)

### Compatibilité
- [ ] Chrome/Edge: OK
- [ ] Firefox: OK
- [ ] Safari >= 16.4: OK
- [ ] Safari < 16.4: Fallback OK

### Monitoring
- [ ] Sentry/LogRocket intégré (optional)
- [ ] Performance API utilisé
- [ ] Métriques exportées vers analytics

## 🚀 Rollout Strategy

### Phase 1: Canary (Semaine 1)
- [ ] 5% trafic sur version optimisée
- [ ] Monitoring métriques 24/7
- [ ] Rollback plan prêt

### Phase 2: Gradual (Semaine 2-3)
- [ ] 25% → 50% → 75% trafic
- [ ] Comparaison A/B metrics
- [ ] Bug reports triés

### Phase 3: Full (Semaine 4)
- [ ] 100% trafic
- [ ] Ancienne version supprimée
- [ ] Documentation mise à jour

## 📝 Notes de Migration

### Breaking Changes
- Signature hook changée: `{ frame, ready }` → `{ frame, ready, metrics }`
- Props overlay: `metrics` est maintenant requis pour stats
- Worker CDN: nécessite accès https://cdn.jsdelivr.net

### Deprecated
- `useMediaPipe` → `useMediaPipeOptimized`
- `LandmarkOverlay` → `LandmarkOverlayOptimized`
- `mediapipe.ts` → `mediapipe-optimized.ts`

### Kept for Backward Compatibility
- Anciens hooks/components restent fonctionnels
- Pas de suppression forcée
- Migration volontaire recommandée

## 🎓 Formation Équipe

- [ ] Présentation optimisations (30min)
- [ ] Workshop intégration (1h)
- [ ] Documentation partagée
- [ ] Canal Slack #mediapipe-optimizations créé

---

**Date de migration:** _____________________

**Reviewer:** _____________________

**Production deploy:** _____________________
