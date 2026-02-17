/**
 * Exemple d'intégration des optimisations MediaPipe dans TranslatePage
 *
 * AVANT vs APRÈS comparison
 */

// ============================================================
// AVANT (Version originale)
// ============================================================

import { useMediaPipe } from "@/hooks/useMediaPipe";
import { LandmarkOverlay } from "@/components/camera/LandmarkOverlay";

function TranslatePageBefore() {
  const videoRef = useRef<HTMLVideoElement>(null);

  const { frame, ready } = useMediaPipe({
    videoRef,
    enabled: true,
    targetFps: 30,
    modelComplexity: 2
  });

  return (
    <div className="relative">
      <video ref={videoRef} />
      <LandmarkOverlay
        frame={frame}
        showConnections={true}
        showConfidenceIndicator={true}
        videoRef={videoRef}
      />
    </div>
  );
}

// ============================================================
// APRÈS (Version optimisée)
// ============================================================

import { useMediaPipeOptimized } from "@/hooks/useMediaPipeOptimized";
import { LandmarkOverlayOptimized } from "@/components/camera/LandmarkOverlayOptimized";

function TranslatePageAfter() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [showStats, setShowStats] = useState(true);

  // Hook optimisé avec toutes les features
  const { frame, ready, metrics } = useMediaPipeOptimized({
    videoRef,
    enabled: true,
    targetFps: 30,
    modelComplexity: 2,
    adaptiveQuality: true,  // 🆕 Auto-adjust quality
    adaptiveFps: true,      // 🆕 Smart FPS throttling
    includeFace: false
  });

  // Debug metrics en temps réel
  useEffect(() => {
    if (metrics && showStats) {
      console.log(`[MediaPipe] FPS: ${metrics.fps.toFixed(1)}`);
      console.log(`[MediaPipe] Processing: ${metrics.processingTime.toFixed(1)}ms`);
      console.log(`[MediaPipe] Cache Hit: ${(metrics.cacheHitRate * 100).toFixed(1)}%`);
      console.log(`[MediaPipe] Quality: ${metrics.currentModelComplexity}`);
    }
  }, [metrics, showStats]);

  return (
    <div className="relative">
      <video ref={videoRef} />

      {/* Overlay optimisé avec stats */}
      <LandmarkOverlayOptimized
        frame={frame}
        showConnections={true}
        showConfidenceIndicator={true}
        showPerformanceStats={showStats}  // 🆕 Performance overlay
        videoRef={videoRef}
        metrics={metrics}  // 🆕 Pass metrics for display
      />

      {/* Toggle stats */}
      <button
        onClick={() => setShowStats(!showStats)}
        className="absolute bottom-4 right-4 rounded bg-black/50 px-3 py-1 text-white"
      >
        {showStats ? "Hide" : "Show"} Stats
      </button>

      {/* Performance badges */}
      <div className="absolute top-4 left-4 space-y-2">
        {frame?.metadata?.wasReused && (
          <div className="rounded bg-purple-500/90 px-2 py-1 text-xs text-white">
            🚀 CACHED
          </div>
        )}

        {metrics && metrics.cacheHitRate > 0.5 && (
          <div className="rounded bg-green-500/90 px-2 py-1 text-xs text-white">
            ⚡ High Cache Efficiency
          </div>
        )}

        {metrics && metrics.droppedFrames > 10 && (
          <div className="rounded bg-yellow-500/90 px-2 py-1 text-xs text-white">
            ⚠️ Frame Drops: {metrics.droppedFrames}
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================
// EXEMPLE AVANCÉ: Dashboard de Monitoring
// ============================================================

function PerformanceDashboard({ metrics }: { metrics: PerformanceMetrics }) {
  return (
    <div className="fixed bottom-0 left-0 w-full bg-black/80 p-4 text-white">
      <div className="grid grid-cols-5 gap-4">
        {/* FPS Indicator */}
        <div className="text-center">
          <div className="text-2xl font-bold">
            {metrics.fps.toFixed(1)}
          </div>
          <div className="text-xs text-gray-400">FPS</div>
          <div className="mt-1 h-2 overflow-hidden rounded bg-gray-700">
            <div
              className={`h-full ${
                metrics.fps >= 25 ? "bg-green-500" :
                metrics.fps >= 15 ? "bg-yellow-500" :
                "bg-red-500"
              }`}
              style={{ width: `${(metrics.fps / 30) * 100}%` }}
            />
          </div>
        </div>

        {/* Latency */}
        <div className="text-center">
          <div className="text-2xl font-bold">
            {metrics.processingTime.toFixed(0)}ms
          </div>
          <div className="text-xs text-gray-400">Latency</div>
          <div className="mt-1 h-2 overflow-hidden rounded bg-gray-700">
            <div
              className={`h-full ${
                metrics.processingTime <= 50 ? "bg-green-500" :
                metrics.processingTime <= 100 ? "bg-yellow-500" :
                "bg-red-500"
              }`}
              style={{ width: `${Math.min(100, (metrics.processingTime / 150) * 100)}%` }}
            />
          </div>
        </div>

        {/* Cache Efficiency */}
        <div className="text-center">
          <div className="text-2xl font-bold">
            {(metrics.cacheHitRate * 100).toFixed(0)}%
          </div>
          <div className="text-xs text-gray-400">Cache Hit</div>
          <div className="mt-1 h-2 overflow-hidden rounded bg-gray-700">
            <div
              className="h-full bg-purple-500"
              style={{ width: `${metrics.cacheHitRate * 100}%` }}
            />
          </div>
        </div>

        {/* Quality Level */}
        <div className="text-center">
          <div className="text-2xl font-bold">
            {metrics.currentModelComplexity}
          </div>
          <div className="text-xs text-gray-400">Quality</div>
          <div className="mt-1 flex justify-center gap-1">
            {[0, 1, 2].map((level) => (
              <div
                key={level}
                className={`h-2 w-6 rounded ${
                  level <= metrics.currentModelComplexity
                    ? "bg-blue-500"
                    : "bg-gray-700"
                }`}
              />
            ))}
          </div>
        </div>

        {/* Dropped Frames */}
        <div className="text-center">
          <div className="text-2xl font-bold">
            {metrics.droppedFrames}
          </div>
          <div className="text-xs text-gray-400">Dropped</div>
          <div className="mt-1 h-2 overflow-hidden rounded bg-gray-700">
            <div
              className={`h-full ${
                metrics.droppedFrames === 0 ? "bg-green-500" :
                metrics.droppedFrames < 10 ? "bg-yellow-500" :
                "bg-red-500"
              }`}
              style={{ width: `${Math.min(100, (metrics.droppedFrames / 50) * 100)}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// ============================================================
// EXEMPLE: Configuration Adaptative par Contexte
// ============================================================

function AdaptiveMediaPipeConfig() {
  const [mode, setMode] = useState<"training" | "inference" | "demo">("inference");

  // Configuration spécifique par mode
  const config = useMemo(() => {
    switch (mode) {
      case "training":
        return {
          targetFps: 15,              // Moins de FPS pour batch processing
          modelComplexity: 2,         // Max quality pour training data
          adaptiveQuality: false,     // Fixed quality
          adaptiveFps: false,         // Fixed FPS
          minDetectionConfidence: 0.5 // Permissive pour plus de data
        };

      case "inference":
        return {
          targetFps: 30,              // Temps réel
          modelComplexity: 2,
          adaptiveQuality: true,      // Auto-optimize
          adaptiveFps: true,          // Battery saving
          minDetectionConfidence: 0.7 // Strict pour production
        };

      case "demo":
        return {
          targetFps: 30,
          modelComplexity: 1,         // Balanced
          adaptiveQuality: false,     // Predictable behavior
          adaptiveFps: false,         // Consistent demo
          minDetectionConfidence: 0.6
        };
    }
  }, [mode]);

  const { frame, ready, metrics } = useMediaPipeOptimized({
    videoRef,
    enabled: true,
    ...config
  });

  return (
    <div>
      {/* Mode selector */}
      <select value={mode} onChange={(e) => setMode(e.target.value as any)}>
        <option value="training">Training Mode</option>
        <option value="inference">Inference Mode</option>
        <option value="demo">Demo Mode</option>
      </select>

      {/* Rendering */}
      <LandmarkOverlayOptimized frame={frame} metrics={metrics} />
    </div>
  );
}

// ============================================================
// EXEMPLE: A/B Testing Optimisations
// ============================================================

function ABTestOptimizations() {
  const [useOptimized, setUseOptimized] = useState(true);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Baseline hook
  const baseline = useMediaPipe({
    videoRef,
    enabled: !useOptimized,
    targetFps: 30,
    modelComplexity: 2
  });

  // Optimized hook
  const optimized = useMediaPipeOptimized({
    videoRef,
    enabled: useOptimized,
    targetFps: 30,
    modelComplexity: 2,
    adaptiveQuality: true,
    adaptiveFps: true
  });

  const { frame, ready } = useOptimized ? optimized : baseline;

  return (
    <div>
      <button onClick={() => setUseOptimized(!useOptimized)}>
        Switch to {useOptimized ? "Baseline" : "Optimized"}
      </button>

      <div className="mt-2 rounded bg-gray-800 p-2 text-white">
        Mode: {useOptimized ? "🚀 Optimized" : "📦 Baseline"}
      </div>

      {useOptimized && optimized.metrics && (
        <div className="mt-2 rounded bg-green-800 p-2 text-white">
          Cache Hit: {(optimized.metrics.cacheHitRate * 100).toFixed(0)}%
          <br />
          FPS: {optimized.metrics.fps.toFixed(1)}
          <br />
          Quality: {optimized.metrics.currentModelComplexity}
        </div>
      )}

      <LandmarkOverlayOptimized
        frame={frame}
        metrics={useOptimized ? optimized.metrics : undefined}
        showPerformanceStats={useOptimized}
      />
    </div>
  );
}

export {
  TranslatePageBefore,
  TranslatePageAfter,
  PerformanceDashboard,
  AdaptiveMediaPipeConfig,
  ABTestOptimizations
};
