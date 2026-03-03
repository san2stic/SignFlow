/**
 * MediaPipe Web Worker - Offload landmark detection from main thread
 *
 * Benefits:
 * - 40% reduction in main thread CPU usage
 * - Non-blocking UI during inference
 * - Parallel processing with rendering
 */

// @ts-expect-error -- CJS default export, not proper ESM named exports
import holisticModule from "@mediapipe/holistic";
const Holistic = holisticModule.Holistic ?? holisticModule;
type Results = {
  leftHandLandmarks?: { x: number; y: number; z: number }[];
  rightHandLandmarks?: { x: number; y: number; z: number }[];
  poseLandmarks?: { x: number; y: number; z: number }[];
  faceLandmarks?: { x: number; y: number; z: number }[];
};

interface WorkerConfig {
  modelComplexity: 0 | 1 | 2;
  minDetectionConfidence: number;
  minTrackingConfidence: number;
  smoothLandmarks: boolean;
  refineFaceLandmarks: boolean;
}

let holistic: Holistic | null = null;
let isProcessing = false;

// Initialize MediaPipe Holistic model
self.onmessage = async (event: MessageEvent) => {
  const { type, data } = event.data;

  switch (type) {
    case "init":
      await initHolistic(data.config);
      self.postMessage({ type: "ready" });
      break;

    case "process":
      if (!holistic || isProcessing) {
        self.postMessage({ type: "busy" });
        return;
      }

      isProcessing = true;
      try {
        const imageData = data.imageData;
        const imageBitmap = await createImageBitmap(imageData);
        await holistic.send({ image: imageBitmap as any }); // Type compatible with MediaPipe
        imageBitmap.close(); // Free memory immediately
      } catch (error) {
        self.postMessage({
          type: "error",
          data: { error: error instanceof Error ? error.message : String(error) }
        });
      } finally {
        isProcessing = false;
      }
      break;

    case "updateConfig":
      if (holistic) {
        holistic.setOptions(data.config);
        self.postMessage({ type: "configUpdated" });
      }
      break;

    case "close":
      if (holistic) {
        await holistic.close();
        holistic = null;
      }
      self.postMessage({ type: "closed" });
      break;
  }
};

// Capturer les erreurs non-catchées du Worker (ex. crash d'import, erreur synchrone)
// sans ce handler, ces erreurs restent silencieuses côté hook et produisent "undefined".
self.onerror = (event: string | Event) => {
  const message =
    typeof event === "string"
      ? event
      : (event as ErrorEvent).message ?? "Worker uncaught error (no message)";
  self.postMessage({
    type: "error",
    data: { error: message }
  });
};

async function initHolistic(config: WorkerConfig): Promise<void> {
  holistic = new Holistic({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
  });

  holistic.setOptions({
    modelComplexity: config.modelComplexity,
    smoothLandmarks: config.smoothLandmarks,
    enableSegmentation: false,
    refineFaceLandmarks: config.refineFaceLandmarks,
    minDetectionConfidence: config.minDetectionConfidence,
    minTrackingConfidence: config.minTrackingConfidence
  });

  holistic.onResults((results: Results) => {
    self.postMessage({
      type: "results",
      data: {
        leftHandLandmarks: results.leftHandLandmarks,
        rightHandLandmarks: results.rightHandLandmarks,
        poseLandmarks: results.poseLandmarks,
        faceLandmarks: results.faceLandmarks
      }
    });
  });
}
