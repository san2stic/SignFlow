/**
 * Type declarations for MediaPipe optimizations
 */

declare module "@/lib/mediapipe-optimized" {
  export interface LandmarkFrame {
    timestamp: number;
    frame_idx: number;
    hands: {
      left: number[][];
      right: number[][];
    };
    pose: number[][];
    face?: number[][];
    metadata?: {
      leftHandVisible: boolean;
      rightHandVisible: boolean;
      poseVisible: boolean;
      faceVisible: boolean;
      averageConfidence: number;
      detectionQuality: "high" | "medium" | "low";
      wasReused: boolean;
    };
  }

  export class FramePool {
    acquire(): LandmarkFrame;
    release(frame: LandmarkFrame): void;
  }

  export class PredictionCache {
    shouldReuse(current: any): boolean;
    update(frame: LandmarkFrame): void;
    getStats(): {
      hits: number;
      misses: number;
      hitRate: number;
    };
    clear(): void;
  }

  export const framePool: FramePool;
  export const predictionCache: PredictionCache;

  export function frameFromHolisticResult(
    result: any,
    frameIndex: number,
    includeFace?: boolean,
    reuseFrame?: LandmarkFrame
  ): LandmarkFrame;

  export function serializeLandmarkFrame(frame: LandmarkFrame): LandmarkFrame;
}

declare module "@/hooks/useMediaPipeOptimized" {
  import type { RefObject } from "react";
  import type { LandmarkFrame } from "@/lib/mediapipe-optimized";

  export interface PerformanceMetrics {
    fps: number;
    processingTime: number;
    cacheHitRate: number;
    droppedFrames: number;
    currentModelComplexity: 0 | 1 | 2;
  }

  export interface UseMediaPipeOptimizedOptions {
    videoRef: RefObject<HTMLVideoElement>;
    enabled: boolean;
    targetFps?: number;
    includeFace?: boolean;
    modelComplexity?: 0 | 1 | 2;
    minDetectionConfidence?: number;
    minTrackingConfidence?: number;
    adaptiveQuality?: boolean;
    adaptiveFps?: boolean;
  }

  export function useMediaPipeOptimized(
    options: UseMediaPipeOptimizedOptions
  ): {
    frame: LandmarkFrame | null;
    ready: boolean;
    metrics: PerformanceMetrics;
  };

  export type { LandmarkFrame };
}

declare module "@/components/camera/LandmarkOverlayOptimized" {
  import type { RefObject } from "react";
  import type { LandmarkFrame } from "@/lib/mediapipe-optimized";
  import type { PerformanceMetrics } from "@/hooks/useMediaPipeOptimized";

  export interface LandmarkOverlayOptimizedProps {
    frame: LandmarkFrame | null;
    showConnections?: boolean;
    showConfidenceIndicator?: boolean;
    showPerformanceStats?: boolean;
    videoRef?: RefObject<HTMLVideoElement>;
    fit?: "cover" | "contain";
    metrics?: PerformanceMetrics;
  }

  export function LandmarkOverlayOptimized(
    props: LandmarkOverlayOptimizedProps
  ): JSX.Element;
}

declare module "@/workers/mediapipe.worker" {
  export interface WorkerConfig {
    modelComplexity: 0 | 1 | 2;
    minDetectionConfidence: number;
    minTrackingConfidence: number;
    smoothLandmarks: boolean;
    refineFaceLandmarks: boolean;
  }

  export interface WorkerMessage {
    type: "init" | "process" | "updateConfig" | "close";
    data?: any;
  }

  export interface WorkerResponse {
    type: "ready" | "results" | "busy" | "error" | "configUpdated" | "closed";
    data?: any;
    error?: string;
  }
}

declare module "@/utils/mediapipe-benchmark" {
  export interface BenchmarkResult {
    name: string;
    duration: number;
    fps: number;
    avgLatency: number;
    p95Latency: number;
    p99Latency: number;
    droppedFrames: number;
    memoryUsed: number;
    gcCount: number;
    cacheHitRate?: number;
  }

  export interface BenchmarkScenario {
    name: string;
    duration: number;
    config: {
      targetFps: number;
      modelComplexity: 0 | 1 | 2;
      adaptiveQuality?: boolean;
      adaptiveFps?: boolean;
    };
    videoSource: "static" | "dynamic" | "rapid" | "lowlight";
  }

  export class MediaPipeBenchmark {
    runScenario(scenario: BenchmarkScenario): Promise<BenchmarkResult>;
    runAll(): Promise<void>;
    getReport(): string;
    exportJSON(): string;
    clear(): void;
  }

  export const benchmark: MediaPipeBenchmark;
}

// Global type augmentations
declare global {
  interface Performance {
    memory?: {
      usedJSHeapSize: number;
      totalJSHeapSize: number;
      jsHeapSizeLimit: number;
    };
  }

  interface OffscreenCanvas {
    transferToImageBitmap(): ImageBitmap;
  }
}

export {};
