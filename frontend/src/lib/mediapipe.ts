export interface LandmarkFrame {
  timestamp: number;
  frame_idx: number;
  hands: { left: number[][]; right: number[][] };
  pose: number[][];
  face?: number[][];
  // Metadata for confidence and visibility tracking
  metadata?: {
    leftHandVisible: boolean;
    rightHandVisible: boolean;
    poseVisible: boolean;
    faceVisible: boolean;
    averageConfidence: number; // Average visibility score across all landmarks
  };
}

interface LandmarkLike {
  x: number;
  y: number;
  z: number;
  visibility?: number; // MediaPipe provides visibility score (0-1)
}

interface HolisticResultLike {
  leftHandLandmarks?: LandmarkLike[];
  rightHandLandmarks?: LandmarkLike[];
  poseLandmarks?: LandmarkLike[];
  faceLandmarks?: LandmarkLike[];
}

/**
 * Filter and convert landmarks to XYZ format, applying visibility threshold.
 * Only keeps landmarks with visibility >= threshold (default: 0.5).
 */
function toXYZ(
  points: LandmarkLike[] | undefined,
  expected: number,
  visibilityThreshold = 0.5
): number[][] {
  if (!points || points.length === 0) {
    return [];
  }

  const mapped = points.slice(0, expected).map((point) => {
    // Filter out low-visibility landmarks (replace with zeros)
    const visibility = point.visibility ?? 1.0; // Default to visible if no visibility data
    if (visibility < visibilityThreshold) {
      return [0, 0, 0]; // Low confidence point -> set to origin
    }
    return [point.x, point.y, point.z];
  });

  if (mapped.length < expected) {
    return [...mapped, ...Array.from({ length: expected - mapped.length }, () => [0, 0, 0])];
  }
  return mapped;
}

/**
 * Calculate average visibility score from landmarks.
 */
function calculateAverageVisibility(points: LandmarkLike[] | undefined): number {
  if (!points || points.length === 0) return 0.0;

  const visibilityScores = points
    .map(p => p.visibility ?? 1.0)
    .filter(v => v > 0); // Ignore zero visibility points

  if (visibilityScores.length === 0) return 0.0;

  return visibilityScores.reduce((sum, v) => sum + v, 0) / visibilityScores.length;
}

export function frameFromHolisticResult(
  result: HolisticResultLike,
  frameIndex: number,
  includeFace = false,
  visibilityThreshold = 0.5
): LandmarkFrame {
  const leftHand = toXYZ(result.leftHandLandmarks, 21, visibilityThreshold);
  const rightHand = toXYZ(result.rightHandLandmarks, 21, visibilityThreshold);
  const pose = toXYZ(result.poseLandmarks, 33, visibilityThreshold);
  const face = includeFace ? toXYZ(result.faceLandmarks, 468, visibilityThreshold) : undefined;

  // Calculate metadata
  const leftHandVisible = (result.leftHandLandmarks?.length ?? 0) > 0;
  const rightHandVisible = (result.rightHandLandmarks?.length ?? 0) > 0;
  const poseVisible = (result.poseLandmarks?.length ?? 0) > 0;
  const faceVisible = includeFace && (result.faceLandmarks?.length ?? 0) > 0;

  // Calculate overall confidence from all visible landmarks
  const allVisibilities = [
    ...(result.leftHandLandmarks || []),
    ...(result.rightHandLandmarks || []),
    ...(result.poseLandmarks || []),
    ...(includeFace ? (result.faceLandmarks || []) : [])
  ];
  const averageConfidence = calculateAverageVisibility(allVisibilities);

  return {
    timestamp: Date.now() / 1000,
    frame_idx: frameIndex,
    hands: { left: leftHand, right: rightHand },
    pose,
    face,
    metadata: {
      leftHandVisible,
      rightHandVisible,
      poseVisible,
      faceVisible,
      averageConfidence
    }
  };
}

export function serializeLandmarkFrame(frame: LandmarkFrame): LandmarkFrame {
  return frame;
}
