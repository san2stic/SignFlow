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
    mouthDetected: boolean;
    mouthOpen: boolean;
    mouthOpenRatio: number;
    facialExpressionIntensity: number;
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

const FACE_IDX_MOUTH_LEFT = 61;
const FACE_IDX_MOUTH_RIGHT = 291;
const FACE_IDX_UPPER_LIP = 13;
const FACE_IDX_LOWER_LIP = 14;
const FACE_IDX_LEFT_EYE_UPPER = 159;
const FACE_IDX_LEFT_EYE_LOWER = 145;
const FACE_IDX_RIGHT_EYE_UPPER = 386;
const FACE_IDX_RIGHT_EYE_LOWER = 374;
const FACE_IDX_LEFT_BROW = 70;
const FACE_IDX_RIGHT_BROW = 300;

interface MouthMetrics {
  mouthDetected: boolean;
  mouthOpen: boolean;
  mouthOpenRatio: number;
  facialExpressionIntensity: number;
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

function distance(a: LandmarkLike, b: LandmarkLike): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function calculateMouthMetrics(
  faceLandmarks: LandmarkLike[] | undefined,
  visibilityThreshold = 0.5
): MouthMetrics {
  if (!faceLandmarks || faceLandmarks.length === 0) {
    return {
      mouthDetected: false,
      mouthOpen: false,
      mouthOpenRatio: 0,
      facialExpressionIntensity: 0
    };
  }

  const get = (index: number): LandmarkLike | null => {
    const point = faceLandmarks[index];
    if (!point) return null;
    const visibility = point.visibility ?? 1;
    return visibility >= visibilityThreshold ? point : null;
  };

  const mouthLeft = get(FACE_IDX_MOUTH_LEFT);
  const mouthRight = get(FACE_IDX_MOUTH_RIGHT);
  const upperLip = get(FACE_IDX_UPPER_LIP);
  const lowerLip = get(FACE_IDX_LOWER_LIP);
  const leftEyeUpper = get(FACE_IDX_LEFT_EYE_UPPER);
  const leftEyeLower = get(FACE_IDX_LEFT_EYE_LOWER);
  const rightEyeUpper = get(FACE_IDX_RIGHT_EYE_UPPER);
  const rightEyeLower = get(FACE_IDX_RIGHT_EYE_LOWER);
  const leftBrow = get(FACE_IDX_LEFT_BROW);
  const rightBrow = get(FACE_IDX_RIGHT_BROW);

  const required = [mouthLeft, mouthRight, upperLip, lowerLip];
  const mouthDetected = required.every((point) => point !== null);
  if (!mouthDetected) {
    return {
      mouthDetected: false,
      mouthOpen: false,
      mouthOpenRatio: 0,
      facialExpressionIntensity: 0
    };
  }

  const mouthOpenDistance = distance(upperLip!, lowerLip!);
  const mouthWidth = distance(mouthLeft!, mouthRight!);
  const mouthOpenRatio = mouthOpenDistance / Math.max(mouthWidth, 1e-6);
  const mouthOpen = mouthOpenRatio > 0.12 && mouthOpenDistance > 0.01;

  const eyeOpenLeft = leftEyeUpper && leftEyeLower ? distance(leftEyeUpper, leftEyeLower) : 0;
  const eyeOpenRight = rightEyeUpper && rightEyeLower ? distance(rightEyeUpper, rightEyeLower) : 0;
  const browRaiseLeft =
    leftBrow && leftEyeUpper ? Math.max(0, leftEyeUpper.y - leftBrow.y) : 0;
  const browRaiseRight =
    rightBrow && rightEyeUpper ? Math.max(0, rightEyeUpper.y - rightBrow.y) : 0;

  const facialExpressionIntensity = Math.min(
    1,
    (
      mouthOpenRatio +
      eyeOpenLeft +
      eyeOpenRight +
      browRaiseLeft +
      browRaiseRight
    ) / 5
  );

  return {
    mouthDetected: true,
    mouthOpen,
    mouthOpenRatio,
    facialExpressionIntensity
  };
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
  const mouthMetrics = includeFace
    ? calculateMouthMetrics(result.faceLandmarks, visibilityThreshold)
    : {
      mouthDetected: false,
      mouthOpen: false,
      mouthOpenRatio: 0,
      facialExpressionIntensity: 0
    };

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
      averageConfidence,
      mouthDetected: mouthMetrics.mouthDetected,
      mouthOpen: mouthMetrics.mouthOpen,
      mouthOpenRatio: mouthMetrics.mouthOpenRatio,
      facialExpressionIntensity: mouthMetrics.facialExpressionIntensity
    }
  };
}

export function serializeLandmarkFrame(frame: LandmarkFrame): LandmarkFrame {
  return frame;
}
