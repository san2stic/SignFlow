export interface LandmarkFrame {
  timestamp: number;
  frame_idx: number;
  hands: { left: number[][]; right: number[][] };
  pose: number[][];
  face?: number[][];
}

interface LandmarkLike {
  x: number;
  y: number;
  z: number;
}

interface HolisticResultLike {
  leftHandLandmarks?: LandmarkLike[];
  rightHandLandmarks?: LandmarkLike[];
  poseLandmarks?: LandmarkLike[];
  faceLandmarks?: LandmarkLike[];
}

function toXYZ(points: LandmarkLike[] | undefined, expected: number): number[][] {
  if (!points || points.length === 0) {
    return [];
  }
  const mapped = points.slice(0, expected).map((point) => [point.x, point.y, point.z]);
  if (mapped.length < expected) {
    return [...mapped, ...Array.from({ length: expected - mapped.length }, () => [0, 0, 0])];
  }
  return mapped;
}

export function frameFromHolisticResult(
  result: HolisticResultLike,
  frameIndex: number,
  includeFace = false
): LandmarkFrame {
  return {
    timestamp: Date.now() / 1000,
    frame_idx: frameIndex,
    hands: {
      left: toXYZ(result.leftHandLandmarks, 21),
      right: toXYZ(result.rightHandLandmarks, 21)
    },
    pose: toXYZ(result.poseLandmarks, 33),
    face: includeFace ? toXYZ(result.faceLandmarks, 468) : undefined
  };
}

export function serializeLandmarkFrame(frame: LandmarkFrame): LandmarkFrame {
  return frame;
}
