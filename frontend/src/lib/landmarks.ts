import type { LandmarkFrame } from "./mediapipe";

function hasVisiblePoints(points: number[][]): boolean {
  return points.some((point) => point[0] !== 0 || point[1] !== 0 || point[2] !== 0);
}

export function countVisibleHands(frame: LandmarkFrame | null): number {
  if (!frame) return 0;
  let count = 0;
  if (hasVisiblePoints(frame.hands.left)) count += 1;
  if (hasVisiblePoints(frame.hands.right)) count += 1;
  return count;
}
