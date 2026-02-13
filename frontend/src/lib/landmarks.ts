import type { LandmarkFrame } from "./mediapipe";

export function countVisibleHands(frame: LandmarkFrame | null): number {
  if (!frame) return 0;
  let count = 0;
  if (frame.hands.left.length > 0) count += 1;
  if (frame.hands.right.length > 0) count += 1;
  return count;
}
