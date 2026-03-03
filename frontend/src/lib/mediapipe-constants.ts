/**
 * MediaPipe landmark connection constants.
 *
 * Defined locally to avoid CJS/ESM interop issues with @mediapipe/holistic
 * when Vite excludes the package from pre-bundling (CDN loading for worker).
 * These arrays are stable and never change across MediaPipe versions.
 */

/** 21 hand landmarks — connection pairs for drawing skeleton. */
export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // index
  [5, 9], [9, 10], [10, 11], [11, 12],  // middle
  [9, 13], [13, 14], [14, 15], [15, 16], // ring
  [13, 17], [17, 18], [18, 19], [19, 20], // pinky
  [0, 17],                                // palm base
];

/** 33 pose landmarks — connection pairs for drawing body skeleton. */
export const POSE_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 7],       // left eye
  [0, 4], [4, 5], [5, 6], [6, 8],       // right eye
  [9, 10],                                // mouth
  [11, 12],                               // shoulders
  [11, 13], [13, 15],                     // left arm
  [12, 14], [14, 16],                     // right arm
  [15, 17], [15, 19], [15, 21], [17, 19], // left hand
  [16, 18], [16, 20], [16, 22], [18, 20], // right hand
  [11, 23], [12, 24],                     // torso
  [23, 24],                               // hips
  [23, 25], [25, 27],                     // left leg
  [24, 26], [26, 28],                     // right leg
  [27, 29], [29, 31], [27, 31],           // left foot
  [28, 30], [30, 32], [28, 32],           // right foot
];
