import { useEffect, useRef } from "react";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { HAND_CONNECTIONS, POSE_CONNECTIONS } from "@mediapipe/holistic";

import type { LandmarkFrame } from "../../lib/mediapipe";

interface LandmarkOverlayProps {
  frame: LandmarkFrame | null;
  showConnections?: boolean;
}

/**
 * Canvas overlay for visualizing MediaPipe landmarks and connections.
 *
 * Draws:
 * - Hand landmarks (21 points per hand) with connections
 * - Pose landmarks (33 points) with skeleton connections
 * - Color-coded: left hand (cyan), right hand (yellow), pose (green)
 */
export function LandmarkOverlay({
  frame,
  showConnections = true,
}: LandmarkOverlayProps): JSX.Element {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !frame) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Match canvas resolution to display size
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width;
    canvas.height = height;

    // Clear previous frame
    ctx.clearRect(0, 0, width, height);

    // Helper to convert normalized landmarks to canvas coordinates
    const toLandmarkList = (points: number[][]) => {
      return points.map((p) => ({ x: p[0], y: p[1], z: p[2] || 0 }));
    };

    // Draw pose (body skeleton)
    if (frame.pose && frame.pose.length > 0) {
      const poseLandmarks = toLandmarkList(frame.pose);

      if (showConnections) {
        drawConnectors(ctx, poseLandmarks, POSE_CONNECTIONS, {
          color: "#10B981",
          lineWidth: 2,
        });
      }

      drawLandmarks(ctx, poseLandmarks, {
        color: "#10B981",
        fillColor: "#10B981",
        radius: 3,
      });
    }

    // Draw left hand
    if (frame.hands.left && frame.hands.left.length > 0) {
      const leftHandLandmarks = toLandmarkList(frame.hands.left);

      if (showConnections) {
        drawConnectors(ctx, leftHandLandmarks, HAND_CONNECTIONS, {
          color: "#06B6D4",
          lineWidth: 2,
        });
      }

      drawLandmarks(ctx, leftHandLandmarks, {
        color: "#06B6D4",
        fillColor: "#06B6D4",
        radius: 4,
      });
    }

    // Draw right hand
    if (frame.hands.right && frame.hands.right.length > 0) {
      const rightHandLandmarks = toLandmarkList(frame.hands.right);

      if (showConnections) {
        drawConnectors(ctx, rightHandLandmarks, HAND_CONNECTIONS, {
          color: "#F59E0B",
          lineWidth: 2,
        });
      }

      drawLandmarks(ctx, rightHandLandmarks, {
        color: "#F59E0B",
        fillColor: "#F59E0B",
        radius: 4,
      });
    }
  }, [frame, showConnections]);

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none absolute inset-0 h-full w-full rounded-card"
    />
  );
}
