import { useEffect, useRef } from "react";
import { drawConnectors, drawLandmarks } from "@mediapipe/drawing_utils";
import { HAND_CONNECTIONS, POSE_CONNECTIONS } from "@mediapipe/holistic";

import type { LandmarkFrame } from "../../lib/mediapipe";

interface LandmarkOverlayProps {
  frame: LandmarkFrame | null;
  showConnections?: boolean;
  showConfidenceIndicator?: boolean; // New: show confidence level
}

/**
 * Canvas overlay for visualizing MediaPipe landmarks and connections.
 *
 * Draws:
 * - Hand landmarks (21 points per hand) with connections
 * - Pose landmarks (33 points) with skeleton connections
 * - Color-coded: left hand (cyan), right hand (yellow), pose (green)
 * - Confidence indicator (when enabled)
 *
 * Enhanced with visibility filtering to show only high-confidence landmarks.
 */
export function LandmarkOverlay({
  frame,
  showConnections = true,
  showConfidenceIndicator = true
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
    // Filters out zero-coordinate points (low confidence filtered landmarks)
    const toLandmarkList = (points: number[][]) => {
      return points
        .map((p) => ({ x: p[0], y: p[1], z: p[2] || 0 }))
        .filter((p) => p.x !== 0 || p.y !== 0 || p.z !== 0); // Remove filtered points
    };

    // Draw confidence indicator
    if (showConfidenceIndicator && frame.metadata) {
      const confidence = frame.metadata.averageConfidence;
      const confidencePercent = Math.round(confidence * 100);

      // Draw confidence bar in top-left corner
      const barWidth = 120;
      const barHeight = 8;
      const padding = 10;

      // Background bar
      ctx.fillStyle = "rgba(0, 0, 0, 0.5)";
      ctx.fillRect(padding, padding, barWidth, barHeight);

      // Confidence bar (color changes based on level)
      const confidenceColor =
        confidence >= 0.8 ? "#10B981" : // Green for high confidence
        confidence >= 0.5 ? "#F59E0B" : // Yellow for medium
        "#EF4444"; // Red for low

      ctx.fillStyle = confidenceColor;
      ctx.fillRect(padding, padding, barWidth * confidence, barHeight);

      // Confidence text
      ctx.fillStyle = "#FFFFFF";
      ctx.font = "12px monospace";
      ctx.fillText(`${confidencePercent}%`, padding + barWidth + 8, padding + barHeight);

      // Draw detection status indicators
      const statusY = padding + barHeight + 20;
      ctx.font = "10px monospace";

      const indicators = [
        { label: "L", visible: frame.metadata.leftHandVisible, color: "#06B6D4" },
        { label: "R", visible: frame.metadata.rightHandVisible, color: "#F59E0B" },
        { label: "P", visible: frame.metadata.poseVisible, color: "#10B981" },
        { label: "F", visible: frame.metadata.faceVisible, color: "#8B5CF6" }
      ];

      indicators.forEach((indicator, i) => {
        const x = padding + i * 25;
        ctx.fillStyle = indicator.visible ? indicator.color : "#666666";
        ctx.fillText(indicator.label, x, statusY);
      });
    }

    // Draw pose (body skeleton)
    if (frame.pose && frame.pose.length > 0) {
      const poseLandmarks = toLandmarkList(frame.pose);

      if (showConnections) {
        drawConnectors(ctx, poseLandmarks, POSE_CONNECTIONS, {
          color: "#10B981",
          lineWidth: 3, // Increased from 2 to 3 for better visibility
        });
      }

      drawLandmarks(ctx, poseLandmarks, {
        color: "#10B981",
        fillColor: "#10B981",
        radius: 4, // Increased from 3 to 4
      });
    }

    // Draw left hand (cyan)
    if (frame.hands.left && frame.hands.left.length > 0) {
      const leftHandLandmarks = toLandmarkList(frame.hands.left);

      if (showConnections) {
        drawConnectors(ctx, leftHandLandmarks, HAND_CONNECTIONS, {
          color: "#06B6D4",
          lineWidth: 3, // Increased from 2 to 3 for better visibility
        });
      }

      drawLandmarks(ctx, leftHandLandmarks, {
        color: "#06B6D4",
        fillColor: "#06B6D4",
        radius: 5, // Increased from 4 to 5 for better visibility
      });
    }

    // Draw right hand (orange/yellow)
    if (frame.hands.right && frame.hands.right.length > 0) {
      const rightHandLandmarks = toLandmarkList(frame.hands.right);

      if (showConnections) {
        drawConnectors(ctx, rightHandLandmarks, HAND_CONNECTIONS, {
          color: "#F59E0B",
          lineWidth: 3, // Increased from 2 to 3 for better visibility
        });
      }

      drawLandmarks(ctx, rightHandLandmarks, {
        color: "#F59E0B",
        fillColor: "#F59E0B",
        radius: 5, // Increased from 4 to 5 for better visibility
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
