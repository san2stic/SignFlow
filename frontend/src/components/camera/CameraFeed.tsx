import { forwardRef } from "react";

interface CameraFeedProps {
  className?: string;
  fit?: "cover" | "contain";
}

export const CameraFeed = forwardRef<HTMLVideoElement, CameraFeedProps>(function CameraFeed(
  { className, fit = "cover" },
  ref
): JSX.Element {
  const fitClass = fit === "contain" ? "object-contain" : "object-cover";

  return (
    <video
      ref={ref}
      autoPlay
      muted
      playsInline
      aria-label="Camera feed"
      className={`h-full w-full rounded-card bg-black ${fitClass} ${className ?? ""}`}
    />
  );
});
