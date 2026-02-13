import { forwardRef } from "react";

interface CameraFeedProps {
  className?: string;
}

export const CameraFeed = forwardRef<HTMLVideoElement, CameraFeedProps>(function CameraFeed(
  { className },
  ref
): JSX.Element {
  return (
    <video
      ref={ref}
      autoPlay
      muted
      playsInline
      aria-label="Camera feed"
      className={`h-full w-full rounded-card bg-black object-cover ${className ?? ""}`}
    />
  );
});
