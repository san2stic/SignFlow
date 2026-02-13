import { useEffect, useRef, useState } from "react";
import { UnlabeledVideo } from "../../api/videos";
import { apiBaseUrl } from "../../api/client";

interface LabelingModalProps {
  video: UnlabeledVideo | null;
  onClose: () => void;
  onLabel: (videoId: string, signId: string) => void;
  onSignSelect: (signId: string | null) => void;
  selectedSignId: string | null;
  children: React.ReactNode;
}

export function LabelingModal({
  video,
  onClose,
  onLabel,
  onSignSelect,
  selectedSignId,
  children,
}: LabelingModalProps): JSX.Element | null {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (!video) return;
    onSignSelect(null);
    setIsSubmitting(false);
  }, [video, onSignSelect]);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [onClose]);

  if (!video) return null;

  const handleSubmit = async () => {
    if (!selectedSignId) return;
    setIsSubmitting(true);
    try {
      await onLabel(video.id, selectedSignId);
      onClose();
    } catch (error) {
      console.error("Failed to label video:", error);
      setIsSubmitting(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="bg-slate-900 rounded-lg shadow-2xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-700">
          <h2 className="text-xl font-semibold text-slate-100">Label Video</h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 transition-colors"
            aria-label="Close modal"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Video player */}
          <div className="aspect-video bg-slate-800 rounded-lg overflow-hidden">
            <video
              ref={videoRef}
              src={`${apiBaseUrl()}${video.file_path}`}
              controls
              loop
              className="w-full h-full"
            />
          </div>

          {/* Video metadata */}
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-slate-400">Duration:</span>
              <span className="ml-2 text-slate-200">
                {Math.floor(video.duration_ms / 1000)}s
              </span>
            </div>
            <div>
              <span className="text-slate-400">Resolution:</span>
              <span className="ml-2 text-slate-200">{video.resolution}</span>
            </div>
            <div>
              <span className="text-slate-400">FPS:</span>
              <span className="ml-2 text-slate-200">{video.fps}</span>
            </div>
          </div>

          {/* Sign selector (passed as children) */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Select Sign
            </label>
            {children}
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-end gap-3 p-4 border-t border-slate-700">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-slate-100 transition-colors"
            disabled={isSubmitting}
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            disabled={!selectedSignId || isSubmitting}
            className="px-4 py-2 text-sm font-medium bg-primary text-white rounded hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isSubmitting ? "Labeling..." : "Label Video"}
          </button>
        </div>
      </div>
    </div>
  );
}
