import { useState } from "react";
import { SuggestedVideo } from "../../api/videos";
import { SuggestionCard } from "./SuggestionCard";

interface SuggestionViewProps {
  suggestions: SuggestedVideo[];
  signName: string;
  onApply: (videoIds: string[]) => void;
  onSkip: () => void;
}

export function SuggestionView({ suggestions, signName, onApply, onSkip }: SuggestionViewProps): JSX.Element {
  const [selectedVideoIds, setSelectedVideoIds] = useState<Set<string>>(
    new Set(suggestions.map((v) => v.id))
  );
  const [isApplying, setIsApplying] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleToggleSelect = (videoId: string) => {
    setSelectedVideoIds((prev) => {
      const next = new Set(prev);
      if (next.has(videoId)) {
        next.delete(videoId);
      } else {
        next.add(videoId);
      }
      return next;
    });
  };

  const handleApply = async () => {
    if (selectedVideoIds.size === 0) return;
    setIsApplying(true);
    setError(null);
    try {
      await onApply(Array.from(selectedVideoIds));
    } catch (err) {
      console.error("Failed to apply suggestions:", err);
      setError("Failed to apply labels. Please try again.");
      setIsApplying(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-slate-900 rounded-lg shadow-2xl max-w-6xl w-full mx-4 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="p-6 border-b border-slate-700">
          <h2 className="text-2xl font-semibold text-slate-100 mb-2">
            Smart Labeling Suggestions
          </h2>
          <p className="text-slate-400">
            We found {suggestions.length} similar video{suggestions.length !== 1 ? "s" : ""} that may also be <span className="text-primary font-medium">{signName}</span>.
            Click to select/deselect videos to label.
          </p>
        </div>

        {/* Suggestions grid */}
        <div className="p-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {suggestions.map((video) => (
              <SuggestionCard
                key={video.id}
                video={video}
                isSelected={selectedVideoIds.has(video.id)}
                onToggleSelect={handleToggleSelect}
              />
            ))}
          </div>
        </div>

        {/* Error message */}
        {error && (
          <div className="mx-6 mb-4 p-3 bg-red-900/20 border border-red-500/30 rounded text-red-300 text-sm">
            {error}
          </div>
        )}

        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t border-slate-700">
          <div className="text-sm text-slate-400">
            {selectedVideoIds.size} video{selectedVideoIds.size !== 1 ? "s" : ""} selected
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={onSkip}
              className="px-4 py-2 text-sm font-medium text-slate-300 hover:text-slate-100 transition-colors"
              disabled={isApplying}
            >
              Skip
            </button>
            <button
              onClick={handleApply}
              disabled={selectedVideoIds.size === 0 || isApplying}
              className="px-4 py-2 text-sm font-medium bg-primary text-white rounded hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {isApplying ? "Applying..." : `Apply Labels (${selectedVideoIds.size})`}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
