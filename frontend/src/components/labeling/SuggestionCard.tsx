import { SuggestedVideo } from "../../api/videos";

interface SuggestionCardProps {
  video: SuggestedVideo;
  isSelected: boolean;
  onToggleSelect: (videoId: string) => void;
}

export function SuggestionCard({ video, isSelected, onToggleSelect }: SuggestionCardProps): JSX.Element {
  const formatDuration = (ms: number): string => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
  };

  const similarityPercentage = Math.round(video.similarity_score * 100);

  return (
    <div
      className={`relative cursor-pointer rounded-lg border-2 transition-all ${
        isSelected ? "border-primary ring-2 ring-primary/20" : "border-slate-700 hover:border-slate-600"
      }`}
      onClick={() => onToggleSelect(video.id)}
    >
      {/* Similarity badge */}
      <div className="absolute top-2 right-2 z-10 px-2 py-1 bg-primary/90 text-white text-xs font-semibold rounded">
        {similarityPercentage}% match
      </div>

      {/* Thumbnail */}
      <div className="aspect-video bg-slate-800 rounded-t-lg overflow-hidden">
        {video.thumbnail_path ? (
          <img
            src={`http://localhost:8000${video.thumbnail_path}`}
            alt="Video thumbnail"
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="flex items-center justify-center h-full text-slate-500">
            <span className="text-4xl">🎥</span>
          </div>
        )}
      </div>

      {/* Metadata */}
      <div className="p-3 space-y-1">
        <div className="flex items-center justify-between text-sm">
          <span className="text-slate-400">Duration</span>
          <span className="text-slate-200">{formatDuration(video.duration_ms)}</span>
        </div>
      </div>

      {/* Selection indicator */}
      {isSelected && (
        <div className="absolute inset-0 bg-primary/10 rounded-lg flex items-center justify-center">
          <div className="w-12 h-12 bg-primary rounded-full flex items-center justify-center text-white text-2xl">
            ✓
          </div>
        </div>
      )}
    </div>
  );
}
