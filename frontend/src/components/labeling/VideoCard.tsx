import { UnlabeledVideo } from "../../api/videos";

interface VideoCardProps {
  video: UnlabeledVideo;
  isSelected: boolean;
  onSelect: (video: UnlabeledVideo) => void;
}

export function VideoCard({ video, isSelected, onSelect }: VideoCardProps): JSX.Element {
  const formatDuration = (ms: number): string => {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
  };

  return (
    <div
      className={`relative cursor-pointer rounded-lg border-2 transition-all ${
        isSelected ? "border-primary ring-2 ring-primary/20" : "border-slate-700 hover:border-slate-600"
      }`}
      onClick={() => onSelect(video)}
    >
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
      <div className="p-3 space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-slate-400">Duration</span>
          <span className="text-slate-200">{formatDuration(video.duration_ms)}</span>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-slate-400">Resolution</span>
          <span className="text-slate-200">{video.resolution}</span>
        </div>
        <div className="flex items-center justify-between text-sm">
          <span className="text-slate-400">Landmarks</span>
          <span className={video.landmarks_extracted ? "text-green-400" : "text-yellow-400"}>
            {video.landmarks_extracted ? "✓ Extracted" : "⏳ Pending"}
          </span>
        </div>
      </div>
    </div>
  );
}
