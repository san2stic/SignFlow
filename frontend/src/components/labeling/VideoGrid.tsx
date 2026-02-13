import { useMemo, useState } from "react";
import { UnlabeledVideo } from "../../api/videos";
import { VideoCard } from "./VideoCard";

interface VideoGridProps {
  videos: UnlabeledVideo[];
  selectedVideo: UnlabeledVideo | null;
  onSelectVideo: (video: UnlabeledVideo) => void;
  isRefreshing?: boolean;
}

type SortOption = "date-desc" | "date-asc" | "duration-desc" | "duration-asc";

export function VideoGrid({
  videos,
  selectedVideo,
  onSelectVideo,
  isRefreshing = false,
}: VideoGridProps): JSX.Element {
  const [sortBy, setSortBy] = useState<SortOption>("date-desc");

  const sortedVideos = useMemo(() => {
    const sorted = [...videos];

    switch (sortBy) {
      case "date-desc":
        return sorted.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
      case "date-asc":
        return sorted.sort((a, b) => new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
      case "duration-desc":
        return sorted.sort((a, b) => b.duration_ms - a.duration_ms);
      case "duration-asc":
        return sorted.sort((a, b) => a.duration_ms - b.duration_ms);
      default:
        return sorted;
    }
  }, [videos, sortBy]);

  if (videos.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[400px] text-center">
        <div className="text-6xl mb-4">🎉</div>
        <h3 className="text-xl font-semibold text-slate-300 mb-2">All videos labeled!</h3>
        <p className="text-slate-400">Great work! All training videos have been labeled.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4 relative">
      {/* Loading overlay */}
      {isRefreshing && (
        <div className="absolute inset-0 bg-slate-900/50 backdrop-blur-sm z-10 flex items-center justify-center rounded-lg">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-2"></div>
            <p className="text-sm text-slate-400">Refreshing...</p>
          </div>
        </div>
      )}

      {/* Header with sorting */}
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-200">
          Unlabeled Videos ({videos.length})
        </h2>
        <div className="flex items-center gap-2">
          <label htmlFor="sort" className="text-sm text-slate-400">
            Sort by:
          </label>
          <select
            id="sort"
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as SortOption)}
            className="bg-slate-800 border border-slate-700 text-slate-200 text-sm rounded px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-primary"
          >
            <option value="date-desc">Newest first</option>
            <option value="date-asc">Oldest first</option>
            <option value="duration-desc">Longest first</option>
            <option value="duration-asc">Shortest first</option>
          </select>
        </div>
      </div>

      {/* Video grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {sortedVideos.map((video) => (
          <VideoCard
            key={video.id}
            video={video}
            isSelected={selectedVideo?.id === video.id}
            onSelect={onSelectVideo}
          />
        ))}
      </div>
    </div>
  );
}
