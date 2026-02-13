import { useEffect, useMemo, useState } from "react";
import { useLocation } from "react-router-dom";

import { useCamera } from "../hooks/useCamera";
import { TrainingWizard } from "../components/training/TrainingWizard";
import { TabsWithContext, Tab } from "../components/common/Tabs";
import { useLabelingStore } from "../stores/labelingStore";
import { VideoGrid } from "../components/labeling/VideoGrid";
import { LabelingModal } from "../components/labeling/LabelingModal";
import { SignSelector } from "../components/labeling/SignSelector";
import { SuggestionView } from "../components/labeling/SuggestionView";
import { Sign } from "../api/signs";

interface TrainPageLocationState {
  assignedSign?: {
    id: string;
    name: string;
    trainingSampleCount?: number;
    videoCount?: number;
  };
}

export function TrainPage(): JSX.Element {
  const location = useLocation();
  const { videoRef, attachVideoRef } = useCamera();
  const [activeTab, setActiveTab] = useState<"record" | "label">("record");
  const {
    unlabeledVideos,
    selectedVideo,
    selectVideo,
    labelVideo,
    loadUnlabeledVideos,
    recentSigns,
    addToRecentSigns,
    suggestions,
    applySuggestions,
    clearSuggestions,
    isLoading,
    error,
  } = useLabelingStore();
  const [selectedSignForLabeling, setSelectedSignForLabeling] = useState<string | null>(null);

  const initialAssignedSign = useMemo(() => {
    const state = location.state as TrainPageLocationState | null;
    if (!state?.assignedSign?.id || !state.assignedSign.name) {
      return undefined;
    }
    return state.assignedSign;
  }, [location.state]);

  useEffect(() => {
    loadUnlabeledVideos();
  }, [loadUnlabeledVideos]);

  return (
    <div className="container mx-auto px-4 py-6">
      <TabsWithContext activeTab={activeTab} onTabChange={(tab) => setActiveTab(tab as "record" | "label")}>
        <Tab value="record">Record</Tab>
        <Tab value="label" badge={unlabeledVideos.length}>Label Videos</Tab>
      </TabsWithContext>

      <div className="mt-6">
        {activeTab === "record" && (
          <TrainingWizard videoRef={videoRef} cameraRef={attachVideoRef} initialAssignedSign={initialAssignedSign} />
        )}
        {activeTab === "label" && (
          <>
            {isLoading && unlabeledVideos.length === 0 && (
              <div className="flex items-center justify-center min-h-[400px]">
                <div className="text-center">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto mb-4"></div>
                  <p className="text-slate-400">Loading videos...</p>
                </div>
              </div>
            )}

            {error && (
              <div className="p-4 bg-red-900/20 border border-red-500/30 rounded text-red-300 mb-4">
                <p className="font-semibold mb-1">Error loading videos</p>
                <p className="text-sm">{error}</p>
                <button
                  onClick={() => loadUnlabeledVideos()}
                  className="mt-2 text-sm underline hover:no-underline"
                >
                  Try again
                </button>
              </div>
            )}

            {!isLoading && !error && (
              <VideoGrid
                videos={unlabeledVideos}
                selectedVideo={selectedVideo}
                onSelectVideo={selectVideo}
                isRefreshing={isLoading && unlabeledVideos.length > 0}
              />
            )}
          </>
        )}
      </div>

      <LabelingModal
        video={selectedVideo}
        onClose={() => selectVideo(null)}
        onLabel={labelVideo}
        onSignSelect={setSelectedSignForLabeling}
        selectedSignId={selectedSignForLabeling}
      >
        <SignSelector
          recentSigns={recentSigns}
          selectedSignId={selectedSignForLabeling}
          onSelectSign={setSelectedSignForLabeling}
          onSignCreated={(signId, signName) => {
            setSelectedSignForLabeling(signId);
            addToRecentSigns({ id: signId, name: signName } as Sign);
          }}
        />
      </LabelingModal>

      {suggestions.length > 0 && selectedSignForLabeling && (
        <SuggestionView
          suggestions={suggestions}
          signName={
            recentSigns.find((s) => s.id === selectedSignForLabeling)?.name || "this sign"
          }
          onApply={async (videoIds) => {
            await applySuggestions(videoIds, selectedSignForLabeling);
            clearSuggestions();
          }}
          onSkip={() => clearSuggestions()}
        />
      )}
    </div>
  );
}
