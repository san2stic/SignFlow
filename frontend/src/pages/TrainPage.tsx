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
    <section className="space-y-5">
      <header className="card p-5">
        <p className="text-xs uppercase tracking-[0.16em] text-text-tertiary">Entrainement</p>
        <h1 className="mt-2 font-display text-2xl font-semibold text-white">Capture et labeling video</h1>
        <p className="mt-1 text-sm text-text-secondary">
          Enregistrez de nouveaux exemples et attribuez rapidement les clips non labels.
        </p>
      </header>

      <TabsWithContext
        activeTab={activeTab}
        onTabChange={(tab) => setActiveTab(tab as "record" | "label")}
        className="card p-2"
      >
        <Tab value="record">Enregistrer</Tab>
        <Tab value="label" badge={unlabeledVideos.length}>
          Labeling
        </Tab>
      </TabsWithContext>

      <div>
        {activeTab === "record" && (
          <TrainingWizard videoRef={videoRef} cameraRef={attachVideoRef} initialAssignedSign={initialAssignedSign} />
        )}
        {activeTab === "label" && (
          <>
            {isLoading && unlabeledVideos.length === 0 && (
              <div className="card flex min-h-[360px] items-center justify-center">
                <div className="text-center">
                  <div className="mx-auto mb-4 h-12 w-12 animate-spin rounded-full border-b-2 border-primary"></div>
                  <p className="text-sm text-slate-300">Chargement des videos...</p>
                </div>
              </div>
            )}

            {error && (
              <div className="mb-4 rounded-btn border border-red-500/30 bg-red-900/20 p-4 text-red-300">
                <p className="mb-1 font-semibold">Erreur de chargement</p>
                <p className="text-sm">{error}</p>
                <button
                  onClick={() => loadUnlabeledVideos()}
                  className="mt-2 text-sm underline hover:no-underline"
                >
                  Reessayer
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
            recentSigns.find((s) => s.id === selectedSignForLabeling)?.name || "ce signe"
          }
          onApply={async (videoIds) => {
            await applySuggestions(videoIds, selectedSignForLabeling);
            clearSuggestions();
          }}
          onSkip={() => clearSuggestions()}
        />
      )}
    </section>
  );
}
