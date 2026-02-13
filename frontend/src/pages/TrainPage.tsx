import { useEffect, useMemo, useState } from "react";
import { useLocation } from "react-router-dom";

import { useCamera } from "../hooks/useCamera";
import { TrainingWizard } from "../components/training/TrainingWizard";
import { TabsWithContext, Tab } from "../components/common/Tabs";
import { useLabelingStore } from "../stores/labelingStore";
import { VideoGrid } from "../components/labeling/VideoGrid";
import { LabelingModal } from "../components/labeling/LabelingModal";

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
  const { unlabeledVideos, selectedVideo, selectVideo, labelVideo, loadUnlabeledVideos } = useLabelingStore();
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
          <VideoGrid
            videos={unlabeledVideos}
            selectedVideo={selectedVideo}
            onSelectVideo={selectVideo}
          />
        )}
      </div>

      <LabelingModal
        video={selectedVideo}
        onClose={() => selectVideo(null)}
        onLabel={labelVideo}
        onSignSelect={setSelectedSignForLabeling}
        selectedSignId={selectedSignForLabeling}
      >
        {/* Temporary placeholder for SignSelector (Task 10) */}
        <div className="p-4 bg-slate-800 rounded border border-slate-700 text-slate-400 text-center">
          SignSelector component will go here (Task 10)
        </div>
      </LabelingModal>
    </div>
  );
}
