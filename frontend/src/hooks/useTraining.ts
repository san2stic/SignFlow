import { useCallback } from "react";

import { deployTrainingSession, startTraining, stopTraining } from "../api/training";
import { useTrainingStore } from "../stores/trainingStore";

export function useTraining() {
  const setSession = useTrainingStore((state) => state.setSession);

  const startFewShot = useCallback(async (signId: string, minDeployAccuracy = 0.85) => {
    const session = await startTraining({
      sign_id: signId,
      mode: "few-shot",
      config: { epochs: 50, learning_rate: 1e-4, augmentation: true, min_deploy_accuracy: minDeployAccuracy }
    });
    setSession(session.id);
    return session;
  }, [setSession]);

  const stop = useCallback(async (sessionId: string) => {
    await stopTraining(sessionId);
  }, []);

  const deploy = useCallback(async (sessionId: string) => {
    return deployTrainingSession(sessionId);
  }, []);

  return { startFewShot, stop, deploy };
}
