import { motion } from "framer-motion";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import { useTrainingStore } from "../../stores/trainingStore";
import { DeploymentReadinessBadge } from "./DeploymentReadinessBadge";
import { RecommendedActionMessage } from "./RecommendedActionMessage";

export function TrainingProgress(): JSX.Element {
  const progress = useTrainingStore((state) => state.progress);
  const chartData = progress.metrics_history ?? [];
  const deploymentAccuracy =
    progress.final_val_accuracy !== null && progress.final_val_accuracy !== undefined
      ? progress.final_val_accuracy
      : progress.metrics.val_accuracy;

  return (
    <div className="card space-y-4 p-4">
      <h3 className="font-heading text-lg">Training Progress</h3>
      <div className="h-3 w-full rounded-full bg-slate-700">
        <motion.div
          className="h-3 rounded-full bg-secondary"
          animate={{ width: `${progress.progress}%` }}
          transition={{ duration: 0.3 }}
        />
      </div>
      <p className="font-mono text-sm">{progress.progress.toFixed(1)}%</p>
      <div className="grid grid-cols-3 gap-2 text-xs text-slate-300 sm:text-sm">
        <p>Loss: {progress.metrics.loss.toFixed(3)}</p>
        <p>Acc: {(progress.metrics.accuracy * 100).toFixed(1)}%</p>
        <p>Val: {(progress.metrics.val_accuracy * 100).toFixed(1)}%</p>
      </div>
      <div className="grid grid-cols-2 gap-1 text-xs text-slate-300 sm:text-sm">
        <p>Epoch: {progress.metrics.current_epoch ?? 0}</p>
        <p>Threshold: {Math.round((progress.deploy_threshold ?? 0.85) * 100)}%</p>
      </div>

      {chartData.length > 0 && (
        <div className="rounded-btn border border-slate-700 bg-slate-900/40 p-3">
          <p className="mb-2 text-xs uppercase tracking-wide text-slate-400">Training metrics</p>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={chartData}>
              <XAxis dataKey="epoch" tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <YAxis domain={[0, 1]} tick={{ fill: "#94a3b8", fontSize: 11 }} />
              <Tooltip
                contentStyle={{ backgroundColor: "#0f172a", border: "1px solid #334155", borderRadius: 8 }}
                labelStyle={{ color: "#94a3b8" }}
              />
              <Line type="monotone" dataKey="loss" name="Loss" stroke="#f59e0b" strokeWidth={2} dot={false} />
              <Line
                type="monotone"
                dataKey="val_accuracy"
                name="Val accuracy"
                stroke="#10b981"
                strokeWidth={2}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <DeploymentReadinessBadge
        ready={Boolean(progress.deployment_ready)}
        accuracy={deploymentAccuracy}
        threshold={progress.deploy_threshold ?? 0.85}
      />
      <RecommendedActionMessage action={progress.recommended_next_action ?? "wait"} />
      <div className="text-xs text-slate-400">
        {progress.status === "completed" ? "Training complete. You can move to validation." : "Live updates every epoch."}
      </div>
    </div>
  );
}
