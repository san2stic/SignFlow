export type RecommendedAction = "deploy" | "collect_more_examples" | "wait" | "review_error";

interface RecommendedActionMessageProps {
  action: RecommendedAction;
}

export function RecommendedActionMessage({ action }: RecommendedActionMessageProps): JSX.Element {
  const message =
    action === "deploy"
      ? { prefix: "OK", text: "Model ready - validate and deploy.", color: "text-secondary" }
      : action === "collect_more_examples"
        ? { prefix: "WARN", text: "Add 3-5 more clips to improve accuracy.", color: "text-accent" }
        : action === "review_error"
          ? { prefix: "ERR", text: "Training failed - review logs.", color: "text-red-400" }
          : { prefix: "WAIT", text: "Training in progress...", color: "text-slate-300" };

  return (
    <p className={`text-xs sm:text-sm ${message.color}`}>
      <span className="mr-1 font-semibold uppercase">{message.prefix}</span>
      {message.text}
    </p>
  );
}
