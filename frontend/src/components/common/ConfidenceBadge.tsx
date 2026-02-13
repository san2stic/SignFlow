import { motion } from "framer-motion";

interface ConfidenceBadgeProps {
  confidence: number;
}

export function ConfidenceBadge({ confidence }: ConfidenceBadgeProps): JSX.Element {
  const pct = Math.round(confidence * 100);

  // Determine color and gradient based on confidence level
  const getConfidenceStyle = (percentage: number) => {
    if (percentage >= 80) {
      return {
        gradient: "from-accent via-accent-light to-accent",
        textColor: "text-accent",
        glowColor: "shadow-[0_0_20px_rgba(16,185,129,0.4)]",
        label: "EXCELLENT"
      };
    } else if (percentage >= 60) {
      return {
        gradient: "from-primary via-primary-light to-primary",
        textColor: "text-primary",
        glowColor: "shadow-[0_0_20px_rgba(14,165,233,0.4)]",
        label: "BON"
      };
    } else if (percentage >= 40) {
      return {
        gradient: "from-yellow-500 via-yellow-400 to-yellow-500",
        textColor: "text-yellow-400",
        glowColor: "shadow-[0_0_20px_rgba(234,179,8,0.4)]",
        label: "MOYEN"
      };
    } else {
      return {
        gradient: "from-red-500 via-red-400 to-red-500",
        textColor: "text-red-400",
        glowColor: "shadow-[0_0_20px_rgba(239,68,68,0.4)]",
        label: "FAIBLE"
      };
    }
  };

  const style = getConfidenceStyle(pct);

  return (
    <div className="relative space-y-3">
      {/* Percentage display with glow effect */}
      <div className="flex items-baseline justify-between">
        <p className="text-xs font-medium uppercase tracking-wider text-text-tertiary">Confiance</p>
        <motion.p
          key={pct}
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className={`font-display text-2xl font-bold ${style.textColor}`}
        >
          {pct}
          <span className="text-sm font-medium">%</span>
        </motion.p>
      </div>

      {/* Progress bar with gradient and animation */}
      <div className="relative h-3 overflow-hidden rounded-full bg-surface-secondary/80">
        {/* Background shimmer effect */}
        <div className="absolute inset-0 shimmer opacity-30" />

        {/* Animated progress fill */}
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className={`relative h-full rounded-full bg-gradient-to-r ${style.gradient} ${style.glowColor}`}
        >
          {/* Inner glow effect */}
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent" />
        </motion.div>

        {/* Animated pulse indicator at the end of the bar */}
        {pct > 5 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="absolute top-1/2 -translate-y-1/2"
            style={{ left: `${pct}%` }}
          >
            <div className={`h-5 w-5 -translate-x-1/2 rounded-full ${style.textColor} animate-pulse`}>
              <div className="absolute inset-0 rounded-full bg-current opacity-60 blur-sm" />
              <div className="absolute inset-1 rounded-full bg-current" />
            </div>
          </motion.div>
        )}
      </div>

      {/* Confidence label */}
      <div className="flex items-center justify-between">
        <motion.span
          key={style.label}
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          className={`rounded-full px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${style.textColor} bg-current/10`}
        >
          {style.label}
        </motion.span>

        {/* Visual indicator dots */}
        <div className="flex gap-1">
          {[1, 2, 3, 4, 5].map((level) => (
            <motion.div
              key={level}
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              transition={{ delay: level * 0.05 }}
              className={`h-1.5 w-1.5 rounded-full ${
                pct >= level * 20
                  ? `${style.textColor.replace('text-', 'bg-')} shadow-glow`
                  : 'bg-surface-tertiary'
              }`}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
