import { useEffect } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useFeedbackStore } from "../../stores/feedbackStore";

const AUTO_DISMISS_MS = 4000;

/**
 * Toast global pour les événements feedback (réentraînement déclenché, etc.).
 * Position : coin bas-droit. Auto-dismiss en 4 secondes.
 */
export function FeedbackToast(): JSX.Element {
  const toast = useFeedbackStore((s) => s.toast);
  const clearToast = useFeedbackStore((s) => s.clearToast);

  useEffect(() => {
    if (!toast) return;
    const tid = window.setTimeout(clearToast, AUTO_DISMISS_MS);
    return () => window.clearTimeout(tid);
  }, [toast, clearToast]);

  const colorMap: Record<"success" | "error" | "info", string> = {
    success:
      "border-accent/40 bg-accent/15 text-accent shadow-[0_4px_24px_rgba(16,185,129,0.25)]",
    error:
      "border-red-400/40 bg-red-500/15 text-red-300 shadow-[0_4px_24px_rgba(239,68,68,0.25)]",
    info: "border-primary/40 bg-primary/15 text-primary shadow-[0_4px_24px_rgba(14,165,233,0.25)]",
  };

  const iconMap: Record<"success" | "error" | "info", JSX.Element> = {
    success: (
      <svg
        className="h-5 w-5 shrink-0"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={2}
        aria-hidden="true"
      >
        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
      </svg>
    ),
    error: (
      <svg
        className="h-5 w-5 shrink-0"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={2}
        aria-hidden="true"
      >
        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
      </svg>
    ),
    info: (
      <svg
        className="h-5 w-5 shrink-0"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={2}
        aria-hidden="true"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
        />
      </svg>
    ),
  };

  return (
    <div
      className="pointer-events-none fixed bottom-6 right-6 z-[80] flex flex-col items-end gap-2"
      aria-live="polite"
    >
      <AnimatePresence>
        {toast && (
          <motion.div
            key="feedback-toast"
            role="status"
            initial={{ opacity: 0, x: 40, scale: 0.9 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 40, scale: 0.9 }}
            transition={{ type: "spring", stiffness: 340, damping: 28 }}
            className={`pointer-events-auto flex max-w-sm items-center gap-3 rounded-2xl border px-4 py-3 text-sm font-medium backdrop-blur-md ${colorMap[toast.type]}`}
          >
            {iconMap[toast.type]}
            <span>{toast.message}</span>
            <button
              type="button"
              onClick={clearToast}
              aria-label="Fermer la notification"
              className="ml-2 rounded-full p-0.5 opacity-70 transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-current"
            >
              <svg
                className="h-4 w-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
                aria-hidden="true"
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
