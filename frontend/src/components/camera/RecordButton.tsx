interface RecordButtonProps {
  isRecording: boolean;
  onClick: () => void;
}

export function RecordButton({ isRecording, onClick }: RecordButtonProps): JSX.Element {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`touch-btn rounded-full font-bold ${
        isRecording ? "bg-red-600 text-white" : "bg-accent text-slate-950"
      }`}
      aria-label={isRecording ? "Stop recording" : "Start recording"}
    >
      {isRecording ? "Stop" : "REC"}
    </button>
  );
}
