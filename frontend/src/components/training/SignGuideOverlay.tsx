export function SignGuideOverlay(): JSX.Element {
  return (
    <svg
      className="pointer-events-none absolute inset-0 h-full w-full opacity-35"
      viewBox="0 0 100 100"
      preserveAspectRatio="xMidYMid meet"
      aria-hidden
    >
      <ellipse cx="50" cy="40" rx="15" ry="25" fill="none" stroke="white" strokeWidth="0.7" />
      <circle cx="30" cy="50" r="10" fill="none" stroke="#06b6d4" strokeWidth="0.7" />
      <circle cx="70" cy="50" r="10" fill="none" stroke="#f59e0b" strokeWidth="0.7" />
      <line x1="50" y1="28" x2="50" y2="72" stroke="white" strokeWidth="0.4" opacity="0.6" />
      <line x1="30" y1="50" x2="70" y2="50" stroke="white" strokeWidth="0.4" opacity="0.6" />
    </svg>
  );
}
