interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

export function SearchBar({ value, onChange, placeholder }: SearchBarProps): JSX.Element {
  return (
    <input
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className="w-full rounded-btn border border-slate-700 bg-slate-900/60 px-3 py-3 text-base"
      placeholder={placeholder ?? "Search"}
      aria-label="Search"
    />
  );
}
